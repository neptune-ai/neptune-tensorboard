#
# Copyright (c) 2019, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import print_function

import os
import random
import re
import traceback

import click
import tensorflow as tf
from future.moves import collections, sys

from neptune_tensorboard.integration.tensorflow_integration import TensorflowIntegrator
from neptune_tensorboard.sync.internal.path_parser import parse_path_to_experiment_name, parse_path_to_hostname


class TensorflowDataSync(object):
    _RECORD = collections.namedtuple('Record', 'x value')

    def __init__(self, project, path):
        self._project = project
        self._path = path

    def run(self):
        for root, _, run_files in os.walk(self._path):
            for run_file in run_files:
                try:
                    self._load_single_run(os.path.join(root, run_file))
                except Exception as e:
                    print("Cannot load run from file '{}'. ".format(run_file) + str(e), file=sys.stderr)
                    try:
                        traceback.print_exc(e)
                    except: # pylint: disable=bare-except
                        pass

    def _load_single_run(self, path):
        click.echo("Loading {}...".format(path))
        run_path = os.path.relpath(path, self._path)
        run_id = re.sub(r'[^0-9A-Za-z_\-]', '_', run_path)
        exp_name = parse_path_to_experiment_name(run_path)
        hostname = parse_path_to_hostname(run_path)
        if not self._experiment_exists(run_id, exp_name):
            with self._project.create_experiment(name=exp_name,
                                                 properties={
                                                     'tf/run/path': run_path
                                                 },
                                                 tags=[run_id],
                                                 upload_source_files=[],
                                                 abort_callback=lambda *args: None,
                                                 upload_stdout=False,
                                                 upload_stderr=False,
                                                 send_hardware_metrics=False,
                                                 run_monitoring_thread=False,
                                                 handle_uncaught_exceptions=True,
                                                 hostname=hostname or None) as exp:
                tf_integrator = TensorflowIntegrator(lambda *args: exp)
                self._load_single_file(exp, path, tf_integrator)
            click.echo("{} was saved as {}".format(run_path, exp.id))
        else:
            click.echo("{} is already synced".format(run_path))

    def _experiment_exists(self, run_id, run_path):
        existing_experiments = self._project.get_experiments(tag=run_id)
        return any(exp.name == run_path and exp.state == 'succeeded' for exp in existing_experiments)

    @staticmethod
    def _load_single_file(experiment, path, tf_integrator):
        tag_buckets = {}
        for record in tf.train.summary_iterator(path):
            if record.graph_def != "":
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(record.graph_def)
                tf_integrator.add_graph_def(graph_def, path)
            TensorflowDataSync._apply_limit(experiment, record.step, record.summary, tag_buckets)

        for tag in tag_buckets:
            for record in tag_buckets[tag]:
                tf_integrator.add_value(record.x, record.value)

    @staticmethod
    def _apply_limit(experiment, step, summary, tag_buckets):
        for value in summary.value:
            if value.tag not in tag_buckets:
                tag_buckets[value.tag] = []

            bucket = tag_buckets[value.tag]
            bucket.append(TensorflowDataSync._RECORD(step, value))

            field = value.WhichOneof('value')
            if field == 'simple_value':
                channel_type = 'numeric'
            elif field == 'image':
                channel_type = 'image'
            elif field == 'tensor' and value.tensor.dtype == tf.string:
                channel_type = 'text'
            else:
                continue

            if len(bucket) > experiment.limits['channels'][channel_type]:
                if channel_type == 'numeric':
                    del bucket[random.randint(1, len(bucket) - 2)]
                else:
                    bucket.pop(0)
