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
import re
import traceback

import click
import tensorflow as tf
from future.moves import sys
from tensorboard.backend.event_processing.event_accumulator import COMPRESSED_HISTOGRAMS, IMAGES, \
    AUDIO, SCALARS, HISTOGRAMS, TENSORS

from neptune_tensorboard.integration.tensorflow_integration import TensorflowIntegrator
from neptune_tensorboard.sync.internal.events_loader import NeptuneEventAccumulator
from neptune_tensorboard.sync.internal.path_parser import parse_path_to_experiment_name, parse_path_to_hostname


class TensorflowDataSync(object):

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
                    except:  # pylint: disable=bare-except
                        pass

    def _load_single_run(self, path):
        click.echo("Loading {}...".format(path))
        run_path = os.path.relpath(path, self._path)
        run_id = re.sub(r'[^0-9A-Za-z_\-]', '_', run_path).lower()
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
        accumulator = NeptuneEventAccumulator(path, size_guidance={
            COMPRESSED_HISTOGRAMS: 0,
            IMAGES: experiment.limits['channels']['image'],
            AUDIO: 0,
            SCALARS: experiment.limits['channels']['numeric'],
            HISTOGRAMS: 0,
            TENSORS: experiment.limits['channels']['text']
        })

        accumulator.Reload()

        tags = accumulator.Tags()

        # load scalars
        for tag in tags[SCALARS]:
            for event in accumulator.Scalars(tag):
                tf_integrator.send_numeric(
                    tag=tag,
                    step=event.step,
                    value=event.value,
                    wall_time=event.wall_time)

        # load images
        for tag in tags[IMAGES]:
            for event in accumulator.Images(tag):
                tf_integrator.send_image(
                    tag=tag,
                    step=event.step,
                    encoded_image_string=event.encoded_image_string,
                    wall_time=event.wall_time)

        # load tensors (actually only strings, see: NeptuneEventAccumulator._ProcessTensor)
        for tag in tags[TENSORS]:
            for event in accumulator.Tensors(tag):
                string_values = []
                for _ in range(0, len(event.tensor_proto.string_val)):
                    string_value = event.tensor_proto.string_val.pop()
                    try:
                        string_values.append(tf.compat.as_text(string_value))
                    except UnicodeDecodeError:
                        # ignore invalid strings
                        pass

                tf_integrator.send_text(
                    tag=tag,
                    step=event.step,
                    text=', '.join(string_values),
                    wall_time=event.wall_time)
