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
from __future__ import unicode_literals

import io
import os
import time

import tensorflow as tf
from PIL import Image
from future.builtins import object
from neptune.exceptions import NeptuneException
from pkg_resources import parse_version
try:
    from tensorflow_core.core.framework import summary_pb2  # pylint:disable=no-name-in-module
except Exception as ignore:
    from tensorflow.core.framework import summary_pb2  # pylint:disable=no-name-in-module

_integrated_with_tensorflow = False


def integrate_with_tensorflow(experiment_getter):
    global _integrated_with_tensorflow  # pylint:disable=global-statement

    if _integrated_with_tensorflow:
        return
    _integrate_with_tensorflow(experiment_getter)
    _integrated_with_tensorflow = True


class TensorflowIntegrator(object):

    def __init__(self, experiment_getter=None):
        self._experiment_holder = experiment_getter

    def add_summary(self, summary, global_step=None):

        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ

        x = self._calculate_x_value(global_step)
        for value in summary.value:
            try:
                self.add_value(x, value)
            except NeptuneException:
                pass

    def add_value(self, x, value):
        field = value.WhichOneof('value')

        if field == 'simple_value':
            self.send_numeric(tag=value.tag,
                              step=x,
                              value=value.simple_value,
                              wall_time=time.time())
            return

        if field == 'image':
            self.send_image(tag=value.tag,
                            step=x,
                            encoded_image_string=value.image.encoded_image_string,
                            wall_time=time.time())
            return

        if field == 'tensor' and value.tensor.dtype == tf.string:
            string_values = []
            for _ in range(0, len(value.tensor.string_val)):
                string_value = value.tensor.string_val.pop()
                string_values.append(string_value.decode("utf-8"))

            self.send_text(tag=value.tag,
                           step=x,
                           text=", ".join(string_values),
                           wall_time=time.time())
            return

    def send_numeric(self, tag, step, value, wall_time):
        self._experiment_holder().send_metric(channel_name=tag,
                                              x=step,
                                              y=value,
                                              timestamp=wall_time)

    def send_image(self, tag, step, encoded_image_string, wall_time):
        image_desc = "({}. Step {})".format(tag, step)
        self._experiment_holder().send_image(channel_name=tag,
                                             x=step,
                                             y=Image.open(io.BytesIO(encoded_image_string)),
                                             name=image_desc,
                                             description=image_desc,
                                             timestamp=wall_time)

    def send_text(self, tag, step, text, wall_time):
        self._experiment_holder().send_text(channel_name=tag,
                                            x=step,
                                            y=text,
                                            timestamp=wall_time)

    @staticmethod
    def get_writer_name(log_dir):
        return os.path.basename(os.path.normpath(log_dir))

    @staticmethod
    def _calculate_x_value(global_step):
        if global_step is not None:
            return int(global_step)
        else:
            return time.time()


def _integrate_with_tensorflow(experiment_getter):
    tensorflow_integrator = TensorflowIntegrator(experiment_getter=experiment_getter)

    try:
        # noinspection PyUnresolvedReferences
        version = parse_version(tf.version.VERSION)

        # Tensorflow 2.x
        if version >= parse_version('2.0.0-rc0'):
            return _patch_tensorflow_2x(experiment_getter)

        # Tensorflow 1.x
        if version >= parse_version('1.0.0'):
            return _patch_tensorflow_1x(tensorflow_integrator)

    except AttributeError:
        pass

    raise ValueError("Unsupported tensorflow version")


# pylint: disable=no-member, protected-access, no-name-in-module, import-error
def _patch_tensorflow_1x(tensorflow_integrator):
    _add_summary_method = tf.summary.FileWriter.add_summary

    def _neptune_add_summary(summary_writer, summary, global_step=None):
        tensorflow_integrator.add_summary(summary, global_step)
        _add_summary_method(summary_writer, summary, global_step=global_step)

    tf.summary.FileWriter.add_summary = _neptune_add_summary
    tf.summary.FileWriter._original_no_neptune_add_summary = _add_summary_method

    return tensorflow_integrator


def _patch_tensorflow_2x(experiment_getter):
    _scalar = tf.summary.scalar
    _text = tf.summary.text
    _image = tf.summary.image

    def scalar(name, data, step=None, description=None):
        if step is None:
            step = tf.summary.experimental.get_step()
        experiment_getter().log_metric(name, x=step, y=data)
        _scalar(name, data, step, description)

    def image(name, data, step=None, max_outputs=3, description=None):
        if step is None:
            step = tf.summary.experimental.get_step()
        experiment_getter().log_image(name, x=step, y=data, description=description)
        _image(name, data, step, max_outputs, description)

    def text(name, data, step=None, description=None):
        if step is None:
            step = tf.summary.experimental.get_step()
        experiment_getter().log_text(name, x=step, y=data)
        _text(name, data, step, description)

    tf.summary.scalar = scalar
    tf.summary._original_no_neptune_scalar = _scalar

    tf.summary.image = image
    tf.summary._original_no_neptune_image = _image

    tf.summary.text = text
    tf.summary._original_no_neptune_text = _text
