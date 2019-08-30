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
            self._send_numeric_value(value.tag, x, value.simple_value)
        elif field == 'image':
            self._send_image(value.tag, x, value.image.encoded_image_string)
        elif field == 'tensor' and value.tensor.dtype == tf.string:
            string_values = []
            for _ in range(0, len(value.tensor.string_val)):
                string_value = value.tensor.string_val.pop()
                string_values.append(string_value.decode("utf-8"))
                self._send_text(value.tag, x, ", ".join(string_values))

    def _send_numeric_value(self, value_tag, x, simple_value):
        self._experiment_holder().send_metric(channel_name=value_tag,
                                              x=x,
                                              y=simple_value)

    def _send_image(self, image_tag, x, encoded_image):
        image_desc = "({}. Step {})".format(image_tag, x)
        self._experiment_holder().send_image(channel_name=image_tag,
                                             x=x,
                                             y=Image.open(io.BytesIO(encoded_image)),
                                             name=image_desc,
                                             description=image_desc)

    def _send_text(self, value_tag, x, text):
        self._experiment_holder().send_text(channel_name=value_tag,
                                            x=x,
                                            y=text)

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

    # pylint: disable=no-member, protected-access, no-name-in-module, import-error
    _add_summary_method = tf.summary.FileWriter.add_summary

    def _neptune_add_summary(summary_writer, summary, global_step=None):
        tensorflow_integrator.add_summary(summary, global_step)
        _add_summary_method(summary_writer, summary, global_step=global_step)

    tf.summary.FileWriter.add_summary = _neptune_add_summary

    return tensorflow_integrator
