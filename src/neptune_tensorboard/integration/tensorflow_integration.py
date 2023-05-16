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

import tensorflow as tf
from pkg_resources import parse_version

_integrated_with_tensorflow = False


def enable_tensorboard_logging(run, *, base_namespace="tensorboard"):
    global _integrated_with_tensorflow

    if _integrated_with_tensorflow:
        return
    _integrate_with_tensorflow(run, base_namespace)
    _integrated_with_tensorflow = True


def _integrate_with_tensorflow(run, base_namespace):
    try:
        version = "<unknown>"

        # noinspection PyUnresolvedReferences
        version = parse_version(tf.version.VERSION)

        if version >= parse_version("2.0.0-rc0"):
            return _patch_tensorflow_2x(run, base_namespace)
    except AttributeError:
        message = (
            "Unrecognized tensorflow version: {}. Please consider "
            "upgrading your neptune and neptune-tensorboard libraries"
        )
        raise Exception(message.format(version))


def _patch_tensorflow_2x(run, base_namespace):
    def get_channel_name(name):
        return name

    _scalar = tf.summary.scalar
    _text = tf.summary.text
    _image = tf.summary.image

    def scalar(name, data, step=None, description=None):
        if step is None:
            step = tf.summary.experimental.get_step()
        run[base_namespace]["scalar"][get_channel_name(name)].append(data)
        _scalar(name, data, step, description)

    def image(name, data, step=None, max_outputs=3, description=None):
        from neptune.types import File

        if step is None:
            step = tf.summary.experimental.get_step()
        # expecting 2 or 3 dimensional tensor. If tensor is 4-dimentional,
        # as in https://www.tensorflow.org/api_docs/python/tf/summary/image
        # iterate over first dimension to send all images
        shape = tf.shape(data)
        if len(shape) >= 4:
            for num in range(0, shape[0]):
                run[base_namespace]["image"][get_channel_name(name)].append(File.as_image(data[num]))
        else:
            run[base_namespace]["image"][get_channel_name(name)] = File.as_image(data)
        _image(name, data, step, max_outputs, description)

    def text(name, data, step=None, description=None):
        run[base_namespace]["text"][get_channel_name(name)] = data
        _text(name, data, step, description)

    tf.summary.scalar = scalar
    tf.summary._original_no_neptune_scalar = _scalar

    tf.summary.image = image
    tf.summary._original_no_neptune_image = _image

    tf.summary.text = text
    tf.summary._original_no_neptune_text = _text

    # Tensorflow 2.3 renames the internal method from `_log_metrics` to `_log_epoch_metrics`
    # and changes its parameters. The conditional below handles both versions.
    if hasattr(tf.keras.callbacks.TensorBoard, "_log_metrics"):

        _tb_log_metrics = tf.keras.callbacks.TensorBoard._log_metrics

        def _log_metrics(instance, logs, prefix, step):

            for (name, value) in logs.items():
                if name in ("batch", "size", "num_steps"):
                    continue
                run[base_namespace][get_channel_name(name)].append(value)

            _tb_log_metrics(instance, logs, prefix, step)

        tf.keras.callbacks.TensorBoard._log_metrics = _log_metrics

    elif hasattr(tf.keras.callbacks.TensorBoard, "_log_epoch_metrics"):

        _tb_log_epoch_metrics = tf.keras.callbacks.TensorBoard._log_epoch_metrics

        def _log_epoch_metrics(instance, epoch, logs):
            for (name, value) in logs.items():
                if name in ("batch", "size", "num_steps"):
                    continue
                run[base_namespace][get_channel_name(name)].append(value)

            _tb_log_epoch_metrics(instance, epoch, logs)

        tf.keras.callbacks.TensorBoard._log_epoch_metrics = _log_epoch_metrics

    else:
        raise AttributeError("Unsupported Tensorflow version")
