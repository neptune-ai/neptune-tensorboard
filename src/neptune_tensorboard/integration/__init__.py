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

__all__ = ["enable_tensorboard_logging", "__version__"]

import warnings

from pkg_resources import parse_version

from neptune_tensorboard.integration.version import __version__

# NOTE: We don't use `importlib.find_spec` here as
#       TF can be installed from multiple packages
#       like tensorflow, tensorflow-macos, etc.
IS_TF_AVAILABLE = True
try:
    import tensorflow as tf  # noqa
except ModuleNotFoundError:
    IS_TF_AVAILABLE = False

if IS_TF_AVAILABLE:
    from neptune_tensorboard.integration.tensorflow_integration import patch_tensorflow

    def integrate_with_tensorflow(run, base_namespace):
        version = "<unknown>"
        try:
            # noinspection PyUnresolvedReferences
            version = parse_version(tf.version.VERSION)

            if version >= parse_version("2.0.0-rc0"):
                patch_tensorflow(run, base_namespace)
        except AttributeError:
            message = "Unrecognized tensorflow version: {}. Please make sure " "that the tensorflow version is >=2.0"
            raise Exception(message.format(version))


def enable_tensorboard_logging(run, *, base_namespace="tensorboard"):
    if IS_TF_AVAILABLE:
        integrate_with_tensorflow(run, base_namespace)
    else:
        msg = "neptune-tensorboard: Tensorflow was not found, please ensure that it is available."
        warnings.warn(msg)
