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
from importlib.util import find_spec

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

IS_PYT_AVAILABLE = find_spec("torch")
IS_TENSORBOARDX_AVAILABLE = find_spec("tensorboardX")

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


if IS_PYT_AVAILABLE:
    import torch

    from neptune_tensorboard.integration.pytorch_integration import patch_pytorch

    def integrate_with_pytorch(run, base_namespace):
        version = "<unknown>"
        try:
            # noinspection PyUnresolvedReferences
            version = parse_version(torch.__version__)

            if version >= parse_version("1.9.0"):
                patch_pytorch(run, base_namespace)
        except AttributeError:
            message = "Unrecognized PyTorch version: {}. Please make sure " "that the PyTorch version is >=1.9.0"
            raise Exception(message.format(version))


if IS_TENSORBOARDX_AVAILABLE:
    import tensorboardX

    from neptune_tensorboard.integration.tensorboardx_integration import patch_tensorboardx

    def integrate_with_tensorboardx(run, base_namespace):
        version = "<unknown>"
        try:
            # noinspection PyUnresolvedReferences
            version = parse_version(tensorboardX.__version__)

            if version >= parse_version("2.2.0"):
                patch_tensorboardx(run, base_namespace)
        except AttributeError:
            message = "Unrecognized tensorboardX version: {}. Please make sure " "that the PyTorch version is >=2.2.0"
            raise Exception(message.format(version))


def enable_tensorboard_logging(run, *, base_namespace="tensorboard"):
    if IS_TF_AVAILABLE:
        integrate_with_tensorflow(run, base_namespace)
    if IS_PYT_AVAILABLE:
        integrate_with_pytorch(run, base_namespace)
    if IS_TENSORBOARDX_AVAILABLE:
        integrate_with_tensorboardx(run, base_namespace)

    if not (IS_PYT_AVAILABLE or IS_TF_AVAILABLE or IS_TENSORBOARDX_AVAILABLE):
        msg = "neptune-tensorboard: Tensorflow or PyTorch was not found, please ensure that it is available."
        warnings.warn(msg)
