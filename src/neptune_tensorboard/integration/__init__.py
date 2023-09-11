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
from contextlib import (
    contextmanager,
    nullcontext,
)
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
    MIN_TF_VERSION = "2.0.0-rc0"
    from neptune_tensorboard.integration.tensorflow_integration import (
        NeptuneTensorflowTracker,
        patch_tensorflow,
    )

    def check_tf_version():
        version = "<unknown>"
        try:
            # noinspection PyUnresolvedReferences
            version = parse_version(tf.version.VERSION)

            if version >= parse_version(MIN_TF_VERSION):
                return
        except AttributeError:
            # user facing
            message = (
                f"Unrecognized TensorFlow version: {version}. Please make sure "
                "that the TensorFlow version is >={MIN_TF_VERSION}"
            )
            raise Exception(message)


if IS_PYT_AVAILABLE:
    MIN_PT_VERSION = "1.9.0"
    import torch

    from neptune_tensorboard.integration.pytorch_integration import (
        NeptunePytorchTracker,
        patch_pytorch,
    )

    def check_pytorch_version():
        version = "<unknown>"
        try:
            # noinspection PyUnresolvedReferences
            version = parse_version(torch.__version__)

            if version >= parse_version(MIN_PT_VERSION):
                return
        except AttributeError:
            # user facing
            message = (
                f"Unrecognized PyTorch version: {version}. Please make sure "
                "that the PyTorch version is >={MIN_PT_VERSION}"
            )
            raise Exception(message)


if IS_TENSORBOARDX_AVAILABLE:
    import tensorboardX

    MIN_TBX_VERSION = "2.2.0"
    from neptune_tensorboard.integration.tensorboardx_integration import (
        NeptuneTensorboardXTracker,
        patch_tensorboardx,
    )

    def check_tensorboardx_version():
        version = "<unknown>"
        try:
            # noinspection PyUnresolvedReferences
            version = parse_version(tensorboardX.__version__)

            if version >= parse_version(MIN_TBX_VERSION):
                return
        except AttributeError:
            # user facing
            message = (
                f"Unrecognized tensorboardX version: {version}. Please make sure "
                "that the tensorboardX version is >={MIN_TBX_VERSION}"
            )
            raise Exception(message)


# user facing
FRAMEWORK_NOT_FOUND_WARNING_MSG = (
    "neptune-tensorboard: TensorFlow or PyTorch or tensorboardX was not found, ",
    "please ensure that it is available.",
)


def enable_tensorboard_logging(run, *, base_namespace="tensorboard"):
    """Logs the tracked metadata to both the tensorboard directory and a Neptune run.

    Args:
        run: An existing run reference, as returned by `neptune.init_run()`.
        base_namespace: Namespace under which all metadata logged by the integration will be stored.

    Example:
        >>> import neptune
        >>> from neptune_tensorboard import enable_tensorboard_logging
        >>> run = neptune.init_run()
        >>> enable_tensorboard_logging(run)

    Learn more in the Neptune docs:
    - Integration guide: https://docs.neptune.ai/integrations/tensorboard/
    - API reference: https://docs.neptune.ai/api/integrations/tensorboard/
    """
    if IS_TF_AVAILABLE:
        check_tf_version()
        patch_tensorflow(run, base_namespace)
    if IS_PYT_AVAILABLE:
        check_pytorch_version()
        patch_pytorch(run, base_namespace)
    if IS_TENSORBOARDX_AVAILABLE:
        check_tensorboardx_version()
        patch_tensorboardx(run, base_namespace)

    if not (IS_PYT_AVAILABLE or IS_TF_AVAILABLE or IS_TENSORBOARDX_AVAILABLE):
        warnings.warn(FRAMEWORK_NOT_FOUND_WARNING_MSG)


@contextmanager
def enable_tensorboard_logging_ctx(run, *, base_namespace="tensorboard"):
    tf_tracker, pt_tracker, tbx_tracker = nullcontext(), nullcontext(), nullcontext()
    if IS_TF_AVAILABLE:
        check_tf_version()
        tf_tracker = NeptuneTensorflowTracker(run, base_namespace)

    if IS_PYT_AVAILABLE:
        check_pytorch_version()
        pt_tracker = NeptunePytorchTracker(run, base_namespace)

    if IS_TENSORBOARDX_AVAILABLE:
        check_tensorboardx_version()
        tbx_tracker = NeptuneTensorboardXTracker(run, base_namespace)

    if not (IS_PYT_AVAILABLE or IS_TF_AVAILABLE or IS_TENSORBOARDX_AVAILABLE):
        warnings.warn(FRAMEWORK_NOT_FOUND_WARNING_MSG)

    with pt_tracker, tf_tracker, tbx_tracker:
        yield
