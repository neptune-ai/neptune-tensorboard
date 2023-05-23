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

from importlib.util import find_spec

from neptune_tensorboard.integration.tensorflow_integration import patch_tensorflow
from neptune_tensorboard.integration.version import __version__


def enable_tensorboard_logging(run, *, base_namespace="tensorboard"):
    if find_spec("tensorflow"):
        patch_tensorflow(run, base_namespace=base_namespace)
