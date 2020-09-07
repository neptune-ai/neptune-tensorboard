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
import neptune


def integrate_with_tensorflow(prefix=False):
    from neptune_tensorboard.integration.tensorflow_integration import integrate_with_tensorflow as integration
    integration(neptune.get_experiment, prefix)


def integrate_with_keras():
    msg = """integrate_with_keras() was removed in neptune-tensorboard > 0.3.14.
    You should use NeptuneMonitor() callback from neptunecontrib.monitoring.keras instead.
    Read about it here https://docs.neptune.ai/integrations/keras.html"""
    raise Exception(msg)
