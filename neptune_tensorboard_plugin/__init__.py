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

import click


@click.command('tensorboard')
@click.option('--project', '-p', help='Project name')
@click.argument('path', required=True)
def sync(project, path):
    """Import TensorFlow event files to Neptune\n
    PATH is a directory where Neptune will look to find TensorFlow event files that it can import.
    Neptune will recursively walk the directory structure rooted at logdir, looking for .*tfevents.* files.

    Examples:

        neptune tensorboard .

        neptune tensorboard /path

        neptune tensorboard /path --project username/sandbox

    """

    # suppress numpy deprecation warnings
    # tensorflow uses deprecated api
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # We do not want to import anything if process was executed for autocompletion purposes.
    from neptune_tensorboard.sync.sync import sync as run_sync
    return run_sync(project=project, path=path)
