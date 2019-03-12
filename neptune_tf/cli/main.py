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
import os

import click
import neptune

from neptune_tf.sync import TensorflowDataSync


@click.group()
def main():
    pass


@main.command('sync')
@click.option('--api-token', '-a', help='Neptune Authorization Token')
@click.option('--project', '-p', help='Project name')
@click.argument('path', required=True)
def sync(api_token, project, path):
    neptune.init(api_token=api_token, project_qualified_name=project)

    if not os.path.exists(path):
        click.echo("ERROR: Provided path doesn't exist", err=True)
        return

    loader = TensorflowDataSync(neptune.project, path)
    loader.run()


if __name__ == '__main__':
    main()