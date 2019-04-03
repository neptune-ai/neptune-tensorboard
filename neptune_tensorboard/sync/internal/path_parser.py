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

_EVENTS_FILE_PREFIX = "events.out.tfevents."


def parse_path_to_experiment_name(path):
    """
    Parses a relative path to tensorflow event file to a reasonable neptune-compatible name

    Arguments:
        path(str): a path relative to entrypoint directory

    Returns: a string representing sanitized project name based on the path, or "untitled-tensorboard" if name cannot
        be determined
    """
    experiment_name = os.path.dirname(path)
    if experiment_name:
        return experiment_name
    else:
        return "untitled-tensorboard"


def parse_path_to_hostname(path):
    """
    Parses a relative path to tensorflow event file to a hostname

    Arguments:
        path(str): a path relative to entrypoint directory

    Returns: a hostname or None if the file name did not match tensorflow events file
    """
    file_name = os.path.basename(path)
    if file_name.startswith(_EVENTS_FILE_PREFIX):
        timestamp_and_hostname = file_name[len(_EVENTS_FILE_PREFIX):]
        separator_index = timestamp_and_hostname.find('.')
        if separator_index >= 0:
            return timestamp_and_hostname[(separator_index + 1):]
        else:
            return None
    else:
        return None
