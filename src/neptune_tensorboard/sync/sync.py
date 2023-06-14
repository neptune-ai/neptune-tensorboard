from __future__ import print_function

import os
import traceback

import neptune
from future.moves import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    import tbparse
except ModuleNotFoundError:
    raise ModuleNotFoundError("neptune-tensorboard: require `tbparse` for exporting logs (pip install tbparse)")

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
        timestamp_and_hostname = file_name[len(_EVENTS_FILE_PREFIX) :]
        separator_index = timestamp_and_hostname.find(".")
        if separator_index >= 0:
            return timestamp_and_hostname[(separator_index + 1) :]
        else:
            return None
    else:
        return None


class DataSync(object):
    def __init__(self, project, path):
        self._project = project
        self._path = path

    def run(self):
        for root, _, run_files in os.walk(self._path):
            for run_file in run_files:
                try:
                    self._export_to_neptune_run(os.path.join(root, run_file))
                except Exception as e:
                    print("Cannot load run from file '{}'. ".format(run_file) + str(e), file=sys.stderr)
                    try:
                        traceback.print_exc(e)
                    except:  # noqa: E722
                        pass

    def _is_valid_tf_event_file(self, path):
        accumulator = EventAccumulator(path)
        accumulator.Reload()
        try:
            accumulator.FirstEventTimestamp()
        except ValueError:
            return False
        return True

    def _export_to_neptune_run(self, path):
        if not self._is_valid_tf_event_file(path):
            return

        run_path = os.path.relpath(path, self._path)

        run = neptune.init_run(project=self._project)
        run["tensorboard_path"] = run_path

        namespace_handler = run["tensorboard"]

        reader = tbparse.SummaryReader(path)

        # Read scalars
        for scalar in reader.scalars.itertuples():
            namespace_handler["scalar"][scalar.tag].append(scalar.value)

        # Read images (and figures)
        for image in reader.images.itertuples():
            namespace_handler["image"][image.tag].append(neptune.types.File.as_image(image.value))

        # Read text
        for text in reader.text.itertuples():
            namespace_handler["text"][text.tag].append(text.value)

        # Read hparams
        for hparam in reader.hparams.itertuples():
            namespace_handler["hparams"][hparam.tag].append(hparam.value)
