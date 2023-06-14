from __future__ import print_function

import io
import os
import re
import traceback

import neptune
from future.moves import sys
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import (
    IMAGES,
    SCALARS,
    EventAccumulator,
)

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
                    self._load_single_run(os.path.join(root, run_file))
                except Exception as e:
                    print("Cannot load run from file '{}'. ".format(run_file) + str(e), file=sys.stderr)
                    try:
                        traceback.print_exc(e)
                    except:  # noqa: E722
                        pass

    def _does_file_describe_experiment_run(self, path):
        accumulator = EventAccumulator(path)
        accumulator.Reload()
        try:
            accumulator.FirstEventTimestamp()
        except ValueError:
            return False
        return True

    def _load_single_run(self, path):
        run_path = os.path.relpath(path, self._path)
        run_id = re.sub(r"[^0-9A-Za-z_\-]", "_", run_path).lower()

        run = neptune.init_run(with_id=run_id, project=self._project, mode="debug")
        accumulator = EventAccumulator(path)
        self._load_single_file(accumulator, run["tensorboard"])
        # if not self._experiment_exists(run_id, exp_name):
        #     if not self._does_file_describe_experiment_run(path):
        #         return
        #     with self._project.create_experiment(
        #         name=exp_name,
        #         properties={"tf/run/path": run_path},
        #         tags=[run_id],
        #         upload_source_files=[],
        #         abort_callback=lambda *args: None,
        #         upload_stdout=False,
        #         upload_stderr=False,
        #         send_hardware_metrics=False,
        #         run_monitoring_thread=False,
        #         handle_uncaught_exceptions=True,
        #         hostname=hostname or None,
        #     ) as exp:
        #         click.echo("Loading {}...".format(path))
        #         accumulator = self._new_accumulator(path, exp)
        #         tf_integrator = TensorflowIntegrator(False, lambda *args: exp)
        #         self._load_single_file(accumulator, tf_integrator)
        #     click.echo("{} was saved as {}".format(run_path, exp.id))
        # else:
        #     click.echo("{} is already synced".format(run_path))

    @staticmethod
    def _load_single_file(accumulator, run):

        accumulator.Reload()

        tags = accumulator.Tags()

        print(tags)
        # load scalars
        for tag in tags[SCALARS]:
            for event in accumulator.Scalars(tag):
                run["scalar"][tag].append(event.value)

        # load images (corresponds to image, images, figure)
        for tag in tags[IMAGES]:
            for event in accumulator.Images(tag):
                img = Image.open(io.BytesIO(event.encoded_image_string))
                run["image"][tag].append(img)

        # load tensors (text is stored as tensor with string dtype)
        # for tag in tags[TENSORS]:
        #     for event in accumulator.Tensors(tag):
        #         if event.tensor_proto.dtype == tf.string:
        #             run["text"][tag] = event.tensor_proto.string_val
