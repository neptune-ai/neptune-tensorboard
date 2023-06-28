import hashlib
import pathlib
import traceback

import click
import neptune
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    import tbparse
except ModuleNotFoundError:
    # user facing
    raise ModuleNotFoundError(
        "neptune-tensorboard: tbparse is required for exporting logs. Install it with: pip install tbparse"
    )


def compute_md5_hash(path):
    return hashlib.md5(path.encode()).hexdigest()


class DataSync:
    def __init__(self, project, api_token, path):
        self._project = project
        self._api_token = api_token
        self._path = path

    def run(self):
        # NOTE: Fetching custom_run_ids is not a trivial operation, so
        #       we cache the custom_run_ids here.
        self._existing_custom_run_ids = self._get_existing_neptune_custom_run_ids()
        # Inspect if files correspond to EventFiles.
        for path in pathlib.Path(self._path).glob("**/*tfevents*"):
            try:
                # methods below expect path to be str.
                str_path = str(path)

                # only try export for valid files i.e. files which EventAccumulator
                # can actually read.
                if self._is_valid_tf_event_file(str_path):
                    self._export_to_neptune_run(str_path)
            except Exception as e:
                # user facing
                click.echo("Cannot load run from file '{}'. ".format(path) + "Error: " + str(e))
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

    def _get_existing_neptune_custom_run_ids(self):
        with neptune.init_project(project=self._project, api_token=self._api_token) as project:
            try:
                return {run_id for run_id in project.fetch_runs_table().to_pandas()["sys/custom_run_id"].to_list()}
            except KeyError:
                # empty project
                return set()

    def _experiment_exists(self, hash_run_id, run_path):
        return hash_run_id in self._existing_custom_run_ids

    def _export_to_neptune_run(self, path):
        # custom_run_id supports str with max length of 32.
        hash_run_id = compute_md5_hash(path)

        if self._experiment_exists(hash_run_id, self._project):
            # user facing
            click.echo(f"{path} was already synchronized")
            return

        with neptune.init_run(
            custom_run_id=hash_run_id, project=self._project, api_token=self._api_token, capture_hardware_metrics=False
        ) as run:
            run["tensorboard_path"] = path

            namespace_handler = run["tensorboard"]

            # parse events file
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

            # user facing
            click.echo(f"{path} was exported with run_id: {hash_run_id}")
