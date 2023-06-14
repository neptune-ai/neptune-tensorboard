import hashlib
import os
import traceback

import click
import neptune
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    import tbparse
except ModuleNotFoundError:
    raise ModuleNotFoundError("neptune-tensorboard: require `tbparse` for exporting logs (pip install tbparse)")


class DataSync(object):
    def __init__(self, project, path):
        self._project = project
        self._path = path

    def run(self):
        # Inspect if files correspond to EventFiles.
        for root, _, run_files in os.walk(self._path):
            for run_file in run_files:
                try:
                    path = os.path.join(root, run_file)
                    # Skip if not a valid file
                    if not self._is_valid_tf_event_file(path):
                        continue
                    self._export_to_neptune_run(path)
                except Exception as e:
                    click.echo("Cannot load run from file '{}'. ".format(run_file) + "Error: " + str(e))
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

    def _experiment_exists(self, has_run_id, run_path):
        runs_table = neptune.init_project(project=self._project)
        runs_df = runs_table.fetch_runs_table(tag=[has_run_id]).to_pandas()
        if len(runs_df) >= 1:
            return True

        return False

    def _export_to_neptune_run(self, path):
        # custom_run_id supports str with max length of 32.
        hash_run_id = hashlib.md5(path.encode()).hexdigest()

        exists = self._experiment_exists(hash_run_id, self._project)
        if exists:
            click.echo(f"{path} was already synced")
            return

        run = neptune.init_run(custom_run_id=hash_run_id, project=self._project, tags=[hash_run_id])
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

        click.echo(f"{path} was exported with run_id: {hash_run_id}")
