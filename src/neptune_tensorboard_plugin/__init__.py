import os

import click


@click.command("tensorboard")
@click.option("--project", help="Project name")
@click.argument("log_dir", required=True)
def sync(project, log_dir):
    if not os.path.exists(log_dir):
        click.echo("ERROR: Provided `log_dir` path doesn't exist", err=True)
        return

    # We do not want to import anything if process was executed for autocompletion purposes.
    from neptune_tensorboard.sync import DataSync

    DataSync(project, log_dir).run()
