import os

import click


@click.command("tensorboard")
@click.option("--project", help="Neptune Project name")
@click.option("--api_token", help="Neptune API token")
@click.argument("log_dir", required=True)
def sync(project, api_token, log_dir):
    if not os.path.exists(log_dir):
        # user facing
        click.echo("ERROR: Provided `log_dir` path doesn't exist", err=True)
        return

    # We do not want to import anything if process was executed for autocompletion purposes.
    from neptune_tensorboard.sync import DataSync

    DataSync(project, api_token, log_dir).run()
