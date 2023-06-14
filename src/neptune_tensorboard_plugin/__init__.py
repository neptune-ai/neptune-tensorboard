import os

import click


@click.command("tensorboard")
@click.option("--project", "-p", help="Project name")
@click.argument("path", required=True)
def sync(project, path):
    # We do not want to import anything if process was executed for autocompletion purposes.
    from neptune_tensorboard.sync import DataSync

    print("EXPORTER", project, path)
    if not os.path.exists(path):
        click.echo("ERROR: Provided path doesn't exist", err=True)
        return

    loader = DataSync(project, path)
    loader.run()
    return
    # return run_sync(project=project, path=path)
