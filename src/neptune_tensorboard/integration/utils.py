import os
import weakref

from neptune import Run
from neptune.types import File


def safe_upload_visualization(run: Run, name: str, file_name: str):
    # Function to safely upload a file and
    # delete the file on completion of upload.
    # We utilise the weakref.finalize to remove
    # the file once the stream object goes out-of-scope.

    def remove(file_name):
        os.remove(file_name)
        # Also remove graphviz intermediate file.
        os.remove(file_name.replace(".png", ""))

    with open(file_name, "rb") as f:
        weakref.finalize(f, remove, file_name)
        run[name].upload(File.from_stream(f, extension="png"))
