import hashlib
import os
import shutil
import uuid

import neptune
import torch
from tensorboardX.writer import SummaryWriter

from neptune_tensorboard.sync.sync_impl import DataSync


def test_exporter():
    log_dir = str(uuid.uuid4())
    writer = SummaryWriter(log_dir=log_dir)

    writer.add_scalar("tensorboardX_scalar", 0.5)
    writer.add_image("zero", torch.zeros(12, 12, 3), dataformats="HWC")
    writer.add_images("zeros", torch.zeros(4, 12, 12, 3), dataformats="NHWC")
    writer.add_text("my_text", "Hello World")
    writer.add_text("my_text", "Hello World 2")

    writer.flush()
    writer.close()

    DataSync(project=None, path=log_dir).run()

    for fname in os.listdir(log_dir):
        path = os.path.join(log_dir, fname)
        hash_run_id = hashlib.md5(path.encode()).hexdigest()
        break

    with neptune.init_project() as project:
        runs_df = project.fetch_runs_table().to_pandas()
        custom_run_id_map = dict(zip(runs_df["sys/custom_run_id"], runs_df["sys/id"]))
        run_id = custom_run_id_map[hash_run_id]

    with neptune.init_run(with_id=run_id) as run:
        assert run.exists("tensorboard_path")
        assert run.exists("tensorboard/image")
        assert run.exists("tensorboard/scalar")
        assert run.exists("tensorboard/text")

    shutil.rmtree(log_dir)
