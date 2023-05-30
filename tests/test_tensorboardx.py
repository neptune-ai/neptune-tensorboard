import matplotlib.pyplot as plt
import neptune
import torch
from tensorboardX.writer import SummaryWriter

import neptune_tensorboard


def test_tensorboardx():
    with neptune.Run() as run:

        neptune_tensorboard.enable_tensorboard_logging(run)

        writer = SummaryWriter()

        writer.add_scalar("batch_loss", 0.5)
        writer.add_scalar("batch_loss", 0.4)
        writer.add_image("zero", torch.zeros(12, 12, 3), dataformats="HWC")
        writer.add_images("zeros", torch.zeros(4, 12, 12, 3), dataformats="NHWC")
        writer.add_hparams(hparam_dict={"batch_size": "32", "lr": 0.1}, metric_dict={"acc": 100, "loss": 0.001})

        data = torch.randn(2, 100)

        figure, ax = plt.subplots(2, 2, figsize=(5, 5))
        ax[0, 0].hist(data[0])
        ax[1, 0].scatter(data[0], data[1])
        ax[0, 1].plot(data[0], data[1])
        writer.add_figure("my_figure", figure)

        writer.add_text("my_text", "Hello World")
        writer.add_text("my_text", "Hello World 2")

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return torch.nn.functional.sigmoid(x)

        writer.add_graph(MyModel(), torch.randn(3, 3))

        run.sync()

        assert run.exists("tensorboard/scalar/batch_loss")
        assert run.exists("tensorboard/image/zero")
        assert run.exists("tensorboard/images/zeros")
        assert run.exists("tensorboard/hparams/batch_size")
        assert run.exists("tensorboard/hparams/lr")
        assert run.exists("tensorboard/metrics/acc")
        assert run.exists("tensorboard/metrics/loss")
        assert run.exists("tensorboard/figure/my_figure")
        assert run.exists("tensorboard/text/my_text")
