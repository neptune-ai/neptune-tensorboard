import contextlib
import warnings
from functools import partial

import torch
from neptune.types import File
from neptune.utils import stringify_unsupported
from torch.utils.tensorboard.writer import SummaryWriter

from neptune_tensorboard.integration.utils import register_pre_hook

IS_TORCHVIZ_AVAILABLE = True
try:
    import torchviz
except ImportError:
    IS_TORCHVIZ_AVAILABLE = False

__all__ = ["patch_pytorch", "NeptunePytorchTracker"]

_integrated_with_pytorch = False


def patch_pytorch(run, base_namespace):
    global _integrated_with_pytorch

    if not _integrated_with_pytorch:
        NeptunePytorchTracker(run, base_namespace)
        _integrated_with_pytorch = True


def track_scalar(
    summary_writer,
    tag,
    scalar_value,
    global_step=None,
    walltime=None,
    new_style=False,
    double_precision=False,
    run=None,
    base_namespace=None,
):
    run[base_namespace]["scalar"][tag].append(scalar_value)


def track_image(
    summary_writer, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW", run=None, base_namespace=None
):
    if not isinstance(img_tensor, torch.Tensor):
        img_tensor = torch.tensor(img_tensor)
    # dataformats : CHW, HWC, HW, WH
    if dataformats == "CHW":
        # convert to HWC
        img_tensor = img_tensor.movedim(0, 2)
    elif dataformats == "WH":
        # convert to HW1
        img_tensor = img_tensor.movedim(0, 1).unsqueeze(2)
    elif dataformats == "HW":
        # convert to HW1
        img_tensor = img_tensor.unsqueeze(2)

    run[base_namespace]["image"][tag] = File.as_image(img_tensor)


def track_images(
    summary_writer, tag, img_tensor, global_step=None, walltime=None, dataformats="NCHW", run=None, base_namespace=None
):
    if not isinstance(img_tensor, torch.Tensor):
        img_tensor = torch.tensor(img_tensor)
    # dataformats : CHW, HWC, HW, WH
    if dataformats == "NCHW":
        # convert to HWC
        img_tensor = img_tensor.movedim(1, 3)
    elif dataformats == "NHWC":
        pass
    else:
        # user facing
        warnings.warn("neptune-tensorboard: Skipping logging images as  {dataformats} is not supported.")

    for idx in range(img_tensor.shape[0]):
        run[base_namespace]["images"][tag].append(File.as_image(img_tensor[idx]))


def track_figure(
    summary_writer, tag, figure, global_step=None, close=True, walltime=None, run=None, base_namespace=None
):
    run[base_namespace]["figure"][tag].append(figure)


def track_text(summary_writer, tag, text_string, global_step=None, walltime=None, run=None, base_namespace=None):
    run[base_namespace]["text"][tag].append(text_string)


def track_graph(
    summary_writer, model, input_to_model=None, verbose=False, use_strict_trace=True, run=None, base_namespace=None
):
    if not IS_TORCHVIZ_AVAILABLE:
        # user facing
        msg = "neptune-tensorboard: Skipping model visualization because no torchviz installation was found."
        warnings.warn(msg)
        return
    if input_to_model is None:
        # user facing
        msg = "neptune-tensorboard: Skipping model visualization because input_to_model was None."
        warnings.warn(msg)
        return
    output = model(input_to_model)
    graph = torchviz.make_dot(output, params=dict(model.named_parameters()))
    png_bytes = graph.pipe(format="png")
    run[base_namespace]["graph"].upload(File.from_content(png_bytes, extension="png"))


def track_hparam(
    summary_writer, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None, run=None, base_namespace=None
):
    run[base_namespace]["hparams"] = stringify_unsupported(hparam_dict)
    run[base_namespace]["metrics"] = metric_dict


class NeptunePytorchTracker(contextlib.AbstractContextManager):
    def __init__(self, run, base_namespace):
        self.org_add_scalar = SummaryWriter.add_scalar
        self.org_add_image = SummaryWriter.add_image
        self.org_add_images = SummaryWriter.add_images
        self.org_add_figure = SummaryWriter.add_figure
        self.org_add_text = SummaryWriter.add_text
        self.org_add_graph = SummaryWriter.add_graph
        self.org_add_hparams = SummaryWriter.add_hparams

        register_pre_hook_with_run = partial(register_pre_hook, run=run, base_namespace=base_namespace)

        SummaryWriter.add_scalar = register_pre_hook_with_run(
            original=SummaryWriter.add_scalar, neptune_hook=track_scalar
        )

        SummaryWriter.add_image = register_pre_hook_with_run(original=SummaryWriter.add_image, neptune_hook=track_image)

        SummaryWriter.add_images = register_pre_hook_with_run(
            original=SummaryWriter.add_images, neptune_hook=track_images
        )

        SummaryWriter.add_figure = register_pre_hook_with_run(
            original=SummaryWriter.add_figure, neptune_hook=track_figure
        )

        SummaryWriter.add_text = register_pre_hook_with_run(original=SummaryWriter.add_text, neptune_hook=track_text)

        SummaryWriter.add_graph = register_pre_hook_with_run(original=SummaryWriter.add_graph, neptune_hook=track_graph)

        SummaryWriter.add_hparams = register_pre_hook_with_run(
            original=SummaryWriter.add_hparams, neptune_hook=track_hparam
        )

    def __exit__(self, exc_type, exc_value, traceback):
        SummaryWriter.add_scalar = self.org_add_scalar
        SummaryWriter.add_image = self.org_add_image
        SummaryWriter.add_images = self.org_add_images
        SummaryWriter.add_figure = self.org_add_figure
        SummaryWriter.add_text = self.org_add_text
        SummaryWriter.add_graph = self.org_add_graph
        SummaryWriter.add_hparams = self.org_add_hparams
