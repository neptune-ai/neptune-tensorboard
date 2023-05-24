#
# Copyright (c) 2019, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import warnings
from functools import wraps
from importlib.util import find_spec

import tensorflow as tf
from neptune.types import File

IS_GRAPHLIB_AVAILABLE = find_spec("tfgraphviz")
if IS_GRAPHLIB_AVAILABLE:
    import tfgraphviz as tfg

_integrated_with_tensorflow = False

__all__ = ["patch_tensorflow"]


def patch_tensorflow(run, base_namespace):
    global _integrated_with_tensorflow

    if not _integrated_with_tensorflow:
        patch_tensorflow_2x(run, base_namespace)
        _integrated_with_tensorflow = True


def track_scalar(name, data, step=None, description=None, run=None, base_namespace=None):
    run[base_namespace]["scalar"][name].append(data)


def track_image(name, data, step=None, run=None, base_namespace=None):
    # expecting 2 or 3 dimensional tensor. If tensor is 4-dimentional,
    # as in https://www.tensorflow.org/api_docs/python/tf/summary/image
    # iterate over first dimension to send all images
    shape = tf.shape(data)
    if len(shape) >= 4:
        for num in range(0, shape[0]):
            run[base_namespace]["image"][name].append(File.as_image(data[num]))
    else:
        run[base_namespace]["image"][name] = File.as_image(data)


def track_text(name, data, step=None, description=None, run=None, base_namespace=None):
    run[base_namespace]["text"][name] = data


def track_graph(graph_data, run=None, base_namespace=None):
    if IS_GRAPHLIB_AVAILABLE:
        graph = tfg.board(graph_data)
        png_bytes = graph.pipe(format="png")
        # There is only one graph
        run[base_namespace]["graph"].upload(File.from_content(png_bytes, extension="png"))
    else:
        warnings.warn("Skipping model visualization because no tfgraphviz installation was found.")


def register_pre_hook(original, neptune_hook, run, base_namespace):
    @wraps(original)
    def wrapper(*args, **kwargs):
        neptune_hook(*args, **kwargs, run=run, base_namespace=base_namespace)
        return original(*args, **kwargs)

    return wrapper


def patch_tensorflow_2x(run, base_namespace):
    tf.summary.scalar = register_pre_hook(
        original=tf.summary.scalar, neptune_hook=track_scalar, run=run, base_namespace=base_namespace
    )
    tf.summary.image = register_pre_hook(
        original=tf.summary.image, neptune_hook=track_image, run=run, base_namespace=base_namespace
    )
    tf.summary.text = register_pre_hook(
        original=tf.summary.text, neptune_hook=track_text, run=run, base_namespace=base_namespace
    )
    tf.summary.graph = register_pre_hook(
        original=tf.summary.graph, neptune_hook=track_graph, run=run, base_namespace=base_namespace
    )
