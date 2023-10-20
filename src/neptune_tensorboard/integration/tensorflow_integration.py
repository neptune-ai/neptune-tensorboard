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
import contextlib
import warnings
from importlib.util import find_spec

import tensorflow as tf
from neptune.types import File

from neptune_tensorboard.integration.utils import register_pre_hook

IS_GRAPHLIB_AVAILABLE = find_spec("tfgraphviz")
if IS_GRAPHLIB_AVAILABLE:
    import tfgraphviz as tfg

_integrated_with_tensorflow = False

__all__ = ["patch_tensorflow", "NeptuneTensorflowTracker"]


def patch_tensorflow(run, base_namespace):
    global _integrated_with_tensorflow

    if not _integrated_with_tensorflow:
        NeptuneTensorflowTracker(run, base_namespace)
        _integrated_with_tensorflow = True


def track_scalar(name, data, step=None, description=None, run=None, base_namespace=None):
    run[base_namespace]["scalar"][name].append(data)


def track_image(name, data, step=None, run=None, base_namespace=None, description=None):
    # If number of images (tf.shape(data)[0]) > 1, append images as FileSeries, else upload as an image.
    # ref: https://www.tensorflow.org/api_docs/python/tf/summary/image
    k = tf.shape(data)[0]
    if k > 1:
        for num in range(k):
            run[base_namespace]["image"][name].append(File.as_image(data[num]), description=description)
    else:
        if description:
            warnings.warn(f"neptune-tensorboard: Uploading single image ({name}). Description will be ignored.")
        run[base_namespace]["image"][name] = File.as_image(data[0])


def track_text(name, data, step=None, description=None, run=None, base_namespace=None):
    run[base_namespace]["text"][name] = data


def track_graph(graph_data, run=None, base_namespace=None):
    if IS_GRAPHLIB_AVAILABLE:
        graph = tfg.board(graph_data)
        png_bytes = graph.pipe(format="png")
        # There is only one graph
        run[base_namespace]["graph"].upload(File.from_content(png_bytes, extension="png"))
    else:
        # user facing
        warnings.warn("neptune-tensorboard: Skipping model visualization because no tfgraphviz installation was found.")


class NeptuneTensorflowTracker(contextlib.AbstractContextManager):
    def __init__(self, run, base_namespace):
        self.org_scalar = tf.summary.scalar
        self.org_image = tf.summary.image
        self.org_text = tf.summary.text
        self.org_graph = tf.summary.graph

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.summary.scalar = self.org_scalar
        tf.summary.image = self.org_image
        tf.summary.text = self.org_text
        tf.summary.graph = self.org_graph
