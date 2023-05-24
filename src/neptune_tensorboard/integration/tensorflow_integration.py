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
from pkg_resources import parse_version

IS_GRAPHLIB_AVAILABLE = find_spec("tfgraphviz")
if IS_GRAPHLIB_AVAILABLE:
    import tfgraphviz as tfg

_integrated_with_tensorflow = False


def patch_tensorflow(run, base_namespace):
    global _integrated_with_tensorflow

    if _integrated_with_tensorflow:
        return
    _integrate_with_tensorflow(run, base_namespace)
    _integrated_with_tensorflow = True


def _integrate_with_tensorflow(run, base_namespace):
    try:
        version = "<unknown>"

        # noinspection PyUnresolvedReferences
        version = parse_version(tf.version.VERSION)

        if version >= parse_version("2.0.0-rc0"):
            return _patch_tensorflow_2x(run, base_namespace)
    except AttributeError:
        message = "Unrecognized tensorflow version: {}. Please make sure " "that the tensorflow version is >=2.0"
        raise Exception(message.format(version))


def _patch_tensorflow_2x(run, base_namespace):
    _scalar = tf.summary.scalar
    _text = tf.summary.text
    _image = tf.summary.image
    _graph = tf.summary.graph

    @wraps(_scalar)
    def scalar(name, data, step=None, description=None):
        if step is None:
            step = tf.summary.experimental.get_step()
        run[base_namespace]["scalar"][name].append(data)
        _scalar(name, data, step, description)

    @wraps(_image)
    def image(name, data, step=None, max_outputs=3, description=None):
        if step is None:
            step = tf.summary.experimental.get_step()
        # expecting 2 or 3 dimensional tensor. If tensor is 4-dimentional,
        # as in https://www.tensorflow.org/api_docs/python/tf/summary/image
        # iterate over first dimension to send all images
        shape = tf.shape(data)
        if len(shape) >= 4:
            for num in range(0, shape[0]):
                run[base_namespace]["image"][name].append(File.as_image(data[num]))
        else:
            run[base_namespace]["image"][name] = File.as_image(data)
        _image(name, data, step, max_outputs, description)

    @wraps(_text)
    def text(name, data, step=None, description=None):
        run[base_namespace]["text"][name] = data
        _text(name, data, step, description)

    @wraps(_graph)
    def graph(graph_data):
        if IS_GRAPHLIB_AVAILABLE:
            graph = tfg.board(graph_data)
            png_bytes = graph.pipe(format="png")
            # There is only one graph
            run[base_namespace]["graph"].upload(File.from_content(png_bytes, extension="png"))
        else:
            warnings.warn("Skipping model visualization because no tfgraphviz installation was found.")
        _graph(graph_data)

    tf.summary.scalar = scalar
    tf.summary.image = image
    tf.summary.text = text
    tf.summary.graph = graph
