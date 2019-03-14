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
import unittest

import mock
import tensorflow as tf
from bunch import Bunch

from neptune_tensorboard.sync.tensorflow_data_sync import TensorflowDataSync


class TensorflowDataSyncTest(unittest.TestCase):
    _EXPERIMENT = Bunch()
    _EXPERIMENT.limits = {
        'channels': {
            'numeric': 100,
            'text': 10,
            'image': 10
        }
    }

    def test_limiting_simple_value_summaries(self):
        # GIVEN
        summaries = []
        for i in range(1000):
            summaries.append(self.__create_summary(
                self.__create_simple_value('num', i)
            ))
        # AND
        tag_buckets = {}

        # WHEN
        step = 0
        for summary in summaries:
            # pylint: disable=protected-access
            TensorflowDataSync._apply_limit(self._EXPERIMENT, step, summary, tag_buckets)
            step += 1

        # THEN
        self.assertEqual(len(tag_buckets['num']), 100)
        self.assertEqual(tag_buckets['num'][0].value.simple_value, 0)
        self.assertEqual(tag_buckets['num'][0].x, 0)
        self.assertEqual(tag_buckets['num'][99].value.simple_value, 999)
        self.assertEqual(tag_buckets['num'][99].x, 999)

    def test_limiting_image_summaries(self):
        # GIVEN
        summaries = []
        for i in range(100):
            summaries.append(self.__create_summary(
                self.__create_image('image', "imageContent{}".format(i))
            ))
        # AND
        tag_buckets = {}

        # WHEN
        step = 0
        for summary in summaries:
            # pylint: disable=protected-access
            TensorflowDataSync._apply_limit(self._EXPERIMENT, step, summary, tag_buckets)
            step += 1

        # THEN
        # pylint: disable=line-too-long
        self.assertEqual([i.value.image for i in tag_buckets['image']],
                         ["imageContent{}".format(i) for i in range(90, 100)])
        self.assertEqual([i.x for i in tag_buckets['image']], list(range(90, 100)))

    def test_limiting_text_summaries(self):
        # GIVEN
        summaries = []
        for i in range(100):
            summaries.append(self.__create_summary(
                self.__create_text('text', "[text{}]".format(i))
            ))
        # AND
        tag_buckets = {}

        # WHEN
        step = 0
        for summary in summaries:
            # pylint: disable=protected-access
            TensorflowDataSync._apply_limit(self._EXPERIMENT, step, summary, tag_buckets)
            step += 1

        # THEN
        self.assertEqual([i.value.tensor.string_val for i in tag_buckets['text']],
                         ["[text{}]".format(i) for i in range(90, 100)])
        self.assertEqual([i.x for i in tag_buckets['text']], list(range(90, 100)))

    @staticmethod
    def __create_summary(value):
        summary = mock.MagicMock()
        summary.value = [value]
        return summary

    @staticmethod
    def __create_simple_value(tag, simple_value):
        value = mock.MagicMock()
        value.tag = tag
        value.WhichOneof.return_value = 'simple_value'
        value.simple_value = simple_value
        return value

    @staticmethod
    def __create_image(tag, content):
        value = mock.MagicMock()
        value.tag = tag
        value.WhichOneof.return_value = 'image'
        value.image = content
        return value

    @staticmethod
    def __create_text(tag, text):
        value = mock.MagicMock()
        value.tag = tag
        value.WhichOneof.return_value = 'tensor'
        value.tensor = mock.MagicMock()
        value.tensor.dtype = tf.string
        value.tensor.string_val = text
        return value
