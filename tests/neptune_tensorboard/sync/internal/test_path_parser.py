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

from neptune_tensorboard.sync.internal.path_parser import parse_path_to_experiment_name, parse_path_to_hostname


class PathParserTest(unittest.TestCase):

    def test_parsing_experiment_name(self):
        # GIVEN
        test_cases = [
            (
                "events.out.tfevents.1551714927.jan-kowalski-pascal05-test-4vgt2",
                "untitled-tensorboard"
            ),
            (
                "deepsense/events.out.tfevents.1551714927.jan-kowalski-pascal05-test-4vgt2",
                "deepsense"
            ),
            (
                "deepsense/events.out.tfevents.1551719327.adam-timing-signal-benchmark11-worker-0",
                "deepsense"
            ),
            (
                "deepsense/events.out.tfevents.1552057986.adam-sem-prevents-overfit-hypothesis-05-worker-0",
                "deepsense"
            ),
            (
                "deepsense/events.out.tfevents.1552057986 (1).adam-sem-prevents-overfit-hypothesis-05-worker-0",
                "deepsense"
            ),
            (
                "mnist/logs/mnist_with_summaries/test/events.out.tfevents.1551288692.adam-nowak",
                "mnist/logs/mnist_with_summaries/test"
            ),
            (
                "mnist/logs/mnist_with_summaries/train/events.out.tfevents.1551288692.piotr-adamski",
                "mnist/logs/mnist_with_summaries/train"
            ),
            (
                "mnist/something",
                "mnist"
            ),
            (
                "mnist",
                "untitled-tensorboard"
            )
        ]

        for test_case in test_cases:
            # WHEN
            parsed_name = parse_path_to_experiment_name(test_case[0])

            # THEN
            self.assertEqual(parsed_name, test_case[1], "{} should equal {}".format(parsed_name, test_case[1]))

    def test_parsing_hostname(self):
        # GIVEN
        test_cases = [
            (
                "events.out.tfevents.1551714927.jan-kowalski-pascal05-test-4vgt2",
                "jan-kowalski-pascal05-test-4vgt2"
            ),
            (
                "deepsense/events.out.tfevents.1551714927.jan-kowalski-pascal05-test-4vgt2",
                "jan-kowalski-pascal05-test-4vgt2"
            ),
            (
                "deepsense/events.out.tfevents.1551719327.adam-timing-signal-benchmark11-worker-0",
                "adam-timing-signal-benchmark11-worker-0"
            ),
            (
                "deepsense/events.out.tfevents.1552057986.adam-sem-prevents-overfit-hypothesis-05-worker-0",
                "adam-sem-prevents-overfit-hypothesis-05-worker-0"
            ),
            (
                "deepsense/events.out.tfevents.1552057986 (1).adam-sem-prevents-overfit-hypothesis-05-worker-0",
                "adam-sem-prevents-overfit-hypothesis-05-worker-0"
            ),
            (
                "mnist/logs/mnist_with_summaries/test/events.out.tfevents.1551288692.adam-nowak",
                "adam-nowak"
            ),
            (
                "mnist/logs/mnist_with_summaries/train/events.out.tfevents.1551288692.piotr-adamski",
                "piotr-adamski"
            ),
            (
                "mnist/something",
                None
            ),
            (
                "mnist/data/events.out.tfevents.piotr-adamski",
                None
            )
        ]

        for test_case in test_cases:
            # WHEN
            parsed_hostname = parse_path_to_hostname(test_case[0])

            # THEN
            self.assertEqual(parsed_hostname, test_case[1], "{} should equal {}".format(parsed_hostname, test_case[1]))
