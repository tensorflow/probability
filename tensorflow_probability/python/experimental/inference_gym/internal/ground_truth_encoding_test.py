# Lint as: python2, python3
# Copyright 2020 The TensorFlow Probability Authors.
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
#
"""Tests for ground_truth_encoding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import sys

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.inference_gym.internal import ground_truth_encoding
from tensorflow_probability.python.internal import test_util


class GroundTruthEncodingTest(test_util.TestCase):

  @parameterized.named_parameters(
      ("Nested", (0, "a")),
      ("NonNested", ()),
  )
  def testEndToEnd(self, tuple_path):
    if sys.version_info[0] < 3:
      self.skipTest("Python 3 only due to advanced importlib usage.")
    # N.B. typically these would all have the same shape, but for testing
    # convenience we use different shapes here to exercise the encoder/decoder
    # logic a bit more.
    mean = 0.
    sem = np.array([1., 2.]),
    std = np.array([1., 2., 3., 4.]).reshape((2, 2))
    sestd = np.array([], dtype=np.float32)
    name = "test"

    array_strs = ground_truth_encoding.save_ground_truth_part(
        name=name,
        tuple_path=tuple_path,
        mean=mean,
        sem=sem,
        std=std,
        sestd=sestd,
    )
    module_source = ground_truth_encoding.get_ground_truth_module_source(
        target_name="target",
        command_str="",
        array_strs=array_strs,
    )
    module_spec = importlib.util.spec_from_loader("module", loader=None)
    module = importlib.util.module_from_spec(module_spec)
    exec(module_source, module.__dict__)  # pylint: disable=exec-used

    loaded_mean, loaded_sem, loaded_std, loaded_sestd = (
        ground_truth_encoding.load_ground_truth_part(
            module=module,
            name=name,
            tuple_path=tuple_path,
        ))

    self.assertAllEqual(loaded_mean, mean)
    self.assertAllEqual(loaded_sem, sem)
    self.assertAllEqual(loaded_std, std)
    self.assertAllEqual(loaded_sestd, sestd)


if __name__ == "__main__":
  tf.test.main()
