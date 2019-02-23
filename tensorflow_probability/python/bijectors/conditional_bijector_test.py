# Copyright 2018 The TensorFlow Probability Authors.
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
# ============================================================================
"""ConditionalBijector Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


class _TestBijector(tfb.ConditionalBijector):

  def __init__(self):
    super(_TestBijector, self).__init__(
        forward_min_event_ndims=0,
        graph_parents=[],
        is_constant_jacobian=True,
        validate_args=False,
        dtype=tf.float32,
        name="test_bijector")

  def _forward(self, _, arg1, arg2):
    raise ValueError("forward", arg1, arg2)

  def _inverse(self, _, arg1, arg2):
    raise ValueError("inverse", arg1, arg2)

  def _inverse_log_det_jacobian(self, _, arg1, arg2):
    raise ValueError("inverse_log_det_jacobian", arg1, arg2)

  def _forward_log_det_jacobian(self, _, arg1, arg2):
    raise ValueError("forward_log_det_jacobian", arg1, arg2)


@test_util.run_all_in_graph_and_eager_modes
class ConditionalBijectorTest(tf.test.TestCase):

  def testConditionalBijector(self):
    b = _TestBijector()
    for name in ["forward", "inverse"]:
      method = getattr(b, name)
      with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
        method(1., arg1="b1", arg2="b2")

    for name in ["inverse_log_det_jacobian", "forward_log_det_jacobian"]:
      method = getattr(b, name)
      with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
        method(1., event_ndims=0, arg1="b1", arg2="b2")

  def testNestedCondition(self):
    b = _TestBijector()
    for name in ["forward", "inverse"]:
      method = getattr(b, name)
      with self.assertRaisesRegexp(
          ValueError, name + ".*{'b1': 'c1'}, {'b2': 'c2'}"):
        method(1., arg1={"b1": "c1"}, arg2={"b2": "c2"})

    for name in ["inverse_log_det_jacobian", "forward_log_det_jacobian"]:
      method = getattr(b, name)
      with self.assertRaisesRegexp(
          ValueError, name + ".*{'b1': 'c1'}, {'b2': 'c2'}"):
        method(1., event_ndims=0, arg1={"b1": "c1"}, arg2={"b2": "c2"})


if __name__ == "__main__":
  tf.test.main()
