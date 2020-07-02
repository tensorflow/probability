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
"""Tests for Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class InlineBijectorTest(test_util.TestCase):
  """Tests correctness of the inline constructed bijector."""

  def testBijector(self):
    inline = tfb.Inline(
        forward_fn=tf.exp,
        inverse_fn=tf.math.log,
        inverse_log_det_jacobian_fn=lambda y: -tf.math.log(y),
        forward_min_event_ndims=0,
        name='exp')

    self.assertStartsWith(inline.name, 'exp')
    x = [[[1., 2.], [3., 4.], [5., 6.]]]
    y = np.exp(x)
    self.assertAllClose(y, self.evaluate(inline.forward(x)))
    self.assertAllClose(x, self.evaluate(inline.inverse(y)))
    self.assertAllClose(
        -np.sum(np.log(y), axis=-1),
        self.evaluate(inline.inverse_log_det_jacobian(y, event_ndims=1)))
    self.assertAllClose(
        self.evaluate(-inline.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(inline.forward_log_det_jacobian(x, event_ndims=1)))

  def testShapeGetters(self):
    bijector = tfb.Inline(
        forward_event_shape_tensor_fn=lambda x: tf.concat((x, [1]), 0),
        forward_event_shape_fn=lambda x: tensorshape_util.as_list(x) + [1],
        inverse_event_shape_tensor_fn=lambda x: x[:-1],
        inverse_event_shape_fn=lambda x: x[:-1],
        forward_min_event_ndims=0,
        name='shape_only')
    x = tf.TensorShape([1, 2, 3])
    y = tf.TensorShape([1, 2, 3, 1])
    self.assertAllEqual(y, bijector.forward_event_shape(x))
    self.assertAllEqual(
        tensorshape_util.as_list(y),
        self.evaluate(
            bijector.forward_event_shape_tensor(tensorshape_util.as_list(x))))
    self.assertAllEqual(x, bijector.inverse_event_shape(y))
    self.assertAllEqual(
        tensorshape_util.as_list(x),
        self.evaluate(
            bijector.inverse_event_shape_tensor(tensorshape_util.as_list(y))))


if __name__ == '__main__':
  tf.test.main()
