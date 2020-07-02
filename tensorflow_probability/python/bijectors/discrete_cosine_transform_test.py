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
"""Tests for DCT Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from scipy import fftpack
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class DiscreteCosineTransformTest(test_util.TestCase):
  """Tests correctness of the DiscreteCosineTransform bijector."""

  def testBijector(self):
    bijector = tfb.DiscreteCosineTransform(validate_args=True)
    self.assertStartsWith(bijector.name, 'dct')
    x = np.random.randn(6, 5, 4).astype(np.float32)
    y = fftpack.dct(x, norm='ortho').astype(np.float32)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllEqual(
        np.float32(0),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllEqual(
        np.float32(0),
        self.evaluate(bijector.inverse_log_det_jacobian(x, event_ndims=1)))

  def testBijector_dct3(self):
    bijector = tfb.DiscreteCosineTransform(dct_type=3, validate_args=True)
    self.assertStartsWith(bijector.name, 'dct')
    x = np.random.randn(6, 5, 4).astype(np.float32)
    y = fftpack.dct(x, type=3, norm='ortho').astype(np.float32)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllEqual(
        np.float32(0),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllEqual(
        np.float32(0),
        self.evaluate(bijector.inverse_log_det_jacobian(x, event_ndims=1)))

  def testBijectiveAndFinite(self):
    x = np.linspace(-10., 10., num=10).astype(np.float32)
    y = np.linspace(0.01, 0.99, num=10).astype(np.float32)
    for dct_type in 2, 3:
      bijector_test_util.assert_bijective_and_finite(
          tfb.DiscreteCosineTransform(dct_type=dct_type, validate_args=True),
          x,
          y,
          eval_func=self.evaluate,
          event_ndims=1,
          rtol=1e-3)


if __name__ == '__main__':
  tf.test.main()
