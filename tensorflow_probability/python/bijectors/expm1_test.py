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
"""Expm1 Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class Expm1BijectorTest(test_util.TestCase):
  """Tests correctness of the Y = g(X) = expm1(X) transformation."""

  def testBijector(self):
    bijector = tfb.Expm1()
    self.assertStartsWith(bijector.name, 'expm1')
    x = [[[-1.], [1.4]]]
    y = np.expm1(x)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        -np.squeeze(np.log1p(y), axis=-1),
        self.evaluate(bijector.inverse_log_det_jacobian(
            y, event_ndims=1)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(
            np.expm1(x), event_ndims=1)),
        self.evaluate(bijector.forward_log_det_jacobian(
            x, event_ndims=1)))

  def testMatchesExpm1(self):
    bijector = tfb.Expm1()
    x = np.random.randn(30)
    y = np.expm1(x)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    # Disable caching.
    self.assertAllClose(x, self.evaluate(
        bijector.inverse(tf.identity(y))))

  def testScalarCongruency(self):
    bijector = tfb.Expm1()
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-5., upper_x=2.5, eval_func=self.evaluate,
        rtol=0.15)

  def testBijectiveAndFinite(self):
    bijector = tfb.Expm1()
    x = np.linspace(-5, 5, num=20).astype(np.float32)
    y = np.logspace(-5, 5, num=20).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0)


@test_util.test_all_tf_execution_regimes
class Log1pBijectorTest(test_util.TestCase):

  def testBijectorIsInvertExpm1(self):
    x = np.linspace(0., 10., num=200)
    log1p = tfb.Log1p()
    invert_expm1 = tfb.Invert(tfb.Expm1())
    self.assertAllClose(
        self.evaluate(log1p.forward(x)),
        self.evaluate(invert_expm1.forward(x)))
    self.assertAllClose(
        self.evaluate(log1p.inverse(x)),
        self.evaluate(invert_expm1.inverse(x)))
    self.assertAllClose(
        self.evaluate(log1p.forward_log_det_jacobian(x, event_ndims=1)),
        self.evaluate(invert_expm1.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        self.evaluate(log1p.inverse_log_det_jacobian(x, event_ndims=1)),
        self.evaluate(invert_expm1.inverse_log_det_jacobian(x, event_ndims=1)))


if __name__ == '__main__':
  tf.test.main()
