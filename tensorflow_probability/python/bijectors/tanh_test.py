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
"""Tanh Tests."""

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
class TanhBijectorTest(test_util.TestCase):
  """Tests correctness of the Y = g(X) = tanh(X) transformation."""

  def testBijector(self):
    self.assertStartsWith(tfb.Tanh().name, "tanh")
    x = np.linspace(-3., 3., 100).reshape([2, 5, 10]).astype(np.float64)
    y = np.tanh(x)
    ildj = -np.log1p(-np.square(np.tanh(x)))
    bijector = tfb.Tanh()
    self.assertAllClose(
        y, self.evaluate(bijector.forward(x)), atol=0., rtol=1e-2)
    self.assertAllClose(
        x, self.evaluate(bijector.inverse(y)), atol=0., rtol=1e-4)
    self.assertAllClose(
        ildj,
        self.evaluate(bijector.inverse_log_det_jacobian(
            y, event_ndims=0)), atol=0., rtol=1e-6)
    self.assertAllClose(
        -ildj,
        self.evaluate(bijector.forward_log_det_jacobian(
            x, event_ndims=0)), atol=0., rtol=1e-4)

  def testScalarCongruency(self):
    bijector_test_util.assert_scalar_congruency(
        tfb.Tanh(), lower_x=-7., upper_x=7., eval_func=self.evaluate,
        n=int(10e4), rtol=.5)

  def testBijectiveAndFinite(self):
    x = np.linspace(-100., 100., 100).astype(np.float64)
    eps = 1e-3
    y = np.linspace(-1. + eps, 1. - eps, 100).astype(np.float64)
    bijector_test_util.assert_bijective_and_finite(
        tfb.Tanh(), x, y, eval_func=self.evaluate, event_ndims=0, atol=0.,
        rtol=1e-4)

  def testMatchWithAffineTransform(self):
    direct_bj = tfb.Tanh()
    indirect_bj = tfb.Chain([
        tfb.AffineScalar(
            shift=tf.cast(-1.0, dtype=tf.float64),
            scale=tf.cast(2.0, dtype=tf.float64)),
        tfb.Sigmoid(),
        tfb.AffineScalar(scale=tf.cast(2.0, dtype=tf.float64))
    ])

    x = np.linspace(-3.0, 3.0, 100)
    y = np.tanh(x)
    self.assertAllClose(self.evaluate(direct_bj.forward(x)),
                        self.evaluate(indirect_bj.forward(x)))
    self.assertAllClose(self.evaluate(direct_bj.inverse(y)),
                        self.evaluate(indirect_bj.inverse(y)))
    self.assertAllClose(
        self.evaluate(direct_bj.inverse_log_det_jacobian(y, event_ndims=0)),
        self.evaluate(indirect_bj.inverse_log_det_jacobian(y, event_ndims=0)))
    self.assertAllClose(
        self.evaluate(direct_bj.forward_log_det_jacobian(x, event_ndims=0)),
        self.evaluate(indirect_bj.forward_log_det_jacobian(x, event_ndims=0)))


if __name__ == "__main__":
  tf.test.main()
