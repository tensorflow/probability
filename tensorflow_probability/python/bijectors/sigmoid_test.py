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
"""Sigmoid Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import special
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
class SigmoidBijectorTest(tf.test.TestCase):
  """Tests correctness of the Y = g(X) = (1 + exp(-X))^-1 transformation."""

  def testBijector(self):
    self.assertStartsWith(tfb.Sigmoid().name, "sigmoid")
    x = np.linspace(-10., 10., 100).reshape([2, 5, 10]).astype(np.float32)
    y = special.expit(x)
    ildj = -np.log(y) - np.log1p(-y)
    bijector = tfb.Sigmoid()
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
        tfb.Sigmoid(), lower_x=-7., upper_x=7., eval_func=self.evaluate)

  def testBijectiveAndFinite(self):
    x = np.linspace(-100., 100., 100).astype(np.float32)
    eps = 1e-3
    y = np.linspace(eps, 1. - eps, 100).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        tfb.Sigmoid(), x, y, eval_func=self.evaluate, event_ndims=0, atol=0.,
        rtol=1e-4)


if __name__ == "__main__":
  tf.test.main()
