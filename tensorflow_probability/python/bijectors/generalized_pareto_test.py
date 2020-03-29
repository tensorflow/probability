# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for GeneralizedPareto bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class GeneralizedParetoTest(test_util.TestCase):
  """Tests correctness of the `GeneralizedPareto` bijector."""

  def testScalarCongruencyPositiveConcentration(self):
    bijector_test_util.assert_scalar_congruency(
        tfb.GeneralizedPareto(
            loc=1., scale=3., concentration=2., validate_args=True),
        lower_x=-7., upper_x=7., eval_func=self.evaluate, rtol=.15)

  def testScalarCongruencyNegativeConcentration(self):
    bijector_test_util.assert_scalar_congruency(
        tfb.GeneralizedPareto(
            loc=1., scale=3., concentration=-5., validate_args=True),
        lower_x=-7., upper_x=7., eval_func=self.evaluate, rtol=.1)

  def testBijectiveAndFinitePositiveConcentration(self):
    loc = 5.
    x = np.linspace(-10., 20., 20).astype(np.float32)
    y = np.linspace(loc + 1e-3, 20., 20).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        tfb.GeneralizedPareto(
            loc=loc, scale=2., concentration=1., validate_args=True),
        x, y, eval_func=self.evaluate, event_ndims=0, atol=1e-2, rtol=1e-2)

  def testBijectiveAndFiniteNegativeConcentration(self):
    x = np.linspace(-10., 10., 20).astype(np.float32)
    eps = 1e-3
    loc = 5.
    scale = 4.
    concentration = 1.
    upper_bound = loc + scale / concentration
    y = np.linspace(loc + eps, upper_bound - eps, 20).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        tfb.GeneralizedPareto(
            loc=loc, scale=scale, concentration=concentration,
            validate_args=True),
        x, y, eval_func=self.evaluate, event_ndims=0, atol=1e-2, rtol=1e-2)

  def testBijectorValues(self):
    x = [-1., 0., 1.]
    bij = tfb.GeneralizedPareto(
        loc=[1., 2., 3.],
        scale=5.,
        concentration=[[-4., -6., -1.], [2., 4., 0.]],
        validate_args=True)

    y = self.evaluate(bij.forward(x))
    self.assertAllClose(
        y[0],
        self.evaluate(bij._negative_concentration_bijector()(x)[0]),
        rtol=1e-6,
        atol=0)
    self.assertAllClose(
        y[1],
        self.evaluate(bij._non_negative_concentration_bijector(x)),
        rtol=1e-6,
        atol=0)

    y = [1.5, 2.5, 3.5]
    ildj = self.evaluate(bij.inverse_log_det_jacobian(y, event_ndims=0))
    self.assertAllClose(
        ildj[0],
        self.evaluate(
            bij._negative_concentration_bijector().inverse_log_det_jacobian(
                y, event_ndims=0))[0],
        rtol=1e-6,
        atol=0)
    self.assertAllClose(
        ildj[1],
        self.evaluate(
            bij._non_negative_concentration_bijector.inverse_log_det_jacobian(
                y, event_ndims=0)),
        rtol=1e-6,
        atol=0)


if __name__ == '__main__':
  tf.test.main()
