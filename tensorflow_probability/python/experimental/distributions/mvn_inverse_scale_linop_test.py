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
# limitations under the License.
# ============================================================================
"""Tests for tensorflow_probability.python.experimental.distributions.mvn_inverse_scale_linop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfd_e = tfp.experimental.distributions


@test_util.test_all_tf_execution_regimes
class MVNInverseScaleLinOpTest(test_util.TestCase):

  def test_log_prob(self):
    cov = 0.1 * tf.eye(3) + tf.ones((3, 3))
    cholesky = tf.linalg.cholesky(cov)
    loc = [1., 2., 3.]
    linop = tf.linalg.LinearOperatorLowerTriangular(cholesky)

    scale_mvn = tfd.MultivariateNormalTriL(loc, cholesky)
    inverse_scale_mvn = tfd_e.MultivariateNormalInverseScaleLinearOperator(
        loc, inverse_scale=linop.inverse())

    point = tf.linspace(-1., 2., 3)
    self.assertAllClose(scale_mvn.log_prob(point),
                        inverse_scale_mvn.log_prob(point))

    batch_point = tf.reshape(tf.range(0., 6.), (2, 3))
    self.assertAllClose(scale_mvn.log_prob(batch_point),
                        inverse_scale_mvn.log_prob(batch_point))

  def test_log_prob_precision(self):
    cov = 0.1 * tf.eye(3) + tf.ones((3, 3))
    cholesky = tf.linalg.cholesky(cov)
    loc = [1., 2., 3.]
    linop = tf.linalg.LinearOperatorLowerTriangular(cholesky)
    precision = tf.linalg.LinearOperatorFullMatrix(tf.linalg.inv(cov))

    scale_mvn = tfd.MultivariateNormalTriL(loc, cholesky)
    inverse_scale_mvn = tfd_e.MultivariateNormalInverseScaleLinearOperator(
        loc, linop.inverse(), precision=precision)
    point = tf.linspace(-1., 2., 3)
    self.assertAllClose(scale_mvn.log_prob(point),
                        inverse_scale_mvn.log_prob(point))

    batch_point = tf.reshape(tf.range(0., 6.), (2, 3))
    self.assertAllClose(scale_mvn.log_prob(batch_point),
                        inverse_scale_mvn.log_prob(batch_point))


if __name__ == '__main__':
  tf.test.main()
