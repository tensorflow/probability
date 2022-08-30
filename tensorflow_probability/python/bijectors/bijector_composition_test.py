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

import functools

# Dependency imports

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import reciprocal
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.distributions import chi
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class BijectorCompositionTest(test_util.TestCase):

  def testComposeFromChainBijector(self):
    x = tf.constant([-5., 0., 5.])
    sigmoid = functools.reduce(lambda chain, f: chain(f), [
        reciprocal.Reciprocal(),
        shift.Shift(shift=1.),
        exp.Exp(),
        scale.Scale(scale=-1.),
    ])
    self.assertIsInstance(sigmoid, chain.Chain)
    self.assertAllClose(
        *self.evaluate([tf.math.sigmoid(x), sigmoid.forward(x)]),
        atol=0, rtol=1e-3)

  def testComposeFromTransformedDistribution(self):
    actual_log_normal = exp.Exp()(
        transformed_distribution.TransformedDistribution(
            distribution=normal.Normal(0, 1),
            bijector=shift.Shift(shift=0.5)(scale.Scale(scale=2.))))
    expected_log_normal = lognormal.LogNormal(0.5, 2.)
    x = tf.constant([0.1, 1., 5.])
    self.assertAllClose(
        *self.evaluate([actual_log_normal.log_prob(x),
                        expected_log_normal.log_prob(x)]),
        atol=0, rtol=1e-3)

  def testComposeFromTDSubclassWithAlternateCtorArgs(self):
    # This line used to raise an exception.
    identity.Identity()(chi.Chi(df=1., allow_nan_stats=True))

  def testComposeFromNonTransformedDistribution(self):
    actual_log_normal = exp.Exp()(normal.Normal(0.5, 2.))
    expected_log_normal = lognormal.LogNormal(0.5, 2.)
    x = tf.constant([0.1, 1., 5.])
    self.assertAllClose(
        *self.evaluate([actual_log_normal.log_prob(x),
                        expected_log_normal.log_prob(x)]),
        atol=0, rtol=1e-3)

  def testComposeFromTensor(self):
    x = tf.constant([-5., 0., 5.])
    self.assertAllClose(
        *self.evaluate([tf.exp(x), exp.Exp()(x)]), atol=0, rtol=1e-3)

  def testHandlesKwargs(self):
    x = exp.Exp()(sample.Sample(normal.Normal(0., 1.), sample_shape=[4]))
    y = independent.Independent(lognormal.LogNormal(tf.zeros(4), 1), 1)
    z = tf.constant([[1., 2, 3, 4],
                     [0.5, 1.5, 2., 2.5]])
    self.assertAllClose(
        *self.evaluate([y.log_prob(z), x.log_prob(z)]),
        atol=0, rtol=1e-3)


if __name__ == '__main__':
  test_util.main()
