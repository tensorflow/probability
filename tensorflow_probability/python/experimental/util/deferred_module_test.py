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
"""Tests for `tfp.util.DeferredModule`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions


def _gamma_from_loc_scale_positional(loc, log_scale):
  return (loc / tf.exp(log_scale))**2, loc / (tf.exp(log_scale))**2


def _gamma_from_loc_scale_named(loc, log_scale):
  return {'concentration': (loc / tf.exp(log_scale))**2,
          'rate': loc / (tf.exp(log_scale))**2}


@test_util.test_all_tf_execution_regimes
class DeferredModuleTest(test_util.TestCase):

  def testDistributionTapeSafety(self):
    log_precision = tf.Variable(0.)
    mean_times_precision = tf.Variable(1.)

    def to_loc_scale(log_precision, mean_times_precision):
      variance = 1./tf.exp(log_precision)
      mean = mean_times_precision * variance
      return {'loc': mean, 'scale': tf.sqrt(variance)}
    dist = tfp.experimental.util.DeferredModule(
        tfd.Normal, to_loc_scale, log_precision, mean_times_precision)
    self.assertIn(log_precision, dist.trainable_variables)
    self.assertIn(mean_times_precision, dist.trainable_variables)

    with tf.GradientTape() as tape:
      lp = dist.log_prob(0.4)
    grads = tape.gradient(lp, dist.trainable_variables)
    self.assertLen(grads, 2)
    for grad in grads:
      self.assertIsNotNone(grad)

  def testDistributionBatchSlicing(self):

    batch_shape = [4, 3]
    seed0, seed1 = samplers.split_seed(test_util.test_seed(), 2)
    log_precision = tf.Variable(samplers.normal(batch_shape, seed=seed0))
    mean_times_precision = tf.Variable(samplers.normal(batch_shape, seed=seed1))

    def to_loc_scale(log_precision, mean_times_precision):
      variance = 1./tf.exp(log_precision)
      mean = mean_times_precision * variance
      return {'loc': mean, 'scale': tf.sqrt(variance)}
    dist = tfp.experimental.util.DeferredModule(
        tfd.Normal, to_loc_scale, log_precision, mean_times_precision)
    self.assertLen(dist.trainable_variables, 2)

    sliced = dist[:2]
    self.assertEqual(sliced.batch_shape, [2, 3])
    # We do *not* expect the slicing itself to be deferred: like log_prob,
    # sample, and other methods, slicing produces a concrete value (that happens
    # to be a Distribution instance).
    self.assertLen(sliced.trainable_variables, 0)

    sliced_sample = sliced.sample(5)
    self.assertEqual(sliced_sample.shape, [5, 2, 3])

  def testDeferredTransformedDistribution(self):
    log_concentration = tf.Variable(0.)
    log_rate = tf.Variable(0.)
    log_scale = tf.Variable(-1.0)
    dist = tfp.experimental.util.DeferredModule(
        tfd.TransformedDistribution,
        lambda a, b, c: (tfd.Gamma(tf.exp(a),  # pylint: disable=g-long-lambda
                                   tf.exp(b)),
                         tfb.Scale(tf.exp(c))),
        log_concentration,
        log_rate,
        log_scale)
    with tf.GradientTape() as tape:
      x = dist.sample()
    self.assertAllNotNone(
        tape.gradient(x,
                      [log_concentration,
                       log_rate,
                       log_scale]))

  def testBijectorCallDistribution(self):
    log_scale = tf.Variable(-1.0)
    self.evaluate(log_scale.initializer)

    bij = tfp.experimental.util.DeferredModule(tfb.Scale, tf.exp, log_scale)
    self.assertLen(bij.trainable_variables, 1)

    # Calling the deferred bij produces a concretized TransformedDistribution.
    dist = bij(tfd.Normal(0., 1.))
    self.assertLen(dist.trainable_variables, 0)

    # We can also defer the call itself, if we like.
    deferred_dist = tfp.experimental.util.DeferredModule(
        bij, tfd.Normal, loc=0., scale=1.)
    self.assertLen(deferred_dist.trainable_variables, 1)

    lp = dist.log_prob(17.)
    with tf.GradientTape() as tape:
      lp_deferred = deferred_dist.log_prob(17.)
    g = tape.gradient(lp_deferred, log_scale)
    self.assertIsNotNone(g)
    self.assertAllClose(*self.evaluate((lp, lp_deferred)))

  # Check that the args_fn can return either positional or named arguments to
  # initialize the base class.
  @parameterized.named_parameters(
      {'testcase_name': 'positional',
       'args_fn': _gamma_from_loc_scale_positional},
      {'testcase_name': 'named',
       'args_fn': _gamma_from_loc_scale_named})
  def testCallingConventions(self, args_fn):

    loc = tf.Variable(3.)
    log_scale = tf.Variable(1.)
    self.evaluate((loc.initializer, log_scale.initializer))

    # Also check that we can *call* the args_fn with positional or named
    # arguments.
    dist_args = tfp.experimental.util.DeferredModule(
        tfd.Gamma, args_fn, loc, log_scale)
    dist_kwargs = tfp.experimental.util.DeferredModule(
        tfd.Gamma, args_fn, loc=loc, log_scale=log_scale)

    with tf.GradientTape(persistent=True) as tape:
      mean1, log_stddev1 = dist_args.mean(), tf.math.log(dist_args.stddev())
      mean2, log_stddev2 = dist_kwargs.mean(), tf.math.log(dist_kwargs.stddev())
    self.assertAllClose(
        self.evaluate(tape.gradient(mean1, [loc, log_scale])),
        [1., 0.])
    self.assertAllClose(
        self.evaluate(tape.gradient(mean2, [loc, log_scale])),
        [1., 0.])
    self.assertAllClose(
        self.evaluate(tape.gradient(log_stddev1, [loc, log_scale])),
        [0., 1.])
    self.assertAllClose(
        self.evaluate(tape.gradient(log_stddev2, [loc, log_scale])),
        [0., 1.])


if __name__ == '__main__':
  test_util.main()
