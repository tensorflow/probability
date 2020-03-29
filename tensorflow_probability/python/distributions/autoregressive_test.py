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
"""Tests for the Autoregressive distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class AutoregressiveTest(test_util.VectorDistributionTestHelpers,
                         test_util.TestCase):
  """Tests the Autoregressive distribution."""

  def setUp(self):
    super(AutoregressiveTest, self).setUp()
    self._rng = np.random.RandomState(42)

  def _random_scale_tril(self, event_size):
    n = np.int32(event_size * (event_size + 1) // 2)
    p = 2. * self._rng.random_sample(n).astype(np.float32) - 1.
    return tfp.math.fill_triangular(0.25 * p)

  def _normal_fn(self, affine_bijector):
    def _fn(samples):
      scale = tf.exp(affine_bijector.forward(samples))
      return tfd.Independent(
          tfd.Normal(loc=0., scale=scale, validate_args=True),
          reinterpreted_batch_ndims=1,
          validate_args=True)

    return _fn

  def testSampleAndLogProbConsistency(self):
    batch_shape = []
    event_size = 2
    batch_event_shape = np.concatenate([batch_shape, [event_size]], axis=0)
    sample0 = tf.zeros(batch_event_shape)
    affine = tfb.ScaleMatvecTriL(
        scale_tril=self._random_scale_tril(event_size), validate_args=True)
    ar = tfd.Autoregressive(
        self._normal_fn(affine), sample0, validate_args=True)
    if tf.executing_eagerly():
      return
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        ar,
        num_samples=int(1e6),
        radius=1.,
        center=0.,
        rtol=0.01,
        seed=test_util.test_seed())

  def testCompareToBijector(self):
    """Demonstrates equivalence between TD, Bijector approach and AR dist."""
    sample_shape = np.int32([4, 5])
    batch_shape = np.int32([])
    event_size = np.int32(2)
    batch_event_shape = np.concatenate([batch_shape, [event_size]], axis=0)
    sample0 = tf.zeros(batch_event_shape)
    affine = tfb.ScaleMatvecTriL(
        scale_tril=self._random_scale_tril(event_size), validate_args=True)
    ar = tfd.Autoregressive(
        self._normal_fn(affine), sample0, validate_args=True)
    ar_flow = tfb.MaskedAutoregressiveFlow(
        is_constant_jacobian=True,
        shift_and_log_scale_fn=lambda x: [None, affine.forward(x)],
        validate_args=True)
    td = tfd.TransformedDistribution(
        # TODO(b/137665504): Use batch-adding meta-distribution to set the batch
        # shape instead of tf.zeros.
        distribution=tfd.Sample(
            tfd.Normal(tf.zeros(batch_shape), 1.), [event_size]),
        bijector=ar_flow,
        validate_args=True)
    x_shape = np.concatenate([sample_shape, batch_shape, [event_size]], axis=0)
    x = 2. * self._rng.random_sample(x_shape).astype(np.float32) - 1.
    td_log_prob_, ar_log_prob_ = self.evaluate([td.log_prob(x), ar.log_prob(x)])
    self.assertAllClose(td_log_prob_, ar_log_prob_, atol=0., rtol=1e-6)

  def testVariableNumSteps(self):
    def fn(sample=0.):
      return tfd.Normal(loc=tf.zeros_like(sample), scale=1.)

    num_steps = tf.Variable(4, dtype=tf.int64)
    self.evaluate(num_steps.initializer)

    ar = tfd.Autoregressive(fn, num_steps=num_steps, validate_args=True)
    sample = ar.sample(seed=test_util.test_seed())
    log_prob = ar.log_prob(sample)
    self.assertAllEqual([], sample.shape)
    self.assertAllEqual([], log_prob.shape)

    [sample, log_prob] = self.evaluate([sample, log_prob])
    self.assertAllEqual([], sample.shape)
    self.assertAllEqual([], log_prob.shape)

  def testVariableNumStepsAndEventShape(self):
    loc = tf.Variable(np.zeros((4, 2)), shape=tf.TensorShape(None))
    event_ndims = tf.Variable(1)
    def fn(sample=None):
      if sample is not None:
        loc_param = tf.broadcast_to(loc, shape=tf.shape(sample))
      else:
        loc_param = loc
      return tfd.Independent(
          tfd.Normal(loc=loc_param, scale=1.),
          reinterpreted_batch_ndims=event_ndims)

    num_steps = tf.Variable(7)
    self.evaluate([v.initializer for v in [loc, num_steps, event_ndims]])

    ar = tfd.Autoregressive(fn, num_steps=num_steps, validate_args=True)
    sample = self.evaluate(ar.sample(3, seed=test_util.test_seed()))
    self.assertAllEqual([3, 4, 2], sample.shape)
    self.assertAllEqual([2], self.evaluate(ar.event_shape_tensor()))
    self.assertAllEqual([4], self.evaluate(ar.batch_shape_tensor()))

  def testBatchAndEventShape(self):
    loc = tf.Variable(np.zeros((5, 1, 3)), shape=tf.TensorShape(None))
    event_ndims = tf.Variable(2)
    def fn(sample):
      return tfd.Independent(
          tfd.Normal(loc=loc + 0.*sample, scale=1.),
          reinterpreted_batch_ndims=tf.convert_to_tensor(event_ndims))

    self.evaluate([v.initializer for v in [loc, event_ndims]])

    zero = tf.convert_to_tensor(0., dtype=loc.dtype)
    ar = tfd.Autoregressive(fn, num_steps=7, sample0=zero, validate_args=True)

    # NOTE: `ar.event_shape` and `ar.batch_shape` are not known statically,
    # even though the output of `ar.distribution_fn(...)` has statically-known
    # event shape and batch shape.
    self.assertEqual(ar.batch_shape, tf.TensorShape(None))
    self.assertEqual(ar.event_shape, tf.TensorShape(None))
    self.assertAllEqual([5], self.evaluate(ar.batch_shape_tensor()))
    self.assertAllEqual([1, 3], self.evaluate(ar.event_shape_tensor()))
    if tf.executing_eagerly():
      self.assertEqual(tf.TensorShape([5]),
                       ar.distribution_fn(zero).batch_shape)
      self.assertEqual(tf.TensorShape([1, 3]),
                       ar.distribution_fn(zero).event_shape)

    with tf.control_dependencies([
        loc.assign(np.zeros((4, 7))),
        event_ndims.assign(1)
    ]):
      self.assertAllEqual([4], self.evaluate(ar.batch_shape_tensor()))
      self.assertAllEqual([7], self.evaluate(ar.event_shape_tensor()))
      if tf.executing_eagerly():
        self.assertEqual(tf.TensorShape([4]),
                         ar.distribution_fn(zero).batch_shape)
        self.assertEqual(tf.TensorShape([7]),
                         ar.distribution_fn(zero).event_shape)

  def testErrorOnNonScalarNumSteps(self):
    def fn(sample=0.):
      return tfd.Normal(loc=tf.zeros_like(sample), scale=1.)

    num_steps = [4, 4, 4]
    with self.assertRaisesRegexp(Exception,
                                 'Argument `num_steps` must be a scalar'):
      ar = tfd.Autoregressive(fn, num_steps=num_steps, validate_args=True)
      self.evaluate(ar.sample(seed=test_util.test_seed()))

    num_steps = tf.Variable(4, shape=tf.TensorShape(None))
    self.evaluate(num_steps.initializer)
    ar = tfd.Autoregressive(fn, num_steps=num_steps, validate_args=True)
    self.evaluate(ar.sample(seed=test_util.test_seed()))
    with self.assertRaisesRegexp(Exception,
                                 'Argument `num_steps` must be a scalar'):
      with tf.control_dependencies([num_steps.assign([17, 3])]):
        self.evaluate(ar.sample(seed=test_util.test_seed()))

  def testErrorOnNonPositiveNumSteps(self):
    def fn(sample=0.):
      return tfd.Normal(loc=tf.zeros_like(sample), scale=1.)

    num_steps = 0
    with self.assertRaisesRegexp(Exception,
                                 'Argument `num_steps` must be positive'):
      ar = tfd.Autoregressive(fn, num_steps=num_steps, validate_args=True)
      self.evaluate(ar.sample(seed=test_util.test_seed()))

    num_steps = tf.Variable(13, shape=tf.TensorShape(None))
    self.evaluate(num_steps.initializer)
    ar = tfd.Autoregressive(fn, num_steps=num_steps, validate_args=True)
    self.evaluate(ar.sample(seed=test_util.test_seed()))
    with self.assertRaisesRegexp(Exception,
                                 'Argument `num_steps` must be positive'):
      with tf.control_dependencies([num_steps.assign(-9)]):
        self.evaluate(ar.sample(seed=test_util.test_seed()))

  @test_util.tf_tape_safety_test
  def testGradientsThroughParams(self):

    class DistFn(tf.Module):

      def __init__(self):
        self._w = tf.Variable([1., 1.])
        self._b = tf.Variable([0., 0., 0.])

      def __call__(self, sample=None):
        if sample is None:
          sample = tf.convert_to_tensor([0., 0., 0.])
        loc = tf.stack([
            tf.broadcast_to(self._b[0], shape=tf.shape(sample)[:-1]),
            self._b[1] + self._w[0]*sample[..., 0]*sample[..., 2],
            self._b[2] + self._w[1]*sample[..., 0]
        ], axis=-1)
        return tfd.Independent(tfd.Normal(loc, 1.), reinterpreted_batch_ndims=1)

    sample0 = tf.Variable([0., 0., 0.])
    ar = tfd.Autoregressive(
        DistFn(), sample0=sample0, num_steps=3, validate_args=True)

    self.evaluate([v.initializer for v in ar.trainable_variables])
    with tf.GradientTape() as tape:
      loss = tf.reduce_sum(
          tf.square(ar.sample(seed=test_util.test_seed())), axis=-1)
    grad = tape.gradient(loss, ar.trainable_variables)

    self.assertLen(grad, 3)
    self.assertAllNotNone(grad)


if __name__ == '__main__':
  tf.test.main()
