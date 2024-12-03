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

import warnings

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import masked_autoregressive
from tensorflow_probability.python.distributions import autoregressive
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample as sample_lib
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import linalg


@test_util.test_all_tf_execution_regimes
class AutoregressiveTest(test_util.VectorDistributionTestHelpers,
                         test_util.TestCase):
  """Tests the Autoregressive distribution."""

  def setUp(self):
    super(AutoregressiveTest, self).setUp()
    self._rng = np.random.RandomState(42)

  def _random_ar_matrix(self, event_size):
    n = np.int32(event_size * (event_size + 1) // 2)
    p = 2. * self._rng.random_sample(n).astype(np.float32) - 1.
    # Zero-out the diagonal to ensure auto-regressive property.
    return tf.linalg.set_diag(
        linalg.fill_triangular(0.25 * p), tf.zeros(event_size)
    )

  def _normal_fn(self, affine):
    def _fn(samples):
      scale = tf.exp(tf.linalg.matvec(affine, samples))
      return independent.Independent(
          normal.Normal(loc=0., scale=scale, validate_args=True),
          reinterpreted_batch_ndims=1,
          validate_args=True)

    return _fn

  def testSampleAndLogProbConsistency(self):
    batch_shape = np.int32([])
    event_size = 2
    batch_event_shape = np.concatenate([batch_shape, [event_size]], axis=0)
    sample0 = tf.zeros(batch_event_shape)
    ar_matrix = self._random_ar_matrix(event_size)
    ar = autoregressive.Autoregressive(
        self._normal_fn(ar_matrix), sample0, validate_args=True)
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        ar,
        num_samples=int(1e6),
        radius=1.,
        center=0.,
        rtol=0.01,
        seed=test_util.test_seed())

  @test_util.jax_disable_test_missing_functionality('seedless sampling')
  def testSampleIndependenceWithoutSeedBug(self):
    # Under eager, there was a bug where subsequent samples for position 0 would
    # be erroneously independent of the first sample when a seed was not
    # provided.

    # A pithy example: A simple 1-Markov autoregressive sequence for which the
    # first frame is either 0 or 1 with equal probability and all subsequent
    # frames copy the previous frame. Thus the overall sequence-level
    # distribution is 0000 with probability 0.5 and 1111 with probability 0.5.
    def distribution_fn(sample):
      num_frames = sample.shape[-1]
      mask = tf.one_hot(0, num_frames)[:, tf.newaxis]
      probs = tf.roll(tf.one_hot(sample, 3), shift=1, axis=-2)
      probs = probs * (1.0 - mask) + tf.convert_to_tensor([0.5, 0.5, 0]) * mask
      return independent.Independent(
          categorical.Categorical(probs=probs), reinterpreted_batch_ndims=1)

    ar = autoregressive.Autoregressive(
        distribution_fn, sample0=tf.constant([2, 2, 2, 2]), num_steps=4)
    samps = self.evaluate(ar.sample(10))
    for s in samps:
      self.assertIn(np.mean(s), (0., 1.), msg=str(s))

  def testCompareToBijector(self):
    """Demonstrates equivalence between TD, Bijector approach and AR dist."""
    sample_shape = np.int32([4, 5])
    batch_shape = np.int32([])
    event_size = np.int32(2)
    batch_event_shape = np.concatenate([batch_shape, [event_size]], axis=0)
    sample0 = tf.zeros(batch_event_shape)
    ar_matrix = self._random_ar_matrix(event_size)
    ar = autoregressive.Autoregressive(
        self._normal_fn(ar_matrix), sample0, validate_args=True)
    ar_flow = masked_autoregressive.MaskedAutoregressiveFlow(
        is_constant_jacobian=True,
        shift_and_log_scale_fn=lambda x: [None, tf.linalg.matvec(ar_matrix, x)],
        validate_args=True)
    td = transformed_distribution.TransformedDistribution(
        # TODO(b/137665504): Use batch-adding meta-distribution to set the batch
        # shape instead of tf.zeros.
        distribution=sample_lib.Sample(
            normal.Normal(tf.zeros(batch_shape), 1.), [event_size]),
        bijector=ar_flow,
        validate_args=True)
    x_shape = np.concatenate([sample_shape, batch_shape, [event_size]], axis=0)
    x = 2. * self._rng.random_sample(x_shape).astype(np.float32) - 1.
    td_log_prob_, ar_log_prob_ = self.evaluate([td.log_prob(x), ar.log_prob(x)])
    self.assertAllClose(td_log_prob_, ar_log_prob_, atol=0., rtol=1e-6)

  def testVariableNumSteps(self):
    def fn(sample=0.):
      return normal.Normal(loc=tf.zeros_like(sample), scale=1.)

    num_steps = tf.Variable(4, dtype=tf.int64)
    self.evaluate(num_steps.initializer)

    ar = autoregressive.Autoregressive(
        fn, num_steps=num_steps, validate_args=True)
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
      return independent.Independent(
          normal.Normal(loc=loc_param, scale=1.),
          reinterpreted_batch_ndims=event_ndims)

    num_steps = tf.Variable(7)
    self.evaluate([v.initializer for v in [loc, num_steps, event_ndims]])

    ar = autoregressive.Autoregressive(
        fn, num_steps=num_steps, validate_args=True)
    sample = self.evaluate(ar.sample(3, seed=test_util.test_seed()))
    self.assertAllEqual([3, 4, 2], sample.shape)
    self.assertAllEqual([2], self.evaluate(ar.event_shape_tensor()))
    self.assertAllEqual([4], self.evaluate(ar.batch_shape_tensor()))

  def testBatchAndEventShape(self):
    loc = tf.Variable(np.zeros((5, 1, 3)), shape=tf.TensorShape(None))
    event_ndims = tf.Variable(2)
    def fn(sample):
      return independent.Independent(
          normal.Normal(loc=loc + 0. * sample, scale=1.),
          reinterpreted_batch_ndims=tf.convert_to_tensor(event_ndims))

    self.evaluate([v.initializer for v in [loc, event_ndims]])

    zero = tf.convert_to_tensor(0., dtype=loc.dtype)
    ar = autoregressive.Autoregressive(
        fn, num_steps=7, sample0=zero, validate_args=True)

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
      return normal.Normal(loc=tf.zeros_like(sample), scale=1.)

    num_steps = [4, 4, 4]
    with self.assertRaisesRegex(Exception,
                                'Argument `num_steps` must be a scalar'):
      ar = autoregressive.Autoregressive(
          fn, num_steps=num_steps, validate_args=True)
      self.evaluate(ar.sample(seed=test_util.test_seed()))

    num_steps = tf.Variable(4, shape=tf.TensorShape(None))
    self.evaluate(num_steps.initializer)
    ar = autoregressive.Autoregressive(
        fn, num_steps=num_steps, validate_args=True)
    self.evaluate(ar.sample(seed=test_util.test_seed()))
    with self.assertRaisesRegex(Exception,
                                'Argument `num_steps` must be a scalar'):
      with tf.control_dependencies([num_steps.assign([17, 3])]):
        self.evaluate(ar.sample(seed=test_util.test_seed()))

  def testErrorOnNonPositiveNumSteps(self):
    def fn(sample=0.):
      return normal.Normal(loc=tf.zeros_like(sample), scale=1.)

    num_steps = 0
    with self.assertRaisesRegex(Exception,
                                'Argument `num_steps` must be positive'):
      ar = autoregressive.Autoregressive(
          fn, num_steps=num_steps, validate_args=True)
      self.evaluate(ar.sample(seed=test_util.test_seed()))

    num_steps = tf.Variable(13, shape=tf.TensorShape(None))
    self.evaluate(num_steps.initializer)
    ar = autoregressive.Autoregressive(
        fn, num_steps=num_steps, validate_args=True)
    self.evaluate(ar.sample(seed=test_util.test_seed()))
    with self.assertRaisesRegex(Exception,
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
        return independent.Independent(
            normal.Normal(loc, 1.), reinterpreted_batch_ndims=1)

    sample0 = tf.Variable([0., 0., 0.])
    ar = autoregressive.Autoregressive(
        DistFn(), sample0=sample0, num_steps=3, validate_args=True)

    self.evaluate([v.initializer for v in ar.trainable_variables])
    with tf.GradientTape() as tape:
      loss = tf.reduce_sum(
          tf.square(ar.sample(seed=test_util.test_seed())), axis=-1)
    grad = tape.gradient(loss, ar.trainable_variables)

    self.assertLen(grad, 3)
    self.assertAllNotNone(grad)


class SamplerBackwardCompatibilityTest(test_util.TestCase):

  @test_util.jax_disable_test_missing_functionality('stateful samplers')
  @test_util.numpy_disable_test_missing_functionality('stateful samplers')
  def testStatefulDistFn(self):

    class StatefulNormal(distribution.Distribution):

      def __init__(self, loc):
        self._loc = tf.convert_to_tensor(loc)
        super(StatefulNormal, self).__init__(
            dtype=tf.float32,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=False)

      def _batch_shape(self):
        return self._loc.shape

      def _event_shape(self):
        return []

      def _sample_n(self, n, seed=None):
        return self._loc + tf.random.normal(
            tf.concat([[n], tf.shape(self._loc)], axis=0), seed=seed)

    def dist_fn(s):
      return StatefulNormal(loc=s)

    ar = autoregressive.Autoregressive(
        dist_fn,
        sample0=normal.Normal(0., 1.).sample(7, seed=test_util.test_seed()),
        num_steps=7)

    with warnings.catch_warnings(record=True) as triggered:
      self.evaluate(ar.sample(seed=test_util.test_seed()))
    self.assertTrue(
        any('Falling back to stateful sampling for `distribution_fn(sample0)`'
            in str(warning.message) for warning in triggered))

    num_steps = tf.Variable(9)
    self.evaluate(num_steps.initializer)
    with warnings.catch_warnings(record=True) as triggered:
      self.evaluate(ar.copy(num_steps=num_steps).sample(
          seed=test_util.test_seed()))
    self.assertTrue(
        any('Falling back to stateful sampling for `distribution_fn(sample0)`'
            in str(warning.message) for warning in triggered))


if __name__ == '__main__':
  test_util.main()
