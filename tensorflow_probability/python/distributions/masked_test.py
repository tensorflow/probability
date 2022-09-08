# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for masked.Masked."""

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import masked
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import relaxed_onehot_categorical
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


@test_util.test_all_tf_execution_regimes
class MaskedTest(test_util.TestCase):

  def test_shape(self):
    d = masked.Masked(
        normal.Normal(tf.zeros([7, 20]), 1), tf.sequence_mask(15, 20))
    self.assertEqual((), d.event_shape)
    self.assertAllEqual([], d.event_shape_tensor())
    self.assertEqual((7, 20), d.batch_shape)
    self.assertAllEqual([7, 20], d.batch_shape_tensor())

  def test_sampling(self):
    d = masked.Masked(
        normal.Normal(tf.zeros([20]), 1), tf.sequence_mask(15, 20),
        safe_sample_fn=distribution.Distribution.mode)
    self.assertAllEqual(tf.zeros([5]),  # distribution mode
                        d.sample(seed=test_util.test_seed())[15:])

    # Gamma doesn't have mode for concentration<1.
    # Let's verify we don't trigger an assert.
    d = masked.Masked(
        gamma.Gamma(.9, tf.ones([20]), allow_nan_stats=False),
        tf.sequence_mask(13, 20),
        safe_sample_fn=distribution.Distribution.mean)
    self.assertAllEqual(d.distribution.mean()[15:],
                        d.sample(seed=test_util.test_seed())[15:])

    d = masked.Masked(gamma.Gamma(.9, tf.ones([20]), allow_nan_stats=False),
                      tf.sequence_mask(13, 20))
    self.evaluate(d.sample(seed=test_util.test_seed()))

  def test_event_space_bijector(self):
    # Test that the default event space bijector executes.  This is
    # non-trivial, because the event space bijector of this particular
    # component distribution cannot be relied upon to produce finite
    # values in the unconstrained space from samples of `sub_d`.
    sub_d = relaxed_onehot_categorical.ExpRelaxedOneHotCategorical(
        logits=[0., 0., 0.],
        temperature=[0.01, 0.01, 0.01, 0.01],
        validate_args=True)
    d = masked.Masked(sub_d, validity_mask=False, validate_args=True)
    bij = d.experimental_default_event_space_bijector()
    x = bij(tf.zeros(shape=[4, 2]))
    # The error tested for manifests as failed validations due to
    # invalid values.
    self.assertAllNotNan(self.evaluate(x))

  def test_event_space_bijector_fldj(self):
    # Also test forward log det jacobian for the default event space
    # bijector in the same setting, for completeness.
    sub_d = relaxed_onehot_categorical.ExpRelaxedOneHotCategorical(
        logits=[0., 0., 0.],
        temperature=[0.01, 0.01, 0.01, 0.01])
    d = masked.Masked(sub_d, validity_mask=False, validate_args=True)
    bij = d.experimental_default_event_space_bijector()
    fldj = bij.forward_log_det_jacobian(tf.zeros(shape=[4, 2]))
    self.assertAllEqual(fldj, tf.zeros_like(fldj))

  def test_degenerate_scalar_mask(self):
    d0 = masked.Masked(
        normal.Normal(0., 1.), validity_mask=False, validate_args=True)
    d1 = masked.Masked(
        normal.Normal(0., 1.), validity_mask=True, validate_args=True)
    stream = test_util.test_seed_stream()
    self.assertAllEqual(
        d0.sample(seed=stream()), d0.sample(seed=stream()))
    self.assertNotAllEqual(
        d1.sample(seed=stream()), d1.sample(seed=stream()))
    self.assertAllEqual(0., d0.log_prob(123.))

  def test_log_prob(self):
    d = masked.Masked(
        normal.Normal(tf.zeros([20]), 1), tf.sequence_mask(15, 20))
    x = np.linspace(-.1, .1, 20).astype(np.float32)
    self.assertAllClose(
        tf.pad(normal.Normal(0, 1).log_prob(x[:15]), [[0, 5]]),
        d.log_prob(x))

    d = masked.Masked(
        normal.Normal(tf.zeros([20]), 1), (tf.range(20) % 2) > 0)
    x = np.linspace(-.1, .1, 20).astype(np.float32)
    self.assertAllClose(tf.zeros(10), d.log_prob(x)[::2])
    self.assertAllClose(
        normal.Normal(0, 1).log_prob(x)[1::2], d.log_prob(x)[1::2])

  def test_batch_mask(self):
    d = masked.Masked(normal.Normal(tf.zeros([20]), 1),
                      tf.sequence_mask([15, 17], 20))
    self.assertEqual((2, 20), d.batch_shape)
    x = np.linspace(-.1, .1, 20).astype(np.float32)
    self.assertAllClose(
        tf.stack([
            tf.pad(normal.Normal(0, 1).log_prob(x[:15]), [[0, 5]]),
            tf.pad(normal.Normal(0, 1).log_prob(x[:17]), [[0, 3]])]),
        d.log_prob(x))

  def test_kl(self):
    a = mvn_diag.MultivariateNormalDiag([tf.range(3.)] * 4, tf.ones(3))
    b = mvn_diag.MultivariateNormalDiag([tf.range(3.) + .5] * 4, tf.ones(3))
    kl = kullback_leibler.kl_divergence(
        masked.Masked(a, tf.sequence_mask(3, 4)),
        masked.Masked(b, tf.sequence_mask(2, 4)))
    kl2 = kullback_leibler.kl_divergence(
        masked.Masked(a, tf.sequence_mask(2, 4)),
        masked.Masked(b, tf.sequence_mask(3, 4)))
    self.assertAllClose(a.kl_divergence(b)[:2], kl[:2])
    self.assertAllEqual(float('nan'), kl[2])
    self.assertAllEqual(0., kl[3])
    self.assertAllEqual(float('nan'), kl2[2])

  def test_log_prob_ratio(self):
    p = masked.Masked(mvn_diag.MultivariateNormalDiag(tf.zeros([4, 3000]),
                                                      tf.ones([3000])),
                      tf.sequence_mask(2, 4))
    q = masked.Masked(mvn_diag.MultivariateNormalDiag(tf.zeros([4, 3000]),
                                                      tf.ones([3000])),
                      [True, True, False, False])
    stream = test_util.test_seed_stream()
    x = p.sample(seed=stream())
    y = x + 1e-5 * q.sample(seed=stream())
    x, y = self.evaluate((x, y))  # Avoids different samples across evals.
    normal64 = normal.Normal(0, 1)
    self.assertAllClose(
        tf.reduce_sum(normal64.log_prob(x) - normal64.log_prob(y),
                      axis=-1) * tf.sequence_mask(2, 4, dtype=tf.float32),
        tf.cast(log_prob_ratio.log_prob_ratio(p, x, q, y), tf.float64))

  @test_util.numpy_disable_gradient_test
  def test_grad_log_prob_unsafe_val(self):
    def f(loc):
      d = independent.Independent(
          masked.Masked(lognormal.LogNormal(loc, 1), [True, False, False]),
          reinterpreted_batch_ndims=1)
      return d.log_prob([1.1, 0., float('nan')])
    self.assertAllFinite(gradient.value_and_gradient(f, tf.ones([3]))[1])


if __name__ == '__main__':
  test_util.main()
