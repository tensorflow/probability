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
"""Tests for BatchBroadcast."""

# Dependency imports

import hypothesis as hp
import hypothesis.strategies as hps

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import hypothesis_testlib as tfd_hps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class _BatchBroadcastTest(object):

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=5)
  def test_shapes(self, data):
    batch_shape = data.draw(tfp_hps.shapes())
    bcast_arg, dist_batch_shp = data.draw(
        tfp_hps.broadcasting_shapes(batch_shape, 2))
    underlying = data.draw(tfd_hps.distributions(batch_shape=dist_batch_shp))
    if not self.is_static_shape:
      bcast_arg = tf.Variable(bcast_arg)
      self.evaluate(bcast_arg.initializer)
    dist = tfd.BatchBroadcast(underlying, bcast_arg)
    if self.is_static_shape:
      self.assertEqual(batch_shape, dist.batch_shape)
      self.assertEqual(underlying.event_shape, dist.event_shape)
    self.assertAllEqual(batch_shape, dist.batch_shape_tensor())
    self.assertAllEqual(underlying.event_shape_tensor(),
                        dist.event_shape_tensor())

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=5)
  def test_sample(self, data):
    batch_shape = data.draw(tfp_hps.shapes())
    bcast_arg, dist_batch_shp = data.draw(
        tfp_hps.broadcasting_shapes(batch_shape, 2))

    underlying = tfd.Normal(
        loc=tf.reshape(
            tf.range(float(np.prod(tensorshape_util.as_list(dist_batch_shp)))),
            dist_batch_shp),
        scale=0.01)

    if not self.is_static_shape:
      bcast_arg = tf.Variable(bcast_arg)
      self.evaluate(bcast_arg.initializer)
    dist = tfd.BatchBroadcast(underlying, bcast_arg)
    sample_shape = data.draw(hps.one_of(hps.integers(0, 13), tfp_hps.shapes()))
    sample_batch_event = tf.concat([np.int32(sample_shape).reshape([-1]),
                                    batch_shape,
                                    dist.event_shape_tensor()],
                                   axis=0)
    sample = dist.sample(sample_shape,
                         seed=test_util.test_seed(sampler_type='stateless'))
    if self.is_static_shape:
      self.assertEqual(tf.TensorShape(self.evaluate(sample_batch_event)),
                       sample.shape)
    self.assertAllEqual(sample_batch_event, tf.shape(sample))
    # Since the `loc` of the underlying is simply 0...n-1 (reshaped), and the
    # scale is extremely small, then we can verify that these locations are
    # effectively broadcast out to the full batch shape when sampling.
    self.assertAllClose(tf.broadcast_to(dist.distribution.loc,
                                        sample_batch_event),
                        sample,
                        atol=.1)

    # Check that `sample_and_log_prob` also gives a correctly-shaped sample
    # with correct log-prob.
    sample2, lp = dist.experimental_sample_and_log_prob(
        sample_shape, seed=test_util.test_seed(sampler_type='stateless'))
    if self.is_static_shape:
      self.assertEqual(tf.TensorShape(self.evaluate(sample_batch_event)),
                       sample2.shape)
    self.assertAllEqual(sample_batch_event, tf.shape(sample2))
    self.assertAllClose(lp, dist.log_prob(sample2))

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=5)
  def test_log_prob(self, data):
    batch_shape = data.draw(tfp_hps.shapes())
    bcast_arg, dist_batch_shp = data.draw(
        tfp_hps.broadcasting_shapes(batch_shape, 2))

    underlying = tfd.Normal(
        loc=tf.reshape(
            tf.range(float(np.prod(tensorshape_util.as_list(dist_batch_shp)))),
            dist_batch_shp),
        scale=0.01)

    if not self.is_static_shape:
      bcast_arg = tf.Variable(bcast_arg)
      self.evaluate(bcast_arg.initializer)
    dist = tfd.BatchBroadcast(underlying, bcast_arg)
    sample_shape = data.draw(hps.one_of(hps.integers(0, 13), tfp_hps.shapes()))
    sample_batch_event = tf.concat([np.int32(sample_shape).reshape([-1]),
                                    batch_shape,
                                    dist.event_shape_tensor()],
                                   axis=0)

    obsv = tf.broadcast_to(dist.distribution.loc, sample_batch_event)
    self.assertAllTrue(dist.log_prob(obsv) > dist.log_prob(obsv + .5))

  def test_mean(self):
    d = tfd.BatchBroadcast(tfd.Independent(tfd.Normal([0., 1, 2], .5),
                                           reinterpreted_batch_ndims=1),
                           [2])
    expected = tf.broadcast_to(tf.constant([0., 1, 2]), [2, 3])
    self.assertAllEqual(expected, d.mean())

  def test_stddev(self):
    self.assertAllEqual(tf.fill([2, 3], .5),
                        tfd.BatchBroadcast(tfd.Normal(0., .5), [2, 3]).stddev())

  def test_entropy(self):
    u = tfd.Sample(tfd.Normal(0., .5), 4)
    self.assertAllEqual(
        tf.fill([2, 3], u.entropy()),
        tfd.BatchBroadcast(u, [2, 3]).entropy())

  def test_var(self):
    d = tfd.BatchBroadcast(tfd.Normal(0., [[.5], [1.]]), [2, 3])
    expected = tf.broadcast_to(tf.constant([[.25], [1.]]), [2, 3])
    self.assertAllEqual(expected, d.variance())

  def test_cov(self):
    d = tfd.BatchBroadcast(tfd.MultivariateNormalDiag(tf.zeros(2), [.5, 1.]),
                           [5, 3])
    expected = tf.broadcast_to(tf.constant([[.25, 0], [0, 1]]), [5, 3, 2, 2])
    self.assertAllEqual(expected, d.covariance())

  def test_quantile(self):
    d = tfd.BatchBroadcast(tfd.Normal(loc=[0., 1, 2], scale=.5), [2, 1])
    expected = tf.broadcast_to(tf.constant([0., 1, 2]), [2, 3])
    self.assertAllEqual(expected, d.quantile(.5))
    x = d.quantile([[.45], [.55]])
    self.assertAllEqual(expected, tf.round(x))
    self.assertAllTrue(x[0] < x[1])

  def test_bug170030378(self):
    n_item = 50
    n_rater = 7

    stream = test_util.test_seed_stream()
    weight = self.evaluate(
        tfd.Sample(tfd.Dirichlet([0.25, 0.25]), n_item).sample(seed=stream()))
    mixture_dist = tfd.Categorical(probs=weight)  # batch_shape=[50]

    rater_sensitivity = self.evaluate(
        tfd.Sample(tfd.Beta(5., 1.), n_rater).sample(seed=stream()))
    rater_specificity = self.evaluate(
        tfd.Sample(tfd.Beta(2., 5.), n_rater).sample(seed=stream()))

    probs = tf.stack([rater_sensitivity, rater_specificity])[None, ...]

    components_dist = tfd.BatchBroadcast(  # batch_shape=[50, 2]
        tfd.Independent(tfd.Bernoulli(probs=probs),
                        reinterpreted_batch_ndims=1),
        [50, 2])

    obs_dist = tfd.MixtureSameFamily(mixture_dist, components_dist)

    observed = self.evaluate(obs_dist.sample(seed=stream()))
    mixture_logp = obs_dist.log_prob(observed)

    expected_logp = tf.math.reduce_logsumexp(
        tf.math.log(weight) + components_dist.distribution.log_prob(
            observed[:, None, ...]),
        axis=-1)
    self.assertAllClose(expected_logp, mixture_logp)

  def test_docstring_shapes(self):
    d = tfd.BatchBroadcast(tfd.Normal(tf.range(3.), 1.), [2, 3])
    self.assertEqual([2, 3], d.batch_shape)
    self.assertEqual([3], d.distribution.batch_shape)
    self.assertEqual([], d.event_shape)

    df = tfd.Uniform(4., 5.).sample([10, 1], seed=test_util.test_seed())
    d = tfd.BatchBroadcast(tfd.WishartTriL(df=df, scale_tril=tf.eye(3)), [2])
    self.assertEqual([10, 2], d.batch_shape)
    self.assertEqual([10, 1], d.distribution.batch_shape)
    self.assertEqual([3, 3], d.event_shape)

  def test_docstring_example(self):
    stream = test_util.test_seed_stream()
    loc = tfp.random.spherical_uniform([10], 3, seed=stream())
    components_dist = tfd.VonMisesFisher(mean_direction=loc, concentration=50.)
    mixture_dist = tfd.Categorical(
        logits=tf.random.uniform([500, 10], seed=stream()))
    obs_dist = tfd.MixtureSameFamily(
        mixture_dist, tfd.BatchBroadcast(components_dist, [500, 10]))
    test_sites = tfp.random.spherical_uniform([20], 3, seed=stream())
    lp = tfd.Sample(obs_dist, 20).log_prob(test_sites)
    self.assertEqual([500], lp.shape)
    self.evaluate(lp)

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=5)
  def test_default_bijector(self, data):
    batch_shape = data.draw(tfp_hps.shapes())
    bcast_arg, dist_batch_shp = data.draw(
        tfp_hps.broadcasting_shapes(batch_shape, 2))
    underlying = data.draw(
        tfd_hps.distributions(
            batch_shape=dist_batch_shp,
            eligibility_filter=(
                lambda name: name != 'BatchReshape')))  # b/183977243
    if not self.is_static_shape:
      bcast_arg = tf.Variable(bcast_arg)
      self.evaluate(bcast_arg.initializer)
    dist = tfd.BatchBroadcast(underlying, bcast_arg)
    bijector = dist.experimental_default_event_space_bijector()
    hp.assume(bijector is not None)
    shp = bijector.inverse_event_shape_tensor(
        tf.concat([dist.batch_shape_tensor(),
                   dist.event_shape_tensor()],
                  axis=0))
    obs = bijector.forward(tf.random.normal(shp, seed=test_util.test_seed()))
    with tf.control_dependencies(dist._sample_control_dependencies(obs)):
      self.evaluate(tf.identity(obs))

  def test_bcast_to_errors(self):
    with self.assertRaisesRegexp(ValueError, 'is incompatible with'):
      tfd.BatchBroadcast(tfd.Normal(tf.range(3.), 0.), to_shape=[2, 1])

    shp = tf.Variable([2, 1])
    self.evaluate(shp.initializer)
    with self.assertRaisesOpError('is incompatible with underlying'):
      self.evaluate(
          tfd.BatchBroadcast(
              tfd.Normal(tf.range(3.), 0.),
              to_shape=shp,
              validate_args=True).log_prob(0.))

  def test_bcast_with_errors(self):
    with self.assertRaisesRegexp(ValueError, 'Incompatible shapes'):
      tfd.BatchBroadcast(tfd.Normal(tf.range(3.), 0.), with_shape=[2, 4])

    shp = tf.Variable([2, 4])
    self.evaluate(shp.initializer)
    with self.assertRaisesOpError('Incompatible shapes'):
      self.evaluate(
          tfd.BatchBroadcast(
              tfd.Normal(tf.range(3.), 0.),
              with_shape=shp,
              validate_args=True).log_prob(0.))

  def test_bcast_both_error(self):
    with self.assertRaisesRegexp(ValueError, 'Exactly one of'):
      tfd.BatchBroadcast(tfd.Normal(0., 1.), [3], to_shape=[3])

    with self.assertRaisesRegexp(ValueError, 'Exactly one of'):
      tfd.BatchBroadcast(tfd.Normal(0., 1.))


@test_util.test_all_tf_execution_regimes
class BatchBroadcastStaticTest(_BatchBroadcastTest, test_util.TestCase):

  is_static_shape = True


@test_util.test_all_tf_execution_regimes
class BatchBroadcastDynamicTest(_BatchBroadcastTest, test_util.TestCase):

  is_static_shape = False


if __name__ == '__main__':
  test_util.main()
