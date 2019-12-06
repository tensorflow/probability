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
"""Test the MarginalizableJointDistributionCoroutine distribution class."""


# pylint: disable=abstract-method, no-member, unused-variable


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
import tensorflow_probability.python.experimental.marginalize as marginalize
from tensorflow_probability.python.internal import test_util


Root = tfd.JointDistributionCoroutine.Root


def _conform(ts):
  """Broadcast all arguments to a common shape."""

  shape = functools.reduce(
      tf.broadcast_static_shape, [a.shape for a in ts])
  return [tf.broadcast_to(a, shape) for a in ts]


def _cat(*ts):
  return tf.concat(ts, axis=0)


def _stack(*ts):
  return tf.stack(_conform(ts), axis=-1)


def _tree_example(n, n_steps):
  # Computes result for `test_particle_tree` using
  # explicit "hand-constructed" computation based on
  # matrix multiplication.

  init = np.ones([n * n]) / (n * n)
  trans = np.zeros((n*n, n*n), dtype=np.float64)

  def wrap(i):
    return (i + n) % n

  def index(i, j):
    return wrap(i) * n + wrap(j)

  # Compute transition matrix for random walk.
  for i in range(n):
    for j in range(n):
      trans[index(i, j), index(i, j - 1)] = 0.1
      trans[index(i, j), index(i, j + 1)] = 0.2
      trans[index(i, j), index(i - 1, j)] = 0.3
      trans[index(i, j), index(i + 1, j)] = 0.4

  def normal_prob(loc, scale, z):
    return (np.exp(-(z - loc) ** 2/(2. * scale * scale)) /
            np.sqrt(2 * np.pi) / scale)

  # `n_steps` steps in random walk.
  trans10 = np.linalg.matrix_power(trans, n_steps)

  zs = np.arange(0, n)

  def obs(x, y):
    return (normal_prob(zs, 2.0, x)[:, None] *
            normal_prob(zs, 2.0, y)).reshape([n * n])

  v1 = obs(0.0, 0.0)
  v2 = obs(4.0, 4.0)
  v3 = obs(0.0, 4.0)

  p1 = np.matmul(trans10, v1)
  p2 = np.matmul(trans10, v2)
  p3 = np.matmul(trans10, v3)

  p = (init * p1 * p2 * p3).reshape([n, n])

  return p


@test_util.test_all_tf_execution_regimes
class _MarginalizeTest(
    test_util.TestCase):

  def test_basics_einsum(self):
    probs = np.random.rand(20) + 0.001
    probs = probs / np.sum(probs)

    def model():
      i = yield Root(tfd.Categorical(probs=probs, dtype=tf.int32))
      j = yield tfd.Categorical(probs=probs, dtype=tf.int32)
      k = yield tfd.Categorical(probs=probs, dtype=tf.int32)

    dist = marginalize.MarginalizableJointDistributionCoroutine(model)

    p = tf.exp(dist.marginalized_log_prob(['tabulate',
                                           'tabulate',
                                           'tabulate'],
                                          method='einsum'))
    self.assertEqual(p.shape, [20, 20, 20])
    self.assertAllClose(tf.reduce_sum(p), 1.0)

    s = tf.exp(dist.marginalized_log_prob(['marginalize',
                                           'marginalize',
                                           'marginalize'],
                                          method='einsum'))
    self.assertAllClose(s, 1.0)

  def test_basics_logeinsumexp(self):
    probs = np.random.rand(20) + 0.001
    probs = probs / np.sum(probs)

    def model():
      i = yield Root(tfd.Categorical(probs=probs, dtype=tf.int32))
      j = yield tfd.Categorical(probs=probs, dtype=tf.int32)
      k = yield tfd.Categorical(probs=probs, dtype=tf.int32)

    dist = marginalize.MarginalizableJointDistributionCoroutine(model)

    p = tf.exp(dist.marginalized_log_prob(['tabulate',
                                           'tabulate',
                                           'tabulate'],
                                          method='logeinsumexp'))
    self.assertEqual(p.shape, [20, 20, 20])
    self.assertAllClose(tf.reduce_sum(p), 1.0)

    s = tf.exp(dist.marginalized_log_prob(['marginalize',
                                           'marginalize',
                                           'marginalize'],
                                          method='logeinsumexp'))
    self.assertAllClose(s, 1.0)

  def test_simple_network(self):
    # From https://en.wikipedia.org/wiki/Bayesian_network#Example
    def model():
      raining = yield Root(tfd.Bernoulli(probs=0.2, dtype=tf.int32))
      sprinkler_prob = [0.4, 0.01]
      sprinkler_prob = tf.gather(sprinkler_prob, raining)
      sprinkler = yield tfd.Bernoulli(probs=sprinkler_prob, dtype=tf.int32)
      grass_wet_prob = [[0.0, 0.8],
                        [0.9, 0.99]]
      grass_wet_prob = tf.gather_nd(grass_wet_prob, _stack(sprinkler, raining))
      grass_wet = yield tfd.Bernoulli(probs=grass_wet_prob, dtype=tf.int32)

    d = marginalize.MarginalizableJointDistributionCoroutine(model)
    # We want to know the probability that it was raining
    # and we want to marginalize over the state of the sprinkler.
    observations = ['tabulate',     # We want to know the probability it rained.
                    'marginalize',  # We don't know the sprinkler state.
                    1]              # We observed a wet lawn.
    p = tf.exp(d.marginalized_log_prob(observations))
    p = p / tf.reduce_sum(p)
    self.assertAllClose(p[1], 0.357688)

  def test_sample_distribution(self):
    probs = np.random.rand(4) + 0.001
    probs = probs / np.sum(probs)

    def model():
      i = yield Root(
          tfd.Sample(
              tfd.Categorical(probs=probs, dtype=tf.int32), sample_shape=[2]))
      # Note use of scalar `sample_shape` to test expansion of shape
      # to vector.
      j = yield tfd.Sample(
          tfd.Categorical(probs=probs, dtype=tf.int32), sample_shape=2)

    dist = marginalize.MarginalizableJointDistributionCoroutine(model)

    ptt = tf.exp(dist.marginalized_log_prob(['tabulate', 'tabulate']))
    ptm = tf.exp(dist.marginalized_log_prob(['tabulate', 'marginalize']))
    pmt = tf.exp(dist.marginalized_log_prob(['marginalize', 'tabulate']))
    pmm = tf.exp(dist.marginalized_log_prob(['marginalize', 'marginalize']))
    self.assertEqual(ptt.shape, [4, 4, 4, 4])
    self.assertEqual(ptm.shape, [4, 4])
    self.assertEqual(pmt.shape, [4, 4])
    self.assertEqual(pmm.shape, [])

  def test_hmm(self):
    n_steps = 4
    infer_step = 2

    observations = [-1.0, 0.0, 1.0, 2.0]

    initial_prob = tf.constant([0.6, 0.4], dtype=tf.float32)
    transition_matrix = tf.constant([[0.6, 0.4],
                                     [0.3, 0.7]], dtype=tf.float32)
    observation_locs = tf.constant([0.0, 1.0], dtype=tf.float32)
    observation_scale = tf.constant(0.5, dtype=tf.float32)

    dist1 = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Normal(loc=observation_locs,
                                             scale=observation_scale),
                                  num_steps=n_steps)

    p = dist1.posterior_marginals(observations).probs_parameter()[infer_step]

    def model():
      i = yield Root(tfd.Categorical(probs=initial_prob,
                                     dtype=tf.int32))
      z = yield tfd.Normal(loc=tf.gather(observation_locs, i),
                           scale=observation_scale)

      for t in range(n_steps - 1):
        i = yield tfd.Categorical(probs=tf.gather(transition_matrix, i),
                                  dtype=tf.int32)
        yield tfd.Normal(loc=tf.gather(observation_locs, i),
                         scale=observation_scale)

    dist2 = marginalize.MarginalizableJointDistributionCoroutine(model)
    full_observations = list(
        itertools.chain(*zip(
            ['tabulate' if i == infer_step else 'marginalize'
             for i in range(n_steps)],
            observations)))
    q = tf.exp(dist2.marginalized_log_prob(full_observations))
    q = q / tf.reduce_sum(q)

    self.assertAllClose(p, q)

  def test_markov_chain_stationary(self):
    n_steps = 52
    initial_prob = tf.constant([0.6, 0.4], dtype=tf.float64)
    transition_matrix = tf.constant([[0.6, 0.4],
                                     [0.3, 0.7]], dtype=tf.float64)

    def model():
      i = yield Root(tfd.Categorical(probs=initial_prob,
                                     dtype=tf.int32))

      for t in range(n_steps - 1):
        i = yield tfd.Categorical(probs=tf.gather(transition_matrix, i),
                                  dtype=tf.int32)

      yield tfd.Deterministic(i)

    dist = marginalize.MarginalizableJointDistributionCoroutine(model)
    # Compute probability of ending in state 1 by marginalizing out
    # all intermediate states.
    # Limiting distribution as `n_steps` -> infinity is exactly 4/7.
    observations = n_steps * ['marginalize'] + [1]
    p = tf.exp(dist.marginalized_log_prob(observations,
                                          internal_type=tf.float64))
    self.assertAllClose(p, 4./7.)

  def test_particle_tree(self):
    # m particles are born at the same random location on an n x n grid.
    # They independently take `n_steps` steps of a random walk going N, S,
    # E or W at each step. At the end we observe their positions subject
    # to Gaussian noise. Computes posterior distribution for the birthplace.

    n = 16
    m = 3
    n_steps = 16

    def model():
      # Shared birthplace
      x_start = yield Root(tfd.Categorical(probs=tf.ones(n) / n,
                                           dtype=tf.int32))
      y_start = yield tfd.Categorical(probs=tf.ones(n) / n,
                                      dtype=tf.int32)

      x = m * [x_start]
      y = m * [y_start]

      for t in range(n_steps):
        for i in range(m):
          # Construct PDF for next step in walk
          # Start with PDF for all mass on current point.
          ox = tf.one_hot(x[i], n)
          oy = tf.one_hot(y[i], n)
          o = ox[..., :, None] * oy[..., None, :]

          # Deliberate choice of non-centered distribution as
          # reduced symmetry lends itself to better testing.
          p = (0.1 * tf.roll(o, shift=[0, -1], axis=[-2, -1]) +
               0.2 * tf.roll(o, shift=[0, 1], axis=[-2, -1]) +
               0.3 * tf.roll(o, shift=[-1, 0], axis=[-2, -1]) +
               0.4 * tf.roll(o, shift=[1, 0], axis=[-2, -1]))

          # Reshape just last two dimensions.
          p = tf.reshape(p, _cat(p.shape[:-2], [-1]))
          xy = yield tfd.Categorical(probs=p, dtype=tf.int32)
          x[i] = xy // n
          y[i] = xy % n

      # 2 * m noisy 2D observations at end
      for i in range(m):
        yield tfd.Normal(tf.cast(x[i], dtype=tf.float32), scale=2.0)
        yield tfd.Normal(tf.cast(y[i], dtype=tf.float32), scale=2.0)

    d = marginalize.MarginalizableJointDistributionCoroutine(model)
    final_observations = [0.0, 0.0, 4.0, 4.0, 0.0, 4.0]
    observations = (['tabulate', 'tabulate'] +
                    n_steps * ['marginalize',
                               'marginalize',
                               'marginalize'] +
                    final_observations)
    # Using `method='logeinsumexp' here creates a large intermediate
    # that impacts performance. Because we are only setting
    # `STEPS == 16` the probabilities involved aren't small enough
    # to result in underflow.
    p = tf.exp(d.marginalized_log_prob(observations, method='einsum'))
    q = _tree_example(n, n_steps)

    # Note that while p and q should be close in value there is a large
    # difference in computation time. I would expect the p
    # to be slower by a factor of around `n_steps/log2(n_steps)` (because
    # `numpy` compute matrix powers by repeated squaring) but it seems
    # to be even slower. This likely means future versions of
    # `marginalized_log_prob` will run faster when the elimination
    # order chosen by `tf.einsum` closer matches `_tree_example` above.
    self.assertAllClose(p, q)

  def test_marginalized_gradient(self):
    n = 10

    mu1 = tf.Variable(3.)
    mu2 = tf.Variable(3.)

    def model():
      change_year = yield Root(tfd.Categorical(probs=tf.ones(n) / n))
      for year in range(n):
        post_change_year = tf.cast(year >= change_year, dtype=tf.int32)
        mu = tf.gather([mu1, mu2], post_change_year)
        accidents = yield tfd.Poisson(mu)

    counts = np.ones([n], dtype=np.float32)
    dist = marginalize.MarginalizableJointDistributionCoroutine(model)
    obs = ['marginalize'] + list(counts)
    with tf.GradientTape() as tape:
      loss = -dist.marginalized_log_prob(obs)
    grad = tape.gradient(loss, [mu1, mu2])
    self.assertLen(grad, 2)
    self.assertAllNotNone(grad)


if __name__ == '__main__':
  tf.test.main()
