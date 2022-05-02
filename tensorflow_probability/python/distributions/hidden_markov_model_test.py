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
"""The HiddenMarkovModel distribution class."""


# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class _HiddenMarkovModelTest(
    test_util.VectorDistributionTestHelpers,
    test_util.DiscreteScalarDistributionTestHelpers,
    test_util.TestCase):

  def make_placeholders(self, constants):
    variables = [tf.Variable(c, shape=tf.TensorShape(None)) for c in constants]
    self.evaluate([v.initializer for v in variables])
    return variables

  def test_reproducibility(self):
    initial_prob_data = tf.constant([0.01, 0.99], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.6, 0.4],
                                          [0.3, 0.7]], dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([30])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True)

    seed = test_util.test_seed()
    s1 = self.evaluate(model.sample(5, seed=seed))
    if tf.executing_eagerly():
      tf.random.set_seed(seed)
    s2 = self.evaluate(model.sample(5, seed=seed))
    self.assertAllEqual(s1, s2)

  def test_supports_dynamic_observation_size(self):
    initial_prob_data = tf.constant([0.01, 0.99], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.6, 0.4],
                                          [0.3, 0.7]], dtype=self.dtype)
    observation_locs_data = tf.constant([[0.0, 1.0],
                                         [1.0, 0.0]], dtype=self.dtype)
    observation_scale_data = tf.constant([0.5, 0.5], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([30])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.MultivariateNormalDiag(loc=observation_locs,
                                   scale_diag=observation_scale),
        num_steps=num_steps,
        validate_args=True)

    self.evaluate(model.sample(5, seed=test_util.test_seed()))
    observation_data = tf.constant(30 * [[0.5, 0.5]], dtype=self.dtype)
    self.evaluate(model.log_prob(observation_data))
    self.evaluate(model.posterior_marginals(observation_data).probs_parameter())
    self.evaluate(model.posterior_mode(observation_data))

  def test_consistency(self):
    initial_prob_data = tf.constant([0.01, 0.99], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.6, 0.4],
                                          [0.3, 0.7]], dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([3])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs,
                   scale=observation_scale),
        num_steps=num_steps,
        validate_args=True)

    self.run_test_sample_consistent_log_prob(
        self.evaluate, model,
        num_samples=100000,
        center=0.5, radius=0.5,
        rtol=0.05, seed=test_util.test_seed())

  def test_broadcast_initial_probs(self):
    initial_prob_data = tf.constant([0.01, 0.99], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.6, 0.4],
                                          [0.3, 0.7]], dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([3])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs,
                   scale=observation_scale),
        num_steps=num_steps,
        validate_args=True)

    self.run_test_sample_consistent_log_prob(
        self.evaluate, model,
        num_samples=100000,
        center=0.5, radius=1.,
        rtol=0.02, seed=test_util.test_seed())

  def test_broadcast_transitions(self):
    initial_prob_data = tf.constant([0.01, 0.99], dtype=self.dtype)
    transition_matrix_data = tf.constant([[[0.8, 0.2],
                                           [0.3, 0.7]],
                                          [[0.9, 0.1],
                                           [0.2, 0.8]]],
                                         dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([3])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs,
                   scale=observation_scale),
        num_steps=num_steps,
        validate_args=True)

    self.run_test_sample_consistent_log_prob(
        self.evaluate, model,
        num_samples=100000,
        center=0.5, radius=1.,
        rtol=2e-2, seed=test_util.test_seed())

  def test_broadcast_observations(self):
    initial_prob_data = tf.constant([0.01, 0.99], dtype=self.dtype)
    transition_matrix_data = tf.constant([[[0.8, 0.2],
                                           [0.3, 0.7]],
                                          [[0.9, 0.1],
                                           [0.2, 0.8]]], dtype=self.dtype)
    observation_locs_data = tf.constant([[0.9, 0.1],
                                         [0.2, 0.8]], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([3])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs,
                   scale=observation_scale),
        num_steps=num_steps,
        validate_args=True)

    self.run_test_sample_consistent_log_prob(
        self.evaluate, model,
        num_samples=100000,
        center=0.5, radius=1.,
        rtol=2e-2, seed=test_util.test_seed())

  def test_edge_case_sample_n_no_transitions(self):
    initial_prob_data = tf.constant([0.9, 0.1], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([1])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    x = model.sample(2, seed=test_util.test_seed())
    x_shape = self.evaluate(tf.shape(x))

    self.assertAllEqual(x_shape, [2, 1])

  def test_edge_case_log_prob_no_transitions(self):
    initial_prob_data = tf.constant([0.9, 0.1], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])
    (initial_prob, transition_matrix,
     observation_probs) = ([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([1])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    x = model.log_prob([0])

    self.assertAllClose(x, np.log(0.9), rtol=1e-5, atol=0.0)

  def test_edge_case_mean_no_transitions(self):
    initial_prob_data = tf.constant([0.9, 0.1], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([1])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs,
                   scale=observation_scale),
        num_steps=num_steps,
        validate_args=True)

    x = model.mean()
    x_shape = self.evaluate(tf.shape(x))

    self.assertAllEqual(x_shape, [1])

  def test_num_states(self):
    initial_prob_data = tf.constant([0.9, 0.1], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[0.5, 0.0, 0.5],
                                          [0.0, 1.0, 0.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    x = model.num_states_tensor()

    self.assertAllEqual(x, 2)

  def test_coin_tosses(self):
    initial_prob_data = tf.constant([0.5, 0.5], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    x = model.log_prob([0, 0, 0, 0, 0])

    self.assertAllClose(x, np.log(.5**5), rtol=1e-5, atol=0.0)

  def test_coin_toss_batch(self):
    initial_prob_data = tf.constant([0.5, 0.5], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    initial_prob = tf.broadcast_to(initial_prob, [3, 2, 2])
    transition_matrix = tf.broadcast_to(transition_matrix, [3, 2, 2, 2])
    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    examples = tf.stack(
        [tf.zeros(5, dtype=tf.int32), tf.ones(5, dtype=tf.int32)], axis=0)
    examples = tf.broadcast_to(examples, [7, 3, 2, 5])
    computed_log_prob = model.log_prob(examples)

    expected_log_prob = tf.broadcast_to(tf.constant([np.log(.5**5)]), [7, 3, 2])
    self.assertAllClose(computed_log_prob, expected_log_prob,
                        rtol=1e-4, atol=0.0)

  def test_mean_shape(self):
    initial_prob_data = tf.constant([0.8, 0.2], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.7, 0.3],
                                          [0.2, 0.8]], dtype=self.dtype)
    observation_locs_data = tf.constant([[0.0, 0.0],
                                         [10.0, 10.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data])

    [num_steps] = self.make_placeholders([7])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.MultivariateNormalDiag(
            loc=observation_locs),
        num_steps=num_steps,
        validate_args=True)

    x = model.mean()
    x_shape = self.evaluate(tf.shape(x))

    self.assertAllEqual(x_shape, [7, 2])

  def test_batch_mean_shape(self):
    initial_prob_data = tf.constant([[0.8, 0.2],
                                     [0.5, 0.5],
                                     [0.2, 0.8]], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.7, 0.3],
                                          [0.2, 0.8]], dtype=self.dtype)
    observation_locs_data = tf.constant([[0.0, 0.0],
                                         [10.0, 10.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data])

    [num_steps] = self.make_placeholders([7])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.MultivariateNormalDiag(
            loc=observation_locs),
        num_steps=num_steps,
        validate_args=True)

    x = model.mean()
    x_shape = self.evaluate(tf.shape(x))

    self.assertAllEqual(x_shape, [3, 7, 2])

  def test_mean_and_variance(self):
    initial_prob_data = tf.constant([0.6, 0.1, 0.3], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.2, 0.6, 0.2],
                                          [0.5, 0.3, 0.2],
                                          [0.0, 1.0, 0.0]], dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0, 2.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True)

    self.run_test_sample_consistent_mean_variance(self.evaluate, model,
                                                  num_samples=100000,
                                                  rtol=0.03)

  def test_single_sequence_posterior_marginals(self):

    # In this test we have a 9-vertex graph with precisely one
    # 7-vertex path from vertex 0 to vertex 8.
    # The hidden Markov model is a random walk on this
    # graph where the only observations are
    # "observed at 0", "observed in {1, 2, ..., 7}",
    # "observed at 8".
    # The purpose of this test is to ensure that transition
    # and observation matrices with many log probabilities
    # equal to -infinity, and where the result contains many
    # -infinities, are handled correctly.

    initial_prob = tf.constant(np.ones(9) / 9.0, dtype=self.dtype)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4),
             (4, 6), (2, 5), (5, 6), (6, 7),
             (6, 8)]
    transition_matrix = np.zeros((9, 9))
    for (i, j) in edges:
      transition_matrix[i, j] = 1.
      transition_matrix[j, i] = 1.
    transition_matrix = tf.constant(
        transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True),
        dtype=self.dtype)
    observation_probs = tf.constant(
        np.block([[1, 0, 0],
                  [np.zeros((7, 1)), np.ones((7, 1)), np.zeros((7, 1))],
                  [0, 0, 1]]),
        dtype=self.dtype)

    [num_steps] = self.make_placeholders([7])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    observations = tf.constant([0, 1, 1, 1, 1, 1, 2])

    probs = self.evaluate(
        model.posterior_marginals(observations).probs_parameter())
    expected_probs = np.eye(9)[[0, 1, 2, 3, 4, 6, 8]]

    self.assertAllClose(probs, expected_probs, rtol=1e-4, atol=0.0)

  @parameterized.parameters(
      (3, 2, 1, 0),
      (1, 2, 3, 0),
      (1, 0, 2, 3))
  def test_posterior_marginals_high_rank(self, rank_o, rank_t, rank_i, rank_s):
    def increase_rank(n, x):
      # By choosing prime number dimensions we make it less
      # likely that a test will pass for accidental reasons.
      primes = [3, 5, 7]
      for i in range(n):
        x = primes[i] * [x]
      return x

    observation_locs_data = tf.identity(
        increase_rank(rank_o, tf.eye(4, dtype=self.dtype)))
    observation_scales_data = tf.constant(
        [0.25, 0.25, 0.25, 0.25],
        dtype=self.dtype)
    transition_matrix_data = tf.constant(
        increase_rank(rank_t, [[0.8, 0.1, 0.1, 0.0],
                               [0.1, 0.8, 0.0, 0.1],
                               [0.1, 0.0, 0.8, 0.1],
                               [0.0, 0.1, 0.1, 0.8]]),
        dtype=self.dtype)
    initial_prob_data = tf.constant(
        increase_rank(rank_i, [0.25, 0.25, 0.25, 0.25]),
        dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scales) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scales_data])

    observations = tf.constant(
        increase_rank(rank_s,
                      [[[0.91, 0.11], [0.21, 0.09]],
                       [[0.11, 0.97], [0.12, 0.08]],
                       [[0.01, 0.12], [0.92, 0.11]],
                       [[0.02, 0.11], [0.77, 0.11]],
                       [[0.81, 0.15], [0.21, 0.03]],
                       [[0.01, 0.13], [0.23, 0.91]],
                       [[0.11, 0.12], [0.23, 0.79]],
                       [[0.13, 0.11], [0.91, 0.29]]]),
        dtype=self.dtype)

    observation_distribution = tfp.distributions.TransformedDistribution(
        tfd.MultivariateNormalDiag(observation_locs,
                                   scale_diag=observation_scales),
        tfp.bijectors.Reshape((2, 2)))

    [num_steps] = self.make_placeholders([8])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        observation_distribution,
        num_steps=num_steps,
        validate_args=True)

    inferred_probs = self.evaluate(
        model.posterior_marginals(observations).probs_parameter())
    rank_e = max(rank_o, rank_t, rank_i, rank_s)
    expected_probs = increase_rank(rank_e,
                                   [[0.99994, 0.00000, 0.00006, 0.00000],
                                    [0.45137, 0.01888, 0.52975, 0.00000],
                                    [0.00317, 0.00002, 0.98112, 0.01570],
                                    [0.00000, 0.00001, 0.99998, 0.00001],
                                    [0.00495, 0.00001, 0.94214, 0.05289],
                                    [0.00000, 0.00083, 0.00414, 0.99503],
                                    [0.00000, 0.00000, 0.00016, 0.99984],
                                    [0.00000, 0.00000, 0.99960, 0.00039]])
    self.assertAllClose(inferred_probs, expected_probs, rtol=0., atol=1e-4)

  def test_posterior_mode_basic_example(self):
    observation_locs_data = tf.constant([0.0, 1.0, 2.0, 3.0],
                                        dtype=self.dtype)
    observation_scale_data = tf.constant(0.25, dtype=self.dtype)

    transition_matrix_data = tf.constant([[0.9, 0.1, 0.0, 0.0],
                                          [0.1, 0.8, 0.1, 0.0],
                                          [0.0, 0.1, 0.8, 0.1],
                                          [0.0, 0.0, 0.1, 0.9]],
                                         dtype=self.dtype)
    initial_prob_data = tf.constant([0.25, 0.25, 0.25, 0.25],
                                    dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    observations = tf.constant([0.1, 0.2, 0.3, 0.4, 0.5,
                                3.0, 2.9, 2.8, 2.7, 2.6],
                               dtype=self.dtype)

    [num_steps] = self.make_placeholders([10])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True)

    inferred_states = model.posterior_mode(observations)
    expected_states = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    self.assertAllEqual(inferred_states, expected_states)

  @parameterized.parameters(
      (3, 2, 1, 0),
      (1, 2, 3, 0),
      (1, 0, 2, 3))
  def test_posterior_mode_high_rank(self, rank_o, rank_t, rank_i, rank_s):
    def increase_rank(n, x):
      # By choosing prime number dimensions we make it less
      # likely that a test will pass for accidental reasons.
      primes = [3, 5, 7]
      for i in range(n):
        x = primes[i] * [x]
      return x

    observation_locs_data = tf.constant(increase_rank(rank_o,
                                                      [[1.0, 0.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 1.0]]),
                                        dtype=self.dtype)
    observation_scales_data = tf.constant(
        [0.25, 0.25, 0.25, 0.25],
        dtype=self.dtype)
    transition_matrix_data = tf.constant(
        increase_rank(rank_t, [[0.8, 0.1, 0.1, 0.0],
                               [0.1, 0.8, 0.0, 0.1],
                               [0.1, 0.0, 0.8, 0.1],
                               [0.0, 0.1, 0.1, 0.8]]),
        dtype=self.dtype)
    initial_prob_data = tf.constant(
        increase_rank(rank_i, [0.25, 0.25, 0.25, 0.25]),
        dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scales) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scales_data])

    observations = tf.constant(
        increase_rank(rank_s,
                      [[[0.91, 0.11], [0.21, 0.09]],
                       [[0.11, 0.97], [0.12, 0.08]],
                       [[0.01, 0.12], [0.92, 0.11]],
                       [[0.02, 0.11], [0.77, 0.11]],
                       [[0.81, 0.15], [0.21, 0.03]],
                       [[0.01, 0.13], [0.23, 0.91]],
                       [[0.11, 0.12], [0.23, 0.79]],
                       [[0.13, 0.11], [0.91, 0.29]]]),
        dtype=self.dtype)

    observation_distribution = tfp.distributions.TransformedDistribution(
        tfd.MultivariateNormalDiag(observation_locs,
                                   scale_diag=observation_scales),
        tfp.bijectors.Reshape((2, 2)))

    [num_steps] = self.make_placeholders([8])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        observation_distribution,
        num_steps=num_steps,
        validate_args=True)

    inferred_states = model.posterior_mode(observations)
    rank_e = max(rank_o, rank_t, rank_i, rank_s)
    expected_states = increase_rank(rank_e, [0, 2, 2, 2, 2, 3, 3, 2])
    self.assertAllEqual(inferred_states, expected_states)

  def test_posterior_mode_high_rank_batch(self):
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]],
                                         dtype=self.dtype)
    transition_matrix_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]],
                                         dtype=self.dtype)
    initial_prob_data = tf.constant([0.5, 0.5],
                                    dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    observations = tf.constant(2*[3*[[5*[0], 5*[1]]]])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    inferred_states = model.posterior_mode(observations)
    expected_states = 2*[3*[[5*[0], 5*[1]]]]
    self.assertAllEqual(inferred_states, expected_states)

  # Check that the Viterbi algorithm is invariant under permutations of the
  # names of the observations of the HMM (when there is a unique most
  # likely sequence of hidden states).
  def test_posterior_mode_invariance_observations(self):
    observation_probs_data = tf.constant([[0.09, 0.48, 0.52, 0.11],
                                          [0.31, 0.21, 0.21, 0.27]],
                                         dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.90, 0.10],
                                          [0.30, 0.70]],
                                         dtype=self.dtype)
    initial_prob_data = tf.constant([0.65, 0.35],
                                    dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    permutations = tf.identity(np.array([np.random.permutation(4)
                                         for _ in range(8)]))
    inverse_permutations = tf.argsort(permutations)

    observation_probs_permuted = tf.transpose(
        a=tf.gather(tf.transpose(a=observation_probs),
                    inverse_permutations),
        perm=[0, 2, 1])

    observations = tf.constant([1, 0, 3, 1, 3, 0, 2, 1, 2, 1, 3, 0, 0, 1, 1, 2])
    observations_permuted = tf.transpose(
        a=tf.gather(tf.transpose(a=permutations)[..., tf.newaxis],
                    observations,
                    batch_dims=(
                        tensorshape_util.rank(observations.shape) - 1))[..., 0])

    [num_steps] = self.make_placeholders([16])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs_permuted),
        num_steps=num_steps,
        validate_args=True)

    inferred_states = model.posterior_mode(observations_permuted)
    expected_states = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    self.assertAllEqual(inferred_states, 8*[expected_states])

  # Check that the Viterbi algorithm is invariant under permutations of the
  # names of the hidden states of the HMM (when there is a unique most
  # likely sequence of hidden states).
  def test_posterior_mode_invariance_states(self):
    observation_probs_data = tf.constant([[0.12, 0.48, 0.5, 0.1],
                                          [0.4, 0.1, 0.5, 0.0],
                                          [0.1, 0.2, 0.3, 0.4]],
                                         dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.21, 0.49, 0.3],
                                          [0.18, 0.12, 0.7],
                                          [0.75, 0.15, 0.1]],
                                         dtype=self.dtype)
    initial_prob_data = tf.constant([0.8, 0.13, 0.07],
                                    dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    permutations = tf.identity(np.array([np.random.permutation(3)
                                         for _ in range(8)]))
    inverse_permutations = tf.argsort(permutations)

    initial_prob_permuted = tf.gather(initial_prob, inverse_permutations)

    # Permute rows of observation matrix
    observation_probs_permuted = tf.gather(observation_probs,
                                           inverse_permutations)

    # Permute both rows and columns of transition matrix
    transition_matrix_permuted = tf.transpose(
        tf.gather(tf.transpose(transition_matrix), inverse_permutations),
        perm=[0, 2, 1])
    transition_matrix_permuted = tf.gather(
        transition_matrix_permuted,
        inverse_permutations,
        batch_dims=1)

    observations = tf.constant([1, 0, 3, 1, 3, 0, 2, 1, 2, 1, 3, 0, 0, 1, 1, 2])

    [num_steps] = self.make_placeholders([16])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob_permuted),
        tfd.Categorical(probs=transition_matrix_permuted),
        tfd.Categorical(probs=observation_probs_permuted),
        num_steps=num_steps,
        validate_args=True)

    inferred_states = model.posterior_mode(observations)
    expected_states = [0, 1, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0, 1, 2, 0, 1]
    expected_states_permuted = tf.transpose(
        tf.gather(
            tf.transpose(permutations)[..., tf.newaxis],
            expected_states)[..., 0])
    self.assertAllEqual(inferred_states, expected_states_permuted)

  def test_posterior_mode_missing_continuous_observations(self):
    initial_prob_data = tf.constant([0.9, 0.1], dtype=self.dtype)
    transition_matrix_data = tf.constant([[[0.6, 0.4],
                                           [0.6, 0.4]],
                                          [[0.4, 0.6],
                                           [0.4, 0.6]]], dtype=self.dtype)
    observation_locs_data = tf.constant([[0.0, 0.0],
                                         [10.0, 10.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data])

    [num_steps] = self.make_placeholders([3])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.MultivariateNormalDiag(loc=observation_locs),
        num_steps=num_steps,
        validate_args=True)

    observations = tf.constant([[0.0, 0.0],
                                [0.0, 0.0],
                                [10.0, 10.0]], dtype=self.dtype)

    # We test two different transition matrices as well as two
    # different masks.
    # As a result we have a 2x2 tensor of sequences of states
    # returned by `posterior_mode`.
    x = model.posterior_mode(observations, mask=[[[False, True, False]],
                                                 [[False, False, False]]])

    self.assertAllEqual(x, [[[0, 0, 1], [0, 1, 1]],
                            [[0, 0, 1], [0, 0, 1]]])

  def test_posterior_mode_missing_discrete_observations(self):
    initial_prob = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=self.dtype)

    # This test uses a model with a random walk that can make a change of
    # of -1, 0 or +1 at each step.
    transition_data = (0.5 * np.diag(np.ones(4)) +
                       0.25*np.diag(np.ones(3), -1) +
                       0.25*np.diag(np.ones(3), 1))
    transition_data[0, 0] += 0.25
    transition_data[3, 3] += 0.25
    transition_matrix = tf.constant(transition_data, dtype=self.dtype)

    # Observations of the random walk are unreliable and give the
    # correct position with probability `0.25 + 0.75 * reliability`
    def observation_fn(reliability):
      return np.array(reliability * np.diag(np.ones(4)) +
                      (1 - reliability) * 0.25 * np.ones((4, 4)))

    observation_data = np.array(
        [observation_fn(reliability)
         for reliability in [0.993, 0.994, 0.995, 0.996]])
    observation_probs = tf.constant(observation_data, dtype=self.dtype)

    [num_steps] = self.make_placeholders([7])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    observations = tf.constant([0, 1, 2, 3, 2, 1, 0])
    mask = tf.constant([False, True, True, False, True, True, False])

    inferred_states = model.posterior_mode(observations, mask)

    # This example has been tuned so that there are two local maxima in the
    # space of paths.
    # As the `reliability` parameter increases, the mode switches from one of
    # the two paths to the other.
    expected_states = [[0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 2, 3, 2, 1, 0],
                       [0, 1, 2, 3, 2, 1, 0]]
    self.assertAllEqual(inferred_states, expected_states)

  def test_posterior_marginals_missing_observations(self):
    initial_prob = tf.constant([1., 0., 0., 0.], dtype=self.dtype)

    # This test uses a model with a random walk that can make a change of
    # of -1, 0 or +1 at each step.
    transition_data = [[0.75, 0.25, 0., 0.],
                       [0.25, 0.5, 0.25, 0.],
                       [0., 0.25, 0.5, 0.25],
                       [0.0, 0.0, 0.25, 0.75]]
    transition_matrix = tf.constant(transition_data, dtype=self.dtype)

    observation_data = np.array(np.eye(4))
    observation_probs = tf.constant(observation_data, dtype=self.dtype)

    [num_steps] = self.make_placeholders([7])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    observations = tf.constant([0, 1, 2, 3, 2, 1, 0])
    mask = tf.constant([False, True, True, True, True, True, False])

    marginals = self.evaluate(
        model.posterior_marginals(observations, mask).probs_parameter())
    expected_marginals = [[1., 0., 0., 0.],
                          [21./26, 5./26, 0., 0.],
                          [105./143, 35./143, 3./143, 0.],
                          [1225./1716, 147./572, 49./1716, 1./1716],
                          [105./143, 35./143, 3./143, 0.],
                          [21./26, 5./26, 0., 0.],
                          [1., 0., 0., 0.]]
    self.assertAllClose(marginals, expected_marginals)

  def test_posterior_mode_edge_case_no_transitions(self):
    # Test all eight combinations of a single state that is
    # 1. unmasked/masked
    # 2. observed at state 0/state 1
    # 3. more likely started at state 0/state 1
    initial_prob_data = tf.constant([[0.9, 0.1], [0.1, 0.9]], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([1])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    inferred_state = model.posterior_mode(
        observations=[[[0]], [[1]]],
        mask=[[[[True]]], [[[False]]]])

    expected_state = [[[[0], [1]], [[0], [1]]],
                      [[[0], [0]], [[1], [1]]]]

    self.assertAllEqual(inferred_state, expected_state)

  def test_posterior_marginals_edge_case_no_transitions(self):
    # Test all eight combinations of a single state that is
    # 1. unmasked/masked
    # 2. observed at state 0/state 1
    # 3. more likely started at state 0/state 1
    initial_prob_data = tf.constant([[0.9, 0.1], [0.1, 0.9]], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([1])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    inferred_marginals = self.evaluate(
        model.posterior_marginals(
            observations=tf.constant([[[0]], [[1]]]),
            mask=[[[[True]]], [[[False]]]]).probs_parameter())

    # Result is a [2,2,2] batch of sequences of length 1 of
    # [2]-vectors of probabilities.
    expected_marginals = [[[[[0.9, 0.1]],
                            [[0.1, 0.9]]],
                           [[[0.9, 0.1]],
                            [[0.1, 0.9]]]],
                          [[[[1., 0.]],
                            [[1., 0.]]],
                           [[[0., 1.]],
                            [[0., 1.]]]]]

    self.assertAllClose(inferred_marginals, expected_marginals)

  def test_prior_many_steps(self):
    initial_prob_data = tf.constant([[0.9, 0.1], [0.1, 0.9]], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([3])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    inferred_priors = self.evaluate(model.prior_marginals().probs_parameter())

    expected_priors = [
        [[0.9, 0.1], [0.5, 0.5], [0.5, 0.5]],
        [[0.1, 0.9], [0.5, 0.5], [0.5, 0.5]]
    ]
    self.assertAllClose(inferred_priors, expected_priors)

  def test_prior_one_step(self):
    initial_prob_data = tf.constant([[0.9, 0.1], [0.1, 0.9]], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([1])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    inferred_priors = self.evaluate(model.prior_marginals().probs_parameter())

    expected_priors = [
        [[0.9, 0.1]],
        [[0.1, 0.9]]
    ]

    self.assertAllClose(inferred_priors, expected_priors)

  def test_prior_multiple_steps(self):
    initial_prob_data = tf.constant([[0.9, 0.1], [0.1, 0.9]], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.9, 0.1],
                                          [0.7, 0.3]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([33])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)

    inferred_priors = self.evaluate(
        model.prior_marginals().probs_parameter())[:, -1]

    expected_priors = [[0.875, 0.125], [0.875, 0.125]]

    self.assertAllClose(inferred_priors, expected_priors)

  def test_prior_dynamic_transition(self):
    initial_prob_data = tf.constant([[0.5, 0.5], [0.1, 0.9]], dtype=self.dtype)
    transition_matrix_data = tf.constant([
        [[[0.6, 0.4], [0.5, 0.5]],
         [[0.7, 0.3], [0.4, 0.6]],
         [[0.9, 0.1], [0.3, 0.7]]],
        [[[0.5, 0.5], [0.5, 0.5]],
         [[0.1, 0.9], [0.1, 0.9]],
         [[0.5, 0.5], [0.5, 0.5]]]
    ], dtype=self.dtype)

    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([4])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        time_varying_transition_distribution=True,
        num_steps=num_steps,
        validate_args=True)

    inferred_priors = self.evaluate(model.prior_marginals().probs_parameter())

    expected_priors = [
        [[0.5, 0.5], [0.55, 0.45], [0.565, 0.435], [0.639, 0.361]],
        [[0.1, 0.9], [0.5, 0.5], [0.1, 0.9], [0.5, 0.5]]
    ]

    self.assertAllClose(inferred_priors, expected_priors)

  def test_prior_dynamic_transition_broadcat_init(self):
    initial_prob_data = tf.constant([[0.5, 0.5]], dtype=self.dtype)
    transition_matrix_data = tf.constant([
        [[[0.6, 0.4], [0.5, 0.5]],
         [[0.7, 0.3], [0.4, 0.6]],
         [[0.9, 0.1], [0.3, 0.7]]],
        [[[0.5, 0.5], [0.5, 0.5]],
         [[0.1, 0.9], [0.1, 0.9]],
         [[0.5, 0.5], [0.5, 0.5]]]
    ], dtype=self.dtype)  # [BS, 3, K, K]

    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    [num_steps] = self.make_placeholders([4])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        time_varying_transition_distribution=True,
        num_steps=num_steps,
        validate_args=True)

    inferred_priors = self.evaluate(model.prior_marginals().probs_parameter())

    expected_priors = [
        [[0.5, 0.5], [0.55, 0.45], [0.565, 0.435], [0.639, 0.361]],
        [[0.5, 0.5], [0.5, 0.5], [0.1, 0.9], [0.5, 0.5]]
    ]

    self.assertAllClose(inferred_priors, expected_priors)

  def test_time_dimension_observation_sample_consistent_mean_variance(self):
    initial_prob_data = tf.constant([0.6, 0.1, 0.3], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.2, 0.6, 0.2],
                                          [0.5, 0.3, 0.2],
                                          [0.0, 1.0, 0.0]], dtype=self.dtype)
    observation_locs_data = tf.constant([[0.0, 1.0, 2.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 1.0, 2.0]], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_observation_distribution=True)

    self.run_test_sample_consistent_mean_variance(self.evaluate, model,
                                                  num_samples=100000,
                                                  rtol=0.03)

  def test_dynamic_observation_sample_consistent_mean_variance(self):
    initial_prob_data = tf.constant([0.6, 0.1, 0.3], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.2, 0.6, 0.2],
                                          [0.5, 0.3, 0.2],
                                          [0.0, 1.0, 0.0]], dtype=self.dtype)
    observation_locs_data = tf.constant([[0.0, 1.0, 2.0],
                                         [0.0, 2.0, 1.0],
                                         [2.0, 1.0, 0.0],
                                         [0.0, 1.0, 2.0],
                                         [2.0, 1.0, 0.0]], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_observation_distribution=True)

    self.run_test_sample_consistent_mean_variance(self.evaluate, model,
                                                  num_samples=100000,
                                                  rtol=0.03)

  def test_high_rank_dynamic_observation_mean_shape(self):
    initial_prob_data = tf.constant([[0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3]], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.2, 0.6, 0.2],
                                          [0.5, 0.3, 0.2],
                                          [0.0, 1.0, 0.0]],
                                         dtype=self.dtype)
    observation_locs_data = tf.constant([[0.0, 1.0, 2.0],
                                         [0.0, 2.0, 1.0],
                                         [2.0, 1.0, 0.0],
                                         [0.0, 1.0, 2.0],
                                         [2.0, 1.0, 0.0]],
                                        dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_observation_distribution=True)

    x = model.mean()
    x_shape = self.evaluate(tf.shape(x))

    self.assertAllEqual(x_shape, [7, 5])

  def test_dynamic_observation_mean_and_variance_match(self):
    initial_prob_data = tf.constant([0.6, 0.1, 0.3], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.2, 0.6, 0.2],
                                          [0.5, 0.3, 0.2],
                                          [0.0, 1.0, 0.0]], dtype=self.dtype)
    static_observation_locs_data = tf.constant([0.0, 1.0, 2.0],
                                               dtype=self.dtype)
    dynamic_observation_locs_data = tf.constant(
        [[0.0, 1.0, 2.0],
         [0.0, 1.0, 2.0],
         [0.0, 1.0, 2.0],
         [0.0, 1.0, 2.0],
         [0.0, 1.0, 2.0]], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix, static_observation_locs,
     dynamic_observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         static_observation_locs_data, dynamic_observation_locs_data,
         observation_scale_data])

    [num_steps] = self.make_placeholders([5])
    static_model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=static_observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True)
    dynamic_model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=dynamic_observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_observation_distribution=True)

    self.assertAllClose(self.evaluate(static_model.mean()),
                        self.evaluate(dynamic_model.mean()),
                        0.03)
    self.assertAllClose(self.evaluate(static_model.variance()),
                        self.evaluate(dynamic_model.variance()),
                        0.03)

  def test_dynamic_observation_posterior_is_appropriate(self):
    # This test forces evaluation of the _observation_log_probs method.
    initial_prob_data = tf.constant([0.9, 0.1], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.9, 0.1],
                                          [0.1, 0.9]], dtype=self.dtype)
    observation_scale_data = tf.constant(0.1, dtype=self.dtype)
    observation_loc_data = tf.constant([[-2., 2.],
                                        [-1., 1.],
                                        [0., 0.],
                                        [1., -1.],
                                        [2., -2.]], dtype=self.dtype)
    observations_data = tf.range(5, dtype=self.dtype) - 2

    (initial_prob, transition_matrix, observation_scale, observations,
     observation_locs) = self.make_placeholders(
         [initial_prob_data, transition_matrix_data, observation_scale_data,
          observations_data, observation_loc_data])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_observation_distribution=True)

    # Observations are a line from -2 to 2. Model has two states, one that
    # matches the observations, and one that is the negated version. The
    # maximum confusion should occur on the third step, but state 0 should
    # always be much more probable.
    marginals = self.evaluate(model.posterior_marginals(observations).logits)
    self.assertAllClose(marginals[:, 0], np.full((5,), 0.), atol=0.03)
    self.assertAllLess(marginals[:, 1], -4.)

    # marginals[0:2, 0] both round to 0, so only compare them to marginals[2, 0]
    self.assertGreater(marginals[0, 0], marginals[2, 0])
    self.assertGreater(marginals[1, 0], marginals[2, 0])
    self.assertLess(marginals[2, 0], marginals[3, 0])
    self.assertLess(marginals[2, 0], marginals[4, 0])

    self.assertLess(marginals[0, 1], marginals[1, 1])
    self.assertLess(marginals[1, 1], marginals[2, 1])
    self.assertGreater(marginals[2, 1], marginals[3, 1])
    self.assertGreater(marginals[3, 1], marginals[4, 1])

    mode = self.evaluate(model.posterior_mode(observations))
    self.assertAllEqual(mode, np.full((5,), 0))

  def test_batched_observations_with_dynamic_observation(self):
    initial_prob_data = tf.constant([0.6, 0.1, 0.3],
                                    dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.2, 0.6, 0.2],
                                          [0.5, 0.3, 0.2],
                                          [0.0, 1.0, 0.0]],
                                         dtype=self.dtype)
    observation_locs_data = tf.constant([[0.0, 1.0, 2.0],
                                         [0.0, 2.0, 1.0],
                                         [2.0, 1.0, 0.0],
                                         [0.0, 1.0, 2.0],
                                         [2.0, 1.0, 0.0]],
                                        dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_observation_distribution=True)

    observations = tf.constant(np.random.normal(
        size=[12, 17, 5]), dtype=self.dtype)
    self.evaluate(model.log_prob(observations))
    self.evaluate(model.posterior_marginals(observations).logits)
    self.evaluate(model.posterior_mode(observations))

  def test_dynamic_transition_log_prob_and_posterior_is_appropriate(self):
    initial_prob_data = tf.constant([0.999, 0.001], dtype=self.dtype)
    transition_matrix_data = tf.constant([[[0.999, 0.001],
                                           [0.001, 0.999]],
                                          [[0.999, 0.001],
                                           [0.001, 0.999]],
                                          [[0.001, 0.999],
                                           [0.999, 0.001]]],
                                         dtype=self.dtype)
    observation_scale_data = tf.constant(0.1, dtype=self.dtype)
    observation_loc_data = tf.constant([0.0, 1.0], dtype=self.dtype)
    observations_data_unlikely = tf.constant([0.0, 0.0, 1.0, 1.0],
                                             dtype=self.dtype)
    observations_data_likely = tf.constant([0.0, 0.0, 0.0, 1.0],
                                           dtype=self.dtype)

    (initial_prob, transition_matrix, observation_scale, observations_unlikely,
     observations_likely, observation_locs) = self.make_placeholders(
         [initial_prob_data, transition_matrix_data, observation_scale_data,
          observations_data_unlikely, observations_data_likely,
          observation_loc_data])

    [num_steps] = self.make_placeholders([4])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_transition_distribution=True)
    likely_log_prob = self.evaluate(model.log_prob(observations_likely))
    unlikely_log_prob = self.evaluate(model.log_prob(observations_unlikely))
    self.assertLess(likely_log_prob, 1e1)
    self.assertLess(unlikely_log_prob, 1e-6)
    marginals = self.evaluate(model.posterior_marginals(
        observations_data_likely).probs_parameter())
    self.assertAllClose(marginals, [[1., 0.],
                                    [1., 0.],
                                    [1., 0.],
                                    [0., 1.]], 1e-2)
    mode = self.evaluate(model.posterior_mode(observations_data_likely))
    self.assertAllClose(mode, [0., 0., 0., 1.], 1e-2)

  def test_dynamic_transition_sample_consistent_mean_variance(self):
    initial_prob_data = tf.constant([0.6, 0.1, 0.3], dtype=self.dtype)
    transition_matrix_data = tf.constant([[[0.2, 0.6, 0.2],
                                           [0.5, 0.3, 0.2],
                                           [0.0, 1.0, 0.0]],
                                          [[0.998, 0.001, 0.001],
                                           [0.001, 0.998, 0.001],
                                           [0.001, 0.001, 0.998]]],
                                         dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0, 2.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([3])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_transition_distribution=True)

    self.run_test_sample_consistent_mean_variance(self.evaluate, model,
                                                  num_samples=100000,
                                                  rtol=0.03)

  def test_high_rank_dynamic_transition_sample_consistent_mean_variance(self):
    initial_prob_data = tf.constant([[0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3]], dtype=self.dtype)
    transition_matrix_data = tf.constant([[[0.2, 0.6, 0.2],
                                           [0.5, 0.3, 0.2],
                                           [0.0, 1.0, 0.0]],
                                          [[0.998, 0.001, 0.001],
                                           [0.001, 0.998, 0.001],
                                           [0.001, 0.001, 0.998]]],
                                         dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0, 2.0],
                                        dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([3])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_transition_distribution=True)

    x = model.mean()
    x_shape = self.evaluate(tf.shape(x))

    self.assertAllEqual(x_shape, [7, 3])

  def test_dynamic_transition_mean_and_variance_match(self):
    initial_prob_data = tf.constant([[0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3],
                                     [0.6, 0.1, 0.3]], dtype=self.dtype)
    transition_matrix_data = tf.constant([[[0.2, 0.6, 0.2],
                                           [0.5, 0.3, 0.2],
                                           [0.0, 1.0, 0.0]],
                                          [[0.2, 0.6, 0.2],
                                           [0.5, 0.3, 0.2],
                                           [0.0, 1.0, 0.0]]],
                                         dtype=self.dtype)
    observation_locs_data = tf.constant([[0.0, 1.0, 2.0]],
                                        dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([3])
    static_model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix[0, :, :]),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_transition_distribution=False)
    dynamic_model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_transition_distribution=True)

    self.assertAllClose(self.evaluate(static_model.mean()),
                        self.evaluate(dynamic_model.mean()),
                        0.03)
    self.assertAllClose(self.evaluate(static_model.variance()),
                        self.evaluate(dynamic_model.variance()),
                        0.03)

  def test_batched_observations_with_dynamic_transition(self):
    initial_prob_data = tf.constant([0.6, 0.1, 0.3],
                                    dtype=self.dtype)
    transition_matrix_data = tf.constant(4 * [[[0.2, 0.6, 0.2],
                                               [0.5, 0.3, 0.2],
                                               [0.0, 1.0, 0.0]]],
                                         dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0, 2.0],
                                        dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_transition_distribution=True)

    observations = tf.constant(np.random.normal(
        size=[12, 17, 5]), dtype=self.dtype)
    self.evaluate(model.log_prob(observations))
    self.evaluate(model.posterior_marginals(observations).logits)
    self.evaluate(model.posterior_mode(observations))

  def test_batched_observations_with_dynamic_transition_and_observation(self):
    initial_prob_data = tf.constant([0.6, 0.1, 0.3],
                                    dtype=self.dtype)
    transition_matrix_data = tf.constant(4 * [[[0.2, 0.6, 0.2],
                                               [0.5, 0.3, 0.2],
                                               [0.0, 1.0, 0.0]]],
                                         dtype=self.dtype)
    observation_locs_data = tf.constant(5 * [[0.0, 1.0, 2.0]],
                                        dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    [num_steps] = self.make_placeholders([5])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_transition_distribution=True,
        time_varying_observation_distribution=True)

    observations = tf.constant(np.random.normal(
        size=[12, 17, 5]), dtype=self.dtype)
    self.evaluate(model.log_prob(observations))
    self.evaluate(model.posterior_marginals(observations).logits)
    self.evaluate(model.posterior_mode(observations))

  def test_dynamic_distributions_log_prob_and_posterior_is_appropriate(self):
    initial_prob_data = tf.constant([0.999, 0.001], dtype=self.dtype)
    transition_matrix_data = tf.constant([[[0.999, 0.001],
                                           [0.001, 0.999]],
                                          [[0.999, 0.001],
                                           [0.001, 0.999]],
                                          [[0.001, 0.999],
                                           [0.999, 0.001]]],
                                         dtype=self.dtype)
    observation_scale_data = tf.constant(0.1, dtype=self.dtype)
    observation_loc_data = tf.constant([[0.0, 1.0],
                                        [0.0, 1.0],
                                        [1.0, 0.0],
                                        [1.0, 0.0]], dtype=self.dtype)
    observations_data_unlikely = tf.constant([0.0, 0.0, 0.0, 1.0],
                                             dtype=self.dtype)
    observations_data_likely = tf.constant([0.0, 0.0, 1.0, 0.0],
                                           dtype=self.dtype)

    (initial_prob, transition_matrix, observation_scale, observations_unlikely,
     observations_likely, observation_locs) = self.make_placeholders(
         [initial_prob_data, transition_matrix_data, observation_scale_data,
          observations_data_unlikely, observations_data_likely,
          observation_loc_data])

    [num_steps] = self.make_placeholders([4])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_transition_distribution=True,
        time_varying_observation_distribution=True)

    likely_log_prob = self.evaluate(model.log_prob(observations_likely))
    unlikely_log_prob = self.evaluate(model.log_prob(observations_unlikely))
    self.assertLess(likely_log_prob, 1e1)
    self.assertLess(unlikely_log_prob, 1e-6)
    marginals = self.evaluate(model.posterior_marginals(
        observations_data_likely).probs_parameter())
    self.assertAllClose(marginals, [[1., 0.],
                                    [1., 0.],
                                    [1., 0.],
                                    [0., 1.]], 1e-2)
    mode = self.evaluate(model.posterior_mode(observations_data_likely))
    self.assertAllClose(mode, [0., 0., 0., 1.], 1e-2)

  def test_batch_dynamic_distributions_log_prob_and_posterior(self):
    initial_prob_data = tf.constant([0.999, 0.001], dtype=self.dtype)
    transition_matrix_data = tf.constant(6 * [[[[0.999, 0.001],
                                                [0.001, 0.999]],
                                               [[0.999, 0.001],
                                                [0.001, 0.999]],
                                               [[0.001, 0.999],
                                                [0.999, 0.001]]]],
                                         dtype=self.dtype)
    observation_scale_data = tf.constant(0.1, dtype=self.dtype)
    observation_loc_data = tf.constant(1 * [[[0.0, 1.0],
                                             [0.0, 1.0],
                                             [1.0, 0.0],
                                             [1.0, 0.0]]], dtype=self.dtype)
    observations_data_unlikely = tf.constant([0.0, 0.0, 0.0, 1.0],
                                             dtype=self.dtype)
    observations_data_likely = tf.constant(5 * [6 * [[0.0, 0.0, 1.0, 0.0]]],
                                           dtype=self.dtype)

    (initial_prob, transition_matrix, observation_scale, observations_unlikely,
     observations_likely, observation_locs) = self.make_placeholders(
         [initial_prob_data, transition_matrix_data, observation_scale_data,
          observations_data_unlikely, observations_data_likely,
          observation_loc_data])

    [num_steps] = self.make_placeholders([4])
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True,
        time_varying_transition_distribution=True,
        time_varying_observation_distribution=True)

    likely_log_prob = self.evaluate(model.log_prob(observations_likely))
    unlikely_log_prob = self.evaluate(model.log_prob(observations_unlikely))
    self.assertLess(likely_log_prob[0, 0], 1e1)
    self.assertLess(unlikely_log_prob[0], 1e-6)
    marginals = self.evaluate(model.posterior_marginals(
        observations_data_likely).probs_parameter())
    self.assertAllClose(marginals[0, 0], [[1., 0.],
                                          [1., 0.],
                                          [1., 0.],
                                          [0., 1.]], 1e-2)
    mode = self.evaluate(model.posterior_mode(observations_data_likely))
    self.assertAllClose(mode[0, 0], [0., 0., 0., 1.], 1e-2)

  @parameterized.named_parameters(('', False),
                                  ('_dynamic', True))
  def test_log_prob_mask(self, dynamic):
    num_steps = 4
    initial_prob = tf.constant([0.999, 0.001], dtype=self.dtype)
    transition_matrix = tf.constant([[0.999, 0.001],
                                     [0.001, 0.999]],
                                    dtype=self.dtype)
    observation_scale = tf.constant(0.1, dtype=self.dtype)
    observation_locs = tf.constant([0.0, 1.0], dtype=self.dtype)
    observations_unlikely = tf.constant([0.0, 1.0, 0.0, 1.0],
                                        dtype=self.dtype)
    mask_none = tf.constant([False, False, False, False])
    mask_all = tf.constant([True, True, True, True])
    mask_unlikely = tf.constant([False, True, False, True])
    mask_batch = tf.stack([mask_none, mask_all, mask_unlikely], axis=0)

    if dynamic:
      (initial_prob, transition_matrix, observation_scale,
       observations_unlikely, observation_locs, mask_none, mask_all,
       mask_unlikely, mask_batch, num_steps) = self.make_placeholders(
           [initial_prob, transition_matrix, observation_scale,
            observations_unlikely, observation_locs,
            mask_none, mask_all, mask_unlikely, mask_batch, num_steps])

    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        mask=mask_unlikely,
        validate_args=True)
    log_prob_mask_unlikely = self.evaluate(
        model.log_prob(observations_unlikely))
    log_prob_mask_unlikely_explicit = self.evaluate(
        model.log_prob(observations_unlikely, mask=mask_unlikely))
    self.assertAllEqual(log_prob_mask_unlikely, log_prob_mask_unlikely_explicit)
    log_prob_mask_all = self.evaluate(
        model.log_prob(observations_unlikely, mask=mask_all))
    self.assertAllClose(log_prob_mask_all, 0.)
    log_prob_mask_none = self.evaluate(
        model.log_prob(observations_unlikely, mask=mask_none))
    self.assertLess(log_prob_mask_none, log_prob_mask_unlikely)

    log_prob_mask_batch = self.evaluate(
        model.log_prob(observations_unlikely, mask=mask_batch))
    self.assertAllClose(log_prob_mask_batch,
                        [log_prob_mask_none,
                         log_prob_mask_all,
                         log_prob_mask_unlikely])

    batch_model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(loc=observation_locs, scale=observation_scale),
        num_steps=num_steps,
        mask=mask_batch,
        validate_args=True)
    self.assertAllEqual(batch_model.batch_shape_tensor(), [3])
    self.assertAllClose(
        log_prob_mask_batch,
        batch_model.log_prob(observations_unlikely))


class HiddenMarkovModelTestFloat32(_HiddenMarkovModelTest):
  dtype = tf.float32


class HiddenMarkovModelTestFloat64(_HiddenMarkovModelTest):
  dtype = tf.float64


del _HiddenMarkovModelTest


class _HiddenMarkovModelAssertionTest(
    test_util.VectorDistributionTestHelpers,
    test_util.DiscreteScalarDistributionTestHelpers,
    test_util.TestCase):

  def test_integer_initial_state_assertion(self):
    transition_matrix = np.array([[0.9, 0.1],
                                  [0.1, 0.9]])
    observation_probs = np.array([[1.0, 0.0],
                                  [0.0, 1.0]])

    num_steps = 2
    message = 'is not over integers'
    with self.assertRaisesRegexp(Exception, message):
      model = tfd.HiddenMarkovModel(
          tfd.Normal(loc=0.0, scale=1.0),
          tfd.Categorical(probs=transition_matrix),
          tfd.Categorical(probs=observation_probs),
          num_steps=num_steps,
          validate_args=True)
      _ = self.evaluate(model.sample())

  def test_integer_transition_state_assertion(self):
    initial_prob = np.array([0.9, 0.1])
    observation_probs = np.array([[1.0, 0.0],
                                  [0.0, 1.0]])

    num_steps = 2
    message = 'is not over integers'
    with self.assertRaisesRegexp(Exception, message):
      model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                    tfd.Normal(loc=0.0, scale=1.0),
                                    tfd.Categorical(probs=observation_probs),
                                    num_steps=num_steps,
                                    validate_args=True)
      _ = self.evaluate(model.sample())

  def test_scalar_num_steps_assertion(self):
    initial_prob = np.array([0.9, 0.1])
    transition_matrix = np.array([[0.9, 0.1],
                                  [0.1, 0.9]])
    observation_probs = np.array([[1.0, 0.0],
                                  [0.0, 1.0]])

    num_steps = np.array([2, 3])
    message = '`num_steps` must be a scalar'
    with self.assertRaisesRegexp(Exception, message):
      model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                    tfd.Categorical(probs=transition_matrix),
                                    tfd.Categorical(probs=observation_probs),
                                    num_steps=num_steps,
                                    validate_args=True)
      _ = self.evaluate(model.sample())

  def test_variable_num_steps_assertion(self):
    initial_prob = np.array([0.9, 0.1])
    transition_matrix = np.array([[0.9, 0.1],
                                  [0.1, 0.9]])
    observation_probs = np.array([[1.0, 0.0],
                                  [0.0, 1.0]])

    num_steps = tf.Variable(np.array([2, 3]))
    message = '`num_steps` must be a scalar'
    with self.assertRaisesRegexp(Exception, message):
      model = tfd.HiddenMarkovModel(
          tfd.Categorical(probs=initial_prob),
          tfd.Categorical(probs=transition_matrix),
          tfd.Categorical(probs=observation_probs),
          num_steps=num_steps,
          validate_args=True)
      _ = self.evaluate(model.sample())

  def test_num_steps_greater1_assertion(self):
    initial_prob = np.array([0.9, 0.1])
    transition_matrix = np.array([[0.9, 0.1],
                                  [0.1, 0.9]])
    observation_probs = np.array([[1.0, 0.0],
                                  [0.0, 1.0]])

    num_steps = 0
    message = '`num_steps` must be at least 1'
    with self.assertRaisesRegexp(Exception, message):
      model = tfd.HiddenMarkovModel(
          tfd.Categorical(probs=initial_prob),
          tfd.Categorical(probs=transition_matrix),
          tfd.Categorical(probs=observation_probs),
          num_steps=num_steps,
          validate_args=True)
      _ = self.evaluate(model.sample())

  def test_initial_scalar_assertion(self):
    initial_prob = np.array([0.9, 0.1])
    transition_matrix = np.array([[0.9, 0.1],
                                  [0.1, 0.9]])
    observation_probs = np.array([[1.0, 0.0],
                                  [0.0, 1.0]])

    num_steps = 2
    message = 'must have scalar'
    with self.assertRaisesRegexp(Exception, message):
      model = tfd.HiddenMarkovModel(
          tfd.Sample(tfd.Categorical(probs=initial_prob), sample_shape=2),
          tfd.Categorical(probs=transition_matrix),
          tfd.Categorical(probs=observation_probs),
          num_steps=num_steps,
          validate_args=True)
      _ = self.evaluate(model.sample())

  def test_batch_agreement_assertion(self):
    initial_prob = np.array([[0.9, 0.1],
                             [0.1, 0.9]])
    transition_matrix = np.array([[1.0]])
    observation_probs = np.array([[1.0, 0.0],
                                  [0.0, 1.0]])

    num_steps = 1
    message = 'must agree on'
    with self.assertRaisesRegexp(Exception, message):
      model = tfd.HiddenMarkovModel(
          tfd.Categorical(probs=initial_prob),
          tfd.Categorical(probs=transition_matrix),
          tfd.Categorical(probs=observation_probs),
          num_steps=num_steps,
          validate_args=True)
      _ = self.evaluate(model.sample())

  def test_variable_batch_agreement_assertion(self):
    initial_prob = np.array([[0.9, 0.1],
                             [0.1, 0.9]])
    transition_matrix_data = np.array([[1.0]])
    observation_probs_data = np.array([[1.0, 0.0],
                                       [0.0, 1.0]])
    transition_matrix = tf.Variable(transition_matrix_data)
    observation_probs = tf.Variable(observation_probs_data)
    self.evaluate(transition_matrix.initializer)
    self.evaluate(observation_probs.initializer)

    num_steps = 1
    message = 'must agree on'
    with self.assertRaisesRegexp(Exception, message):
      model = tfd.HiddenMarkovModel(
          tfd.Categorical(probs=initial_prob),
          tfd.Categorical(probs=transition_matrix),
          tfd.Categorical(probs=observation_probs),
          num_steps=num_steps,
          validate_args=True)
      _ = self.evaluate(model.sample())

  def test_modified_variable_batch_agreement_assertion(self):
    initial_prob = np.array([[0.9, 0.1],
                             [0.1, 0.9]])
    transition_matrix_data = np.array([[1.0, 0.0],
                                       [0.0, 1.0]])
    transition_matrix_data2 = np.array([[1.0]])
    observation_probs_data = np.array([[1.0, 0.0],
                                       [0.0, 1.0]])
    transition_matrix = tf.Variable(transition_matrix_data,
                                    shape=tf.TensorShape(None))
    observation_probs = tf.Variable(observation_probs_data,
                                    shape=tf.TensorShape(None))
    self.evaluate(transition_matrix.initializer)
    self.evaluate(observation_probs.initializer)

    num_steps = 1
    message = 'transition_distribution` and `observation_distribution` must'
    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs),
        num_steps=num_steps,
        validate_args=True)
    with self.assertRaisesRegexp(Exception, message):
      with tf.control_dependencies([
          transition_matrix.assign(transition_matrix_data2)]):
        _ = self.evaluate(model.sample())

  def test_non_scalar_transition_batch(self):
    initial_prob = tf.constant([0.6, 0.4])
    # The HMM class expect a `Categorical` distribution for each state.
    # This test provides only a single scalar distribution.
    # For this test to pass it must raise an appropriate exception.
    transition_matrix = tf.constant([0.6, 0.4])
    observation_locs = tf.constant(0.0)
    observation_scale = tf.constant(0.5)

    num_steps = 4

    with self.assertRaisesRegexp(Exception, 'can\'t have scalar batches'):
      model = tfd.HiddenMarkovModel(
          tfd.Categorical(probs=initial_prob),
          tfd.Categorical(probs=transition_matrix),
          tfd.Normal(observation_locs, scale=observation_scale),
          num_steps=num_steps,
          validate_args=True)
      self.evaluate(model.mean())

  def test_variable_non_scalar_transition_batch(self):
    initial_prob = tf.constant([0.6, 0.4])
    # The HMM class expect a `Categorical` distribution for each state.
    # This test provides only a single scalar distribution.
    # For this test to pass it must raise an appropriate exception.
    transition_matrix_data = tf.constant([0.6, 0.4])
    transition_matrix = tf.Variable(transition_matrix_data)
    self.evaluate(transition_matrix.initializer)
    observation_locs = tf.constant([0.0, 1.0])
    observation_scale = tf.constant([0.5, 0.5])

    num_steps = 4

    with self.assertRaisesRegexp(Exception, 'can\'t have scalar batches'):
      model = tfd.HiddenMarkovModel(
          tfd.Categorical(probs=initial_prob),
          tfd.Categorical(probs=transition_matrix),
          tfd.Normal(loc=observation_locs, scale=observation_scale),
          num_steps=num_steps,
          validate_args=True)
      self.evaluate(model.mean())

  def test_modified_variable_non_scalar_transition_batch(self):
    initial_prob = tf.constant([0.6, 0.4])
    transition_matrix_data = tf.constant([[0.6, 0.4], [0.5, 0.5]])
    transition_matrix = tf.Variable(
        transition_matrix_data,
        shape=tf.TensorShape(None))
    transition_matrix_data2 = tf.constant([0.6, 0.4])
    self.evaluate(transition_matrix.initializer)
    observation_locs = tf.constant([0.0, 1.0])
    observation_scale = tf.constant([0.5, 0.5])

    num_steps = 4

    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Normal(observation_locs, scale=observation_scale),
        num_steps=num_steps,
        validate_args=True)

    with self.assertRaisesRegexp(
        Exception,
        'have scalar batches'):
      with tf.control_dependencies([
          transition_matrix.assign(transition_matrix_data2)]):
        self.evaluate(model.mean())

  def test_github_issue_854(self):
    nstates = 3
    data = np.random.randint(low=0, high=10, size=(5, 7, 11))
    p_init = tfd.Categorical(probs=np.float32(np.ones(nstates) / nstates))
    pswitch = 0.05
    pt = pswitch / (nstates - 1) * np.ones([nstates, nstates])
    np.fill_diagonal(pt, 1 - pswitch)
    p_trans = tfd.Categorical(probs=np.float32(pt))
    # prior on NB probability
    p_nb = self.evaluate(tfd.Beta(2, 5).sample([nstates, data.shape[-1]],
                                               seed=test_util.test_seed()))
    p_emission = tfd.Independent(tfd.NegativeBinomial(1, probs=p_nb),
                                 reinterpreted_batch_ndims=1)
    hmm = tfd.HiddenMarkovModel(
        initial_distribution=p_init,
        transition_distribution=p_trans,
        observation_distribution=p_emission,
        num_steps=data.shape[-2])

    self.assertAllEqual(data.shape[-2:],
                        tf.shape(hmm.sample(seed=test_util.test_seed())))
    self.assertAllEqual(data.shape[:1],
                        tf.shape(hmm.log_prob(data)))

  def test_time_varying_transition_batch_size(self):
    initial_prob = tf.constant([0.6, 0.4])
    transition_matrix = tf.constant(7 * [[[0.6, 0.4], [0.6, 0.4]]])
    observation_locs = tf.constant([0.0, 1.0])
    observation_scale = tf.constant(0.5)

    num_steps = 5

    with self.assertRaisesRegexp(Exception, 'matches num_steps - 1.'):
      model = tfd.HiddenMarkovModel(
          tfd.Categorical(probs=initial_prob),
          tfd.Categorical(probs=transition_matrix),
          tfd.Normal(observation_locs, scale=observation_scale),
          num_steps=num_steps,
          time_varying_transition_distribution=True,
          validate_args=True)
      self.evaluate(model.mean())

  def test_time_varying_observation_batch_size(self):
    initial_prob = tf.constant([0.6, 0.4])
    transition_matrix = tf.constant([[0.6, 0.4], [0.6, 0.4]])
    observation_locs = tf.constant(7 * [[0.0, 1.0]])
    observation_scale = tf.constant(0.5)

    num_steps = 6

    with self.assertRaisesRegexp(Exception, 'matches num_steps.'):
      model = tfd.HiddenMarkovModel(
          tfd.Categorical(probs=initial_prob),
          tfd.Categorical(probs=transition_matrix),
          tfd.Normal(observation_locs, scale=observation_scale),
          num_steps=num_steps,
          time_varying_observation_distribution=True,
          validate_args=True)
      self.evaluate(model.mean())


class HiddenMarkovModelJaxTest(test_util.TestCase):

  def test_jit(self):
    if not JAX_MODE:
      self.skipTest('JAX-only test')

    import jax  # pylint: disable=g-import-not-at-top

    @jax.jit
    def test(data):
      p_c = tf.constant([0.1, 0.2])
      p_e = tf.constant([0.2, 0.3])
      one = tf.ones_like(p_c)
      zero = tf.zeros_like(p_c)

      dist = tfd.HiddenMarkovModel(
          initial_distribution=tfd.Bernoulli(probs=0.),
          transition_distribution=tfd.Bernoulli(probs=tf.stack([p_c, one], -1)),
          observation_distribution=tfd.Bernoulli(
              probs=tf.stack([p_e, zero], -1)),
          num_steps=data.shape[-1])
      lp = dist.log_prob(data)
      pom = dist.posterior_mode(data)
      s = dist.sample(seed=jax.random.PRNGKey(0))
      prm = dist.prior_marginals()
      pom2 = dist.posterior_marginals(data)
      return lp, pom, s, prm, pom2

    data = tf.ones(5)
    test(data)


if __name__ == '__main__':
  test_util.main()
