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

from __future__ import absolute_import
from __future__ import division

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
tfd = tfp.distributions
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

# pylint: disable=no-member


@test_util.run_all_in_graph_and_eager_modes
class _HiddenMarkovModelTest(
    tfp_test_util.VectorDistributionTestHelpers,
    tfp_test_util.DiscreteScalarDistributionTestHelpers):

  @staticmethod
  def make_placeholders(constants):
    return [
        tf.compat.v1.placeholder_with_default(constant, shape=None)
        for constant in constants
    ]

  def test_non_agreeing_states(self):
    initial_prob_data = tf.constant([0.6, 0.4], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.6, 0.4],
                                          [0.3, 0.7]], dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0, 2.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    with self.assertRaisesWithPredicateMatch(Exception,
                                             lambda e: "must agree" in str(e)):
      model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                    tfd.Categorical(probs=transition_matrix),
                                    tfd.Normal(observation_locs,
                                               scale=observation_scale),
                                    num_steps=4,
                                    validate_args=True)
      self.evaluate(model.mean())

  def test_non_scalar_transition_batch(self):
    initial_prob_data = tf.constant([0.6, 0.4], dtype=self.dtype)
    transition_matrix_data = tf.constant([0.6, 0.4], dtype=self.dtype)
    observation_locs_data = tf.constant(0.0, dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    with self.assertRaisesWithPredicateMatch(
        Exception,
        lambda e: "scalar batches" in str(e)):
      model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                    tfd.Categorical(probs=transition_matrix),
                                    tfd.Normal(observation_locs,
                                               scale=observation_scale),
                                    num_steps=4,
                                    validate_args=True)
      self.evaluate(model.mean())

  def test_consistency(self):
    initial_prob_data = tf.constant([0.6, 0.4], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.6, 0.4],
                                          [0.3, 0.7]], dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Normal(loc=observation_locs,
                                             scale=observation_scale),
                                  num_steps=3,
                                  validate_args=True)

    self.run_test_sample_consistent_log_prob(self.evaluate, model,
                                             num_samples=100000,
                                             center=0.5, radius=0.5,
                                             rtol=0.05)

  def test_broadcast_initial_probs(self):
    initial_prob_data = tf.constant([0.6, 0.4], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.6, 0.4],
                                          [0.3, 0.7]], dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Normal(loc=observation_locs,
                                             scale=observation_scale),
                                  num_steps=3)

    self.run_test_sample_consistent_log_prob(self.evaluate, model,
                                             num_samples=100000,
                                             center=0.5, radius=1.,
                                             rtol=0.02)

  def test_broadcast_transitions(self):
    initial_prob_data = tf.constant([0.6, 0.4], dtype=self.dtype)
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

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Normal(loc=observation_locs,
                                             scale=observation_scale),
                                  num_steps=3)

    self.run_test_sample_consistent_log_prob(self.evaluate, model,
                                             num_samples=100000,
                                             center=0.5, radius=1.,
                                             rtol=2e-2)

  def test_broadcast_observations(self):
    initial_prob_data = tf.constant([0.6, 0.4], dtype=self.dtype)
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

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Normal(loc=observation_locs,
                                             scale=observation_scale),
                                  num_steps=3)

    self.run_test_sample_consistent_log_prob(self.evaluate, model,
                                             num_samples=100000,
                                             center=0.5, radius=1.,
                                             rtol=2e-2)

  def test_edge_case_sample_n_no_transitions(self):
    initial_prob_data = tf.constant([0.5, 0.5], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Categorical(probs=observation_probs),
                                  num_steps=1)

    x = model._sample_n(1)
    x_shape = self.evaluate(tf.shape(input=x))

    self.assertAllEqual(x_shape, [1, 1])

  def test_edge_case_log_prob_no_transitions(self):
    initial_prob_data = tf.constant([0.5, 0.5], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_probs_data = tf.constant([[1.0, 0.0],
                                          [0.0, 1.0]], dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_probs) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_probs_data])

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Categorical(probs=observation_probs),
                                  num_steps=1)

    x = model.log_prob([0])

    self.assertAllClose(x, np.log(0.5), rtol=1e-5, atol=0.0)

  def test_edge_case_mean_no_transitions(self):
    initial_prob_data = tf.constant([0.5, 0.5], dtype=self.dtype)
    transition_matrix_data = tf.constant([[0.5, 0.5],
                                          [0.5, 0.5]], dtype=self.dtype)
    observation_locs_data = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_data = tf.constant(0.5, dtype=self.dtype)

    (initial_prob, transition_matrix,
     observation_locs, observation_scale) = self.make_placeholders([
         initial_prob_data, transition_matrix_data,
         observation_locs_data, observation_scale_data])

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Normal(loc=observation_locs,
                                             scale=observation_scale),
                                  num_steps=1)

    x = model.mean()
    x_shape = self.evaluate(tf.shape(input=x))

    self.assertAllEqual(x_shape, [1])

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

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Categorical(probs=observation_probs),
                                  num_steps=5)

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
    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Categorical(probs=observation_probs),
                                  num_steps=5)

    examples = [tf.zeros(5, dtype=tf.int32), tf.ones(5, dtype=tf.int32)]
    examples = tf.broadcast_to(examples, [7, 3, 2, 5])
    computed_log_prob = model.log_prob(examples)

    expected_log_prob = tf.broadcast_to([np.log(.5**5)], [7, 3, 2])
    self.assertAllClose(computed_log_prob, expected_log_prob,
                        rtol=1e-4, atol=0.0)

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

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.MultivariateNormalDiag(
                                      loc=observation_locs),
                                  num_steps=7)

    x = model.mean()
    x_shape = self.evaluate(tf.shape(input=x))

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

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Normal(loc=observation_locs,
                                             scale=observation_scale),
                                  num_steps=5)

    self.run_test_sample_consistent_mean_variance(self.evaluate, model,
                                                  num_samples=100000,
                                                  rtol=0.02)

  def test_single_path_posterior_marginals(self):

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

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Categorical(probs=observation_probs),
                                  num_steps=7)

    observations = [0, 1, 1, 1, 1, 1, 2]

    probs = model.posterior_marginals(observations).probs
    expected_probs = np.eye(9)[[0, 1, 2, 3, 4, 6, 8]]

    self.assertAllClose(probs, expected_probs, rtol=1e-4, atol=0.0)

  def test_broadcast_posterior_marginals(self):
    def compute_posterior_marginals(initial_prob, transition_matrix,
                                    observation_probs, observations):
      model = tfd.HiddenMarkovModel(
          tfd.Categorical(probs=tf.constant(initial_prob,
                                            dtype=self.dtype)),
          tfd.Categorical(probs=tf.constant(transition_matrix,
                                            dtype=self.dtype)),
          tfd.Categorical(probs=tf.constant(observation_probs,
                                            dtype=self.dtype)),
          num_steps=5)

      return model.posterior_marginals(observations).probs

    initial_prob = [0.5, 0.5]
    transition_matrix = [[0.7, 0.3],
                         [0.3, 0.7]]
    observation_probs = [[0.9, 0.1],
                         [0.2, 0.8]]
    observations = [0, 0, 1, 0, 0]

    # Test broadcasting by using a scalar batch for every argument apart
    # from one which is a batch with two elements.
    # Test all four ways we can pick one argument to be a size two batch.
    # They should all give the same result.
    args = [initial_prob, transition_matrix, observation_probs, observations]

    def repeat_ith(i):
      return compute_posterior_marginals(*[[arg, arg] if i == j else arg
                                           for j, arg in enumerate(args)])

    results = [repeat_ith(i) for i in range(len(args))]

    expected_probs = [[0.8673, 0.8204, 0.3075, 0.8204, 0.8673],
                      [0.1327, 0.1796, 0.6925, 0.1796, 0.1327]]

    probs = tf.identity(results)
    expected_probs = tf.broadcast_to(
        tf.transpose(a=expected_probs), [4, 2, 5, 2])

    self.assertAllClose(probs, expected_probs, rtol=1e-3, atol=0.0)


class HiddenMarkovModelTestFloat32(tf.test.TestCase, _HiddenMarkovModelTest):
  dtype = tf.float32


class HiddenMarkovModelTestFloat64(tf.test.TestCase, _HiddenMarkovModelTest):
  dtype = tf.float64

if __name__ == "__main__":
  tf.test.main()
