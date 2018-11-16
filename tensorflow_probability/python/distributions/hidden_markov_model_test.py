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

from tensorflow_probability.python.internal import test_util
tfd = tfp.distributions
tfe = tf.contrib.eager

# pylint: disable=no-member


@tfe.run_all_tests_in_graph_and_eager_modes
class _HiddenMarkovModelTest(test_util.VectorDistributionTestHelpers,
                             test_util.DiscreteScalarDistributionTestHelpers):

  def test_non_agreeing_states(self):
    initial_prob_ = tf.constant([0.6, 0.4], dtype=self.dtype)
    transition_matrix_ = tf.constant([[0.6, 0.4],
                                      [0.3, 0.7]], dtype=self.dtype)
    observation_locs_ = tf.constant([0.0, 1.0, 2.0], dtype=self.dtype)
    observation_scale_ = tf.constant(0.5, dtype=self.dtype)

    initial_prob = tf.placeholder_with_default(initial_prob_,
                                               shape=None)
    transition_matrix = tf.placeholder_with_default(transition_matrix_,
                                                    shape=None)
    observation_locs = tf.placeholder_with_default(observation_locs_,
                                                   shape=None)
    observation_scale = tf.placeholder_with_default(observation_scale_,
                                                    shape=None)

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
    initial_prob_ = tf.constant([0.6, 0.4], dtype=self.dtype)
    transition_matrix_ = tf.constant([0.6, 0.4], dtype=self.dtype)
    observation_locs_ = tf.constant(0.0, dtype=self.dtype)
    observation_scale_ = tf.constant(0.5, dtype=self.dtype)

    initial_prob = tf.placeholder_with_default(initial_prob_,
                                               shape=None)
    transition_matrix = tf.placeholder_with_default(transition_matrix_,
                                                    shape=None)
    observation_locs = tf.placeholder_with_default(observation_locs_,
                                                   shape=None)
    observation_scale = tf.placeholder_with_default(observation_scale_,
                                                    shape=None)

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
    initial_prob_ = tf.constant([0.6, 0.4], dtype=self.dtype)
    transition_matrix_ = tf.constant([[0.6, 0.4],
                                      [0.3, 0.7]], dtype=self.dtype)
    observation_locs_ = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_ = tf.constant(0.5, dtype=self.dtype)

    initial_prob = tf.placeholder_with_default(initial_prob_,
                                               shape=None)
    transition_matrix = tf.placeholder_with_default(transition_matrix_,
                                                    shape=None)
    observation_locs = tf.placeholder_with_default(observation_locs_,
                                                   shape=None)
    observation_scale = tf.placeholder_with_default(observation_scale_,
                                                    shape=None)

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
    initial_prob_ = tf.constant([0.6, 0.4], dtype=self.dtype)
    transition_matrix_ = tf.constant([[0.6, 0.4],
                                      [0.3, 0.7]], dtype=self.dtype)
    observation_locs_ = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_ = tf.constant(0.5, dtype=self.dtype)

    initial_prob = tf.placeholder_with_default(initial_prob_,
                                               shape=None)
    transition_matrix = tf.placeholder_with_default(transition_matrix_,
                                                    shape=None)
    observation_locs = tf.placeholder_with_default(observation_locs_,
                                                   shape=None)
    observation_scale = tf.placeholder_with_default(observation_scale_,
                                                    shape=None)

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
    initial_prob_ = tf.constant([0.6, 0.4], dtype=self.dtype)
    transition_matrix_ = tf.constant([[[0.8, 0.2],
                                       [0.3, 0.7]],
                                      [[0.9, 0.1],
                                       [0.2, 0.8]]], dtype=self.dtype)
    observation_locs_ = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale_ = tf.constant(0.5, dtype=self.dtype)

    initial_prob = tf.placeholder_with_default(initial_prob_,
                                               shape=None)
    transition_matrix = tf.placeholder_with_default(transition_matrix_,
                                                    shape=None)
    observation_locs = tf.placeholder_with_default(observation_locs_,
                                                   shape=None)
    observation_scale = tf.placeholder_with_default(observation_scale_,
                                                    shape=None)

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
    initial_prob_ = tf.constant([0.6, 0.4], dtype=self.dtype)
    transition_matrix_ = tf.constant([[[0.8, 0.2],
                                       [0.3, 0.7]],
                                      [[0.9, 0.1],
                                       [0.2, 0.8]]], dtype=self.dtype)
    observation_locs_ = tf.constant([[0.9, 0.1],
                                     [0.2, 0.8]], dtype=self.dtype)
    observation_scale_ = tf.constant(0.5, dtype=self.dtype)

    initial_prob = tf.placeholder_with_default(initial_prob_,
                                               shape=None)
    transition_matrix = tf.placeholder_with_default(transition_matrix_,
                                                    shape=None)
    observation_locs = tf.placeholder_with_default(observation_locs_,
                                                   shape=None)
    observation_scale = tf.placeholder_with_default(observation_scale_,
                                                    shape=None)

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
    initial_prob = tf.constant([0.5, 0.5], dtype=self.dtype)
    transition_matrix = tf.constant([[0.5, 0.5],
                                     [0.5, 0.5]], dtype=self.dtype)
    observation_probs = tf.constant([[1.0, 0.0],
                                     [0.0, 1.0]], dtype=self.dtype)

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Categorical(probs=observation_probs),
                                  num_steps=1)

    x = model._sample_n(1)
    x_shape = self.evaluate(tf.shape(x))

    self.assertAllEqual(x_shape, [1, 1])

  def test_edge_case_log_prob_no_transitions(self):
    initial_prob = tf.constant([0.5, 0.5], dtype=self.dtype)
    transition_matrix = tf.constant([[0.5, 0.5],
                                     [0.5, 0.5]], dtype=self.dtype)
    observation_probs = tf.constant([[1.0, 0.0],
                                     [0.0, 1.0]], dtype=self.dtype)

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Categorical(probs=observation_probs),
                                  num_steps=1)

    x = model.log_prob([0])

    self.assertAllClose(x, np.log(0.5), rtol=1e-5, atol=0.0)

  def test_edge_case_mean_no_transitions(self):
    initial_prob = tf.constant([0.5, 0.5], dtype=self.dtype)
    transition_matrix = tf.constant([[0.5, 0.5],
                                     [0.5, 0.5]], dtype=self.dtype)
    observation_locs = tf.constant([0.0, 1.0], dtype=self.dtype)
    observation_scale = tf.constant(0.5, dtype=self.dtype)

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Normal(loc=observation_locs,
                                             scale=observation_scale),
                                  num_steps=1)

    x = model.mean()
    x_shape = self.evaluate(tf.shape(x))

    self.assertAllEqual(x_shape, [1])

  def test_coin_tosses(self):
    initial_prob = tf.constant([0.5, 0.5], dtype=self.dtype)
    transition_matrix = tf.constant([[0.5, 0.5],
                                     [0.5, 0.5]], dtype=self.dtype)
    observation_probs = tf.constant([[1.0, 0.0],
                                     [0.0, 1.0]], dtype=self.dtype)

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Categorical(probs=observation_probs),
                                  num_steps=5)

    x = model.log_prob([0, 0, 0, 0, 0])

    self.assertAllClose(x, np.log(.5**5), rtol=1e-5, atol=0.0)

  def test_coin_toss_batch(self):
    initial_prob_ = tf.constant([0.5, 0.5], dtype=self.dtype)
    transition_matrix = tf.constant([[0.5, 0.5],
                                     [0.5, 0.5]], dtype=self.dtype)
    observation_matrix = tf.constant([[1.0, 0.0],
                                      [0.0, 1.0]], dtype=self.dtype)

    initial_prob = tf.broadcast_to(initial_prob_, [3, 2, 2])
    transition_matrix = tf.broadcast_to(transition_matrix, [3, 2, 2, 2])
    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Categorical(probs=observation_matrix),
                                  num_steps=5)

    examples = [tf.zeros(5, dtype=tf.int32), tf.ones(5, dtype=tf.int32)]
    examples = tf.broadcast_to(examples, [7, 3, 2, 5])
    computed_log_prob = model.log_prob(examples)

    expected_log_prob = tf.broadcast_to([np.log(.5**5)], [7, 3, 2])
    self.assertAllClose(computed_log_prob, expected_log_prob,
                        rtol=1e-4, atol=0.0)

  def test_batch_mean_shape(self):
    initial_prob = tf.constant([[0.8, 0.2],
                                [0.5, 0.5],
                                [0.2, 0.8]], dtype=self.dtype)
    transition_matrix = tf.constant([[0.7, 0.3],
                                     [0.2, 0.8]], dtype=self.dtype)
    observation_locs = tf.constant([[0.0, 0.0],
                                    [10.0, 10.0]], dtype=self.dtype)

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.MultivariateNormalDiag(
                                      loc=observation_locs),
                                  num_steps=7)

    x = model.mean()
    x_shape = self.evaluate(tf.shape(x))

    self.assertAllEqual(x_shape, [3, 7, 2])

  def test_mean_and_variance(self):
    initial_prob = tf.constant([0.6, 0.1, 0.3], dtype=self.dtype)
    transition_matrix = tf.constant([[0.2, 0.6, 0.2],
                                     [0.5, 0.3, 0.2],
                                     [0.0, 1.0, 0.0]], dtype=self.dtype)
    observation_locs = tf.constant([0.0, 1.0, 2.0], dtype=self.dtype)
    observation_scale = tf.constant(0.5, dtype=self.dtype)

    model = tfd.HiddenMarkovModel(tfd.Categorical(probs=initial_prob),
                                  tfd.Categorical(probs=transition_matrix),
                                  tfd.Normal(loc=observation_locs,
                                             scale=observation_scale),
                                  num_steps=5)

    self.run_test_sample_consistent_mean_variance(self.evaluate, model,
                                                  num_samples=100000,
                                                  rtol=0.02)


class HiddenMarkovModelTestFloat32(tf.test.TestCase, _HiddenMarkovModelTest):
  dtype = tf.float32


class HiddenMarkovModelTestFloat64(tf.test.TestCase, _HiddenMarkovModelTest):
  dtype = tf.float64

if __name__ == "__main__":
  tf.test.main()

