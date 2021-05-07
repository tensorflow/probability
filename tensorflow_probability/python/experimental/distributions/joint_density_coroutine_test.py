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
"""Tests for JointDensityCoroutine."""

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfde = tfp.experimental.distributions


@test_util.test_graph_and_eager_modes
class JointDensityCoroutineTest(test_util.TestCase):

  def test_constructor(self):
    root = tfd.JointDistributionCoroutine.Root

    @tfde.JointDensityCoroutine
    def model():
      w = yield root(tfd.Normal(0., 1.))
      yield tfd.Normal(w, 1.)
      yield tfde.IncrementLogProb(5.)

    return model

  def test_dtype(self):
    root = tfd.JointDistributionCoroutine.Root

    @tfde.JointDensityCoroutine
    def model():
      w = yield root(tfd.Normal(0., 1.))
      yield tfd.Normal(w, 1.)
      yield root(tfde.IncrementLogProb(5.))

    self.assertEqual((tf.float32, tf.float32, tf.float32), model.dtype)

  def test_sample_shape(self):
    root = tfd.JointDistributionCoroutine.Root

    @tfde.JointDensityCoroutine
    def model():
      w = yield root(tfd.Normal(0., 1.))
      yield tfd.Normal(w, 1.)
      yield root(tfde.IncrementLogProb(5.))

    seed = test_util.test_seed()
    sample_shape = [10]
    _, _, increment_log_prob_sample = model.sample(
        sample_shape=sample_shape, seed=seed)
    self.assertEqual(increment_log_prob_sample.shape,
                     tf.TensorShape(sample_shape) + tf.TensorShape([0]))

  def test_log_prob(self):
    root = tfd.JointDistributionCoroutine.Root

    @tfde.JointDensityCoroutine
    def model():
      w = yield root(tfd.Normal(0., 1.))
      yield tfd.Normal(w, 1.)

    with self.assertRaises(AttributeError):
      self.evaluate(model.log_prob(2., 3.))

  def test_joint_distribution_coroutine(self):
    # This tests that ImplementLogProb doesn't work with
    # Joint>>>Distribution<<<Coroutine.
    root = tfd.JointDistributionCoroutine.Root
    increment_amount = 5.

    @tfd.JointDistributionCoroutine
    def model():
      w = yield root(tfd.Normal(0., 1.))
      yield tfd.Normal(w, 1.)
      yield root(tfde.IncrementLogProb(increment_amount))

    with self.assertRaises(AttributeError):
      self.evaluate(model.log_prob(2., 3., ()))

  def test_unnormalized_log_prob(self):
    root = tfd.JointDistributionCoroutine.Root
    increment_amount = 5.

    @tfde.JointDensityCoroutine
    def model_with_increment():
      w = yield root(tfd.Normal(0., 1.))
      yield tfd.Normal(w, 1.)
      yield root(tfde.IncrementLogProb(increment_amount))

    @tfd.JointDistributionCoroutine
    def model_without_increment():
      w = yield root(tfd.Normal(0., 1.))
      yield tfd.Normal(w, 1.)

    self.assertAlmostEqual(
        self.evaluate(model_with_increment.unnormalized_log_prob((2., 3., ()))),
        self.evaluate(model_without_increment.log_prob(
            (2., 3.))) + increment_amount)

  def test_unnormalized_log_prob_and_root(self):
    root = tfd.JointDistributionCoroutine.Root
    increment_amount = 5.

    @tfde.JointDensityCoroutine
    def model_with_increment():
      yield root(tfde.IncrementLogProb(increment_amount))
      w = yield root(tfd.Normal(0., 1.))
      yield tfd.Normal(w, 1.)

    @tfd.JointDistributionCoroutine
    def model_without_increment():
      w = yield root(tfd.Normal(0., 1.))
      yield tfd.Normal(w, 1.)

    self.assertAlmostEqual(
        self.evaluate(model_with_increment.unnormalized_log_prob(((), 2., 3.))),
        self.evaluate(model_without_increment.log_prob(
            (2., 3.))) + increment_amount,
        places=6)

  def test_unnormalized_log_prob_batch(self):
    root = tfd.JointDistributionCoroutine.Root
    increment_amount = tf.convert_to_tensor([5., 6.])

    @tfde.JointDensityCoroutine
    def model_with_increment():
      w = yield root(tfd.Normal([0., 1.], 2.))
      yield tfd.Normal(w, 3.)
      yield root(tfde.IncrementLogProb(increment_amount))

    @tfd.JointDistributionCoroutine
    def model_without_increment():
      w = yield root(tfd.Normal([0., 1.], 2.))
      yield tfd.Normal(w, 3.)

    self.assertAllClose(
        self.evaluate(
            model_with_increment.unnormalized_log_prob(([4., 5.], [6.,
                                                                   7.], ()))),
        self.evaluate(model_without_increment.log_prob(
            ([4., 5.], [6., 7.]))) + increment_amount)


if __name__ == '__main__':
  tf.test.main()
