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
"""Tests for IncrementLogProb."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfed = tfp.experimental.distributions


@test_util.test_graph_and_eager_modes
class JointDensityCoroutineTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('_tensor', 5.),
      ('_callable', lambda: 5.),
  )
  def test_dtype(self, increment_amount):

    @tfd.JointDistributionCoroutine
    def model():
      w = yield tfd.Normal(0., 1.)
      yield tfd.Normal(w, 1.)
      yield tfed.IncrementLogProb(increment_amount)

    self.assertEqual((tf.float32, tf.float32, tf.float32), model.dtype)

  @parameterized.named_parameters(
      ('_tensor', [1., 5.]),
      ('_callable', lambda: [1., 5.]),
  )
  def test_batch_shape(self, increment_amount):
    model = tfed.IncrementLogProb(increment_amount)
    self.assertEqual([2], model.batch_shape)
    self.assertEqual([2], model.batch_shape_tensor())

  def test_sample_shape(self):
    root = tfd.JointDistributionCoroutine.Root

    @tfd.JointDistributionCoroutine
    def model():
      w = yield root(tfd.Normal(0., 1.))
      yield tfd.Normal(w, 1.)
      yield root(tfed.IncrementLogProb(5.))

    seed = test_util.test_seed()
    sample_shape = [10]
    _, _, increment_log_prob_sample = model.sample(
        sample_shape=sample_shape, seed=seed)
    self.assertEqual(increment_log_prob_sample.shape,
                     tf.TensorShape(sample_shape) + tf.TensorShape([0]))

  def test_unnormalized_log_prob(self):
    increment_amount = 5.

    @tfd.JointDistributionCoroutine
    def model_with_increment():
      w = yield tfd.Normal(0., 1.)
      yield tfd.Normal(w, 1.)
      yield tfed.IncrementLogProb(increment_amount)

    @tfd.JointDistributionCoroutine
    def model_without_increment():
      w = yield tfd.Normal(0., 1.)
      yield tfd.Normal(w, 1.)

    self.assertAlmostEqual(
        self.evaluate(model_with_increment.unnormalized_log_prob((2., 3., ()))),
        self.evaluate(model_without_increment.log_prob(
            (2., 3.))) + increment_amount)

  def test_unnormalized_log_prob_kwargs(self):
    dist = tfed.IncrementLogProb(lambda value: value, value=5.)
    self.assertAllClose(5., dist.unnormalized_log_prob(()))

  def test_unnormalized_log_prob_batch(self):
    increment_amount = tf.convert_to_tensor([5., 6.])

    @tfd.JointDistributionCoroutine
    def model_with_increment():
      w = yield tfd.Normal([0., 1.], 2.)
      yield tfd.Normal(w, 3.)
      yield tfed.IncrementLogProb(increment_amount)

    @tfd.JointDistributionCoroutine
    def model_without_increment():
      w = yield tfd.Normal([0., 1.], 2.)
      yield tfd.Normal(w, 3.)

    self.assertAllClose(
        self.evaluate(
            model_with_increment.unnormalized_log_prob(([4., 5.], [6.,
                                                                   7.], ()))),
        self.evaluate(model_without_increment.log_prob(
            ([4., 5.], [6., 7.]))) + increment_amount)

  def test_custom_log_prob_ratio(self):

    def log_prob_ratio_fn(p_kwargs, q_kwargs):
      return 2 * (p_kwargs['value'] - q_kwargs['value'])

    p = tfed.IncrementLogProb(
        lambda value: value, log_prob_ratio_fn=log_prob_ratio_fn, value=3.)
    q = tfed.IncrementLogProb(
        lambda value: value, log_prob_ratio_fn=log_prob_ratio_fn, value=1.)
    q2 = tfed.IncrementLogProb(
        lambda value: value, value=1.)

    self.assertAllClose(2. * (3. - 1.), tfed.log_prob_ratio(p, (), q, ()))
    self.assertAllClose((3. - 1.), tfed.log_prob_ratio(p, (), q2, ()))


if __name__ == '__main__':
  test_util.main()
