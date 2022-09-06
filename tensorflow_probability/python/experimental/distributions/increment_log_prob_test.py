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

from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.experimental.distributions import increment_log_prob
from tensorflow_probability.python.internal import test_util


@test_util.test_graph_and_eager_modes
class IncrementLogProbTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('_tensor', 5.),
      ('_callable', lambda: 5.),
  )
  def test_dtype(self, increment_amount):
    dist = yield increment_log_prob.IncrementLogProb(increment_amount)
    self.assertEqual(tf.float32, dist.dtype)

  @parameterized.named_parameters(
      ('_tensor', [1., 5.]),
      ('_callable', lambda: [1., 5.]),
  )
  def test_batch_shape(self, increment_amount):
    model = increment_log_prob.IncrementLogProb(increment_amount)
    self.assertEqual([2], model.batch_shape)
    self.assertEqual([2], self.evaluate(model.batch_shape_tensor()))

  def test_sample_shape(self):
    dist = increment_log_prob.IncrementLogProb(5.)
    seed = test_util.test_seed()
    sample_shape = [10]
    increment_log_prob_sample = dist.sample(
        sample_shape=sample_shape, seed=seed)
    self.assertEqual(increment_log_prob_sample.shape,
                     tf.TensorShape(sample_shape) + tf.TensorShape([0]))

  def test_unnormalized_log_prob(self):
    increment_amount = 5.
    dist = increment_log_prob.IncrementLogProb(increment_amount)
    self.assertAlmostEqual(increment_amount, self.evaluate(dist.log_prob(())))

  def test_unnormalized_log_prob_kwargs(self):
    dist = increment_log_prob.IncrementLogProb(
        lambda value: value, log_prob_increment_kwargs=dict(value=5.))
    self.assertAllClose(5., dist.unnormalized_log_prob(()))

  def test_unnormalized_log_prob_batch(self):
    increment_amount = tf.convert_to_tensor([5., 6.])
    dist = increment_log_prob.IncrementLogProb(increment_amount)
    self.assertAllClose(
        self.evaluate(increment_amount), self.evaluate(dist.log_prob(())))

  def test_log_prob_sample_shape(self):
    increment_amount = 5.
    dist = increment_log_prob.IncrementLogProb(increment_amount)
    self.assertEqual([3], list(dist.log_prob(tf.zeros([3, 0])).shape))

  def test_custom_log_prob_ratio(self):

    def log_prob_ratio_fn(p_kwargs, q_kwargs):
      return 2 * (p_kwargs['value'] - q_kwargs['value'])

    p = increment_log_prob.IncrementLogProb(
        lambda value: value,
        log_prob_ratio_fn=log_prob_ratio_fn,
        log_prob_increment_kwargs=dict(value=3.))
    q = increment_log_prob.IncrementLogProb(
        lambda value: value,
        log_prob_ratio_fn=log_prob_ratio_fn,
        log_prob_increment_kwargs=dict(value=1.))
    q2 = increment_log_prob.IncrementLogProb(
        lambda value: value, log_prob_increment_kwargs=dict(value=1.))

    self.assertAllClose(2. * (3. - 1.),
                        log_prob_ratio.log_prob_ratio(p, (), q, ()))
    self.assertAllClose((3. - 1.), log_prob_ratio.log_prob_ratio(p, (), q2, ()))

  def test_tf_function(self):

    @tf.function
    def make_dist():
      return increment_log_prob.IncrementLogProb(1.)

    d = make_dist()
    self.assertAllClose(1., d.log_prob(d.sample(seed=test_util.test_seed())))

  def test_tf_function_callable(self):

    @tf.function
    def make_dist():
      return increment_log_prob.IncrementLogProb(
          lambda v: v, log_prob_increment_kwargs={'v': tf.constant(1.)})

    d = make_dist()
    self.assertAllClose(
        1., self.evaluate(d.log_prob(d.sample(seed=test_util.test_seed()))))


if __name__ == '__main__':
  test_util.main()
