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
"""Tests for tensorflow_probability.python.experimental.bijectors.sharded."""
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import sigmoid
from tensorflow_probability.python.distributions import beta
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.experimental.bijectors import sharded
from tensorflow_probability.python.experimental.distribute import sharded as sharded_lib
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib as test_lib
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient

JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class ShardedTest(test_lib.DistributedTest):

  def test_sharded_log_det_jacobian(self):

    def log_prob(y):
      return tf.reduce_sum(beta.Beta(2., 2.).log_prob(y))

    def transform_log_prob(log_prob, bijector):

      def new_log_prob(x):
        y = bijector.forward(x)
        return log_prob(y) + bijector.forward_log_det_jacobian(x, len(x.shape))

      return new_log_prob

    @tf.function
    def lp_grad(x):
      untransformed_log_prob = distribute_lib.make_sharded_log_prob_parts(
          log_prob, self.axis_name)
      transformed_log_prob = transform_log_prob(
          untransformed_log_prob,
          sharded.Sharded(sigmoid.Sigmoid(), shard_axis_name=self.axis_name))
      lp, g = gradient.value_and_gradient(transformed_log_prob, (x,))
      return lp, g

    def true_lp_grad(x):
      transformed_log_prob = transform_log_prob(log_prob, sigmoid.Sigmoid())
      lp, g = gradient.value_and_gradient(transformed_log_prob, (x,))
      return lp, g

    y = tf.convert_to_tensor(
        np.linspace(0.2, 0.7, test_lib.NUM_DEVICES, dtype=np.float32))
    x = self.evaluate(sigmoid.Sigmoid().inverse(y))
    sharded_x = self.shard_values(x)
    lp, g = self.evaluate(
        self.per_replica_to_tensor(self.strategy_run(lp_grad, (sharded_x,))))
    true_lp, true_g = self.evaluate(true_lp_grad(x))
    self.assertAllClose(true_lp, lp[0])
    self.assertAllClose(true_lp, lp[1])
    self.assertAllClose(true_g, g)

  def test_sharded_distribution_sharded_bijector(self):

    td = transformed_distribution.TransformedDistribution(
        normal.Normal(loc=0, scale=1), sigmoid.Sigmoid())
    sharded_td = transformed_distribution.TransformedDistribution(
        sharded_lib.Sharded(
            normal.Normal(loc=0, scale=1), shard_axis_name=self.axis_name),
        sharded.Sharded(sigmoid.Sigmoid(), shard_axis_name=self.axis_name))

    x = self.evaluate(td.sample(test_lib.NUM_DEVICES, seed=self.key))
    sharded_x = self.shard_values(x)

    def true_lp_grad(x):

      def log_prob(x):
        return tf.reduce_sum(td.log_prob(x))

      lp, g = gradient.value_and_gradient(log_prob, (x,))
      return lp, g

    @tf.function
    def sharded_lp_grad(x):
      lp, g = gradient.value_and_gradient(sharded_td.log_prob, (x,))
      return lp, g

    true_lp, true_grad = self.evaluate(true_lp_grad(x))
    sharded_lp, sharded_grad = self.evaluate(
        self.per_replica_to_tensor(
            self.strategy_run(sharded_lp_grad, (sharded_x,))))
    self.assertAllClose(true_lp, sharded_lp[0])
    self.assertAllClose(true_lp, sharded_lp[1])
    self.assertAllClose(true_grad, sharded_grad)


if __name__ == '__main__':
  test_util.main()
