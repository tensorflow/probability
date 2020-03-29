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
"""Tests for SuccessiveGradientsAreUncorrelated convergence criterion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd


from tensorflow_probability.python.internal import test_util


class SuccessiveGradientsAreUncorrelatedTests(test_util.TestCase):

  # We don't `test_all_execution_regimes` because `eager_no_tf_function`
  # is *painfully* slow in the optimization step, and it's redundant since the
  # convergence criterion itself is only invoked outside of tf.function.
  @test_util.test_graph_and_eager_modes
  def test_stochastic_optimization(self):
    seed = test_util.test_seed_stream()
    tf.random.set_seed(seed())

    # Example of fitting one normal to another using a
    # Monte Carlo variational loss.
    locs = tf.Variable(tf.random.normal([10], seed=seed()))
    scales = tfp.util.TransformedVariable(
        tf.nn.softplus(tf.random.normal([10], seed=seed())),
        tfb.Softplus())
    trained_dist = tfd.Normal(locs, scales)
    target_dist = tfd.Normal(loc=-0.4, scale=1.2)

    optimizer = tf.optimizers.Adam(learning_rate=0.1)
    @tf.function(autograph=False)
    def optimization_step():
      with tf.GradientTape() as tape:
        loss = tfp.vi.monte_carlo_variational_loss(
            target_log_prob_fn=target_dist.log_prob,
            surrogate_posterior=trained_dist,
            sample_size=20,
            seed=seed())
      grads = tape.gradient(loss, trained_dist.trainable_variables)
      optimizer.apply_gradients(zip(grads, trained_dist.trainable_variables))
      return loss, grads

    criterion = (
        tfp.optimizer.convergence_criteria.SuccessiveGradientsAreUncorrelated(
            window_size=10, min_num_steps=20))

    loss, grads = optimization_step()
    self.evaluate(tf1.global_variables_initializer())
    auxiliary_state = criterion.bootstrap(
        loss, grads, trained_dist.trainable_variables)
    for step in range(1, 100):
      loss, grads = optimization_step()
      has_converged, auxiliary_state = criterion.one_step(
          step, loss, grads,
          trained_dist.trainable_variables,
          auxiliary_state)

      has_converged_ = self.evaluate(has_converged)
      if has_converged_:
        break

    # Check that the criterion successfully stopped the optimization
    # (at step 32 with the test seed as of this writing).
    self.assertLess(step, 99)  # pylint: disable=undefined-loop-variable

    # Because this is a stochastic optimization with no learning rate decay,
    # we will not converge to the true values, just to a stationary distribution
    # that (hopefully) includes them.
    self.assertLess(self.evaluate(tf.reduce_sum(loss)), 1.5)
    self.assertAllClose(*self.evaluate((
        tf.reduce_mean(locs), target_dist.mean())), atol=0.5)
    self.assertAllClose(*self.evaluate((
        tf.reduce_mean(scales), target_dist.stddev())), atol=0.5)

if __name__ == '__main__':
  tf.test.main()
