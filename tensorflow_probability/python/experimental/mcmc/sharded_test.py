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
"""Tests for tensorflow_probability.python.experimental.mcmc.sharded."""

import tensorflow as tf

from tensorflow_probability.python.experimental.mcmc import sharded
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc import kernel as tk


class RandomWalk(tk.TransitionKernel):

  @property
  def is_calibrated(self):
    return False

  def bootstrap_results(self, _):
    return ()

  def one_step(self, current_state, previous_kernel_results, seed=None):
    seed = samplers.sanitize_seed(seed, salt='random_walk')
    current_state = current_state + samplers.normal(
        current_state.shape, seed=seed)
    return current_state, previous_kernel_results


@test_util.disable_test_for_backend(
    disable_numpy=True, reason='No distributed support for NumPy')
@test_util.test_all_tf_execution_regimes
class ShardedTest(distribute_test_lib.DistributedTest):

  def test_sharded_kernel_produces_independent_chains(self):

    def run(seed):
      kernel = sharded.Sharded(RandomWalk(), self.axis_name)
      state = tf.convert_to_tensor(0.)
      kr = kernel.bootstrap_results(state)
      state, _ = kernel.one_step(state, kr, seed=seed)
      return state
    states = self.evaluate(self.per_replica_to_tensor(
        self.strategy_run(run, args=(samplers.zeros_seed(),), in_axes=None)))
    for i in range(distribute_test_lib.NUM_DEVICES):
      for j in range(i + 1, distribute_test_lib.NUM_DEVICES):
        self.assertNotAllClose(states[i], states[j])


if __name__ == '__main__':
  tf.test.main()
