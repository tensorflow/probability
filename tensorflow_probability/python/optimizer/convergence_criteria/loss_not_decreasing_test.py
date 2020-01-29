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
"""Tests for LossNotDecreasing stopping rule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class LossNotDecreasingTests(test_util.TestCase):

  @parameterized.named_parameters(
      ('atol_only', 1.0, None, 8, False),
      ('rtol_only', None, 1e-1, 7, False),
      ('rtol_with_atol', 1.0, 1e-1, 7, True),
      ('atol_with_rtol', 1.0, 1e-6, 8, True))
  def test_follows_tolerances(self,
                              atol,
                              rtol,
                              expected_num_steps,
                              pass_inputs_as_tensors):

    with self.assertRaisesRegexp(ValueError, 'Must specify at least one of'):
      tfp.optimizer.convergence_criteria.LossNotDecreasing()

    window_size = 2
    criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
        window_size=window_size,
        atol=atol, rtol=rtol, min_num_steps=4)

    losses = (1e2, 30., 10., 9., 8.95, 8.96, 8.93, 8.92, 8.93, 8.94, 8.91)
    step = 0
    auxiliary_state = criterion.bootstrap(
        tf.convert_to_tensor(losses[0]), [], [])
    for step, loss in zip(range(1, len(losses)), losses[1:]):
      has_converged, auxiliary_state = criterion.one_step(
          tf.convert_to_tensor(step) if pass_inputs_as_tensors else step,
          tf.convert_to_tensor(loss) if pass_inputs_as_tensors else loss,
          [], [], auxiliary_state)
      has_converged = self.evaluate(has_converged)
      if has_converged:
        break
    self.assertEqual(has_converged, True)
    self.assertEqual(step, expected_num_steps - 1)

    decreases_in_loss = -np.diff(losses)
    average_decreases = [1./window_size * decreases_in_loss[0]]
    for loss_decrease in decreases_in_loss[1:step]:
      average_decreases.append(
          ((window_size - 1) * average_decreases[-1] + loss_decrease) /
          window_size)
    self.assertAllClose(self.evaluate((
        auxiliary_state.previous_loss,
        auxiliary_state.average_decrease_in_loss,
        auxiliary_state.average_initial_decrease_in_loss)),
                        [losses[step],
                         average_decreases[-1],
                         average_decreases[window_size - 1]])

  def test_batch_inputs_produce_batch_results(self):

    batch_shape = [3]
    num_timesteps = 50
    # Build synthetic loss curves: a quadratic decrease, scaled by random
    # positive values.
    losses = (tf.range(num_timesteps, 0, -1, dtype=tf.float32)**2 *
              tf.exp(
                  1e-2 * tf.random.normal(
                      batch_shape + [num_timesteps],
                      seed=test_util.test_seed())))

    # Test that a convergence criterion with batched params matches the
    # results of separately constructed criteria.
    window_sizes = [5, 6, 4]
    atols = [0.1, 15.0, 40.0]
    rtols = [0.4, 1e-2, 1e-6]
    min_num_steps = [6, num_timesteps-1, 10]
    batch_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
        window_size=window_sizes,
        atol=atols,
        rtol=rtols,
        min_num_steps=min_num_steps)

    component_criteria = [
        tfp.optimizer.convergence_criteria.LossNotDecreasing(
            window_size=ws, atol=at, rtol=rt, min_num_steps=mns)
        for (ws, at, rt, mns) in zip(window_sizes, atols, rtols, min_num_steps)
    ]

    # Initialize the auxiliary states for the batch and component criteria.
    batch_auxiliary_state = batch_criterion.bootstrap(losses[..., 0], [], [])
    component_auxiliary_state = [
        c.bootstrap(losses[i, 0], [], [])
        for (i, c) in enumerate(component_criteria)]

    # Run all criteria forward and check that the results match.
    batch_has_converged = []
    batch_auxiliary_states = [batch_auxiliary_state]
    component_has_converged = []
    component_auxiliary_states = [component_auxiliary_state]
    for step in range(1, num_timesteps):
      has_converged, auxiliary_state = batch_criterion.one_step(
          step, losses[..., step], [], [], batch_auxiliary_states[-1])
      batch_has_converged.append(has_converged)
      batch_auxiliary_states.append(auxiliary_state)

      has_converged, auxiliary_state = zip(*[
          c.one_step(step, losses[i, step], [], [], cs)
          for i, (c, cs) in enumerate(
              zip(component_criteria, component_auxiliary_states[-1]))])
      component_has_converged.append(has_converged)
      component_auxiliary_states.append(auxiliary_state)

    (batch_has_converged_,
     component_has_converged_,
     batch_auxiliary_states_,
     component_auxiliary_states_) = self.evaluate([
         batch_has_converged,
         component_has_converged,
         batch_auxiliary_states,
         component_auxiliary_states])

    for i in range(len(component_criteria)):
      self.assertAllEqual(
          [x[i] for x in component_has_converged_],
          [x[i] for x in batch_has_converged_])
      self.assertAllEqual(
          [x[i] for x in component_auxiliary_states_],
          [tf.nest.map_structure(lambda x: x[i], batch_state)
           for batch_state in batch_auxiliary_states_])

      # Assert that we've set parameters so that the test is interesting.
      self.assertEqual(component_has_converged_[-1][i], True)

if __name__ == '__main__':
  tf.test.main()
