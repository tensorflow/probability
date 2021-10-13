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
"""Convergence criterion from correlation between successive gradients."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.optimizer.convergence_criteria import convergence_criterion


class SuccessiveGradientsAreUncorrelated(
    convergence_criterion.ConvergenceCriterion):
  """Convergence criterion based on inner products between successive gradients.

  Let `g[t]` be the gradient vector at step `t`, and `g[t-1]` the previous
  gradient. Their inner product:

  ```python
  grad_inner_product[t] = sum_i(g[t, i] * g[t - 1, i])
  ```

  measures correlation between successive optimization steps.
  We expect this to be positive if the optimization is making progress;
  conversely, it can be shown to be negative in expectation at the stationary
  distribution of constant-step-size SGD [(Pflug, 1990)][2].

  This criterion detects convergence when an exponentially-weighted moving
  average of `grad_inner_product` becomes negative; intuitively, when there has
  been no consistent direction to the most recent `window_size` steps.

  Theoretical analysis shows that with no decay (`window_size=np.inf`), this
  rule stops in finite time almost surely, for constant-step-size SGD under
  standard assumptions ([Pflug, 1990][2]; [Chee and Toulis, 2017][1]). In
  practice, it is often more efficient to use a decaying moving average.

  **Batch semantics**: because this criterion does not depend on the loss,
  vector-valued losses will not produce vector-valued convergence indicators.
  Instead, the returned `has_converged` is always scalar, and is computed from
  the inner product summed across gradients from all variables being
  optimized.

  TODO(b/150151919): If per-batch convergence indicators are important to you,
  please contact `tfprobability@tensorflow.org`.

  #### References

  [1] Jerry Chee and Panos Toulis. Convergence diagnostics for stochastic
  gradient descent with constant step size. _arXiv preprint arXiv:1710.06382,
  2017. https://arxiv.org/abs/1710.06382

  [2] Georg Ch. Pflug. Non-asymptotic confidence bounds for stochastic
  approximation algorithms with constant step size. Monatshefte fur Mathematik,
  110(3-4), pp.297-314, 1990.

  """

  def __init__(self, window_size=10, min_num_steps=20, name=None):
    """Constructs a SuccessiveGradientsAreUncorrelated convergence criterion.

    Args:
      window_size: int `Tensor` effective window size for the moving average.
        The moving average inner product is computed as
        `moving_average[t] = grad_inner_product[t] + decay *
        (moving_average[t - 1] - grad_inner_product[t])` where
        `decay = 1. - 1. / window_size`. The non-decaying (`decay = 1.`)
        setting can therefore be recovered by passing `window_size=np.inf`.
        Default value: `10`.
      min_num_steps: int `Tensor` minimum number of steps before convergence.
        The criterion will not return `has_converged=True` until
        `step >= min_num_steps`. This should generally be a larger value than
        `window_size`.
        Default value: `20`.
      name: optional Python `str` name prefixed to ops created by this class.
    """
    self._window_size = window_size
    super(SuccessiveGradientsAreUncorrelated, self).__init__(
        min_num_steps=min_num_steps,
        name=name or 'SuccessiveGradientsAreUncorrelated')

  @property
  def window_size(self):
    return self._window_size

  def _bootstrap(self, loss, grads, parameters):
    initial_moving_product = tf.zeros(tf.shape(self.window_size),
                                      dtype=loss.dtype)
    return (grads, initial_moving_product)

  def _one_step(self, step, loss, grads, parameters, auxiliary_state):
    previous_grads, moving_product = auxiliary_state
    decay = 1. - 1. / tf.cast(self.window_size, loss.dtype)

    grad_inner_product = sum(tf.reduce_sum(g1 * g2)
                             for g1, g2 in zip(previous_grads, grads))
    updated_moving_product = (
        grad_inner_product + decay * (moving_product - grad_inner_product))

    has_converged = updated_moving_product < 0.
    return has_converged, (previous_grads, updated_moving_product)
