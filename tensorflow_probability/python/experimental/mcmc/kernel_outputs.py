# Copyright 2020 The TensorFlow Probability Authors.
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
"""Convenience wrapper around step_kernel outputs."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.mcmc import tracing_reducer
from tensorflow_probability.python.internal import unnest
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

# Notes #

# REMC doesn't have inner_kernel (has multiple inner results too)

# TODO(leben): support for preconditioning / adaptation
# TODO(leben): check vs prior target_accept_prob
# TODO(leben): empirical covariance check / preconditioning check
# TODO(leben): divergence check
# TODO(leben): other core kernel diagnostics
# TODO(leben): better tracing support


__all__ = [
    'KernelOutputs',
]


class KernelOutputs:
  """Facade around outputs of `step_kernel`.

  Processes results and extracts useful data for analysis and further sampling.
  """

  def __init__(self, kernel, state, results):
    """Construct `KernelOutputs`.

    This processes the results, including calling `finalize` on all reductions.

    Args:
      kernel: The `TransitionKernel` which generated the outputs.
      state: The final chain state as returned by `step_kernel`.
      results: The final kernel results as returned by `step_kernel`.
    """
    # parameters
    self.kernel = kernel
    self.current_state = state
    self.results = results
    # derived goodness
    self.reductions = self.all_states = self.trace = None
    self.new_step_size = None
    # go!
    self._process_results()

  def _process_results(self):
    """Process outputs to extract useful data."""
    if unnest.has_nested(self.kernel, 'reducer'):
      reducers = unnest.get_outermost(self.kernel, 'reducer')
      # Finalize streaming calculations.
      self.reductions = nest.map_structure_up_to(
          reducers,
          lambda r, s: r.finalize(s),
          reducers,
          unnest.get_outermost(self.results, 'reduction_results'),
          check_types=False)

      # Grab useful reductions.
      def process_reductions(reducer, reduction):
        if isinstance(reducer, tracing_reducer.TracingReducer):
          self.all_states, self.trace = reduction

      nest.map_structure_up_to(
          reducers,
          process_reductions,
          reducers,
          self.reductions,
          check_types=False)

    if unnest.has_nested(self.results, 'new_step_size'):
      self.new_step_size = unnest.get_outermost(self.results, 'new_step_size')

  def get_diagnostics(self):
    """Generate diagnostics on the outputs."""
    diagnostics = {}
    acceptance_rate = self.realized_acceptance_rate()
    if acceptance_rate is not None:
      diagnostics['realized_acceptance_rate'] = acceptance_rate

    return diagnostics

  def realized_acceptance_rate(self):
    """Return realized acceptance rate of the samples."""
    try:
      is_accepted = unnest.get_outermost(self.trace, 'is_accepted')
    except AttributeError:
      return
    return tf.math.reduce_mean(
        tf.cast(is_accepted, tf.float32), axis=0)
