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
"""Drivers for streaming reductions framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
# Dependency imports

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    'step_kernel',
]


def step_kernel(
    num_steps,
    current_state,
    previous_kernel_results=None,
    kernel=None,
    return_final_kernel_results=False,
    parallel_iterations=10,
    seed=None,
    name=None,
):
  """Takes `num_steps` repeated `TransitionKernel` steps from `current_state`.

  This is meant to be a minimal driver for executing `TransitionKernel`s; for
  something more featureful, see `sample_chain`.

  Args:
    num_steps: Integer number of Markov chain steps.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s).
    previous_kernel_results: A `Tensor` or a nested collection of `Tensor`s.
      Warm-start for the auxiliary state needed by the given `kernel`.
      If not supplied, `step_kernel` will cold-start with
      `kernel.bootstrap_results`.
    kernel: An instance of `tfp.mcmc.TransitionKernel` which implements one step
      of the Markov chain.
    return_final_kernel_results: If `True`, then the final kernel results are
      returned alongside the chain state after `num_steps` steps are taken.
      This can be useful to inspect the final auxiliary state, or for a later
      warm restart.
    parallel_iterations: The number of iterations allowed to run in parallel. It
      must be a positive integer. See `tf.while_loop` for more details.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'mcmc_step_kernel').

  Returns:
    next_state: Markov chain state after `num_step` steps are taken, of
      identical type as `current_state`.
    final_kernel_results: kernel results, as supplied by `kernel.one_step` after
      `num_step` steps are taken. This is only returned if
      `return_final_kernel_results` is `True`.
  """
  is_seeded = seed is not None
  seed = samplers.sanitize_seed(seed, salt='experimental.mcmc.step_kernel')

  if not kernel.is_calibrated:
    warnings.warn('supplied `TransitionKernel` is not calibrated. Markov '
                  'chain may not converge to intended target distribution.')
  with tf.name_scope(name or 'mcmc_step_kernel'):
    num_steps = tf.convert_to_tensor(
        num_steps, dtype=tf.int32, name='num_steps')
    current_state = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(x, name='current_state'),
        current_state)
    if previous_kernel_results is None:
      previous_kernel_results = kernel.bootstrap_results(current_state)

    def _seeded_one_step(seed, *state_and_results):
      step_seed, passalong_seed = (
          samplers.split_seed(seed) if is_seeded else (None, seed))
      one_step_kwargs = dict(seed=step_seed) if is_seeded else {}
      return [passalong_seed] + list(
          kernel.one_step(*state_and_results, **one_step_kwargs))

    _, next_state, final_kernel_results = mcmc_util.smart_for_loop(
        loop_num_iter=num_steps,
        body_fn=_seeded_one_step,
        initial_loop_vars=list((seed, current_state, previous_kernel_results)),
        parallel_iterations=parallel_iterations)

    # return semantics are simple enough to not warrant the use of named tuples
    # as in `sample_chain`
    if return_final_kernel_results:
      return next_state, final_kernel_results
    else:
      return next_state
