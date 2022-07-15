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
"""Contains probabilistic program kernels for MCMC."""
import jax
from jax import lax
from jax import random
from jax import tree_util
import jax.numpy as np


from oryx import distributions as bd
from oryx.core import ppl
from oryx.core import primitive
from oryx.core import state as st

__all__ = [
    'random_walk',
    'metropolis',
    'metropolis_hastings',
    'hmc',
    'mala',
    'sample_chain',
]


LogProbFunction = ppl.LogProbFunction
Program = ppl.Program
MCMC_METRICS = 'mcmc_metrics'


def random_walk(scale=1.) -> Program:
  """Returns a probabilistic program that takes a Gaussian step with a provided variance."""

  def step(key, state):
    flat_state, state_tree = tree_util.tree_flatten(state)
    keys = tree_util.tree_unflatten(state_tree,
                                    random.split(key, len(flat_state)))

    def _sample(key, state):
      return ppl.random_variable(
          bd.Independent(  # pytype: disable=module-attr
              bd.Normal(state, scale),  # pytype: disable=module-attr
              reinterpreted_batch_ndims=np.ndim(state)))(
                  key)

    return tree_util.tree_map(_sample, keys, state)

  return step


def metropolis(unnormalized_log_prob: LogProbFunction,
               inner_step: Program) -> Program:
  """Returns a program that takes a Metropolis step with an inner kernel.

  The Metropolis algorithm is a special case of Metropolis-Hastings for
  symmetric proposal distributions. This algorithm assumes the `inner_step`
  program is symmetric (i.e. p(y | x) = p(x | y)).
  Args:
    unnormalized_log_prob: A function that computes the log probability of a
      state.
    inner_step: A probabilistic program that acts as the proposal distribution
      for a Metropolis step.
  Returns:
    A program that proposes a new state and accepts or rejects according to the
      unnormalized log probability.
  """

  def step(key, state, init_key=None):
    transition_key, accept_key = random.split(key)
    next_state = st.init(inner_step)(init_key, transition_key, state)(
        transition_key, state)
    # TODO(sharadmv): add log probabilities to the state to avoid recalculation.
    state_log_prob = unnormalized_log_prob(state)
    next_state_log_prob = unnormalized_log_prob(next_state)
    log_unclipped_accept_prob = next_state_log_prob - state_log_prob
    accept_prob = np.clip(np.exp(log_unclipped_accept_prob), 0., 1.)
    u = primitive.tie_in(accept_prob, random.uniform(accept_key))
    accept = np.log(u) < log_unclipped_accept_prob
    return tree_util.tree_map(lambda n, s: np.where(accept, n, s), next_state,
                              state)

  return step


def metropolis_hastings(unnormalized_log_prob: LogProbFunction,
                        inner_step: Program) -> Program:
  """Returns a program that takes a Metropolis-Hastings step.

  The Metropolis-Hastings algorithm takes a proposal distribution (`inner_step`)
  and iteratively accepts or rejects proposals from it according to an accept
  ratio calculated from `unnormalized_log_prob` and `log_prob(inner_step)`. This
  creates a Markov Chain whose stationary distribution is some target
  distribution (specified by `unnormalized_log_prob`).

  Args:
    unnormalized_log_prob: A function that computes the log probability of a
      state.
    inner_step: A probabilistic program that acts as the proposal distribution
      for a Metropolis-Hasting step.
  Returns:
    A program that proposes a new state and accepts or rejects according to the
    unnormalized log probability and proposal distribution transition
    probabilities.
  """

  def step(key, state):
    transition_key, accept_key = random.split(key)
    next_state = inner_step(transition_key, state)
    forward_transition_log_prob = ppl.log_prob(inner_step)(state, next_state)
    backward_transition_log_prob = ppl.log_prob(inner_step)(next_state, state)
    # TODO(sharadmv): add log probabilities to the state to avoid recalculation.
    state_log_prob = unnormalized_log_prob(state)
    next_state_log_prob = unnormalized_log_prob(next_state)
    log_unclipped_accept_prob = (
        next_state_log_prob + backward_transition_log_prob - state_log_prob -
        forward_transition_log_prob)
    accept_prob = np.clip(np.exp(log_unclipped_accept_prob), 0., 1.)
    u = primitive.tie_in(accept_prob, random.uniform(accept_key))
    accept = np.log(u) < log_unclipped_accept_prob
    return tree_util.tree_map(lambda n, s: np.where(accept, n, s), next_state,
                              state)

  return step


def hmc(unnormalized_log_prob: LogProbFunction,
        num_leapfrog_steps: int = 5,
        step_size: float = 1e-1) -> Program:
  """Makes a Hamiltonian Monte Carlo step function."""

  def inner(key, state):
    del key  # leapfrog steps are deterministic
    # TODO(sharadmv): add gradients to the state to avoid recalculation.
    def leapfrog(carry, _):
      state, momentum = carry
      momentum = tree_util.tree_map(lambda m, g: m + 0.5 * step_size * g,
                                    momentum,
                                    jax.grad(unnormalized_log_prob)(state))
      state = tree_util.tree_map(lambda s, m: s + step_size * m, state,
                                 momentum)
      momentum = tree_util.tree_map(lambda m, g: m + 0.5 * step_size * g,
                                    momentum,
                                    jax.grad(unnormalized_log_prob)(state))
      return (state, momentum), ()

    # Use scan since it's differentiable
    return lax.scan(leapfrog, state, np.arange(num_leapfrog_steps))[0]

  def step(key, state):
    mh_key, momentum_key = random.split(key)
    flat_state, state_tree = tree_util.tree_flatten(state)

    def momentum_distribution(key):
      momentum_keys = tree_util.tree_unflatten(
          state_tree, random.split(key, len(flat_state)))

      def _sample(key, s):
        return ppl.random_variable(
            bd.Sample(bd.Normal(0., 1.),  # pytype: disable=module-attr
                      sample_shape=s.shape))(key).astype(s.dtype)

      return tree_util.tree_map(_sample, momentum_keys, state)

    momentum = momentum_distribution(momentum_key)

    def inner_log_prob(state_momentum):
      state, momentum = state_momentum
      momentum_prob = ppl.log_prob(momentum_distribution)(momentum)
      return unnormalized_log_prob(state) + momentum_prob

    state, momentum = metropolis(inner_log_prob, inner)(
        mh_key, (state, momentum))
    return state

  return step


def mala(unnormalized_log_prob: LogProbFunction,
         step_size: float = 1e-1) -> Program:
  """Makes a Metropolis-adjusted Langevin algorithm (MALA) step function."""
  return hmc(unnormalized_log_prob, num_leapfrog_steps=1, step_size=step_size)


def sample_chain(kernel_fn, num_steps, callbacks=None):
  """Runs several steps of MCMC."""
  if callbacks is None:
    callbacks = []

  def step(key, state, init_key=None):
    kernel = st.init(kernel_fn, name='kernel')(init_key, key, state)

    def body(carry, key):
      kernel, state = carry
      state, kernel = kernel.call_and_update(key, state)
      for cb in callbacks:
        kernel, state, _ = primitive.tie_all(kernel, state, cb(kernel, state))
      return (kernel, state), state

    (kernel, _), states = lax.scan(body, (kernel, state),
                                   random.split(key, num_steps))
    return primitive.tie_in(st.assign(kernel, name='kernel'), states)

  return step
