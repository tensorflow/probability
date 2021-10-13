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
"""Model augmentations for particle filtering ."""

import collections

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_named
from tensorflow_probability.python.distributions import joint_distribution_util
from tensorflow_probability.python.internal import prefer_static as ps

__all__ = [
    'StateWithHistory',
    'augment_with_observation_history',
    'augment_with_state_history',
    'augment_prior_with_state_history'
]

StateWithHistory = collections.namedtuple('StateWithHistory',
                                          ['state', 'state_history'])


def augment_with_observation_history(
    observations, history_size, num_transitions_per_observation=1):
  """Decorates a function to take `observation_history`.

  Args:
    observations: a (structure of) Tensors, each of shape
      `concat([[num_observation_steps, b1, ..., bN], event_shape])` with
      optional batch dimensions `b1, ..., bN`.
    history_size: integer `Tensor` number of steps of history to pass.
    num_transitions_per_observation: integer `Tensor` number of
      state transitions between regular observation points. A value of `1`
      indicates that there is an observation at every timestep,
      `2` that every other step is observed, and so on. Values greater than `1`
      may be used with an appropriately-chosen transition function to
      approximate continuous-time dynamics. The initial and final steps
      (steps `0` and `num_timesteps - 1`) are always observed.
      Default value: `1`.
  Returns:
    augment_fn: Python `callable` such that `augmented_fn = augment_fn(fn)`.
      When called, `augmented_fn` invokes `fn`
      with an additional `observation_history` keyword arg, whose value is a
      `Tensor` of shape `concat([[history_size, b1, ..., bN], event_shape])`
      containing up to the most recent `history_size` observations.
  """
  def augment_fn(fn):
    """Set the `observation_history` kwarg of the given fn."""
    def augmented_fn(step, *args, **kwargs):
      with tf.name_scope('augment_with_observation_history'):
        observation_idx = step // num_transitions_per_observation
        observation_history_indices = ps.range(
            ps.maximum(0, observation_idx - history_size),
            observation_idx)
        return fn(step,
                  *args,
                  observation_history=tf.gather(
                      observations, observation_history_indices),
                  **kwargs)
    return augmented_fn
  return augment_fn


def _wrap_as_distributions(structure):
  return tf.nest.map_structure(
      lambda x: independent.Independent(  # pylint: disable=g-long-lambda
          deterministic.Deterministic(x),
          # Particles are a batch dimension.
          reinterpreted_batch_ndims=tf.rank(x) - 1),
      structure)


def augment_with_state_history(fn):
  """Decorates a transition or proposal fn to track state history.

  For example usage, see
  `tfp.experimental.mcmc.augment_prior_with_state_history`.

  Args:
    fn: Python `callable` to wrap, having signature
      `new_state_dist = fn(step, state_with_history, **kwargs)` where
      `state_with_history` is a `StateWithHistory` namedtuple.
  Returns:
    augmented_fn: Python `callable` wrapping `fn`, having signature
      `new_state_with_history_dist = augmented_fn(step, state_with_history,
      **kwargs)`. The return value is a `tfd.JointDistributionNamed` instance
      over`tfp.experimental.mcmc.StateWithHistory` namedtuples, in which the
      `state_history` component is rotated to discard
      the (previously-oldest) state at the initial position and append the
      new state at the final position.
  """
  def augmented_fn(step, state_with_history, **kwargs):
    """Builds history-tracking dist. over `StateWithHistory` instances."""
    with tf.name_scope('augment_with_state_history'):
      new_state_dist = fn(step, state_with_history, **kwargs)

      def new_state_history_dist(state):
        with tf.name_scope('new_state_history_dist'):
          new_state_histories = tf.nest.map_structure(
              lambda h, s: tf.concat([h[:, 1:],  # pylint: disable=g-long-lambda
                                      s[:, tf.newaxis]], axis=1),
              state_with_history.state_history,
              state)
          return (
              joint_distribution_util
              .independent_joint_distribution_from_structure(
                  _wrap_as_distributions(new_state_histories)))

    return joint_distribution_named.JointDistributionNamed(
        StateWithHistory(
            state=new_state_dist,
            state_history=new_state_history_dist))

  return augmented_fn


def augment_prior_with_state_history(prior, history_size):
  """Augments a prior or proposal distribution's state space with history.

  The augmented state space is over `tfp.experimental.mcmc.StateWithHistory`
  namedtuples, which contain the original `state` as well as a `state_history`.
  The `state_history` is a structure of `Tensor`s matching `state`, of shape
  `concat([[num_particles, history_size], state.shape[1:]])`. In other words,
  previous states for each particle are indexed along `axis=1`, to the right
  of the particle indices.

  Args:
    prior: a (joint) distribution over the initial latent state,
      with optional batch shape `[b1, ..., bN]`.
    history_size: integer `Tensor` number of steps of history to pass.
  Returns:
    augmented_prior: a `tfd.JointDistributionNamed` instance whose samples
      are `tfp.experimental.mcmc.StateWithHistory` namedtuples.

  #### Example

  As a toy example, let's see how we'd use state history to experiment with
  stochastic 'Fibonacci sequences'. We'll assume that the sequence starts at a
  value sampled from a Poisson distribution.

  ```python
  initial_state_prior = tfd.Poisson(5.)
  initial_state_with_history_prior = (
    tfp.experimental.mcmc.augment_prior_with_state_history(
      initial_state_prior, history_size=2))
  ```

  Note that we've augmented the state space to include a state history of
  size two. The augmented state space is over instances of
  `tfp.experimental.mcmc.StateWithHistory`. Initially, the state history
  will simply tile the initial state: if
  `s = initial_state_with_history_prior.sample()`, then
  `s.state_history==[s.state, s.state]`.

  Next, we'll define a `transition_fn` that uses the history to
  sample the next integer in the sequence, also from a Poisson distribution.

  ```python
  @tfp.experimental.mcmc.augment_with_state_history
  def fibonacci_transition_fn(_, state_with_history):
    expected_next_element = tf.reduce_sum(
      state_with_history.state_history[:, -2:], axis=1)
    return tfd.Poisson(rate=expected_next_element)
  ```

  Our transition function must accept `state_with_history`,
  so that it can access the history, but it returns a distribution
  only over the next state. Decorating it with `augment_with_state_history`
  ensures that the state history is automatically propagated.

  Note: if we were using an `initial_state_proposal` and/or `proposal_fn`, we
  would need to wrap them similarly to the prior and transition function
  shown here.

  Combined with an observation function (which must also now be defined on the
  augmented `StateWithHistory` space), we can track stochastic Fibonacci
  sequences and, for example, infer the initial value of a sequence:

  ```python

  def observation_fn(_, state_with_history):
    return tfd.Poisson(rate=state_with_history.state)

  trajectories, _ = tfp.experimental.mcmc.infer_trajectories(
    observations=tf.convert_to_tensor([4., 11., 16., 23., 40., 69., 100.]),
    initial_state_prior=initial_state_with_history_prior,
    transition_fn=fibonacci_transition_fn,
    observation_fn=observation_fn,
    num_particles=1024)
  inferred_initial_states = trajectories.state[0]
  print(tf.unique_with_counts(inferred_initial_states))
  ```

  """
  def initialize_state_history(state):
    """Build an initial state history by replicating the initial state."""
    with tf.name_scope('initialize_state_history'):
      initial_state_histories = tf.nest.map_structure(
          lambda x: tf.broadcast_to(  # pylint: disable=g-long-lambda
              tf.expand_dims(x, ps.minimum(ps.rank(x), 1)),
              ps.concat([ps.shape(x)[:1],
                         [history_size],
                         ps.shape(x)[1:]], axis=0)),
          state)
      return (joint_distribution_util
              .independent_joint_distribution_from_structure(
                  _wrap_as_distributions(initial_state_histories)))

  return joint_distribution_named.JointDistributionNamed(
      StateWithHistory(
          state=prior,
          state_history=initialize_state_history))
