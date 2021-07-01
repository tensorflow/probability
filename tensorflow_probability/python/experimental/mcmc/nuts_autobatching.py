# Copyright 2018 The TensorFlow Probability Authors.
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
"""No U-Turn Sampler.

The implementation closely follows [1; Algorithm 3].
The path length is set adaptively; the step size is fixed.

Achieves batch execution across chains by using
`tensorflow_probability.python.internal.auto_batching` internally.

This code is not yet integrated into the tensorflow_probability.mcmc Markov
chain Monte Carlo library.

#### References

[1]: Matthew D. Hoffman, Andrew Gelman. The No-U-Turn Sampler: Adaptively
     Setting Path Lengths in Hamiltonian Monte Carlo.
     In _Journal of Machine Learning Research_, 15(1):1593-1623, 2014.
     http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf
"""

# Maintainer's note: The code may look weird in several respects.  These are due
# to limitations of the auto-batching system (at the time of writing).  See
# https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/internal/auto_batching/README.md.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

# Dependency imports
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python import random as tfp_random
from tensorflow_probability.python.experimental import auto_batching as ab
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.util.seed_stream import SeedStream


# TODO(axch): Sensibly support field references from auto-batched code
truthy = ab.truthy

__all__ = [
    "NoUTurnSampler",
]


# state: List of `Tensor`s representing the "position" states of the NUTS
#   trajectory.
# target_log_prob: Scalar `Tensor` representing the value of
#   `target_log_prob_fn` at the `state`.
# grads_target_log_prob: List of `Tensor`s representing gradient of
#   `target_log_prob` with respect to `state`. Has same shape as `state`.
# momentum: List of `Tensor`s representing the momentums at `state`. Has same
#   shape as `state`.
Point = collections.namedtuple(
    "Point", ["state", "target_log_prob", "grads_target_log_prob", "momentum"])


class NoUTurnSampler(kernel_base.TransitionKernel):
  """Runs one step of the No U-Turn Sampler.

  The No U-Turn Sampler (NUTS) is an adaptive variant of the Hamiltonian Monte
  Carlo (HMC) method for MCMC.  NUTS adapts the distance traveled in response to
  the curvature of the target density.  Conceptually, one proposal consists of
  reversibly evolving a trajectory through the sample space, continuing until
  that trajectory turns back on itself (hence the name, "No U-Turn").  This
  class implements one random NUTS step from a given
  `current_state`.  Mathematical details and derivations can be found in
  [Hoffman, Gelman (2011)][1].

  The `one_step` function can update multiple chains in parallel. It assumes
  that a prefix of leftmost dimensions of `current_state` index independent
  chain states (and are therefore updated independently).  The output of
  `target_log_prob_fn(*current_state)` should sum log-probabilities across all
  event dimensions.  Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0][0, ...]` could have a
  different target distribution from `current_state[0][1, ...]`.  These
  semantics are governed by `target_log_prob_fn(*current_state)`. (The number of
  independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

  TODO(axch): Examples (e.g., a la HMC).  For them to be sensible, need to
  pick sensible step sizes, or implement step size adaptation, or both.

  #### References

  [1] Matthew D. Hoffman, Andrew Gelman.  The No-U-Turn Sampler: Adaptively
  Setting Path Lengths in Hamiltonian Monte Carlo.  2011.
  https://arxiv.org/pdf/1111.4246.pdf.
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               max_tree_depth=10,
               unrolled_leapfrog_steps=1,
               num_trajectories_per_step=1,
               use_auto_batching=True,
               stackless=False,
               backend=None,
               seed=None,
               name=None):
    """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.  Due
        to limitations of the underlying auto-batching system,
        target_log_prob_fn may be invoked with junk data at some batch indexes,
        which it must process without crashing.  (The results at those indexes
        are ignored).
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      max_tree_depth: Maximum depth of the tree implicitly built by NUTS. The
        maximum number of leapfrog steps is bounded by `2**max_tree_depth-1`
        i.e. the number of nodes in a binary tree `max_tree_depth` nodes deep.
        The default setting of 10 takes up to 1023 leapfrog steps.
      unrolled_leapfrog_steps: The number of leapfrogs to unroll per tree
        expansion step. Applies a direct linear multipler to the maximum
        trajectory length implied by max_tree_depth. Defaults to 1. This
        parameter can be useful for amortizing the auto-batching control flow
        overhead.
      num_trajectories_per_step: Python `int` giving the number of NUTS
        trajectories to run as "one" step.  Setting this higher than 1 may be
        favorable for performance by giving the autobatching system the
        opportunity to batch gradients across consecutive trajectories.  The
        intermediate samples are thinned: only the last sample from the run (in
        each batch member) is returned.
      use_auto_batching: Boolean.  If `False`, do not invoke the auto-batching
        system; operate on batch size 1 only.
      stackless: Boolean.  If `True`, invoke the stackless version of
        the auto-batching system.  Only works in Eager mode.
      backend: Auto-batching backend object. Falls back to a default
        TensorFlowBackend().
      seed: Python integer to seed the random number generator.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'nuts_kernel').
    """
    self._parameters = dict(locals())
    del self._parameters["self"]
    self.target_log_prob_fn = target_log_prob_fn
    self.step_size = step_size
    if max_tree_depth < 1:
      raise ValueError(
          "max_tree_depth must be >= 1 but was {}".format(max_tree_depth))
    self.max_tree_depth = max_tree_depth
    self.unrolled_leapfrog_steps = unrolled_leapfrog_steps
    self.num_trajectories_per_step = num_trajectories_per_step
    self.use_auto_batching = use_auto_batching
    self.stackless = stackless
    self.backend = backend
    self._seed_stream = SeedStream(seed, "nuts_one_step")
    self.name = "nuts_kernel" if name is None else name
    # TODO(b/125544625): Identify why we need `use_gradient_tape=True`, i.e.,
    # what's different between `tape.gradient` and `tf.gradient`.
    value_and_gradients_fn = lambda *args: tfp_math.value_and_gradient(  # pylint: disable=g-long-lambda
        self.target_log_prob_fn, args, use_gradient_tape=True)
    self.value_and_gradients_fn = _embed_no_none_gradient_check(
        value_and_gradients_fn)
    max_tree_edges = max_tree_depth - 1
    self.many_steps, self.autobatch_context = _make_evolve_trajectory(
        self.value_and_gradients_fn, max_tree_edges, unrolled_leapfrog_steps,
        self._seed_stream)
    self._block_code_cache = {}

  @property
  def parameters(self):
    return self._parameters

  @property
  def is_calibrated(self):
    return True

  def one_step(self, current_state, previous_kernel_results):
    """Runs one iteration of the No U-Turn Sampler.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s). The first `r` dimensions index
        independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)

    Returns:
      next_state: `Tensor` or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking `self.num_trajectories_per_step`
        steps. Has same type and shape as `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.
    """
    if self.stackless and not tf.executing_eagerly():
      raise ValueError("Cannot use stackless auto-batching in graph mode.")
    current_target_log_prob = previous_kernel_results.target_log_prob
    current_grads_log_prob = previous_kernel_results.grads_target_log_prob
    leapfrogs_taken = previous_kernel_results.leapfrogs_taken
    leapfrogs_computed = previous_kernel_results.leapfrogs_computed
    with tf1.name_scope(
        self.name,
        values=[
            current_state, self.step_size, current_target_log_prob,
            current_grads_log_prob
        ]):
      unwrap_state_list = False
      with tf1.name_scope("initialize"):
        if not tf.nest.is_nested(current_state):
          unwrap_state_list = True
          current_state = [current_state]
        current_state = [tf.convert_to_tensor(value=s) for s in current_state]
        step_size = self.step_size
        if not tf.nest.is_nested(step_size):
          step_size = [step_size]
        step_size = [tf.convert_to_tensor(value=s) for s in step_size]
        if len(step_size) == 1:
          step_size = step_size * len(current_state)
        if len(step_size) != len(current_state):
          raise ValueError("Expected either one step size or {} (size of "
                           "`current_state`), but found {}".format(
                               len(current_state), len(step_size)))

      num_steps = tf.constant([self.num_trajectories_per_step], dtype=tf.int64)
      if self.backend is None:
        if self._seed_stream() is not None:
          # The user wanted reproducible results; limit the parallel iterations
          backend = ab.TensorFlowBackend(while_parallel_iterations=1)
        else:
          backend = ab.TensorFlowBackend()
      else:
        backend = self.backend
      # The `dry_run` and `max_stack_depth` arguments are added by the
      # @ctx.batch decorator, confusing pylint.
      # pylint: disable=unexpected-keyword-arg
      ((next_state, next_target_log_prob, next_grads_target_log_prob),
       new_leapfrogs) = self.many_steps(
           num_steps,
           current_state,
           current_target_log_prob,
           current_grads_log_prob,
           step_size,
           tf.zeros_like(leapfrogs_taken),  # leapfrogs
           dry_run=not self.use_auto_batching,
           stackless=self.stackless,
           backend=backend,
           max_stack_depth=self.max_tree_depth + 4,
           block_code_cache=self._block_code_cache)

      if unwrap_state_list:
        next_state = next_state[0]
      return next_state, NUTSKernelResults(
          next_target_log_prob, next_grads_target_log_prob,
          leapfrogs_taken + new_leapfrogs,
          leapfrogs_computed + tf.math.reduce_max(input_tensor=new_leapfrogs))

  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    if not tf.nest.is_nested(init_state):
      init_state = [init_state]
    with tf1.name_scope("NoUTurnSampler.bootstrap_results"):
      batch_size = tf.shape(input=init_state[0])[0]
      (current_target_log_prob,
       current_grads_log_prob) = self.value_and_gradients_fn(*init_state)
      zeros = tf.dtypes.cast(
          tf.fill(dims=[batch_size], value=0), dtype=tf.int64)
      return NUTSKernelResults(
          current_target_log_prob, current_grads_log_prob,
          leapfrogs_taken=zeros, leapfrogs_computed=zeros)


NUTSKernelResults = collections.namedtuple(
    "NUTSKernelResults",
    ["target_log_prob", "grads_target_log_prob",
     "leapfrogs_taken", "leapfrogs_computed"])


def _make_evolve_trajectory(value_and_gradients_fn, max_depth,
                            unrolled_leapfrog_steps, seed_stream):
  """Constructs an auto-batched NUTS trajectory evolver.

  This indirection with an explicit maker function is necessary because the
  auto-batching system this uses doesn't understand non-Tensor variables.
  Consequently, `target_log_prob_fn` and `seed_stream` have to be passed through
  the lexical context.

  The returned trajectory evolver will invoke `target_log_prob_fn` as many times
  as requested by the longest trajectory.

  Args:
    value_and_gradients_fn: Python callable which takes arguments like
      `*current_state` and returns a batch of its (possibly unnormalized)
      log-densities under the target distribution, and the gradients thereof.
    max_depth: Maximum depth of the recursion tree, in *edges*.
    unrolled_leapfrog_steps: Number of leapfrogs to unroll per tree extension
      step.
    seed_stream: Mutable random number generator.

  Returns:
    evolve_trajectory: Function for running the trajectory evolution.
  """
  ctx = ab.Context()

  def many_steps_type(args):
    _, state_type, prob_type, grad_type, _, leapfrogs_type = args
    return (state_type, prob_type, grad_type), leapfrogs_type

  @ctx.batch(type_inference=many_steps_type)
  def many_steps(
      num_steps,
      current_state,
      current_target_log_prob,
      current_grads_log_prob,
      step_size,
      leapfrogs):
    """Runs `evolve_trajectory` the requested number of times sequentially."""
    current_momentum, log_slice_sample = _start_trajectory_batched(
        current_state, current_target_log_prob, seed_stream)

    current = Point(
        current_state, current_target_log_prob,
        current_grads_log_prob, current_momentum)

    if truthy(num_steps > 0):
      next_, new_leapfrogs = evolve_trajectory(
          current,
          current,
          current,
          step_size,
          log_slice_sample,
          tf.constant([0], dtype=tf.int64),  # depth
          tf.constant([1], dtype=tf.int64),  # num_states
          tf.constant([0], dtype=tf.int64),  # leapfrogs_taken
          True)  # continue_trajectory
      return many_steps(
          num_steps - 1,
          next_.state,
          next_.target_log_prob,
          next_.grads_target_log_prob,
          step_size,
          leapfrogs + new_leapfrogs)
    else:
      return ((current.state, current.target_log_prob,
               current.grads_target_log_prob), leapfrogs)

  def evolve_trajectory_type(args):
    point_type, _, _, _, _, _, _, leapfrogs_type, _ = args
    return point_type, leapfrogs_type

  @ctx.batch(type_inference=evolve_trajectory_type)
  def evolve_trajectory(
      reverse,
      forward,
      next_,
      step_size,
      log_slice_sample,
      depth,
      num_states,
      leapfrogs,
      continue_trajectory):
    """Evolves one NUTS trajectory in progress until a U-turn is encountered.

    This function is coded for one NUTS chain, and automatically batched to
    support several.  The argument descriptions below are written in
    single-chain language.

    This function only exists because the auto-batching system does not (yet)
    support syntactic while loops.  It implements a while loop by calling
    itself at the end.

    Args:
      reverse: `Point` tuple of `Tensor`s representing the "reverse" states of
        the NUTS trajectory.
      forward: `Point` tuple of `Tensor`s representing the "forward" states of
        the NUTS trajectory. Has same shape as `reverse`.
      next_: `Point` tuple of `Tensor`s representing the next states of the
        NUTS trajectory. Has same shape as `reverse`.
      step_size: List of `Tensor`s representing the step sizes for the
        leapfrog integrator. Must have same shape as `current_state`.
      log_slice_sample: The log of an auxiliary slice variable. It is used to
        pay for the posterior value at traversed states to avoid a Metropolis
        correction at the end.
      depth: non-negative integer Tensor that indicates how deep of a tree to
        build at the next trajectory doubling.
      num_states: Number of acceptable candidate states in the initial tree
        built so far. A state is acceptable if it is "in the slice", that is,
        if its log-joint probability with its momentum is greater than
        `log_slice_sample`.
      leapfrogs: Number of leapfrog steps computed so far.
      continue_trajectory: bool determining whether to continue the simulation
        trajectory. The trajectory is continued if no U-turns are encountered
        within the built subtree, and if the log-probability accumulation due
        to integration error does not exceed `max_simulation_error`.

    Returns:
      next_: `Point` tuple of `Tensor`s representing the state this NUTS
        trajectory transitions to.  Has same shape as `reverse`.
      leapfrogs: Number of leapfrog steps computed in the trajectory, as a
        diagnostic.
    """
    if truthy(continue_trajectory):
      # Grow the No-U-Turn Sampler trajectory by choosing a random direction
      # and simulating Hamiltonian dynamics in that direction. This extends
      # either the forward or reverse state.
      direction = _choose_direction_batched(forward, seed_stream)
      into_build_tree = _tf_where(direction < 0, reverse, forward)
      [
          reverse_out,
          forward_out,
          next_in_subtree,
          num_states_in_subtree,
          more_leapfrogs,
          continue_trajectory,
      ] = _build_tree(
          into_build_tree, direction, depth, step_size, log_slice_sample)
      # TODO(b/122732601): Revert back to `if` when the compiler makes the xform
      reverse_in = reverse
      reverse = _tf_where(direction < 0, reverse_out, reverse_in)
      forward_in = forward
      forward = _tf_where(direction < 0, forward_in, forward_out)

      # TODO(b/122732601): Revert back to `if` when the compiler makes the xform
      # If the built tree did not terminate, accept the tree's next state
      # with a certain probability.
      accept_state_in_subtree = _binomial_subtree_acceptance_batched(
          num_states_in_subtree, num_states, seed_stream)
      next_in = next_
      next_ = _tf_where(continue_trajectory & accept_state_in_subtree,
                        next_in_subtree, next_in)

      # Continue the NUTS trajectory if the tree-building did not terminate,
      # and if the reverse-most and forward-most states do not exhibit a
      # U-turn.
      continue_trajectory_in = continue_trajectory
      continue_trajectory = _continue_test_batched(
          continue_trajectory_in & (depth < max_depth), forward, reverse)
      return evolve_trajectory(
          reverse,
          forward,
          next_,
          step_size,
          log_slice_sample,
          depth + 1,
          num_states + num_states_in_subtree,
          leapfrogs + more_leapfrogs,
          continue_trajectory)
    else:
      return next_, leapfrogs

  def _build_tree_type(args):
    point_type, _, _, _, _ = args
    return (point_type, point_type, point_type,
            ab.TensorType(np.int64, ()), ab.TensorType(np.int64, ()),
            ab.TensorType(np.bool_, ()))

  @ctx.batch(type_inference=_build_tree_type)
  def _build_tree(current, direction, depth, step_size, log_slice_sample):
    """Builds a tree at a given tree depth and at a given state.

    The `current` state is immediately adjacent to, but outside of,
    the subtrajectory spanned by the returned `forward` and `reverse` states.

    This function is coded for one NUTS chain, and automatically batched to
    support several.  The argument descriptions below are written in
    single-chain language.

    Args:
      current: `Point` tuple of `Tensor`s representing the current states of
        the NUTS trajectory.
      direction: Integer Tensor that is either -1 or 1. It determines whether
        to perform leapfrog integration backward (reverse) or forward in time
        respectively.
      depth: non-negative integer Tensor that indicates how deep of a tree to
        build.  Each call to `_build_tree` takes `2**depth` leapfrog steps,
        unless stopped early by detecting a U-turn.
      step_size: List of `Tensor`s representing the step sizes for the
        leapfrog integrator. Must have same shape as `current_state`.
      log_slice_sample: The log of an auxiliary slice variable. It is used to
        pay for the posterior value at traversed states to avoid a Metropolis
        correction at the end.

    Returns:
      reverse: `Point` tuple of `Tensor`s representing the state at the
        extreme "backward in time" point of the simulated subtrajectory. Has
        same shape as `current`.
      forward: `Point` tuple of `Tensor`s representing the state at the
        extreme "forward in time" point of the simulated subtrajectory. Has
        same shape as `current`.
      next_: `Point` tuple of `Tensor`s representing the candidate point to
        transition to, sampled from this subtree. Has same shape as
        `current_state`.
      num_states: Number of acceptable candidate states in the subtree. A
        state is acceptable if it is "in the slice", that is, if its log-joint
        probability with its momentum is greater than `log_slice_sample`.
      leapfrogs: Number of leapfrog steps computed in this subtree, as a
        diagnostic.
      continue_trajectory: bool determining whether to continue the simulation
        trajectory. The trajectory is continued if no U-turns are encountered
        within the built subtree, and if the log-probability accumulation due
        to integration error does not exceed `max_simulation_error`.
    """
    if truthy(depth > 0):  # Recursive case
      # Build a tree at the current state.
      (reverse, forward, next_,
       num_states, leapfrogs, continue_trajectory) = _build_tree(
           current, direction, depth - 1, step_size, log_slice_sample)
      more_leapfrogs = 0
      if truthy(continue_trajectory):
        # If the just-built subtree did not terminate, build a second subtree
        # at the forward or reverse state, as appropriate.
        # TODO(b/122732601): Revert back to `if` when compiler makes the xform
        in_ = _tf_where(direction < 0, reverse, forward)
        (reverse_out, forward_out, far,
         far_num_states, more_leapfrogs, far_continue) = _build_tree(
             in_, direction, depth - 1, step_size, log_slice_sample)
        reverse_in = reverse
        reverse = _tf_where(direction < 0, reverse_out, reverse_in)
        forward_in = forward
        forward = _tf_where(direction < 0, forward_in, forward_out)

        # Propose either `next_` (which came from the first subtree and
        # so is nearby) or the new forward/reverse state (which came from the
        # second subtree and so is far away).
        num_states_old = num_states
        num_states = num_states_old + far_num_states
        accept_far_state = _binomial_subtree_acceptance_batched(
            far_num_states, num_states, seed_stream)
        # TODO(b/122732601): Revert back to `if` when compiler makes the xform
        next_in = next_
        next_ = _tf_where(accept_far_state, far, next_in)

        # Continue the NUTS trajectory if the far subtree did not terminate
        # either, and if the reverse-most and forward-most states do not
        # exhibit a U-turn.
        continue_trajectory = _continue_test_batched(
            far_continue, forward, reverse)

      return (reverse, forward, next_,
              num_states, leapfrogs + more_leapfrogs, continue_trajectory)
    else:  # Base case
      # Take a leapfrog step. Terminate the tree-building if the simulation
      # error from the leapfrog integrator is too large. States discovered by
      # continuing the simulation are likely to have very low probability.
      next_ = _leapfrog(
          value_and_gradients_fn=value_and_gradients_fn,
          current=current,
          step_size=step_size,
          direction=direction,
          unrolled_leapfrog_steps=unrolled_leapfrog_steps)
      next_log_joint = _log_joint(next_)
      num_states = _compute_num_states_batched(
          next_log_joint, log_slice_sample)
      # This 1000 is the max_simulation_error.  Inlined instead of named so
      # TensorFlow can infer its dtype from context, b/c the type inference in
      # the auto-batching system gets confused.  TODO(axch): Re-extract.
      continue_trajectory = (next_log_joint > log_slice_sample - 1000.)
      return (next_, next_, next_, num_states, unrolled_leapfrog_steps,
              continue_trajectory)

  return many_steps, ctx


def _embed_no_none_gradient_check(value_and_gradients_fn):
  """Wraps value and gradients function to assist with None gradients."""
  @functools.wraps(value_and_gradients_fn)
  def func_wrapped(*args, **kwargs):
    """Wrapped function which checks for None gradients."""
    value, grads = value_and_gradients_fn(*args, **kwargs)
    if any(grad is None for grad in grads):
      raise ValueError(
          "Gradient is None for a state.", args, kwargs, value, grads)
    return value, grads
  return func_wrapped


def _start_trajectory_batched(
    current_state, current_target_log_prob, seed_stream):
  """Computations needed to start a trajectory."""
  with tf1.name_scope("start_trajectory_batched"):
    batch_size = tf.shape(input=current_state[0])[0]
    current_momentum = []
    for state_tensor in current_state:
      momentum_tensor = tf.random.normal(
          shape=tf.shape(input=state_tensor),
          dtype=state_tensor.dtype,
          seed=seed_stream())
      current_momentum.append(momentum_tensor)

    # Draw a slice variable u ~ Uniform(0, p(initial state, initial
    # momentum)) and compute log u. For numerical stability, we perform this
    # in log space where log u = log (u' * p(...)) = log u' + log
    # p(...) and u' ~ Uniform(0, 1).
    log_slice_sample = tf.math.log1p(-tf.random.uniform(
        shape=[batch_size],
        dtype=current_target_log_prob.dtype,
        seed=seed_stream()))
    log_slice_sample += _log_joint(Point(
        None, current_target_log_prob, None, current_momentum))

    return current_momentum, log_slice_sample


def _batchwise_reduce_sum(x):
  with tf1.name_scope("batchwise_reduce_sum"):
    return tf.reduce_sum(input_tensor=x, axis=tf.range(1, tf.rank(x)))


def _has_no_u_turn(state_one, state_two, momentum):
  """If two given states and momentum do not exhibit a U-turn pattern."""
  with tf1.name_scope("has_no_u_turn"):
    batch_dot_product = sum(
        [_batchwise_reduce_sum((s1 - s2) * m)
         for s1, s2, m in zip(state_one, state_two, momentum)])
    return batch_dot_product > 0


def _leapfrog_base(value_and_gradients_fn,
                   current,
                   step_size,
                   direction,
                   unrolled_leapfrog_steps):
  """Runs `unrolled_leapfrog_steps` steps of leapfrog integration."""
  with tf1.name_scope("leapfrog"):
    step_size = [d * s for d, s in zip(direction, step_size)]
    for _ in range(unrolled_leapfrog_steps):
      mid_momentum = [
          m + 0.5 * step * g for m, step, g in
          zip(current.momentum, step_size, current.grads_target_log_prob)]
      next_state = [
          s + step * m for s, step, m in
          zip(current.state, step_size, mid_momentum)]
      with tf1.name_scope("gradients"):
        [next_target_log_prob,
         next_grads_target_log_prob] = value_and_gradients_fn(*next_state)
      next_momentum = [
          m + 0.5 * step * g for m, step, g in
          zip(mid_momentum, step_size, next_grads_target_log_prob)]
      current = Point(next_state,
                      next_target_log_prob,
                      next_grads_target_log_prob,
                      next_momentum)
    return current


def _leapfrog(
    value_and_gradients_fn,
    current,
    step_size,
    direction,
    unrolled_leapfrog_steps):
  """_leapfrog_base, with input rank padding by force of will."""
  # The purpose of this padding is to simulate broadcasting rank expansion
  # despite a batch dimension.

  # The problem is this: Consider, say, adding two Tensors of shape [2, 3] and
  # [3].  Addition broadcasts, which automatically expands the rank of the
  # second one by padding with size-1 dimensions, to [1, 3], and matches the [2,
  # 3] shape by tiling.  But, what happens if both of these Tensors gain a batch
  # dimension of size 4?  Naively, the shapes become [4, 2, 3] and [4, 3].  The
  # batch dimension of the second Tensor now aligns with the top non-batch
  # dimension of the first, and broadcasting fails.

  # This happens in the leapfrog integrator for matching step sizes with the
  # state Tensors being stepped.  It also happens as a knock-on for the
  # direction of integration, which multiplies the step size.
  step_size = [_expand_dims_under_batch_dim(step, tf.rank(state))
               for step, state in zip(step_size, current.state)]
  direction = [_expand_dims_under_batch_dim(direction, tf.rank(step))
               for step in step_size]
  return _leapfrog_base(
      value_and_gradients_fn,
      current,
      step_size,
      direction,
      unrolled_leapfrog_steps)


def _expand_dims_under_batch_dim(tensor, new_rank):
  """Adds size-1 dimensions below the first until `tensor` has `new_rank`."""
  if not tf.is_tensor(tensor):
    return tensor
  ones = tf.ones([new_rank - tf.rank(tensor)], dtype=tf.int32)
  shape = tf.shape(input=tensor)
  new_shape = tf.concat([shape[:1], ones, shape[1:]], axis=0)
  return tf.reshape(tensor, new_shape)


def _log_joint(current):
  """Log-joint probability given a state's log-probability and momentum."""
  with tf1.name_scope("log_joint"):
    momentum_log_prob = -sum([
        _batchwise_reduce_sum(0.5 * (m ** 2)) for m in current.momentum])
    return current.target_log_prob + momentum_log_prob


def _compute_num_states_batched(next_log_joint, log_slice_sample):
  # Returns the number of states (of necessity, at most one per batch member)
  # represented by the `next_log_joint` Tensor that are good enough to pass the
  # slice variable.
  with tf1.name_scope("compute_num_states_batched"):
    return tf.cast(next_log_joint > log_slice_sample, dtype=tf.int64)


def _random_bernoulli(shape, probs, dtype=tf.int64, seed=None, name=None):
  """Returns samples from a Bernoulli distribution."""
  with tf1.name_scope(name, "random_bernoulli", [shape, probs]):
    probs = tf.convert_to_tensor(value=probs)
    random_uniform = tf.random.uniform(shape, dtype=probs.dtype, seed=seed)
    return tf.cast(tf.less(random_uniform, probs), dtype)


def _continue_test_batched(
    continue_trajectory, forward, reverse):
  with tf1.name_scope("continue_test_batched"):
    return (continue_trajectory &
            _has_no_u_turn(forward.state, reverse.state, forward.momentum) &
            _has_no_u_turn(forward.state, reverse.state, reverse.momentum))


def _binomial_subtree_acceptance_batched(
    num_states_in_subtree, num_states, seed_stream):
  with tf1.name_scope("binomial_subtree_acceptance_batched"):
    batch_size = tf.shape(input=num_states_in_subtree)[0]
    return _random_bernoulli(
        [batch_size],
        probs=tf.minimum(
            tf.cast(num_states_in_subtree, dtype=tf.float32) /
            tf.cast(num_states, dtype=tf.float32), 1.),
        dtype=tf.bool,
        seed=seed_stream())


def _choose_direction_batched(point, seed_stream):
  with tf1.name_scope("choose_direction_batched"):
    batch_size = tf.shape(input=point.state[0])[0]
    dtype = point.state[0].dtype
    return tfp_random.rademacher(
        [batch_size], dtype=dtype, seed=seed_stream())


def _tf_where(condition, x, y):
  return ab.instructions.pattern_map2(
      lambda x_elt, y_elt: tf1.where(condition, x_elt, y_elt), x, y)
