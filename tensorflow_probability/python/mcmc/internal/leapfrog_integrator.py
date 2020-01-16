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
"""Defines the LeapfrogIntegrator class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = [
    'LeapfrogIntegrator',
    'SimpleLeapfrogIntegrator',
    'process_args',
]


@six.add_metaclass(abc.ABCMeta)
class LeapfrogIntegrator(object):
  """Base class for all leapfrog integrators.

  [Leapfrog integrators](https://en.wikipedia.org/wiki/Leapfrog_integration)
  numerically integrate differential equations of the form:

  ```none
  v' = dv/dt = F(x)
  x' = dx/dt = v
  ```

  This class defines minimal requirements for leapfrog integration calculations.
  """

  @abc.abstractmethod
  def __call__(self, momentum_parts, state_parts,
               target=None, target_grad_parts=None,
               name=None):
    """Computes the integration.

    Args:
      momentum_parts: Python `list` of `Tensor`s representing momentume for each
        state part.
      state_parts: Python `list` of `Tensor`s which collectively representing
        the state.
      target: Batch of scalar `Tensor` representing the target (i.e.,
        unnormalized log prob) evaluated at `state_parts`.
      target_grad_parts: Python `list` of `Tensor`s representing the gradient of
        `target` with respect to each of `state_parts`.
      name: Python `str` used to group ops created by this function.

    Returns:
      next_momentum_parts: Python `list` of `Tensor`s representing new momentum.
      next_state_parts: Python `list` of `Tensor`s which collectively
        representing the new state.
      next_target: Batch of scalar `Tensor` representing the target (i.e.,
        unnormalized log prob) evaluated at `next_state_parts`.
      next_target_grad_parts: Python `list` of `Tensor`s representing the
        gradient of `next_target` with respect to each of `next_state_parts`.
    """
    raise NotImplementedError('Integrate logic not implemented.')


class SimpleLeapfrogIntegrator(LeapfrogIntegrator):
  # pylint: disable=line-too-long
  """Simple leapfrog integrator.

  Calling this functor is conceptually equivalent to:

  ```none
  def leapfrog(x, v, eps, L, f, M):
    g = lambda x: gradient(f, x)
    v[0] = v + eps/2 g(x)
    for l = 1...L:
      x[l] = x[l-1] + eps * inv(M) @ v[l-1]
      v[l] = v[l-1] + eps * g(x[l])
    v = v[L] - eps/2 * g(x[L])
    return x[L], v
  ```

  where `M = eye(dims(x))`.
  (In the future we may support arbitrary covariance `M`.)

  #### Examples:

  ```python
  import matplotlib.pyplot as plt
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl
  tf.enable_v2_behavior()

  dims = 10
  dtype = tf.float32

  target_fn = tfp.distributions.MultivariateNormalDiag(
      loc=tf.zeros(dims, dtype)).log_prob

  integrator = leapfrog_impl.SimpleLeapfrogIntegrator(
      target_fn,
      step_sizes=[0.1],
      num_steps=3)

  momentum = [tf.random.normal([dims], dtype=dtype)]
  position = [tf.random.normal([dims], dtype=dtype)]
  target = None
  target_grad_parts = None

  num_iter = int(1e3)
  positions = tf.zeros([num_iter, dims], dtype)
  for i in range(num_iter):
    [momentum, position, target, target_grad_parts] = integrator(
        momentum, position, target, target_grad_parts)
    positions = tf.tensor_scatter_nd_update(positions, [[i]], position)

  plt.plot(positions[:, 0]);  # Sinusoidal.
  ```

  """
  # pylint: enable=line-too-long

  def __init__(self, target_fn, step_sizes, num_steps):
    """Constructs the LeapfrogIntegrator.

    Assumes a simple quadratic kinetic energy function: `0.5 ||momentum||**2`.

    Args:
      target_fn: Python callable which takes an argument like `*state_parts` and
        returns its (possibly unnormalized) log-density under the target
        distribution.
      step_sizes: Python `list` of `Tensor`s representing the step size for the
        leapfrog integrator. Must broadcast with the shape of
        `current_state_parts`.  Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_steps: `int` `Tensor` representing  number of steps to run
        the leapfrog integration. Total progress is roughly proportional to
        `step_size * num_steps`.
    """
    # Note on per-variable step sizes:
    #
    # Using per-variable step sizes is equivalent to using the same step
    # size for all variables and adding a diagonal mass matrix in the
    # kinetic energy term of the Hamiltonian being integrated. This is
    # hinted at by Neal (2011) but not derived in detail there.
    #
    # Let x and v be position and momentum variables respectively.
    # Let g(x) be the gradient of `target_fn(x)`.
    # Let S be a diagonal matrix of per-variable step sizes.
    # Let the Hamiltonian H(x, v) = -target_fn(x) + 0.5 * ||v||**2.
    #
    # Using per-variable step sizes gives the updates:
    #
    #   v' = v0 + 0.5 * S @ g(x0)
    #   x1 = x0 + S @ v'
    #   v1 = v' + 0.5 * S @ g(x1)
    #
    # Let,
    #
    #   u = inv(S) @ v
    #
    # for "u'", "u0", and "u1". Multiplying v by inv(S) in the updates above
    # gives the transformed dynamics:
    #
    #   u' = inv(S) @ v'
    #      = inv(S) @ v0 + 0.5 * g(x)
    #      = u0 + 0.5 * g(x)
    #
    #   x1 = x0 + S @ v'
    #      = x0 + S @ S @ u'
    #
    #   u1 = inv(S) @ v1
    #      = inv(S) @ v' + 0.5 * g(x1)
    #      = u' + 0.5 * g(x1)
    #
    # These are exactly the leapfrog updates for the Hamiltonian
    #
    #   H'(x, u) = -target_fn(x) + 0.5 * (S @ u).T @ (S @ u)
    #            = -target_fn(x) + 0.5 * ||v||**2
    #            = H(x, v).
    #
    # To summarize:
    #
    # * Using per-variable step sizes implicitly simulates the dynamics
    #   of the Hamiltonian H' (which are energy-conserving in H'). We
    #   keep track of v instead of u, but the underlying dynamics are
    #   the same if we transform back.
    # * The value of the Hamiltonian H'(x, u) is the same as the value
    #   of the original Hamiltonian H(x, v) after we transform back from
    #   u to v.
    # * Sampling v ~ N(0, I) is equivalent to sampling u ~ N(0, S**-2).
    #
    # So using per-variable step sizes in HMC will give results that are
    # exactly identical to explicitly using a diagonal mass matrix.
    self._target_fn = target_fn
    self._step_sizes = step_sizes
    self._num_steps = num_steps

  @property
  def target_fn(self):
    return self._target_fn

  @property
  def step_sizes(self):
    return self._step_sizes

  @property
  def num_steps(self):
    return self._num_steps

  def __call__(self,
               momentum_parts,
               state_parts,
               target=None,
               target_grad_parts=None,
               name=None):
    """Applies `num_steps` of the leapfrog integrator.

    Args:
      momentum_parts: Python `list` of `Tensor`s representing momentume for each
        state part.
      state_parts: Python `list` of `Tensor`s which collectively representing
        the state.
      target: Batch of scalar `Tensor` representing the target (i.e.,
        unnormalized log prob) evaluated at `state_parts`.
      target_grad_parts: Python `list` of `Tensor`s representing the gradient of
        `target` with respect to each of `state_parts`.
      name: Python `str` used to group ops created by this function.

    Returns:
      next_momentum_parts: Python `list` of `Tensor`s representing new momentum.
      next_state_parts: Python `list` of `Tensor`s which collectively
        representing the new state.
      next_target: Batch of scalar `Tensor` representing the target (i.e.,
        unnormalized log prob) evaluated at `next_state_parts`.
      next_target_grad_parts: Python `list` of `Tensor`s representing the
        gradient of `next_target` with respect to each of `next_state_parts`.
    """
    with tf.name_scope(name or 'leapfrog_integrate'):
      [
          momentum_parts,
          state_parts,
          target,
          target_grad_parts,
      ] = process_args(
          self.target_fn,
          momentum_parts,
          state_parts,
          target,
          target_grad_parts)

      # See Algorithm 1 of "Faster Hamiltonian Monte Carlo by Learning Leapfrog
      # Scale", https://arxiv.org/abs/1810.04449.

      half_next_momentum_parts = [
          v + 0.5 * tf.cast(eps, v.dtype) * tf.cast(g, v.dtype)
          for v, eps, g
          in zip(momentum_parts, self.step_sizes, target_grad_parts)]

      [
          _,
          next_half_next_momentum_parts,
          next_state_parts,
          next_target,
          next_target_grad_parts,
      ] = tf.while_loop(
          cond=lambda i, *_: i < self.num_steps,
          body=lambda i, *args: [i + 1] + list(_one_step(  # pylint: disable=no-value-for-parameter,g-long-lambda
              self.target_fn, self.step_sizes, *args)),
          loop_vars=[
              tf.zeros_like(self.num_steps, name='iter'),
              half_next_momentum_parts,
              state_parts,
              target,
              target_grad_parts,
          ])

      next_momentum_parts = [
          v - 0.5 * tf.cast(eps, v.dtype) * tf.cast(g, v.dtype)  # pylint: disable=g-complex-comprehension
          for v, eps, g
          in zip(next_half_next_momentum_parts,
                 self.step_sizes,
                 next_target_grad_parts)
      ]

      return (
          next_momentum_parts,
          next_state_parts,
          next_target,
          next_target_grad_parts,
      )


def _one_step(
    target_fn,
    step_sizes,
    half_next_momentum_parts,
    state_parts,
    target,
    target_grad_parts):
  """Body of integrator while loop."""
  with tf.name_scope('leapfrog_integrate_one_step'):
    next_state_parts = [
        x + tf.cast(eps, x.dtype) * tf.cast(v, x.dtype)  # pylint: disable=g-complex-comprehension
        for x, eps, v
        in zip(state_parts, step_sizes, half_next_momentum_parts)
    ]

    [next_target, next_target_grad_parts] = mcmc_util.maybe_call_fn_and_grads(
        target_fn, next_state_parts)
    if any(g is None for g in next_target_grad_parts):
      raise ValueError(
          'Encountered `None` gradient.\n'
          '  state_parts: {}\n'
          '  next_state_parts: {}\n'
          '  next_target_grad_parts: {}'.format(
              state_parts,
              next_state_parts,
              next_target_grad_parts))

    tensorshape_util.set_shape(next_target, target.shape)
    for ng, g in zip(next_target_grad_parts, target_grad_parts):
      tensorshape_util.set_shape(ng, g.shape)

    next_half_next_momentum_parts = [
        v + tf.cast(eps, v.dtype) * tf.cast(g, v.dtype)  # pylint: disable=g-complex-comprehension
        for v, eps, g
        in zip(half_next_momentum_parts, step_sizes, next_target_grad_parts)]

    return [
        next_half_next_momentum_parts,
        next_state_parts,
        next_target,
        next_target_grad_parts,
    ]


def process_args(target_fn, momentum_parts, state_parts,
                 target=None, target_grad_parts=None):
  """Sanitize inputs to `__call__`."""
  with tf.name_scope('process_args'):
    momentum_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='momentum_parts')
        for v in momentum_parts]
    state_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='state_parts')
        for v in state_parts]
    if target is None or target_grad_parts is None:
      [target, target_grad_parts] = mcmc_util.maybe_call_fn_and_grads(
          target_fn, state_parts)
    else:
      target = tf.convert_to_tensor(
          target, dtype_hint=tf.float32, name='target')
      target_grad_parts = [
          tf.convert_to_tensor(
              g, dtype_hint=tf.float32, name='target_grad_part')
          for g in target_grad_parts]
    return momentum_parts, state_parts, target, target_grad_parts
