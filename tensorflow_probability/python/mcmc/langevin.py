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
"""Metropolis-adjusted Langevin algorithm, a gradient-based MCMC algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.math import diag_jacobian
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    'MetropolisAdjustedLangevinAlgorithm',
    'UncalibratedLangevin',
]


class UncalibratedLangevinKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'UncalibratedLangevinKernelResults',
        [
            'grads_target_log_prob',
            # Results are for "next_state".
            'log_acceptance_correction',
            'target_log_prob',
            'volatility',
            'grads_volatility',
            'diffusion_drift',
            'seed',
        ])):
  """Internal state and diagnostics for Uncalibrated Langevin."""
  __slots__ = ()


class MetropolisAdjustedLangevinAlgorithm(kernel_base.TransitionKernel):
  """Runs one step of Metropolis-adjusted Langevin algorithm.

  Metropolis-adjusted Langevin algorithm (MALA) is a Markov chain Monte Carlo
  (MCMC) algorithm that takes a step of a discretised Langevin diffusion as a
  proposal. This class implements one step of MALA using Euler-Maruyama method
  for a given `current_state` and diagonal preconditioning `volatility` matrix.
  Mathematical details and derivations can be found in
  [Roberts and Rosenthal (1998)][1] and [Xifara et al. (2013)][2].

  See `UncalibratedLangevin` class description below for details on the proposal
  generating step of the algorithm.

  The `one_step` function can update multiple chains in parallel. It assumes
  that all leftmost dimensions of `current_state` index independent chain states
  (and are therefore updated independently). The output of
  `target_log_prob_fn(*current_state)` should reduce log-probabilities across
  all event dimensions. Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0, :]` could have a
  different target distribution from `current_state[1, :]`. These semantics are
  governed by `target_log_prob_fn(*current_state)`. (The number of independent
  chains is `tf.size(target_log_prob_fn(*current_state))`.)

  #### Examples:

  ##### Simple chain with warm-up.

  In this example we sample from a standard univariate normal
  distribution using MALA with `step_size` equal to 0.75.

  ```python
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  import numpy as np
  import matplotlib.pyplot as plt

  tf.enable_v2_behavior()

  tfd = tfp.distributions
  dtype = np.float32

  # Target distribution is Standard Univariate Normal
  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  def target_log_prob(x):
    return target.log_prob(x)

  # Define MALA sampler with `step_size` equal to 0.75
  samples = tfp.mcmc.sample_chain(
      num_results=1000,
      current_state=dtype(1),
      kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
          target_log_prob_fn=target_log_prob,
          step_size=0.75),
      num_burnin_steps=500,
      trace_fn=None,
      seed=42)

  sample_mean = tf.reduce_mean(samples, axis=0)
  sample_std = tf.sqrt(
      tf.reduce_mean(
          tf.math.squared_difference(samples, sample_mean),
          axis=0))

  print('sample mean', sample_mean)
  print('sample standard deviation', sample_std)

  plt.title('Traceplot')
  plt.plot(samples.numpy(), 'b')
  plt.xlabel('Iteration')
  plt.ylabel('Position')
  plt.show()
  ```

  ##### Sample from a 3-D Multivariate Normal distribution.

  In this example we also consider a non-constant volatility function.

  ```python
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  import numpy as np

  tf.enable_v2_behavior()

  dtype = np.float32
  true_mean = dtype([0, 0, 0])
  true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
  num_results = 500
  num_chains = 500

  # Target distribution is defined through the Cholesky decomposition
  chol = tf.linalg.cholesky(true_cov)
  target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

  # Here we define the volatility function to be non-constant
  def volatility_fn(x):
    # Stack the input tensors together
    return 1. / (0.5 + 0.1 * tf.math.abs(x))

  # Initial state of the chain
  init_state = np.ones([num_chains, 3], dtype=dtype)

  # Run MALA with normal proposal for `num_results` iterations for
  # `num_chains` independent chains:
  states = tfp.mcmc.sample_chain(
      num_results=num_results,
      current_state=init_state,
      kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
          target_log_prob_fn=target.log_prob,
          step_size=.1,
          volatility_fn=volatility_fn),
      num_burnin_steps=200,
      num_steps_between_results=1,
      trace_fn=None,
      seed=42)

  sample_mean = tf.reduce_mean(states, axis=[0, 1])
  x = (states - sample_mean)[..., tf.newaxis]
  sample_cov = tf.reduce_mean(
      tf.matmul(x, tf.transpose(x, [0, 1, 3, 2])), [0, 1])

  print('sample mean', sample_mean.numpy())
  print('sample covariance matrix', sample_cov.numpy())
  ```

  #### References

  [1]: Gareth Roberts and Jeffrey Rosenthal. Optimal Scaling of Discrete
       Approximations to Langevin Diffusions. _Journal of the Royal Statistical
       Society: Series B (Statistical Methodology)_, 60: 255-268, 1998.
       https://doi.org/10.1111/1467-9868.00123

  [2]: T. Xifara et al. Langevin diffusions and the Metropolis-adjusted
       Langevin algorithm. _arXiv preprint arXiv:1309.2983_, 2013.
       https://arxiv.org/abs/1309.2983
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               volatility_fn=None,
               parallel_iterations=10,
               name=None):
    """Initializes MALA transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      volatility_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns
        volatility value at `current_state`. Should return a `Tensor` or Python
        `list` of `Tensor`s that must broadcast with the shape of
        `current_state` Defaults to the identity function.
      parallel_iterations: the number of coordinates for which the gradients of
        the volatility matrix `volatility_fn` can be computed in parallel.
        Default value: `None` (i.e., use system default).
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'mala_kernel').

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `step_size` or a list with same length as
        `current_state`.
      TypeError: if `volatility_fn` is not callable.
    """
    impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UncalibratedLangevin(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            volatility_fn=volatility_fn,
            parallel_iterations=parallel_iterations,
            name=name))

    self._impl = impl
    parameters = impl.inner_kernel.parameters.copy()
    # Remove `compute_acceptance` parameter as this is not a MALA kernel
    # `__init__` parameter.
    del parameters['compute_acceptance']
    self._parameters = parameters

  @property
  def target_log_prob_fn(self):
    return self._impl.inner_kernel.target_log_prob_fn

  @property
  def volatility_fn(self):
    return self._impl.inner_kernel.volatility_fn

  @property
  def step_size(self):
    return self._impl.inner_kernel.step_size

  @property
  def parallel_iterations(self):
    return self._impl.inner_kernel.parallel_iterations

  @property
  def name(self):
    return self._impl.inner_kernel.name

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return True

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Runs one iteration of MALA.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s). The first `r` dimensions index
        independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)
      seed: Optional, a seed for reproducible sampling.

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `step_size` or a list with same length as
        `current_state` or `diffusion_drift`.
    """
    return self._impl.one_step(current_state, previous_kernel_results,
                               seed=seed)

  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    return self._impl.bootstrap_results(init_state)


class UncalibratedLangevin(kernel_base.TransitionKernel):
  """Runs one step of Uncalibrated Langevin discretized diffusion.

  The class generates a Langevin proposal using `_euler_method` function and
  also computes helper `UncalibratedLangevinKernelResults` for the next
  iteration.

  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use
  `MetropolisAdjustedLangevinAlgorithm(...)` or
  `MetropolisHastings(UncalibratedLangevin(...))`.

  For more details on `UncalibratedLangevin`, see
  `MetropolisAdjustedLangevinAlgorithm`.
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               volatility_fn=None,
               parallel_iterations=10,
               compute_acceptance=True,
               name=None):
    """Initializes Langevin diffusion transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      volatility_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns
        volatility value at `current_state`. Should return a `Tensor` or Python
        `list` of `Tensor`s that must broadcast with the shape of
        `current_state` Defaults to the identity function.
      parallel_iterations: the number of coordinates for which the gradients of
        the volatility matrix `volatility_fn` can be computed in parallel.
      compute_acceptance: Python 'bool' indicating whether to compute the
        Metropolis log-acceptance ratio used to construct
        `MetropolisAdjustedLangevinAlgorithm` kernel.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'mala_kernel').

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `step_size` or a list with same length as
        `current_state`.
      TypeError: if `volatility_fn` is not callable.
    """
    # Default value of `volatility_fn` is the identity function.
    if volatility_fn is None:
      volatility_fn = lambda *args: 1.
    if not callable(volatility_fn):
      raise TypeError('`volatility_fn` must be callable (saw: {})'.format(
          type(volatility_fn)))
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        volatility_fn=volatility_fn,
        compute_acceptance=tf.convert_to_tensor(compute_acceptance),
        parallel_iterations=parallel_iterations,
        name=name)

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def step_size(self):
    return self._parameters['step_size']

  @property
  def volatility_fn(self):
    return self._parameters['volatility_fn']

  @property
  def compute_acceptance(self):
    return self._parameters['compute_acceptance']

  @property
  def parallel_iterations(self):
    return self._parameters['parallel_iterations']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return False

  @mcmc_util.set_doc(MetropolisAdjustedLangevinAlgorithm.one_step.__doc__)
  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(mcmc_util.make_name(self.name, 'mala', 'one_step')):
      with tf.name_scope('initialize'):
        # Prepare input arguments to be passed to `_euler_method`.
        [
            current_state_parts,
            step_size_parts,
            current_target_log_prob,
            _,  # grads_target_log_prob
            current_volatility_parts,
            _,  # grads_volatility
            current_drift_parts,
        ] = _prepare_args(
            self.target_log_prob_fn,
            self.volatility_fn,
            current_state,
            self.step_size,
            previous_kernel_results.target_log_prob,
            previous_kernel_results.grads_target_log_prob,
            previous_kernel_results.volatility,
            previous_kernel_results.grads_volatility,
            previous_kernel_results.diffusion_drift,
            self.parallel_iterations)

        seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
        seeds = samplers.split_seed(
            seed, n=len(current_state_parts), salt='langevin.one_step')

        random_draw_parts = []
        for state_part, part_seed in zip(current_state_parts, seeds):
          random_draw_parts.append(
              samplers.normal(
                  shape=ps.shape(state_part),
                  dtype=dtype_util.base_dtype(state_part.dtype),
                  seed=part_seed))

      # Number of independent chains run by the algorithm.
      independent_chain_ndims = ps.rank(current_target_log_prob)

      # Generate the next state of the algorithm using Euler-Maruyama method.
      next_state_parts = _euler_method(random_draw_parts,
                                       current_state_parts,
                                       current_drift_parts,
                                       step_size_parts,
                                       current_volatility_parts)

      # Compute helper `UncalibratedLangevinKernelResults` to be processed by
      # `_compute_log_acceptance_correction` and in the next iteration of
      # `one_step` function.
      [
          _,  # state_parts
          _,  # step_sizes
          next_target_log_prob,
          next_grads_target_log_prob,
          next_volatility_parts,
          next_grads_volatility,
          next_drift_parts,
      ] = _prepare_args(
          self.target_log_prob_fn,
          self.volatility_fn,
          next_state_parts,
          step_size_parts,
          parallel_iterations=self.parallel_iterations)

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(current_state) else x[0]

      # Decide whether to compute the acceptance ratio
      log_acceptance_correction_compute = _compute_log_acceptance_correction(
          current_state_parts,
          next_state_parts,
          current_volatility_parts,
          next_volatility_parts,
          current_drift_parts,
          next_drift_parts,
          step_size_parts,
          independent_chain_ndims)
      log_acceptance_correction_skip = tf.zeros_like(next_target_log_prob)

      log_acceptance_correction = tf.cond(
          pred=self.compute_acceptance,
          true_fn=lambda: log_acceptance_correction_compute,
          false_fn=lambda: log_acceptance_correction_skip)

      return [
          maybe_flatten(next_state_parts),
          UncalibratedLangevinKernelResults(
              log_acceptance_correction=log_acceptance_correction,
              target_log_prob=next_target_log_prob,
              grads_target_log_prob=next_grads_target_log_prob,
              volatility=maybe_flatten(next_volatility_parts),
              grads_volatility=next_grads_volatility,
              diffusion_drift=next_drift_parts,
              seed=seed,
          ),
      ]

  @mcmc_util.set_doc(
      MetropolisAdjustedLangevinAlgorithm.bootstrap_results.__doc__)
  def bootstrap_results(self, init_state):
    with tf.name_scope(mcmc_util.make_name(
        self.name, 'mala', 'bootstrap_results')):
      init_state_parts = (list(init_state)
                          if mcmc_util.is_list_like(init_state)
                          else [init_state])

      init_state_parts = [
          tf.convert_to_tensor(x) for x in init_state_parts
      ]
      init_volatility = self.volatility_fn(*init_state_parts)  # pylint: disable=not-callable

      [
          _,  # state_parts
          _,  # step_sizes
          init_target_log_prob,
          init_grads_target_log_prob,
          init_volatility,
          init_grads_volatility,
          init_diffusion_drift,
      ] = _prepare_args(
          self.target_log_prob_fn,
          self.volatility_fn,
          state=init_state_parts,
          step_size=self.step_size,
          volatility=init_volatility,
          parallel_iterations=self.parallel_iterations)

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(init_state) else x[0]

      return UncalibratedLangevinKernelResults(
          log_acceptance_correction=tf.zeros_like(init_target_log_prob),
          target_log_prob=init_target_log_prob,
          grads_target_log_prob=init_grads_target_log_prob,
          volatility=maybe_flatten(init_volatility),
          grads_volatility=init_grads_volatility,
          diffusion_drift=init_diffusion_drift,
          # Allow room for one_step's seed.
          seed=samplers.zeros_seed(),
      )


def _euler_method(random_draw_parts,
                  state_parts,
                  drift_parts,
                  step_size_parts,
                  volatility_parts,
                  name=None):
  """Applies one step of Euler-Maruyama method.

  Generates proposal of the form:

  ```python
  tfd.Normal(loc=state_parts + _get_drift(state_parts, ...),
             scale=tf.sqrt(step_size * volatility_fn(current_state)))
  ```

  `_get_drift(state_parts, ..)` is a diffusion drift value at `state_parts`.


  Args:
    random_draw_parts: Python `list` of `Tensor`s containing the value(s) of the
      random perturbation variable(s). Must broadcast with the shape of
      `state_parts`.
    state_parts: Python `list` of `Tensor`s representing the current
      state(s) of the Markov chain(s).
    drift_parts: Python `list` of `Tensor`s representing value of the drift
      `_get_drift(*state_parts, ..)`. Must broadcast with the shape of
      `state_parts`.
    step_size_parts: Python `list` of `Tensor`s representing the step size for
      the Euler-Maruyama method. Must broadcast with the shape of
      `state_parts`.  Larger step sizes lead to faster progress, but
      too-large step sizes make rejection exponentially more likely. When
      possible, it's often helpful to match per-variable step sizes to the
      standard deviations of the target distribution in each variable.
    volatility_parts: Python `list` of `Tensor`s representing the value of
      `volatility_fn(*state_parts)`. Must broadcast with the shape of
      `state_parts`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'mala_euler_method').

  Returns:
    proposed_state_parts: Tensor or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at each result step. Has same shape as
      input `current_state_parts`.
  """
  with tf.name_scope(name or 'mala_euler_method'):
    proposed_state_parts = []
    for random_draw, state, drift, step_size, volatility in zip(
        random_draw_parts,
        state_parts,
        drift_parts,
        step_size_parts,
        volatility_parts):
      proposal = state + drift + volatility * tf.sqrt(step_size) * random_draw
      proposed_state_parts.append(proposal)

    return proposed_state_parts


def _get_drift(step_size_parts, volatility_parts, grads_volatility,
               grads_target_log_prob,
               name=None):
  """Compute diffusion drift at the current location `current_state`.

  The drift of the diffusion at is computed as

  ```none
  0.5 * `step_size` * volatility_parts * `target_log_prob_fn(current_state)`
  + `step_size` * `grads_volatility`
  ```

  where `volatility_parts` = `volatility_fn(current_state)**2` and
  `grads_volatility` is a gradient of `volatility_parts` at the `current_state`.

  Args:
    step_size_parts: Python `list` of `Tensor`s representing the step size for
      Euler-Maruyama method. Must broadcast with the shape of
      `volatility_parts`.  Larger step sizes lead to faster progress, but
      too-large step sizes make rejection exponentially more likely. When
      possible, it's often helpful to match per-variable step sizes to the
      standard deviations of the target distribution in each variable.
    volatility_parts: Python `list` of `Tensor`s representing the value of
      `volatility_fn(*state_parts)`.
    grads_volatility: Python list of `Tensor`s representing the value of the
      gradient of `volatility_parts**2` wrt the state of the chain.
    grads_target_log_prob: Python list of `Tensor`s representing
      gradient of `target_log_prob_fn(*state_parts`) wrt `state_parts`. Must
      have same shape as `volatility_parts`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'mala_get_drift').

  Returns:
    drift_parts: Tensor or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at each result step. Has same shape as
      input `current_state_parts`.
  """

  with tf.name_scope(name or 'mala_get_drift'):

    drift_parts = []

    for step_size, volatility, grad_volatility, grad_target_log_prob in (
        zip(step_size_parts,
            volatility_parts,
            grads_volatility,
            grads_target_log_prob)):
      volatility_squared = tf.square(volatility)
      drift = 0.5 * step_size * (volatility_squared * grad_target_log_prob
                                 + grad_volatility)
      drift_parts.append(drift)

    return drift_parts


def _compute_log_acceptance_correction(current_state_parts,
                                       proposed_state_parts,
                                       current_volatility_parts,
                                       proposed_volatility_parts,
                                       current_drift_parts,
                                       proposed_drift_parts,
                                       step_size_parts,
                                       independent_chain_ndims,
                                       name=None):
  r"""Helper to `kernel` which computes the log acceptance-correction.

  Computes `log_acceptance_correction` as described in `MetropolisHastings`
  class. The proposal density is normal. More specifically,

   ```none
  q(proposed_state | current_state) \sim N(current_state + current_drift,
  step_size * current_volatility**2)

  q(current_state | proposed_state) \sim N(proposed_state + proposed_drift,
  step_size * proposed_volatility**2)
  ```

  The `log_acceptance_correction` is then

  ```none
  log_acceptance_correctio = q(current_state | proposed_state)
  - q(proposed_state | current_state)
  ```

  Args:
    current_state_parts: Python `list` of `Tensor`s representing the value(s) of
      the current state of the chain.
    proposed_state_parts:  Python `list` of `Tensor`s representing the value(s)
      of the proposed state of the chain. Must broadcast with the shape of
      `current_state_parts`.
    current_volatility_parts: Python `list` of `Tensor`s representing the value
      of `volatility_fn(*current_volatility_parts)`. Must broadcast with the
      shape of `current_state_parts`.
    proposed_volatility_parts: Python `list` of `Tensor`s representing the value
      of `volatility_fn(*proposed_volatility_parts)`. Must broadcast with the
      shape of `current_state_parts`
    current_drift_parts: Python `list` of `Tensor`s representing value of the
      drift `_get_drift(*current_state_parts, ..)`. Must broadcast with the
      shape of `current_state_parts`.
    proposed_drift_parts: Python `list` of `Tensor`s representing value of the
      drift `_get_drift(*proposed_drift_parts, ..)`. Must broadcast with the
      shape of `current_state_parts`.
    step_size_parts: Python `list` of `Tensor`s representing the step size for
      Euler-Maruyama method. Must broadcast with the shape of
      `current_state_parts`.
    independent_chain_ndims: Scalar `int` `Tensor` representing the number of
      leftmost `Tensor` dimensions which index independent chains.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'compute_log_acceptance_correction').

  Returns:
    log_acceptance_correction: `Tensor` representing the `log`
      acceptance-correction.  (See docstring for mathematical definition.)
  """

  with tf.name_scope(name or 'compute_log_acceptance_correction'):

    proposed_log_density_parts = []
    dual_log_density_parts = []

    for [
        current_state,
        proposed_state,
        current_volatility,
        proposed_volatility,
        current_drift,
        proposed_drift,
        step_size,
    ] in zip(
        current_state_parts,
        proposed_state_parts,
        current_volatility_parts,
        proposed_volatility_parts,
        current_drift_parts,
        proposed_drift_parts,
        step_size_parts,
    ):
      axis = tf.range(independent_chain_ndims, tf.rank(current_state))

      state_diff = proposed_state - current_state

      current_volatility *= tf.sqrt(step_size)

      proposed_energy = (state_diff - current_drift) / current_volatility

      proposed_volatility *= tf.sqrt(step_size)
      # Compute part of `q(proposed_state | current_state)`
      proposed_energy = (
          tf.reduce_sum(
              mcmc_util.safe_sum(
                  [tf.math.log(current_volatility),
                   0.5 * (proposed_energy**2)]),
              axis=axis))
      proposed_log_density_parts.append(-proposed_energy)

      # Compute part of `q(current_state | proposed_state)`
      dual_energy = (state_diff + proposed_drift) / proposed_volatility
      dual_energy = (
          tf.reduce_sum(
              mcmc_util.safe_sum(
                  [tf.math.log(proposed_volatility), 0.5 * (dual_energy**2)]),
              axis=axis))
      dual_log_density_parts.append(-dual_energy)

    # Compute `q(proposed_state | current_state)`
    proposed_log_density_reduce = tf.add_n(proposed_log_density_parts)
    # Compute `q(current_state | proposed_state)`
    dual_log_density_reduce = tf.add_n(dual_log_density_parts)

    return mcmc_util.safe_sum([
        dual_log_density_reduce, -proposed_log_density_reduce])


def _maybe_call_volatility_fn_and_grads(volatility_fn,
                                        state,
                                        volatility_fn_results=None,
                                        grads_volatility_fn=None,
                                        sample_shape=None,
                                        parallel_iterations=10):
  """Helper which computes `volatility_fn` results and grads, if needed."""
  state_parts = list(state) if mcmc_util.is_list_like(state) else [state]
  needs_volatility_fn_gradients = grads_volatility_fn is None

  # Convert `volatility_fn_results` to a list
  if volatility_fn_results is None:
    volatility_fn_results = volatility_fn(*state_parts)

  volatility_fn_results = (list(volatility_fn_results)
                           if mcmc_util.is_list_like(volatility_fn_results)
                           else [volatility_fn_results])
  if len(volatility_fn_results) == 1:
    volatility_fn_results *= len(state_parts)
  if len(state_parts) != len(volatility_fn_results):
    raise ValueError('`volatility_fn` should return a tensor or a list '
                     'of the same length as `current_state`.')

  # The shape of 'volatility_parts' needs to have the number of chains as a
  # leading dimension. For determinism we broadcast 'volatility_parts' to the
  # shape of `state_parts` since each dimension of `state_parts` could have a
  # different volatility value.

  volatility_fn_results = _maybe_broadcast_volatility(volatility_fn_results,
                                                      state_parts)
  if grads_volatility_fn is None:
    [
        _,
        grads_volatility_fn,
    ] = diag_jacobian(
        xs=state_parts,
        ys=volatility_fn_results,
        sample_shape=sample_shape,
        parallel_iterations=parallel_iterations,
        fn=volatility_fn)

  # Compute gradient of `volatility_parts**2`
  if needs_volatility_fn_gradients:
    grads_volatility_fn = [
        2. * g * volatility if g is not None else tf.zeros_like(fn_arg)
        for g, volatility, fn_arg in zip(
            grads_volatility_fn, volatility_fn_results, state_parts)
    ]

  return volatility_fn_results, grads_volatility_fn


def _maybe_broadcast_volatility(volatility_parts,
                                state_parts):
  """Helper to broadcast `volatility_parts` to the shape of `state_parts`."""
  return [v + tf.zeros_like(sp)
          for v, sp in zip(volatility_parts, state_parts)]


def _prepare_args(target_log_prob_fn,
                  volatility_fn,
                  state,
                  step_size,
                  target_log_prob=None,
                  grads_target_log_prob=None,
                  volatility=None,
                  grads_volatility_fn=None,
                  diffusion_drift=None,
                  parallel_iterations=10):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts = list(state) if mcmc_util.is_list_like(state) else [state]

  [
      target_log_prob,
      grads_target_log_prob,
  ] = mcmc_util.maybe_call_fn_and_grads(
      target_log_prob_fn,
      state_parts,
      target_log_prob,
      grads_target_log_prob)
  [
      volatility_parts,
      grads_volatility,
  ] = _maybe_call_volatility_fn_and_grads(
      volatility_fn,
      state_parts,
      volatility,
      grads_volatility_fn,
      ps.shape(target_log_prob),
      parallel_iterations)

  step_sizes = (list(step_size) if mcmc_util.is_list_like(step_size)
                else [step_size])
  step_sizes = [
      tf.convert_to_tensor(
          value=s, name='step_size', dtype=target_log_prob.dtype)
      for s in step_sizes
  ]
  if len(step_sizes) == 1:
    step_sizes *= len(state_parts)
  if len(state_parts) != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')

  if diffusion_drift is None:
    diffusion_drift_parts = _get_drift(step_sizes, volatility_parts,
                                       grads_volatility,
                                       grads_target_log_prob)
  else:
    diffusion_drift_parts = (list(diffusion_drift)
                             if mcmc_util.is_list_like(diffusion_drift)
                             else [diffusion_drift])
    if len(state_parts) != len(diffusion_drift):
      raise ValueError('There should be exactly one `diffusion_drift` or it '
                       'should have same length as list-like `current_state`.')

  return [
      state_parts,
      step_sizes,
      target_log_prob,
      grads_target_log_prob,
      volatility_parts,
      grads_volatility,
      diffusion_drift_parts,
  ]
