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
"""TransformedTransitionKernel Transition Kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'TransformedTransitionKernel',
]


class TransformedTransitionKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('TransformedTransitionKernelResults',
                           ['transformed_state',
                            'inner_results',
                            ])):
  """Internal state and diagnostics for Transformed kernel."""
  __slots__ = ()


def make_log_det_jacobian_fn(bijector, direction):
  """Makes a function which applies a list of Bijectors' `log_det_jacobian`s."""
  attr = '{}_log_det_jacobian'.format(direction)
  if not mcmc_util.is_list_like(bijector):
    dtype = getattr(bijector, '{}_dtype'.format(direction))()
    if mcmc_util.is_list_like(dtype):
      def multipart_fn(state_parts, event_ndims):
        return getattr(bijector, attr)(state_parts, event_ndims)
      return multipart_fn
    elif tf.nest.is_nested(dtype):
      raise ValueError(
          'Only list-like multi-part bijectors are currently supported, but '
          'got {}.'.format(tf.nest.map_structure(lambda _: '.', dtype)))
    bijector = [bijector]
  def fn(state_parts, event_ndims):
    return sum([
        getattr(b, attr)(sp, event_ndims=e)
        for b, e, sp in zip(bijector, event_ndims, state_parts)
    ])
  return fn


def make_transform_fn(bijector, direction):
  """Makes a function which applies a list of Bijectors' `forward`s."""
  if not mcmc_util.is_list_like(bijector):
    dtype = getattr(bijector, '{}_dtype'.format(direction))()
    if mcmc_util.is_list_like(dtype):
      return getattr(bijector, direction)
    elif tf.nest.is_nested(dtype):
      raise ValueError(
          'Only list-like multi-part bijectors are currently supported, but '
          'got {}.'.format(tf.nest.map_structure(lambda _: '.', dtype)))
    bijector = [bijector]
  def fn(state_parts):
    if len(bijector) != len(state_parts):
      raise ValueError('State has {} parts, but bijector has {}.'.format(
          len(state_parts), len(bijector)))
    transformed_parts = [
        getattr(b, direction)(sp) for b, sp in zip(bijector, state_parts)]
    return tf.nest.pack_sequence_as(state_parts, transformed_parts)
  return fn


def make_transformed_log_prob(
    log_prob_fn, bijector, direction, enable_bijector_caching=True):
  """Transforms a log_prob function using bijectors.

  Note: `direction = 'inverse'` corresponds to the transformation calculation
  done in `tfp.distributions.TransformedDistribution.log_prob`.

  Args:
    log_prob_fn: Python `callable` taking an argument for each state part which
      returns a `Tensor` representing the joint `log` probability of those state
      parts.
    bijector: `tfp.bijectors.Bijector`-like instance (or list thereof)
      corresponding to each state part. When `direction = 'forward'` the
      `Bijector`-like instance must possess members `forward` and
      `forward_log_det_jacobian` (and corresponding when
      `direction = 'inverse'`).
    direction: Python `str` being either `'forward'` or `'inverse'` which
      indicates the nature of the bijector transformation applied to each state
      part.
    enable_bijector_caching: Python `bool` indicating if `Bijector` caching
      should be invalidated.
      Default value: `True`.

  Returns:
    transformed_log_prob_fn: Python `callable` which takes an argument for each
      transformed state part and returns a `Tensor` representing the joint `log`
      probability of the transformed state parts.
  """
  if direction not in {'forward', 'inverse'}:
    raise ValueError('Argument `direction` must be either `"forward"` or '
                     '`"inverse"`; saw "{}".'.format(direction))
  fn = make_transform_fn(bijector, direction)
  ldj_fn = make_log_det_jacobian_fn(bijector, direction)
  def transformed_log_prob_fn(*state_parts):
    """Log prob of the transformed state."""
    if not enable_bijector_caching:
      state_parts = [tf.identity(sp) for sp in state_parts]
    tlp = log_prob_fn(*fn(state_parts))
    tlp_rank = prefer_static.rank(tlp)
    event_ndims = [(prefer_static.rank(sp) - tlp_rank) for sp in state_parts]
    return tlp + ldj_fn(state_parts, event_ndims)
  return transformed_log_prob_fn


def _make_kernel_stack(kernel):
  kernel_stack = [kernel]
  while 'target_log_prob_fn' not in kernel.parameters:
    if 'inner_kernel' not in kernel.parameters:
      raise ValueError('"None of the nested `inner_kernel`s contains a '
                       '`target_log_prob_fn`."')
    kernel = kernel.inner_kernel
    kernel_stack.append(kernel)
  return kernel_stack


def _find_nested_target_log_prob_recursively(kernel):
  kernel_stack = _make_kernel_stack(kernel)
  target_log_prob_fn = kernel_stack[-1].parameters['target_log_prob_fn']
  return target_log_prob_fn


def _update_target_log_prob(kernel, new_target_log_prob):
  """Replaces `target_log_prob_fn` of outermost `inner_kernel` of `kernel`."""
  kernel_stack = _make_kernel_stack(kernel)
  # Update to target_log_prob to `new_target_log_prob`.
  with deprecation.silence():
    prev_kernel = kernel_stack.pop()
    prev_kernel = prev_kernel.copy(
        target_log_prob_fn=new_target_log_prob)

    # Propagate the change upwards by reconstructing wrapper kernels.
    while kernel_stack:
      curr_kernel = kernel_stack.pop()
      updated_kernel = type(prev_kernel)(**prev_kernel.parameters)
      curr_kernel = curr_kernel.copy(inner_kernel=updated_kernel)
      prev_kernel = curr_kernel
    return prev_kernel


class TransformedTransitionKernel(kernel_base.TransitionKernel):
  """TransformedTransitionKernel applies a bijector to the MCMC's state space.

  The `TransformedTransitionKernel` `TransitionKernel` enables fitting
  a `tfp.bijectors.Bijector` which serves to decorrelate the Markov chain Monte
  Carlo (MCMC) event dimensions thus making the chain mix faster. This is
  particularly useful when the geometry of the target distribution is
  unfavorable. In such cases it may take many evaluations of the
  `target_log_prob_fn` for the chain to mix between faraway states.

  The idea of training an affine function to decorrelate chain event dims was
  presented in [Parno and Marzouk (2014)][1]. Used in conjunction with the
  `HamiltonianMonteCarlo` `TransitionKernel`, the [Parno and Marzouk (2014)][1]
  idea is an instance of Riemannian manifold HMC [(Girolami and Calderhead,
  2011)][2].

  The `TransformedTransitionKernel` enables arbitrary bijective transformations
  of arbitrary `TransitionKernel`s, e.g., one could use bijectors
  `tfp.bijectors.ScaleMatvecTriL`, `tfp.bijectors.RealNVP`, etc. with transition
  kernels `tfp.mcmc.HamiltonianMonteCarlo`, `tfp.mcmc.RandomWalkMetropolis`,
  etc.

  ### Transforming nested kernels

  `TransformedTransitionKernel` can operate on multiply nested kernels, as in
  the following example:

  ```python
  tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.SimpleStepSizeAdaptation(
      inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        ... # doesn't matter
      ),
      num_adaptation_steps=9)
    bijector=tfb.Identity()))
  ```

  Upon construction, `TransformedTransitionKernel` searches the given
  `inner_kernel` and the "stack" of nested kernels in any `inner_kernel`
  fields thereof until it finds one with a field called `target_log_prob_fn`,
  and replaces this with the transformed function. If no
  `inner_kernel` has such a target log prob a `ValueError` is raised.

  #### Mathematical Details

  `TransformedTransitionKernel` enables Markov chains which operate in
  "unconstrained space." Since we interpret the bijector as mapping
  "unconstrained space" to "user space", this means that the MCMC transformed
  `target_log_prob` is:

  ```python
  target_log_prob(bij.forward(x)) + bij.forward_log_det_jacobian(x)
  ```

  Recall that `tfp.distributions.TransformedDistribution` uses the `inverse` to
  compute its `log_prob`. Despite this difference, the use of `forward` in
  `TransformedTransitionKernel` is perfectly consistent with
  `TransformedDistribution` following the TFP convention of "sampling" being
  what defines semantics. The apparent difference is because
  `TransformedDistribution.log_prob` is derived from a user provided
  distribution while in `TransformedTransitionKernel` samples are derived from
  `target_log_prob_fn`. That is, in `TransformedDistribution` we do:

  ```python
  x ~ NoiseDistribution()
  y = bij.forward(x)
  log_prob_y = NoiseDistribution().log_prob(bij.inverse(y))
               + bij.inverse_log_det_jacobian(y)
  ```

  yet in `TransformedTransitionKernel` we do:

  ```python
  x ~ MCMC()
  y = bij.forward(x)
  log_prob_y = log_prob(y) + bij.forward_log_det_jacobian(x)
  ```

  In other words (and in general), `tfp.mcmc` is derived from a `log_prob`
  which what induces a *seeming* direction convention change. Aside from TFP
  convention, that Bijectors should adhere to "sample first" semantics is
  important because it mitigates pervasive necessity of `tfp.bijectors.Invert`
  in user code.

  #### Examples

  ##### RealNVP + HamiltonianMonteCarlo

  Note: this example is only meant to illustrate how to wire up a
  `TransformedTransitionKernel`. As it is this won't work well because:
  * a 1-layer RealNVP is a pretty weak density model, since it can't change the
  density of the masked dimensions
  * we're not actually training the bijector to do anything useful.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors

  def make_likelihood(true_variances):
    return tfd.MultivariateNormalDiag(
        scale_diag=tf.sqrt(true_variances))

  dims = 10
  dtype = np.float32
  true_variances = tf.linspace(dtype(1), dtype(3), dims)
  likelihood = make_likelihood(true_variances)

  realnvp_hmc = tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=likelihood.log_prob,
        step_size=0.5,
        num_leapfrog_steps=2),
      bijector=tfb.RealNVP(
        num_masked=2,
        shift_and_log_scale_fn=tfb.real_nvp_default_template(
            hidden_layers=[512, 512])))

  states, kernel_results = tfp.mcmc.sample_chain(
      num_results=1000,
      current_state=tf.zeros(dims),
      kernel=realnvp_hmc,
      num_burnin_steps=500)

  # Compute sample stats.
  sample_mean = tf.reduce_mean(states, axis=0)
  sample_var = tf.reduce_mean(
      tf.squared_difference(states, sample_mean),
      axis=0)
  ```

  #### References

  [1]: Matthew Parno and Youssef Marzouk. Transport map accelerated Markov chain
       Monte Carlo. _arXiv preprint arXiv:1412.5492_, 2014.
       https://arxiv.org/abs/1412.5492

  [2]: Mark Girolami and Ben Calderhead. Riemann manifold langevin and
       hamiltonian monte carlo methods. In _Journal of the Royal Statistical
       Society_, 2011. https://doi.org/10.1111/j.1467-9868.2010.00765.x
  """

  def __init__(self, inner_kernel, bijector, name=None):
    """Instantiates this object.

    Args:
      inner_kernel: `TransitionKernel`-like object that either has a
        `target_log_prob_fn` argument, or wraps around another `inner_kernel`
        with said argument.
      bijector: `tfp.distributions.Bijector` or list of
        `tfp.distributions.Bijector`s. These bijectors use `forward` to map the
        `inner_kernel` state space to the state expected by
        `inner_kernel.target_log_prob_fn`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "transformed_kernel").

    Returns:
      transformed_kernel: Instance of `TransitionKernel` which copies the input
        transition kernel then modifies its `target_log_prob_fn` by applying the
        provided bijector(s).
    """
    self._parameters = dict(
        inner_kernel=inner_kernel,
        bijector=bijector,
        name=name or 'transformed_kernel')
    target_log_prob_fn = _find_nested_target_log_prob_recursively(inner_kernel)
    new_target_log_prob = make_transformed_log_prob(
        target_log_prob_fn,
        bijector,
        direction='forward',
        # TODO(b/72831017): Disable caching until gradient linkage
        # generally works.
        enable_bijector_caching=False)
    self._inner_kernel = _update_target_log_prob(inner_kernel,
                                                 new_target_log_prob)
    # Prebuild `_forward_transform` which is used by `one_step`.
    self._transform_unconstrained_to_target_support = make_transform_fn(
        bijector, direction='forward')
    # Prebuild `_inverse_transform` which is used by `bootstrap_kernel_results`.
    self._transform_target_support_to_unconstrained = make_transform_fn(
        bijector, direction='inverse')

  @property
  def inner_kernel(self):
    return self._parameters['inner_kernel']

  @property
  def bijector(self):
    return self._parameters['bijector']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return self._inner_kernel.is_calibrated

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Runs one iteration of the Transformed Kernel.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s
        representing the current state(s) of the Markov chain(s),
        _after_ application of `bijector.forward`. The first `r`
        dimensions index independent chains,
        `r = tf.rank(target_log_prob_fn(*current_state))`. The
        `inner_kernel.one_step` does not actually use `current_state`,
        rather it takes as input
        `previous_kernel_results.transformed_state` (because
        `TransformedTransitionKernel` creates a copy of the input
        inner_kernel with a modified `target_log_prob_fn` which
        internally applies the `bijector.forward`).
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.
    """
    with tf.name_scope(mcmc_util.make_name(
        self.name, 'transformed_kernel', 'one_step')):
      inner_kwargs = {} if seed is None else dict(seed=seed)
      transformed_prev_state = previous_kernel_results.transformed_state
      transformed_next_state, kernel_results = self._inner_kernel.one_step(
          transformed_prev_state,
          previous_kernel_results.inner_results,
          **inner_kwargs)
      transformed_next_state_parts = (
          transformed_next_state
          if mcmc_util.is_list_like(transformed_next_state) else
          [transformed_next_state])
      next_state_parts = self._transform_unconstrained_to_target_support(
          transformed_next_state_parts)
      next_state = (
          next_state_parts if mcmc_util.is_list_like(transformed_next_state)
          else next_state_parts[0])
      if mcmc_util.is_list_like(transformed_prev_state):
        transformed_next_state = tf.nest.pack_sequence_as(
            transformed_prev_state, transformed_next_state)
      kernel_results = TransformedTransitionKernelResults(
          transformed_state=transformed_next_state,
          inner_results=kernel_results)
      return next_state, kernel_results

  def bootstrap_results(self, init_state=None, transformed_init_state=None):
    """Returns an object with the same type as returned by `one_step`.

    Unlike other `TransitionKernel`s,
    `TransformedTransitionKernel.bootstrap_results` has the option of
    initializing the `TransformedTransitionKernelResults` from either an initial
    state, eg, requiring computing `bijector.inverse(init_state)`, or
    directly from `transformed_init_state`, i.e., a `Tensor` or list
    of `Tensor`s which is interpretted as the `bijector.inverse`
    transformed state.

    Args:
      init_state: `Tensor` or Python `list` of `Tensor`s representing the a
        state(s) of the Markov chain(s). Must specify `init_state` or
        `transformed_init_state` but not both.
      transformed_init_state: `Tensor` or Python `list` of `Tensor`s
        representing the a state(s) of the Markov chain(s). Must specify
        `init_state` or `transformed_init_state` but not both.

    Returns:
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.

    Raises:
      ValueError: if none of the nested `inner_kernel` results contain
        the member "target_log_prob".

    #### Examples

    To use `transformed_init_state` in context of
    `tfp.mcmc.sample_chain`, you need to explicitly pass the
    `previous_kernel_results`, e.g.,

    ```python
    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(...)
    init_state = ...        # Doesnt matter.
    transformed_init_state = ... # Does matter.
    results = tfp.mcmc.sample_chain(
        num_results=...,
        current_state=init_state,
        previous_kernel_results=transformed_kernel.bootstrap_results(
            transformed_init_state=transformed_init_state),
        trace_fn=None,
        kernel=transformed_kernel)
    ```
    """
    if (init_state is None) == (transformed_init_state is None):
      raise ValueError('Must specify exactly one of `init_state` '
                       'or `transformed_init_state`.')
    with tf.name_scope(mcmc_util.make_name(
        self.name, 'transformed_kernel', 'bootstrap_results')):
      if transformed_init_state is None:
        init_state_parts = (init_state if mcmc_util.is_list_like(init_state)
                            else [init_state])
        transformed_init_state_parts = (
            self._transform_target_support_to_unconstrained(init_state_parts))
        transformed_init_state = (
            tf.nest.pack_sequence_as(init_state, transformed_init_state_parts)
            if mcmc_util.is_list_like(init_state)
            else transformed_init_state_parts[0])
      else:
        if mcmc_util.is_list_like(transformed_init_state):
          transformed_init_state = tf.nest.pack_sequence_as(
              transformed_init_state,
              [tf.convert_to_tensor(s, name='transformed_init_state')
               for s in transformed_init_state])
        else:
          transformed_init_state = tf.convert_to_tensor(
              value=transformed_init_state, name='transformed_init_state')
      kernel_results = TransformedTransitionKernelResults(
          transformed_state=transformed_init_state,
          inner_results=self._inner_kernel.bootstrap_results(
              transformed_init_state))
      return kernel_results

  @property
  def experimental_shard_axis_names(self):
    return self.inner_kernel.experimental_shard_axis_names

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(
        inner_kernel=self.inner_kernel.experimental_with_shard_axes(
            shard_axis_names))
