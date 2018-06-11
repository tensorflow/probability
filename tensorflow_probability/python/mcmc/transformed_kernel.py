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

import tensorflow as tf

# Because we use TF's `remove_undocumented`, we must directly import what we
# use.
from tensorflow_probability.python.mcmc.kernel import TransitionKernel
from tensorflow_probability.python.mcmc.util import is_list_like
from tensorflow_probability.python.mcmc.util import make_name


__all__ = [
    'TransformedTransitionKernel',
]

TransformedTransitionKernelResults = collections.namedtuple(
    'TransformedTransitionKernelResults', [
        'transformed_state',
        'inner_results',
    ])


def forward_log_det_jacobian_fn(bijector):
  """Makes a function which applies a list of Bijectors' `log_det_jacobian`s."""
  if not is_list_like(bijector):
    bijector = [bijector]

  def fn(transformed_state_parts, event_ndims):
    return sum([
        b.forward_log_det_jacobian(sp, event_ndims=e)
        for b, e, sp in zip(bijector, event_ndims, transformed_state_parts)
    ])

  return fn


def forward_transform_fn(bijector):
  """Makes a function which applies a list of Bijectors' `forward`s."""
  if not is_list_like(bijector):
    bijector = [bijector]

  def fn(transformed_state_parts):
    return [b.forward(sp) for b, sp in zip(bijector, transformed_state_parts)]

  return fn


def inverse_transform_fn(bijector):
  """Makes a function which applies a list of Bijectors' `inverse`s."""
  if not is_list_like(bijector):
    bijector = [bijector]
  def fn(state_parts):
    return [b.inverse(sp)
            for b, sp in zip(bijector, state_parts)]
  return fn


class TransformedTransitionKernel(TransitionKernel):
  """TransformedTransitionKernel applies a bijector to the MCMC's state space.

  The `TransformedTransitionKernel` `TransitionKernel` enables fitting
  a [Bijector](
  https://www.tensorflow.org/api_docs/python/tf/distributions/bijectors/Bijector)
  which serves to decorrelate the Markov chain Monte Carlo (MCMC)
  event dimensions thus making the chain mix faster. This is
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
  `tfp.distributions.bijectors.Affine`,
  `tfp.distributions.bijectors.RealNVP`, etc. with transition kernels
  `tfp.mcmc.HamiltonianMonteCarlo`, `tfp.mcmc.RandomWalkMetropolis`,
  etc.

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
  tfb = tfd.bijectors

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
      inner_kernel: `TransitionKernel`-like object which has a
        `target_log_prob_fn` argument.
      bijector: `tfp.distributions.Bijector` or list of
        `tfp.distributions.Bijector`s. These bijectors use `forward` to map the
        `inner_kernel` state space to the state expected by
        `inner_kernel.target_log_prob_fn`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "hmc_kernel").

    Returns:
      transformed_kernel: Instance of `TransitionKernel` which copies the input
        transition kernel then modifies its `target_log_prob_fn` by applying the
        provided bijector(s).
    """
    self._parameters = dict(
        inner_kernel=inner_kernel,
        bijector=bijector,
        name=name)
    inner_kernel_kwargs = inner_kernel.parameters
    target_log_prob_fn = inner_kernel_kwargs['target_log_prob_fn']
    self._forward_transform = forward_transform_fn(bijector)
    self._inverse_transform = inverse_transform_fn(bijector)
    self._forward_log_det_jacobian = forward_log_det_jacobian_fn(bijector)

    def new_target_log_prob(*transformed_state_parts):
      """Log prob of the transformed state."""
      # TODO(b/72831017): Use `tf.identity` to disable caching (since HMC takes
      # gradient with respect to input).
      transformed_state_parts = [
          tf.identity(sp) for sp in transformed_state_parts
      ]
      tlp = target_log_prob_fn(
          *self._forward_transform(transformed_state_parts))
      event_ndims = [
          tf.rank(sp) - tf.rank(tlp) for sp in transformed_state_parts
      ]
      return tlp + self._forward_log_det_jacobian(
          transformed_state_parts=transformed_state_parts,
          event_ndims=event_ndims)

    inner_kernel_kwargs.update(target_log_prob_fn=new_target_log_prob)
    self._inner_kernel = type(inner_kernel)(**inner_kernel_kwargs)

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

  def one_step(self, current_state, previous_kernel_results):
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

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.
    """
    with tf.name_scope(
        name=make_name(self.name, 'transformed_kernel', 'one_step'),
        values=[previous_kernel_results]):
      transformed_next_state, kernel_results = self._inner_kernel.one_step(
          previous_kernel_results.transformed_state,
          previous_kernel_results.inner_results)
      transformed_next_state_parts = (
          transformed_next_state
          if is_list_like(transformed_next_state) else [transformed_next_state])
      next_state_parts = self._forward_transform(transformed_next_state_parts)
      next_state = (
          next_state_parts
          if is_list_like(transformed_next_state) else next_state_parts[0])
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
      ValueError: if `inner_kernel` results doesn't contain the member
        "target_log_prob".

    #### Examples

    To use `transformed_init_state` in context of
    `tfp.mcmc.sample_chain`, you need to explicitly pass the
    `previous_kernel_results`, e.g.,

    ```python
    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(...)
    init_state = ...        # Doesnt matter.
    transformed_init_state = ... # Does matter.
    results, _ = tfp.mcmc.sample_chain(
        num_results=...,
        current_state=init_state,
        previous_kernel_results=transformed_kernel.bootstrap_results(
            transformed_init_state=transformed_init_state),
        kernel=transformed_kernel)
    ```
    """
    if (init_state is None) == (transformed_init_state is None):
      raise ValueError('Must specify exactly one of `init_state` '
                       'or `transformed_init_state`.')
    with tf.name_scope(
        name=make_name(self.name, 'transformed_kernel', 'bootstrap_results'),
        values=[init_state, transformed_init_state]):
      if transformed_init_state is None:
        init_state_parts = (init_state if is_list_like(init_state)
                            else [init_state])
        transformed_init_state_parts = self._inverse_transform(init_state_parts)
        transformed_init_state = (
            transformed_init_state_parts
            if is_list_like(init_state) else transformed_init_state_parts[0])
      else:
        if is_list_like(transformed_init_state):
          transformed_init_state = [
              tf.convert_to_tensor(s, name='transformed_init_state')
              for s in transformed_init_state
          ]
        else:
          transformed_init_state = tf.convert_to_tensor(
              transformed_init_state, name='transformed_init_state')
      kernel_results = TransformedTransitionKernelResults(
          transformed_state=transformed_init_state,
          inner_results=self._inner_kernel.bootstrap_results(
              transformed_init_state))
      return kernel_results
