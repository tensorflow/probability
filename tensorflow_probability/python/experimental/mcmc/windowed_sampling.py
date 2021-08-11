# Copyright 2021 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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
"""Windowed adaptation for Markov chain Monte Carlo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import warnings

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import restructure
from tensorflow_probability.python.experimental.mcmc import diagonal_mass_matrix_adaptation as dmma
from tensorflow_probability.python.experimental.mcmc import initialization
from tensorflow_probability.python.experimental.mcmc import preconditioned_hmc as phmc
from tensorflow_probability.python.experimental.mcmc import preconditioned_nuts as pnuts
from tensorflow_probability.python.experimental.mcmc import preconditioning_utils
from tensorflow_probability.python.experimental.stats import sample_stats
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.math import generic as generic_math
from tensorflow_probability.python.mcmc import dual_averaging_step_size_adaptation as dassa
from tensorflow_probability.python.mcmc import sample
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'windowed_adaptive_hmc',
    'windowed_adaptive_nuts'
]

# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.filterwarnings(
    'always', module='tensorflow_probability.*windowed_sampling',
    append=True)  # Don't override user-set filters.


def _default_nuts_trace_fn(state, bijector, is_adapting, pkr):
  """Trace function for `windowed_adaptive_nuts` providing standard diagnostics.

  Specifically, these match up with a number of diagnostics used by ArviZ [1],
  to make diagnostics and analysis easier. The names used follow those used in
  TensorFlow Probability, and will need to be mapped to those used in the ArviZ
  schema.

  References:
    [1]: Kumar, R., Carroll, C., Hartikainen, A., & Martin, O. (2019). ArviZ a
    unified library for exploratory analysis of Bayesian models in Python.
    Journal of Open Source Software, 4(33), 1143.

  Args:
   state: tf.Tensor
     Current sampler state, flattened and unconstrained.
   bijector: tfb.Bijector
     This can be used to untransform the shape to something with the same shape
     as will be returned.
   is_adapting: bool
     Whether this is an adapting step, or may be treated as a valid MCMC draw.
   pkr: UncalibratedPreconditionedHamiltonianMonteCarloKernelResults
     Kernel results from this iteration.

  Returns:
    dict with sampler statistics.
  """
  del state, bijector  # Unused

  energy_diff = unnest.get_innermost(pkr, 'log_accept_ratio')
  return {
      'step_size': unnest.get_innermost(pkr, 'step_size'),
      'tune': is_adapting,
      'target_log_prob': unnest.get_innermost(pkr, 'target_log_prob'),
      'diverging': unnest.get_innermost(pkr, 'has_divergence'),
      'accept_ratio': tf.minimum(1., tf.exp(energy_diff)),
      'variance_scaling':
          unnest.get_innermost(pkr, 'momentum_distribution').variance(),
      'n_steps': unnest.get_innermost(pkr, 'leapfrogs_taken'),
      'is_accepted': unnest.get_innermost(pkr, 'is_accepted')}


def _default_hmc_trace_fn(state, bijector, is_adapting, pkr):
  """Trace function for `windowed_adaptive_hmc` providing standard diagnostics.

  Specifically, these match up with a number of diagnostics used by ArviZ [1],
  to make diagnostics and analysis easier. The names used follow those used in
  TensorFlow Probability, and will need to be mapped to those used in the ArviZ
  schema.

  References:
    [1]: Kumar, R., Carroll, C., Hartikainen, A., & Martin, O. (2019). ArviZ a
    unified library for exploratory analysis of Bayesian models in Python.
    Journal of Open Source Software, 4(33), 1143.

  Args:
   state: tf.Tensor
     Current sampler state, flattened and unconstrained.
   bijector: tfb.Bijector
     This can be used to untransform the shape to something with the same shape
     as will be returned.
   is_adapting: bool
     Whether this is an adapting step, or may be treated as a valid MCMC draw.
   pkr: UncalibratedPreconditionedHamiltonianMonteCarloKernelResults
     Kernel results from this iteration.

  Returns:
    dict with sampler statistics.
  """
  del state, bijector  # Unused

  energy_diff = unnest.get_innermost(pkr, 'log_accept_ratio')
  has_divergence = tf.math.abs(energy_diff) > 500.
  return {
      'step_size': unnest.get_innermost(pkr, 'step_size'),
      'tune': is_adapting,
      'target_log_prob': unnest.get_innermost(pkr, 'target_log_prob'),
      'diverging': has_divergence,
      'log_acceptance_correction':
          unnest.get_innermost(pkr, 'log_acceptance_correction'),
      'accept_ratio': tf.minimum(1., tf.exp(energy_diff)),
      'variance_scaling':
          unnest.get_innermost(pkr, 'momentum_distribution').variance(),
      'is_accepted': unnest.get_innermost(pkr, 'is_accepted')}


def _get_flat_unconstraining_bijector(jd_model):
  """Create a bijector from a joint distribution that flattens and unconstrains.

  The intention is (loosely) to go from a model joint distribution supported on

  U_1 x U_2 x ... U_n, with U_j a subset of R^{n_j}

  to a model supported on R^N, with N = sum(n_j). (This is "loose" in the sense
  of base measures: some distribution may be supported on an m-dimensional
  subset of R^n, and the default transform for that distribution may then
  have support on R^m. See [1] for details.

  Args:
    jd_model: subclass of `tfd.JointDistribution` A JointDistribution for a
      model.

  Returns:
    Two `tfb.Bijector`s where the `.forward` method flattens and unconstrains
    points, and the second may be used to initialize a step size.
  """
  # TODO(b/180396233): This bijector is in general point-dependent.
  event_space_bij = jd_model.experimental_default_event_space_bijector()
  flat_bijector = restructure.pack_sequence_as(jd_model.event_shape_tensor())

  unconstrained_shapes = event_space_bij(
      flat_bijector).inverse_event_shape_tensor(jd_model.event_shape_tensor())

  # this reshaping is required as as split can produce a tensor of shape [1]
  # when the distribution event shape is []
  unsplit = joint_map.JointMap(bijectors=[
      reshape.Reshape(event_shape_out=x, event_shape_in=[-1])
      for x in unconstrained_shapes])

  bij = invert.Invert(chain.Chain([event_space_bij, flat_bijector, unsplit]))
  step_size_bij = invert.Invert(flat_bijector)

  return bij, step_size_bij


def _setup_mcmc(model, n_chains, *, init_position=None, seed=None, **pins):
  """Construct bijector and transforms needed for windowed MCMC.

  This pins the initial model, constructs a bijector that unconstrains and
  flattens each dimension and adds a leading batch shape of `n_chains`,
  initializes a point in the unconstrained space, and constructs a transformed
  log probability using the bijector.

  Note that we must manually construct this target log probability instead of
  using a transformed transition kernel because the TTK assumes the shape
  in is the same as the shape out.

  Args:
    model: `tfd.JointDistribution`
      The model to sample from.
    n_chains: int
      Number of chains (independent examples) to run.
    init_position: Optional
      Structure of tensors at which to initialize sampling. Should have the
      same shape and structure as
      `model.experimental_pin(**pins).sample_unpinned(n_chains)`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    **pins:
      Values passed to `model.experimental_pin`.


  Returns:
    target_log_prob_fn: Callable on the transformed space.
    initial_transformed_position: `tf.Tensor`, sampled from a uniform (-2, 2).
    bijector: `tfb.Bijector` instance, which unconstrains and flattens.
    step_broadcast_fn: Callable to broadcast step size over latent structure.
    batch_shape: Batch shape of the model.
  """
  pinned_model = model.experimental_pin(**pins) if pins else model
  bijector, step_bijector = _get_flat_unconstraining_bijector(pinned_model)

  if init_position is None:
    raw_init_dist = initialization.init_near_unconstrained_zero(pinned_model)
    init_position = initialization.retry_init(
        raw_init_dist.sample,
        target_fn=pinned_model.unnormalized_log_prob,
        sample_shape=[n_chains],
        seed=seed)

  initial_transformed_position = tf.nest.map_structure(
      tf.identity, bijector.forward(init_position))

  batch_shape = pinned_model.batch_shape
  if tf.nest.is_nested(batch_shape):
    batch_shape = functools.reduce(tf.broadcast_static_shape,
                                   tf.nest.flatten(batch_shape))

  lp_static_shape = tensorshape_util.concatenate([n_chains], batch_shape)

  if not tensorshape_util.is_fully_defined(batch_shape):
    batch_shape = pinned_model.batch_shape_tensor()
    if tf.nest.is_nested(batch_shape):
      batch_shape = functools.reduce(tf.broadcast_dynamic_shape,
                                     tf.nest.flatten(batch_shape))

  # This tf.function is not redundant with the ones on _fast_window
  # and _slow_window because the various kernels (like HMC) may invoke
  # `target_log_prob_fn` multiple times within one window.
  @tf.function(autograph=False)
  def target_log_prob_fn(*args):
    lp = pinned_model.unnormalized_log_prob(bijector.inverse(args))
    tensorshape_util.set_shape(lp, lp_static_shape)
    ldj = bijector.inverse_log_det_jacobian(
        args, event_ndims=[1 for _ in initial_transformed_position])
    return lp + ldj

  def step_broadcast(step_size):
    # Only apply the bijector to nested step sizes or non-scalar batches.
    if tf.nest.is_nested(step_size):
      return step_bijector(
          nest_util.broadcast_structure(pinned_model.event_shape_tensor(),
                                        step_size))
    else:
      return step_size

  return (target_log_prob_fn,
          initial_transformed_position,
          bijector,
          step_broadcast,
          ps.convert_to_shape_tensor(batch_shape, name='batch_shape'))


def _make_base_kernel(*, kind, proposal_kernel_kwargs):
  """Construct internal sampling kernel."""
  if kind == 'nuts':
    return pnuts.PreconditionedNoUTurnSampler(**proposal_kernel_kwargs)
  elif kind == 'hmc':
    return phmc.PreconditionedHamiltonianMonteCarlo(**proposal_kernel_kwargs)
  else:
    raise TypeError(
        '`kind` must be "nuts" or "hmc" (got {kind})'.format(kind=kind))


def make_fast_adapt_kernel(*,
                           kind,
                           proposal_kernel_kwargs,
                           dual_averaging_kwargs):
  return dassa.DualAveragingStepSizeAdaptation(
      _make_base_kernel(
          kind=kind, proposal_kernel_kwargs=proposal_kernel_kwargs),
      **dual_averaging_kwargs)


def make_slow_adapt_kernel(*,
                           kind,
                           proposal_kernel_kwargs,
                           dual_averaging_kwargs,
                           initial_running_variance):
  return dmma.DiagonalMassMatrixAdaptation(
      make_fast_adapt_kernel(
          kind=kind,
          proposal_kernel_kwargs=proposal_kernel_kwargs,
          dual_averaging_kwargs=dual_averaging_kwargs),
      initial_running_variance=initial_running_variance)


def _fast_window(*,
                 kind,
                 proposal_kernel_kwargs,
                 dual_averaging_kwargs,
                 num_draws,
                 initial_position,
                 bijector,
                 trace_fn,
                 seed):
  """Sample using just step size adaptation."""
  dual_averaging_kwargs = dict(dual_averaging_kwargs)
  dual_averaging_kwargs.update({'num_adaptation_steps': num_draws})
  kernel = make_fast_adapt_kernel(
      kind=kind,
      proposal_kernel_kwargs=proposal_kernel_kwargs,
      dual_averaging_kwargs=dual_averaging_kwargs)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    draws, trace, fkr = sample.sample_chain(
        num_draws,
        initial_position,
        kernel=kernel,
        return_final_kernel_results=True,
        # pylint: disable=g-long-lambda
        trace_fn=lambda state, pkr: trace_fn(
            state, bijector, tf.constant(True), pkr.inner_results),
        seed=seed)
    # pylint: enable=g-long-lambda

  draw_and_chain_axes = [0, 1]
  weighted_running_variance = []
  for d in list(draws):
    prev_mean, prev_var = tf.nn.moments(d[-num_draws // 2:],
                                        axes=draw_and_chain_axes)
    num_samples = tf.cast(
        num_draws / 2,
        dtype=dtype_util.common_dtype([prev_mean, prev_var], tf.float32))
    weighted_running_variance.append(sample_stats.RunningVariance.from_stats(
        num_samples=num_samples,
        mean=prev_mean,
        variance=prev_var))

  step_size = unnest.get_outermost(fkr, 'step_size')
  return draws, trace, step_size, weighted_running_variance


def _slow_window(*,
                 kind,
                 proposal_kernel_kwargs,
                 dual_averaging_kwargs,
                 num_draws,
                 initial_position,
                 initial_running_variance,
                 bijector,
                 trace_fn,
                 seed):
  """Sample using both step size and mass matrix adaptation."""
  dual_averaging_kwargs = dict(dual_averaging_kwargs)
  dual_averaging_kwargs.setdefault('num_adaptation_steps', num_draws)
  kernel = make_slow_adapt_kernel(
      kind=kind,
      proposal_kernel_kwargs=proposal_kernel_kwargs,
      dual_averaging_kwargs=dual_averaging_kwargs,
      initial_running_variance=initial_running_variance)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    draws, trace, fkr = sample.sample_chain(
        num_draws,
        initial_position,
        kernel=kernel,
        return_final_kernel_results=True,
        # pylint: disable=g-long-lambda
        trace_fn=lambda state, pkr: trace_fn(state,
                                             bijector,
                                             tf.constant(True),
                                             pkr.inner_results.inner_results),
        seed=seed)
    # pylint: enable=g-long-lambda

  draw_and_chain_axes = [0, 1]
  weighted_running_variance = []
  for d in list(draws):
    prev_mean, prev_var = tf.nn.moments(d[-num_draws // 2:],
                                        axes=draw_and_chain_axes)
    num_samples = tf.cast(
        num_draws / 2,
        dtype=dtype_util.common_dtype([prev_mean, prev_var], tf.float32))
    weighted_running_variance.append(sample_stats.RunningVariance.from_stats(
        num_samples=num_samples,
        mean=prev_mean,
        variance=prev_var))

  step_size = unnest.get_outermost(fkr, 'step_size')
  momentum_distribution = unnest.get_outermost(fkr, 'momentum_distribution')

  return draws, trace, step_size, weighted_running_variance, momentum_distribution


def _do_sampling(*,
                 kind,
                 proposal_kernel_kwargs,
                 num_draws,
                 initial_position,
                 trace_fn,
                 bijector,
                 return_final_kernel_results,
                 seed):
  """Sample from base HMC kernel."""
  kernel = _make_base_kernel(
      kind=kind,
      proposal_kernel_kwargs=proposal_kernel_kwargs)
  return sample.sample_chain(
      num_draws,
      initial_position,
      kernel=kernel,
      # pylint: disable=g-long-lambda
      trace_fn=lambda state, pkr: trace_fn(state,
                                           bijector,
                                           tf.constant(False),
                                           pkr),
      # pylint: enable=g-long-lambda
      return_final_kernel_results=return_final_kernel_results,
      seed=seed)


def _get_window_sizes(num_adaptation_steps):
  """Hardcoded way to get a reasonable scheme.

  This assumes we do something proportional to

  fast window: 75 steps
  slow window: 25 steps
  slow window: 50 steps
  slow window: 100 steps
  slow window: 200 steps
  fast window: 75 steps

  Which is a total of 525 steps.

  Args:
    num_adaptation_steps: int
      Number of adaptation steps to run

  Returns:
    The first window size, the initial slow window size, the last window size
  """
  slow_window_size = num_adaptation_steps // 21
  first_window_size = 3 * slow_window_size
  last_window_size = (num_adaptation_steps -
                      15 * slow_window_size -
                      first_window_size)
  return first_window_size, slow_window_size, last_window_size


def _get_step_size(initial_transformed_position, log_prob_fn):
  """Heuristic for initializing step size.

  If we (over) optimistically assume good scaling, 1 / sum(event_dims)**0.25
  will be near the optimal step size. We further scale that downwards. See
  Langmore, Ian, Michael Dikovsky, Scott Geraedts, Peter Norgaard, and Rob Von
  Behren. 2019. “A Condition Number for Hamiltonian Monte Carlo.” arXiv
  [stat.CO]. arXiv. http://arxiv.org/abs/1905.09813.

  Args:
    initial_transformed_position: Iterable of arguments to log_prob_fn, in order
      to get a dtype and find a heuristic for an initial step size. We assume
      the Tensor has been flattened so that number of event dimensions is the
      last one.
    log_prob_fn: Target log probability function.

  Returns:
    Scalar float of the same dtype as log_prob_fn.
  """
  # TODO(b/187658871): Update this code after internal kernels can support it.
  dtype = log_prob_fn(*initial_transformed_position).dtype
  return 0.5 * sum([
      tf.cast(ps.shape(state_part)[-1], dtype)
      for state_part in initial_transformed_position])**-0.25


def _init_momentum(initial_transformed_position, *, batch_shape):
  """Initialize momentum so trace_fn can be concatenated."""
  variance_parts = [ps.ones_like(p) for p in initial_transformed_position]
  return preconditioning_utils.make_momentum_distribution(
      state_parts=initial_transformed_position,
      batch_shape=batch_shape,
      running_variance_parts=variance_parts)


def windowed_adaptive_nuts(n_draws,
                           joint_dist,
                           *,
                           n_chains=64,
                           num_adaptation_steps=525,
                           current_state=None,
                           init_step_size=None,
                           dual_averaging_kwargs=None,
                           max_tree_depth=10,
                           max_energy_diff=500.,
                           unrolled_leapfrog_steps=1,
                           parallel_iterations=10,
                           trace_fn=_default_nuts_trace_fn,
                           return_final_kernel_results=False,
                           discard_tuning=True,
                           seed=None,
                           **pins):
  """Adapt and sample from a joint distribution using NUTS, conditioned on pins.

  Step size is tuned using a dual-averaging adaptation, and the kernel is
  conditioned using a diagonal mass matrix, which is estimated using expanding
  windows.

  Args:
    n_draws: int
      Number of draws after adaptation.
    joint_dist: `tfd.JointDistribution`
      A joint distribution to sample from.
    n_chains: int
      Number of independent chains to run MCMC with.
    num_adaptation_steps: int
      Number of draws used to adapt step size and
    current_state: Optional
      Structure of tensors at which to initialize sampling. Should have the
      same shape and structure as
      `model.experimental_pin(**pins).sample(n_chains)`.
    init_step_size: Optional
      Where to initialize the step size for the leapfrog integrator. The
      structure should broadcast with `current_state`. For example, if the
      initial state is
      ```
      {'a': tf.zeros(n_chains),
       'b': tf.zeros([n_chains, n_features])}
       ```
      then any of `1.`, `{'a': 1., 'b': 1.}`, or
      `{'a': tf.ones(n_chains), 'b': tf.ones([n_chains, n_features])}` will
      work. Defaults to the dimension of the log density to the 0.25 power.
    dual_averaging_kwargs: Optional dict
      Keyword arguments to pass to `tfp.mcmc.DualAveragingStepSizeAdaptation`.
      By default, a `target_accept_prob` of 0.85 is set, acceptance
      probabilities across chains are reduced using a harmonic mean, and the
      class defaults are used otherwise.
    max_tree_depth: Maximum depth of the tree implicitly built by NUTS. The
      maximum number of leapfrog steps is bounded by `2**max_tree_depth` i.e.
      the number of nodes in a binary tree `max_tree_depth` nodes deep. The
      default setting of 10 takes up to 1024 leapfrog steps.
    max_energy_diff: Scalar threshold of energy differences at each leapfrog,
      divergence samples are defined as leapfrog steps that exceed this
      threshold. Default to 1000.
    unrolled_leapfrog_steps: The number of leapfrogs to unroll per tree
      expansion step. Applies a direct linear multipler to the maximum
      trajectory length implied by max_tree_depth. Defaults to 1.
    parallel_iterations: The number of iterations allowed to run in parallel.
      It must be a positive integer. See `tf.while_loop` for more details.
    trace_fn: Optional callable
      The trace function should accept the arguments
      `(state, bijector, is_adapting, phmc_kernel_results)`,  where the `state`
      is an unconstrained, flattened float tensor, `bijector` is the
      `tfb.Bijector` that is used for unconstraining and flattening,
      `is_adapting` is a boolean to mark whether the draw is from an adaptation
      step, and `phmc_kernel_results` is the
      `UncalibratedPreconditionedHamiltonianMonteCarloKernelResults` from the
      `PreconditionedHamiltonianMonteCarlo` kernel. Note that
      `bijector.inverse(state)` will provide access to the current draw in the
      untransformed space, using the structure of the provided `joint_dist`.
    return_final_kernel_results: If `True`, then the final kernel results are
      returned alongside the chain state and the trace specified by the
      `trace_fn`.
    discard_tuning: bool
      Whether to return tuning traces and draws.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    **pins:
      These are used to condition the provided joint distribution, and are
      passed directly to `joint_dist.experimental_pin(**pins)`.
  Returns:
    A single structure of draws is returned in case the trace_fn is `None`, and
    `return_final_kernel_results` is `False`. If there is a trace function,
    the return value is a tuple, with the trace second. If the
    `return_final_kernel_results` is `True`, the return value is a tuple of
    length 3, with final kernel results returned last. If `discard_tuning` is
    `True`, the tensors in `draws` and `trace` will have length `n_draws`,
    otherwise, they will have length `n_draws + num_adaptation_steps`.
  """
  if dual_averaging_kwargs is None:
    dual_averaging_kwargs = {}
  dual_averaging_kwargs.setdefault('target_accept_prob', 0.85)
  proposal_kernel_kwargs = {
      'step_size': init_step_size,
      'max_tree_depth': max_tree_depth,
      'max_energy_diff': max_energy_diff,
      'unrolled_leapfrog_steps': unrolled_leapfrog_steps,
      'parallel_iterations': parallel_iterations}
  return _windowed_adaptive_impl(
      n_draws=n_draws,
      joint_dist=joint_dist,
      kind='nuts',
      n_chains=n_chains,
      proposal_kernel_kwargs=proposal_kernel_kwargs,
      current_state=current_state,
      num_adaptation_steps=num_adaptation_steps,
      dual_averaging_kwargs=dual_averaging_kwargs,
      trace_fn=trace_fn,
      return_final_kernel_results=return_final_kernel_results,
      discard_tuning=discard_tuning,
      seed=seed,
      **pins)


def windowed_adaptive_hmc(n_draws,
                          joint_dist,
                          *,
                          num_leapfrog_steps,
                          n_chains=64,
                          num_adaptation_steps=525,
                          current_state=None,
                          init_step_size=None,
                          dual_averaging_kwargs=None,
                          trace_fn=_default_hmc_trace_fn,
                          return_final_kernel_results=False,
                          discard_tuning=True,
                          seed=None,
                          **pins):
  """Adapt and sample from a joint distribution, conditioned on pins.

  This uses Hamiltonian Monte Carlo to do the sampling. Step size is tuned using
  a dual-averaging adaptation, and the kernel is conditioned using a diagonal
  mass matrix, which is estimated using expanding windows.

  Args:
    n_draws: int
      Number of draws after adaptation.
    joint_dist: `tfd.JointDistribution`
      A joint distribution to sample from.
    num_leapfrog_steps: int
      Number of leapfrog steps to use for the Hamiltonian Monte Carlo step.
    n_chains: int
      Number of independent chains to run MCMC with.
    num_adaptation_steps: int
      Number of draws used to adapt step size and
    current_state: Optional
      Structure of tensors at which to initialize sampling. Should have the
      same shape and structure as
      `model.experimental_pin(**pins).sample(n_chains)`.
    init_step_size: Optional
      Where to initialize the step size for the leapfrog integrator. The
      structure should broadcast with `current_state`. For example, if the
      initial state is
      ```
      {'a': tf.zeros(n_chains),
       'b': tf.zeros([n_chains, n_features])}
       ```
      then any of `1.`, `{'a': 1., 'b': 1.}`, or
      `{'a': tf.ones(n_chains), 'b': tf.ones([n_chains, n_features])}` will
      work. Defaults to the dimension of the log density to the 0.25 power.
    dual_averaging_kwargs: Optional dict
      Keyword arguments to pass to `tfp.mcmc.DualAveragingStepSizeAdaptation`.
      By default, a `target_accept_prob` of 0.75 is set, acceptance
      probabilities across chains are reduced using a harmonic mean, and the
      class defaults are used otherwise.
    trace_fn: Optional callable
      The trace function should accept the arguments
      `(state, bijector, is_adapting, phmc_kernel_results)`,  where the `state`
      is an unconstrained, flattened float tensor, `bijector` is the
      `tfb.Bijector` that is used for unconstraining and flattening,
      `is_adapting` is a boolean to mark whether the draw is from an adaptation
      step, and `phmc_kernel_results` is the
      `UncalibratedPreconditionedHamiltonianMonteCarloKernelResults` from the
      `PreconditionedHamiltonianMonteCarlo` kernel. Note that
      `bijector.inverse(state)` will provide access to the current draw in the
      untransformed space, using the structure of the provided `joint_dist`.
    return_final_kernel_results: If `True`, then the final kernel results are
      returned alongside the chain state and the trace specified by the
      `trace_fn`.
    discard_tuning: bool
      Whether to return tuning traces and draws.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    **pins:
      These are used to condition the provided joint distribution, and are
      passed directly to `joint_dist.experimental_pin(**pins)`.
  Returns:
    A single structure of draws is returned in case the trace_fn is `None`, and
    `return_final_kernel_results` is `False`. If there is a trace function,
    the return value is a tuple, with the trace second. If the
    `return_final_kernel_results` is `True`, the return value is a tuple of
    length 3, with final kernel results returned last. If `discard_tuning` is
    `True`, the tensors in `draws` and `trace` will have length `n_draws`,
    otherwise, they will have length `n_draws + num_adaptation_steps`.
  """
  if dual_averaging_kwargs is None:
    dual_averaging_kwargs = {}
  dual_averaging_kwargs.setdefault('target_accept_prob', 0.75)
  proposal_kernel_kwargs = {
      'num_leapfrog_steps': num_leapfrog_steps,
      'step_size': init_step_size,
      'store_parameters_in_results': True}
  return _windowed_adaptive_impl(
      n_draws=n_draws,
      joint_dist=joint_dist,
      kind='hmc',
      n_chains=n_chains,
      proposal_kernel_kwargs=proposal_kernel_kwargs,
      num_adaptation_steps=num_adaptation_steps,
      current_state=current_state,
      dual_averaging_kwargs=dual_averaging_kwargs,
      trace_fn=trace_fn,
      return_final_kernel_results=return_final_kernel_results,
      discard_tuning=discard_tuning,
      seed=seed,
      **pins)


def _windowed_adaptive_impl(n_draws,
                            joint_dist,
                            *,
                            kind,
                            n_chains,
                            proposal_kernel_kwargs,
                            num_adaptation_steps,
                            current_state,
                            dual_averaging_kwargs,
                            trace_fn,
                            return_final_kernel_results,
                            discard_tuning,
                            seed,
                            **pins):
  """Runs windowed sampling using either HMC or NUTS as internal sampler."""
  if trace_fn is None:
    trace_fn = lambda *args: ()
    no_trace = True
  else:
    no_trace = False

  if (tf.executing_eagerly() or
      not control_flow_util.GraphOrParentsInXlaContext(
          tf1.get_default_graph())):
    # A Tensor num_draws argument breaks XLA, which requires static TensorArray
    # trace_fn result allocation sizes.
    num_adaptation_steps = ps.convert_to_shape_tensor(num_adaptation_steps)

  dual_averaging_kwargs.setdefault('reduce_fn',
                                   generic_math.reduce_log_harmonic_mean_exp)
  setup_seed, init_seed, seed = samplers.split_seed(
      samplers.sanitize_seed(seed), n=3)
  (target_log_prob_fn, initial_transformed_position, bijector,
   step_broadcast, batch_shape) = _setup_mcmc(
       joint_dist,
       n_chains=n_chains,
       init_position=current_state,
       seed=setup_seed,
       **pins)

  if proposal_kernel_kwargs.get('step_size') is None:
    if batch_shape.shape != (0,):  # Scalar batch has a 0-vector shape.
      raise ValueError('Batch target density must specify init_step_size. Got '
                       f'batch shape {batch_shape} from joint {joint_dist}.')

    init_step_size = _get_step_size(initial_transformed_position,
                                    target_log_prob_fn)

  else:
    init_step_size = step_broadcast(proposal_kernel_kwargs['step_size'])

  proposal_kernel_kwargs.update({
      'target_log_prob_fn': target_log_prob_fn,
      'step_size': init_step_size,
      'momentum_distribution': _init_momentum(
          initial_transformed_position,
          batch_shape=ps.concat([[n_chains], batch_shape], axis=0))})

  first_window_size, slow_window_size, last_window_size = _get_window_sizes(
      num_adaptation_steps)

  all_traces = []
  # Using tf.function here and on _slow_window_closure caches tracing
  # of _fast_window and _slow_window, respectively, within a single
  # call to windowed sampling.  Why not annotate _fast_window and
  # _slow_window directly?  Two reasons:
  # - Caching across calls to windowed sampling is probably futile,
  #   because the trace function and bijector will be different Python
  #   objects, preventing cache hits.
  # - The cache of a global tf.function sticks around for the lifetime
  #   of the Python process, potentially leaking memory.
  @tf.function(autograph=False)
  def _fast_window_closure(proposal_kernel_kwargs,
                           window_size,
                           initial_position,
                           seed):
    return _fast_window(
        kind=kind,
        proposal_kernel_kwargs=proposal_kernel_kwargs,
        dual_averaging_kwargs=dual_averaging_kwargs,
        num_draws=window_size,
        initial_position=initial_position,
        bijector=bijector,
        trace_fn=trace_fn,
        seed=seed)
  draws, trace, step_size, running_variances = _fast_window_closure(
      proposal_kernel_kwargs=proposal_kernel_kwargs,
      window_size=first_window_size,
      initial_position=initial_transformed_position,
      seed=init_seed)
  proposal_kernel_kwargs.update({'step_size': step_size})

  all_draws = [[d] for d in draws]
  all_traces.append(trace)
  *slow_seeds, seed = samplers.split_seed(seed, n=5)
  @tf.function(autograph=False)
  def _slow_window_closure(proposal_kernel_kwargs,
                           window_size,
                           initial_position,
                           running_variances,
                           seed):
    return _slow_window(
        kind=kind,
        proposal_kernel_kwargs=proposal_kernel_kwargs,
        dual_averaging_kwargs=dual_averaging_kwargs,
        num_draws=window_size,
        initial_position=initial_position,
        initial_running_variance=running_variances,
        bijector=bijector,
        trace_fn=trace_fn,
        seed=seed)
  for idx, slow_seed in enumerate(slow_seeds):
    window_size = slow_window_size * (2**idx)

    # TODO(b/180011931): if num_adaptation_steps is small, this throws an error.
    (draws, trace, step_size, running_variances, momentum_distribution
     ) = _slow_window_closure(
         proposal_kernel_kwargs=proposal_kernel_kwargs,
         window_size=window_size,
         initial_position=[d[-1] for d in draws],
         running_variances=running_variances,
         seed=slow_seed)
    for all_d, d in zip(all_draws, draws):
      all_d.append(d)
    all_traces.append(trace)
    proposal_kernel_kwargs.update(
        {'step_size': step_size,
         'momentum_distribution': momentum_distribution})

  fast_seed, sample_seed = samplers.split_seed(seed)
  draws, trace, step_size, _ = _fast_window_closure(
      proposal_kernel_kwargs=proposal_kernel_kwargs,
      window_size=last_window_size,
      initial_position=[d[-1] for d in draws],
      seed=fast_seed)
  proposal_kernel_kwargs.update({'step_size': step_size})
  for all_d, d in zip(all_draws, draws):
    all_d.append(d)
  all_traces.append(trace)

  ret = _do_sampling(
      kind=kind,
      proposal_kernel_kwargs=proposal_kernel_kwargs,
      num_draws=n_draws,
      initial_position=[d[-1] for d in draws],
      bijector=bijector,
      trace_fn=trace_fn,
      return_final_kernel_results=return_final_kernel_results,
      seed=sample_seed)

  if discard_tuning:
    if return_final_kernel_results:
      draws, trace, fkr = ret
      return sample.CheckpointableStatesAndTrace(
          all_states=bijector.inverse(draws),
          trace=trace,
          final_kernel_results=fkr)
    else:
      draws, trace = ret
      if no_trace:
        return bijector.inverse(draws)
      else:
        return sample.StatesAndTrace(all_states=bijector.inverse(draws),
                                     trace=trace)
  else:
    if return_final_kernel_results:
      draws, trace, fkr = ret
      for all_d, d in zip(all_draws, draws):
        all_d.append(d)
      all_traces.append(trace)
      return sample.CheckpointableStatesAndTrace(
          all_states=bijector.inverse(
              [tf.concat(d, axis=0) for d in all_draws]),
          trace=tf.nest.map_structure(lambda *s: tf.concat(s, axis=0),
                                      *all_traces, expand_composites=True),
          final_kernel_results=fkr)
    else:
      draws, trace = ret
      for all_d, d in zip(all_draws, draws):
        all_d.append(d)
      all_states = bijector.inverse([tf.concat(d, axis=0) for d in all_draws])
      if no_trace:
        return all_states
      else:
        all_traces.append(trace)
        return sample.StatesAndTrace(
            all_states=all_states,
            trace=tf.nest.map_structure(lambda *s: tf.concat(s, axis=0),
                                        *all_traces, expand_composites=True))
