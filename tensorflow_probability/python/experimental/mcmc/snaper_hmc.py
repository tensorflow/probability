# Copyright 2021 The TensorFlow Probability Authors.
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
"""SNAPER-HMC[1] TransitionKernel.

#### References

[1]: Sountsov, P. & Hoffman, M. (2021). Focusing on Difficult Directions for
     Learning HMC Trajectory Lengths. <https://arxiv.org/abs/2110.11576>
"""

import collections
import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import restructure
from tensorflow_probability.python.experimental.mcmc import gradient_based_trajectory_length_adaptation as gbtla
from tensorflow_probability.python.experimental.mcmc import preconditioned_hmc
from tensorflow_probability.python.experimental.mcmc import preconditioning_utils
from tensorflow_probability.python.experimental.mcmc import reducer as reducer_lib
from tensorflow_probability.python.experimental.mcmc import sample_discarding_kernel
from tensorflow_probability.python.experimental.mcmc import sharded
from tensorflow_probability.python.experimental.mcmc import thinning_kernel
from tensorflow_probability.python.experimental.mcmc import with_reductions
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import loop_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.math import generic as math_generic
from tensorflow_probability.python.mcmc import dual_averaging_step_size_adaptation as dassa
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'SNAPERHamiltonianMonteCarlo',
    'SNAPERHamiltonianMonteCarloResults',
    'SampleSNAPERHamiltonianMonteCarloResults',
    'sample_snaper_hmc',
]


class SNAPERHamiltonianMonteCarloResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('GradientBasedTrajectoryLengthAdaptationResults', [
        'inner_results',
        'ema_mean',
        'ema_variance',
        'state_ema_points',
        'ema_principal_component',
        'principal_component_ema_points',
        'seed',
    ])):
  """Internal state of SNAPERHamiltonianMonteCarlo.

  Attributes:
    inner_results: Results of the inner kernel. This is
      `PreconditionedHamiltonianMonteCarloResults` wrapped in
      `GradientBasedTrajectoryLengthAdaptationResults`.
    ema_mean: Exponential moving average cross-chain state mean.
    ema_variance: Exponential moving average cross-chain state variance.
    state_ema_points: Approximate number of points used to compute the
      exponential moving averages.
    ema_principal_component: Exponential moving average cross-chain state
      covariance matrix principal component.
    principal_component__ema_points: Approximate number of points used to
      compute the exponential moving average of the principal component.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details. The random seed
      used by the kernel in the previous step.
  """
  __slots__ = ()


class SNAPERHamiltonianMonteCarlo(kernel_base.TransitionKernel):
  """SNAPER-HMC without step size adaptation.

  This implements the SNAPER-HMC algorithm from [1], without the step size
  adaptation. This kernel learns a diagonal mass matrix and the trajectory
  length parameters of the Hamiltonian Monte Carlo (HMC) sampler using the
  Adaptive MCMC framework [2]. As with all adaptive MCMC algorithms, this kernel
  does not produce samples from the target distribution while adaptation is
  engaged, so be sure to set `num_adaptation_steps` parameter smaller than the
  number of burnin steps.

  This kernel uses the SNAPER criterion (see
  `tfp.experimental.mcmc.snaper_criterion` for details) which has a principal-
  component parameter. This kernel learns it using a batched Oja's algorithm
  with a learning rate of `principal_component_ema_factor / step` where `step`
  is the iteration number.

  The mass matrix is learned using a variant of the Welford's
  algorithm/Exponential Moving Average, with a decay rate set to `step //
  state_ema_factor / (step // state_ema_factor + 1)`.

  Learning the step size is a necessary component of a good HMC sampler, but it
  is not handled by this kernel. That adaptation can be provided by, for
  example, `tfp.mcmc.SimpleStepSizeAdaptation` or
  `tfp.mcmc.DualAveragingSizeAdaptation`.

  To aid algorithm stability, the first few steps are taken with the number of
  leapfrog steps set to 1, turning the algorithm into Metropolis Adjusted
  Langevin Algorithm (MALA). This is controlled by the `num_mala_steps`
  argument.

  Unlike some classical MCMC algorithms, this algorithm behaves best when the
  chains are initialized with very low variance. Initializing them all at one
  point is recommended.

  SNAPER-HMC requires at least two chains to function.

  #### Examples

  Here we apply this kernel to a target with a known covariance structure and
  show that it recovers the principal component and the variances.

  ```python
  num_dims = 8
  num_burnin_steps = 1000
  num_adaptation_steps = int(num_burnin_steps * 0.8)
  num_results = 500
  num_chains = 64
  step_size = 1e-2
  num_mala_steps = 100

  eigenvalues = np.exp(np.linspace(0., 3., num_dims))
  q, r = np.linalg.qr(np.random.randn(num_dims, num_dims))
  q *= np.sign(np.diag(r))
  covariance = (q * eigenvalues).dot(q.T)

  _, eigs = np.linalg.eigh(covariance)
  principal_component = eigs[:, -1]

  gaussian = tfd.MultivariateNormalTriL(
      loc=tf.zeros(num_dims),
      scale_tril=tf.linalg.cholesky(covariance),
  )

  kernel = tfp.experimental.mcmc.SNAPERHamiltonianMonteCarlo(
      gaussian.log_prob,
      step_size=step_size,
      num_adaptation_steps=num_adaptation_steps,
      num_mala_steps=num_mala_steps,
  )
  kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
      kernel, num_adaptation_steps=num_adaptation_steps)

  def trace_fn(_, pkr):
    return {
        'principal_component':
            unnest.get_innermost(pkr, 'ema_principal_component'),
        'variance':
            unnest.get_innermost(pkr, 'ema_variance'),
    }

  init_x = tf.zeros([num_chains, num_dims])

  chain, trace = tfp.mcmc.sample_chain(
              num_results=num_results,
              num_burnin_steps=num_burnin_steps,
              current_state=init_x,
              kernel=kernel,
              trace_fn=trace_fn)

  # Close to `np.diag(covariance)`
  trace['variance'][-1]
  # Close to `principal_component`, up to a sign.
  trace['principal_component'][-1]

  # Compute sampler diagnostics.
  tfp.mcmc.effective_sample_size(chain, cross_chain_dims=1)
  tfp.mcmc.potential_scale_reduction(chain)

  # Compute downstream statistics.
  tf.reduce_mean(chain, [0, 1])
  ```


  #### References

  [1]: Sountsov, P. & Hoffman, M. (2021). Focusing on Difficult Directions for
       Learning HMC Trajectory Lengths. <https://arxiv.org/abs/2110.11576>

  [2]: Andrieu, Christophe, Thoms, Johannes. A tutorial on adaptive MCMC.
       Statistics and Computing, 2008.
       <https://people.eecs.berkeley.edu/~jordan/sail/readings/andrieu-thoms.pdf>.
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_adaptation_steps,
               num_mala_steps=100,
               max_leapfrog_steps=1000,
               trajectory_length_adaptation_rate=0.05,
               principal_component_ema_factor=8,
               state_ema_factor=8,
               experimental_shard_axis_names=None,
               experimental_reduce_chain_axis_names=None,
               preconditioned_hamiltonian_monte_carlo_kwargs=None,
               gradient_based_trajectory_length_adaptation_kwargs=None,
               validate_args=False,
               name=None):
    """Constructs the `SNAPERHamiltonianMonteCarlo` kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: Scalar `float` `Tensor` representing the step size for the
        leapfrog integrator.
      num_adaptation_steps: Scalar `int` `Tensor` number of initial steps during
        which to adjust the hyperparameters.
      num_mala_steps: Scalar `int` `Tensor` number of initial steps during which
        the number of leapfrog steps is clamped to 1, for stability.
      max_leapfrog_steps: Scalar `int` `Tensor`. Clips the number of leapfrog
        steps to this value.
      trajectory_length_adaptation_rate: Scalar `float` `Tensor`. How
        rapidly to adapt the trajectory length.
      principal_component_ema_factor: Scalar `int` `Tensor`. Factor controlling
        the principal component adaptation. Larger number corresponds to faster
        adaptation.
      state_ema_factor: Scalar `int` `Tensor`. Factor controlling
        the mass matrix adaptation. Larger number corresponds to faster
        adaptation.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.
      experimental_reduce_chain_axis_names: A string or list of string names
        indicating which named axes to average cross-chain statistics over.
      preconditioned_hamiltonian_monte_carlo_kwargs: Additional keyword
        arguments to pass to `PreconditionedHamiltonianMonteCarlo` kernel.
      gradient_based_trajectory_length_adaptation_kwargs: Additional keyword
        arguments to pass to `GradientBasedTrajectoryLengthAdaptation` kernel.
      validate_args: Python `bool`. When `True`, kernel parameters are checked
        for validity. When `False`, invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class. Default:
        'snaper_hamiltonian_monte_carlo'.
    """
    with tf.name_scope(
        mcmc_util.make_name(name, 'snaper_hamiltonian_monte_carlo',
                            '__init__')):
      num_adaptation_steps = tf.convert_to_tensor(
          num_adaptation_steps,
          dtype_hint=tf.int32,
          name='num_adaptation_steps')
      num_mala_steps = tf.convert_to_tensor(
          num_mala_steps, dtype_hint=tf.int32, name='num_mala_steps')
      max_leapfrog_steps = tf.convert_to_tensor(
          max_leapfrog_steps, dtype_hint=tf.int32, name='max_leapfrog_steps')

    if preconditioned_hamiltonian_monte_carlo_kwargs is None:
      preconditioned_hamiltonian_monte_carlo_kwargs = {}

    if gradient_based_trajectory_length_adaptation_kwargs is None:
      gradient_based_trajectory_length_adaptation_kwargs = {}

    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_adaptation_steps=num_adaptation_steps,
        num_mala_steps=num_mala_steps,
        max_leapfrog_steps=max_leapfrog_steps,
        trajectory_length_adaptation_rate=trajectory_length_adaptation_rate,
        state_ema_factor=state_ema_factor,
        principal_component_ema_factor=principal_component_ema_factor,
        experimental_reduce_chain_axis_names=(
            experimental_reduce_chain_axis_names),
        experimental_shard_axis_names=experimental_shard_axis_names,
        preconditioned_hamiltonian_monte_carlo_kwargs=(
            preconditioned_hamiltonian_monte_carlo_kwargs),
        gradient_based_trajectory_length_adaptation_kwargs=(
            gradient_based_trajectory_length_adaptation_kwargs),
        validate_args=validate_args,
        name=name,
    )

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def step_size(self):
    return self._parameters['step_size']

  @property
  def num_adaptation_steps(self):
    return self._parameters['num_adaptation_steps']

  @property
  def num_mala_steps(self):
    return self._parameters['num_mala_steps']

  @property
  def max_leapfrog_steps(self):
    return self._parameters['max_leapfrog_steps']

  @property
  def trajectory_length_adaptation_rate(self):
    return self._parameters['trajectory_length_adaptation_rate']

  @property
  def state_ema_factor(self):
    return self._parameters['state_ema_factor']

  @property
  def principal_component_ema_factor(self):
    return self._parameters['principal_component_ema_factor']

  @property
  def experimental_reduce_chain_axis_names(self):
    return self._parameters['experimental_reduce_chain_axis_names']

  @property
  def experimental_shard_axis_names(self):
    return self._parameters['experimental_shard_axis_names']

  @property
  def preconditioned_hamiltonian_monte_carlo_kwargs(self):
    return self._parameters['preconditioned_hamiltonian_monte_carlo_kwargs']

  @property
  def gradient_based_trajectory_length_adaptation_kwargs(self):
    return self._parameters[
        'gradient_based_trajectory_length_adaptation_kwargs']

  @property
  def validate_args(self):
    return self._parameters['validate_args']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    return self._parameters

  def _make_kernel(self, batch_shape, step, state_ema_points, state, mean,
                   variance, principal_component):

    if self.experimental_shard_axis_names is None:
      shard_axis_names = tf.nest.map_structure(lambda _: None, state)
    else:
      shard_axis_names = self.experimental_shard_axis_names

    def _max_part(x, named_axis):
      size = tf.get_static_value(ps.size(x))
      # Support zero-sized states. It's invalid (under JAX) to reduce zero-sized
      # tensors along the zero-sized axis. TF, returns -inf in this case.
      if size is not None and size == 0:
        return tf.ones([], x.dtype)
      else:
        return distribute_lib.reduce_max(
            # all_gather is fine here, since we're reducing locally to a scalar.
            x, None, named_axis, allow_all_gather=True)

    max_variance = tf.reduce_max(
        tf.nest.flatten(
            nest.map_structure_up_to(state, _max_part, variance,
                                     shard_axis_names)),
        axis=0)
    variance = tf.nest.map_structure(lambda x: x / max_variance, variance)

    state_parts = tf.nest.flatten(state)
    mean_parts = tf.nest.flatten(mean)
    variance_parts = tf.nest.flatten(variance)
    principal_component_parts = tf.nest.flatten(principal_component)
    shard_axis_names_parts = nest.flatten_up_to(state, shard_axis_names)
    state_ema_points = tf.cast(state_ema_points, state_parts[0].dtype)

    kernel = preconditioned_hmc.PreconditionedHamiltonianMonteCarlo(
        self.target_log_prob_fn,
        step_size=self.step_size,
        num_leapfrog_steps=1,
        momentum_distribution=preconditioning_utils.make_momentum_distribution(
            state_parts,
            batch_shape=batch_shape,
            running_variance_parts=variance_parts,
            shard_axis_names=shard_axis_names_parts,
        ),
        **self.preconditioned_hamiltonian_monte_carlo_kwargs,
    )
    gbtla_kwargs = (
        self.gradient_based_trajectory_length_adaptation_kwargs.copy())
    gbtla_kwargs.setdefault('averaged_sq_grad_adaptation_rate', 0.5)
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        kernel,
        num_adaptation_steps=self.num_adaptation_steps,
        max_leapfrog_steps=tf.where(step < self.num_mala_steps, 1,
                                    self.max_leapfrog_steps),
        adaptation_rate=self.trajectory_length_adaptation_rate,
        experimental_reduce_chain_axis_names=(
            self.experimental_reduce_chain_axis_names),
        experimental_shard_axis_names=shard_axis_names_parts,
        criterion_fn=functools.partial(
            gbtla.snaper_criterion,
            direction=principal_component_parts,
            state_mean=mean_parts,
            state_mean_weight=state_ema_points / (state_ema_points + 1),
        ),
        validate_args=self.validate_args,
        **gbtla_kwargs,
    )
    return kernel

  def _update_state_ema(
      self,
      reduce_axes,
      state,
      step,
      state_ema_points,
      ema_mean,
      ema_variance,
  ):
    # This is Welford's algorithm where the number of points is clamped to a
    # function that grows slower than N.

    def _one_part(state, mean, variance):
      state_mean = distribute_lib.reduce_mean(
          state,
          reduce_axes,
          self.experimental_reduce_chain_axis_names,
      )
      state_variance = distribute_lib.reduce_mean(
          tf.square(state - state_mean),
          reduce_axes,
          self.experimental_reduce_chain_axis_names,
      )
      mean_diff = state_mean - mean
      mean_diff_sq = tf.square(mean_diff)
      variance_diff = state_variance - variance
      weight = 1. / (tf.cast(state_ema_points, state.dtype) + 1.)

      new_mean = mean + mean_diff * weight
      new_variance = (
          variance + weight * (variance_diff + (1. - weight) * mean_diff_sq))

      return new_mean, new_variance

    mean_variance = tf.nest.map_structure(_one_part, state, ema_mean,
                                          ema_variance)

    new_state_ema_points = tf.minimum(
        state_ema_points + 1, tf.maximum(1, step // self.state_ema_factor))
    new_ema_mean = nest.map_structure_up_to(state, lambda x: x[0],
                                            mean_variance)
    new_ema_variance = nest.map_structure_up_to(state, lambda x: x[1],
                                                mean_variance)
    return tf.nest.map_structure(
        lambda x, y: tf.where(step < self.num_adaptation_steps, x, y),
        (new_state_ema_points, new_ema_mean, new_ema_variance),
        (state_ema_points, ema_mean, ema_variance),
    )

  def _update_principal_component_ema(
      self,
      reduce_axes,
      state,
      step,
      principal_component_ema_points,
      ema_principal_component,
  ):
    # This is a batched version of Oja's algorithm. For the learning rate step,
    # we use Welford's algorithm where the number of points is clamped to a
    # function that grows slower than N.

    event_axes = tf.nest.map_structure(
        lambda x: ps.range(ps.size(reduce_axes), ps.rank(x)) - ps.rank(x),
        state)
    if self.experimental_shard_axis_names is None:
      shard_axis_names = tf.nest.map_structure(lambda _: None, state)
    else:
      shard_axis_names = self.experimental_shard_axis_names

    def _center_part(x):
      return x - distribute_lib.reduce_mean(
          x, reduce_axes, self.experimental_reduce_chain_axis_names)

    state_dot_p = _dot_product(
        tf.nest.map_structure(_center_part, state), ema_principal_component,
        event_axes, shard_axis_names)

    def _weighted_sum_part(x):
      return distribute_lib.reduce_sum(
          bu.left_justified_expand_dims_like(state_dot_p, x) * x, reduce_axes,
          self.experimental_reduce_chain_axis_names)

    new_principal_component = _normalize(
        tf.nest.map_structure(_weighted_sum_part, state), event_axes,
        shard_axis_names)

    def _ema_part(old_x, new_x):
      weight = 1. / (tf.cast(principal_component_ema_points, old_x.dtype) + 1.)
      return old_x + (new_x - old_x) * weight

    new_principal_component_ema_points = tf.minimum(
        principal_component_ema_points + 1,
        tf.maximum(1, step // self.principal_component_ema_factor))
    new_ema_principal_component = _normalize(
        tf.nest.map_structure(_ema_part, ema_principal_component,
                              new_principal_component), event_axes,
        shard_axis_names)
    return tf.nest.map_structure(
        lambda x, y: tf.where(step < self.num_adaptation_steps, x, y),
        (new_principal_component_ema_points, new_ema_principal_component),
        (principal_component_ema_points, ema_principal_component),
    )

  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'snaper_hamiltonian_monte_carlo',
                            'one_step')):
      inner_results = previous_kernel_results.inner_results

      batch_shape = ps.shape(
          unnest.get_innermost(previous_kernel_results, 'target_log_prob'))
      reduce_axes = ps.range(0, ps.size(batch_shape))
      step = inner_results.step
      state_ema_points = previous_kernel_results.state_ema_points

      kernel = self._make_kernel(
          batch_shape=batch_shape,
          step=step,
          state_ema_points=state_ema_points,
          state=current_state,
          mean=previous_kernel_results.ema_mean,
          variance=previous_kernel_results.ema_variance,
          principal_component=previous_kernel_results.ema_principal_component,
      )

      inner_results = unnest.replace_innermost(
          inner_results,
          momentum_distribution=(
              kernel.inner_kernel.parameters['momentum_distribution']),  # pylint: disable=protected-access
      )

      seed = samplers.sanitize_seed(seed)
      state_parts, inner_results = kernel.one_step(
          tf.nest.flatten(current_state),
          inner_results,
          seed=seed,
      )

      state = tf.nest.pack_sequence_as(current_state, state_parts)

      state_ema_points, ema_mean, ema_variance = self._update_state_ema(
          reduce_axes=reduce_axes,
          state=state,
          step=step,
          state_ema_points=state_ema_points,
          ema_mean=previous_kernel_results.ema_mean,
          ema_variance=previous_kernel_results.ema_variance,
      )

      (principal_component_ema_points,
       ema_principal_component) = self._update_principal_component_ema(
           reduce_axes=reduce_axes,
           state=state,
           step=step,
           principal_component_ema_points=(
               previous_kernel_results.principal_component_ema_points),
           ema_principal_component=(
               previous_kernel_results.ema_principal_component),
       )

      kernel_results = previous_kernel_results._replace(
          inner_results=inner_results,
          ema_mean=ema_mean,
          ema_variance=ema_variance,
          state_ema_points=state_ema_points,
          ema_principal_component=ema_principal_component,
          principal_component_ema_points=principal_component_ema_points,
          seed=seed,
      )

      return state, kernel_results

  def bootstrap_results(self, init_state):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'snaper_hamiltonian_monte_carlo',
                            'bootstrap_results')):
      init_state = tf.nest.map_structure(
          lambda x: tf.convert_to_tensor(x, name='init_state'), init_state)

      # It is unfortunate that we need to make this extra call to the TLP here.
      # The issue is that we need this value to even construct the PHMC, and
      # the kernel will call this one itself.
      tlp = self.target_log_prob_fn(*tf.nest.flatten(init_state))
      batch_shape = ps.shape(tlp)
      batch_ndims = ps.rank(tlp)
      if tf.get_static_value(batch_ndims) is None:
        # The issue doesn't live in this file, rather it is the downstream
        # components that fail to work (notably, tfb.Reshape).
        raise ValueError('SNAPERHMC currently requires a statically known '
                         'rank of the target log probability.')

      # We need at least two chains to estimate the principal component.
      # Number of total chains is local batch size * distributed axis size
      reduce_chain_axis_names = distribute_lib.canonicalize_named_axis(
          self.experimental_reduce_chain_axis_names)
      local_axis_size = ps.size(tlp)
      distributed_axis_size = int(
          ps.reduce_prod([
              distribute_lib.get_axis_size(a) for a in reduce_chain_axis_names
          ]))
      num_chains = local_axis_size * distributed_axis_size
      num_chains_ = tf.get_static_value(num_chains)
      if num_chains_ is not None:
        if num_chains_ < 2:
          raise ValueError(
              'SNAPERHMC requires at least 2 chains. Got: {}'.format(
                  num_chains_))
      elif self.validate_args:
        with tf.control_dependencies([
            assert_util.assert_greater_equal(
                num_chains, 2, 'SNAPERHMC requires at least 2 chains.')
        ]):
          init_state = tf.nest.map_structure(tf.identity, init_state)

      event_axes = tf.nest.map_structure(
          lambda x: ps.range(batch_ndims, ps.rank(x)) - ps.rank(x), init_state)
      if self.experimental_shard_axis_names is None:
        shard_axis_names = tf.nest.map_structure(lambda _: None, init_state)
      else:
        shard_axis_names = self.experimental_shard_axis_names

      ema_variance = tf.nest.map_structure(
          lambda x: tf.ones(  # pylint: disable=g-long-lambda
              ps.shape(x)[batch_ndims:],
              dtype=x.dtype,
              name='ema_variance'),
          init_state)
      ema_mean = tf.nest.map_structure(
          lambda x: tf.zeros_like(x, name='ema_mean'), ema_variance)
      ema_principal_component = _normalize(ema_variance, event_axes,
                                           shard_axis_names)
      # These start out at 1 for a bit of smoothing.
      state_ema_points = tf.ones([], tf.int32)
      principal_component_ema_points = tf.ones([], tf.int32)

      kernel = self._make_kernel(
          batch_shape=batch_shape,
          step=tf.zeros([], tf.int32),
          state_ema_points=state_ema_points,
          state=init_state,
          mean=ema_mean,
          variance=ema_variance,
          principal_component=ema_principal_component,
      )

      inner_results = kernel.bootstrap_results(tf.nest.flatten(init_state))

      kernel_results = SNAPERHamiltonianMonteCarloResults(
          inner_results=inner_results,
          ema_mean=ema_mean,
          ema_variance=ema_variance,
          state_ema_points=state_ema_points,
          ema_principal_component=ema_principal_component,
          principal_component_ema_points=principal_component_ema_points,
          seed=samplers.zeros_seed(),
      )
      return kernel_results

  @property
  def is_calibrated(self):
    return True

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(experimental_shard_axis_names=shard_axis_names)


def _init_chain_state(
    model,
    event_space_bijector,
    num_chains,
    event_dtype,
    event_shape,
    init_state,
    experimental_shard_axis_names,
):
  """Initializes chain state for sample_snaper_hmc."""
  # TODO(siege): Consider what, if anything, can be shared with
  # mcmc.init_near_unconstrained_zero.
  with tf.name_scope('init_mcmc_state'):
    if event_space_bijector is None:
      if hasattr(model, 'experimental_default_event_space_bijector'):
        event_space_bijector = model.experimental_default_event_space_bijector()
    else:
      if not isinstance(event_space_bijector, bijector_lib.Bijector):
        event_space_bijector = joint_map.JointMap(event_space_bijector)

    if init_state is not None:
      with tf.name_scope('init_state'):
        init_state = tf.nest.map_structure(tf.convert_to_tensor, init_state)

    # Impute shape/dtype. There's 3 possible states here.
    # 1. init_state is specified.
    # 2. init_state is not specified, and we use a distribution.
    # 3. init_state is not specified, and we use manual annotations.

    if event_dtype is None:
      if init_state is None:
        if hasattr(model, 'dtype'):
          event_dtype = model.dtype  # (2)
        else:
          raise ValueError(
              '`event_dtype` must be specified if `model` does not '
              'have an `event_shape` property and `init_state` is not '
              'specified.')  # (3)
      else:
        event_dtype = tf.nest.map_structure(lambda x: x.dtype,
                                            init_state)  # (1)

    if event_shape is None:
      if init_state is None:
        if hasattr(model, 'event_shape'):
          event_shape = model.event_shape  # (2)
          if not all(
              tf.nest.flatten(
                  tf.nest.map_structure(tensorshape_util.is_fully_defined,
                                        event_shape))):
            event_shape = model.event_shape_tensor
        else:
          raise ValueError(
              '`event_shape` must be specified if `model` does not '
              'have an `event_shape` property and `init_state` is not '
              'specified.')  # (2)
      else:
        event_shape = tf.nest.map_structure(ps.shape, init_state)  # (1)

    # Get a flat current state.
    flat_event_space_bijector = restructure.pack_sequence_as(event_dtype)
    if event_space_bijector is not None:
      flat_event_space_bijector = event_space_bijector(
          flat_event_space_bijector)

    if init_state is None:
      # TODO(siege): See if we can somehow extract this into an API similar to
      # init_near_unconstrained_zero. Key issue is that we do the static shape
      # canonicalization differently.
      if num_chains is None:
        # Default to 64 from the paper, lower might be fine too.
        num_chains = 64
      static_event_shape = nest.map_structure_up_to(event_dtype,
                                                    tf.get_static_value,
                                                    event_shape)
      event_shape_is_static = all(
          map(lambda x: x is not None,
              nest.flatten_up_to(event_dtype, static_event_shape)))
      if event_shape_is_static:
        unconstrained_shape = flat_event_space_bijector.inverse_event_shape(
            static_event_shape)
      else:
        unconstrained_shape = flat_event_space_bijector.inverse_event_shape_tensor(
            event_shape)
      unconstrained_dtype = flat_event_space_bijector.inverse_dtype(event_dtype)

      with tf.name_scope('unconstrained_state'):
        unconstrained_state = nest.map_structure_up_to(
            unconstrained_dtype,
            lambda s, d: tf.zeros(ps.concat([[num_chains], s], 0), d),
            unconstrained_shape, unconstrained_dtype)

    else:
      with tf.name_scope('unconstrained_state'):
        unconstrained_state = tf.nest.map_structure(
            tf.identity, flat_event_space_bijector.inverse(init_state))

    shard_axis_names_parts = None
    if experimental_shard_axis_names is None:
      if hasattr(model, 'experimental_shard_axis_names'):
        experimental_shard_axis_names = model.experimental_shard_axis_names

      # Canonicalize non-sharded axis sharded names to None to skip the check
      # below.
      if not nest.flatten(experimental_shard_axis_names):
        experimental_shard_axis_names = None

    if experimental_shard_axis_names is None:
      shard_axis_names_parts = None
    else:
      # TODO(siege): This assumes that the event bijector doesn't alter the
      # structure of the event. We're missing some sort of
      # forward_sharded_axis_names type of function. This check cannot detect
      # pure shuffles, but they are just as problematic.
      shard_axis_names_parts = nest.flatten_up_to(
          event_dtype, experimental_shard_axis_names)
      if len(shard_axis_names_parts) != len(unconstrained_state):
        raise ValueError('`event_space_bijector`s that alter the state shape '
                         'are not supported when sharded axes are used.')

  return unconstrained_state, flat_event_space_bijector, shard_axis_names_parts


def _init_step_size(target_log_prob_fn, state_parts, shard_axis_names_parts):
  """Initializes step size for sample_snaper_hmc."""
  dtype = dtype_util.common_dtype(state_parts)
  tlp_rank = ps.rank(target_log_prob_fn(*state_parts))
  if shard_axis_names_parts is None:
    shard_axis_names_parts = [None] * len(state_parts)
  num_dims = sum(
      distribute_lib.reduce_sum(
          ps.reduce_prod(ps.shape(x)[tlp_rank:]), named_axis=na)
      for x, na in zip(state_parts, shard_axis_names_parts))
  # See Beskos, A., Pillai, N. S., Roberts, G. O., Sanz-Serna, J. M., & Stuart,
  # A. M. 2010. "Optimal tuning of the Hybrid Monte-Carlo Algorithm."
  return 1e-2 * tf.cast(num_dims, dtype) ** -0.25


def _make_snaper_kernel(
    model,
    reducer,
    unconstrained_state,
    flat_event_space_bijector,
    init_step_size,
    num_burnin_steps,
    num_adaptation_steps,
    dual_averaging_kwargs,
    discard_burnin_steps,
    num_steps_between_results,
    shard_axis_names_parts,
    experimental_reduce_chain_axis_names,
    validate_args,
    snaper_kwargs,
):
  """Initializes the kernel for sample_snaper_hmc."""
  # We don't use TransformedTransition kernel because it has poor support for
  # general event types.
  def flat_target_log_prob_fn(*unconstrained_state):
    if hasattr(model, 'unnormalized_log_prob'):
      target_log_prob_fn = model.unnormalized_log_prob
    else:
      target_log_prob_fn = model
    # Restructure bijector expects a list.
    unconstrained_state = list(unconstrained_state)

    state = flat_event_space_bijector.forward(unconstrained_state)
    tlp = nest_util.call_fn(target_log_prob_fn, state)
    tlp_rank = ps.rank(tlp)
    fldj = flat_event_space_bijector.forward_log_det_jacobian(
        unconstrained_state,
        event_ndims=tf.nest.map_structure(lambda s: ps.rank(s) - tlp_rank,
                                          unconstrained_state))
    return tlp + fldj

  if init_step_size is None:
    init_step_size = _init_step_size(
        target_log_prob_fn=flat_target_log_prob_fn,
        state_parts=unconstrained_state,
        shard_axis_names_parts=shard_axis_names_parts)

  if snaper_kwargs is None:
    snaper_kwargs = {}

  if dual_averaging_kwargs is None:
    dual_averaging_kwargs = {}

  dual_averaging_kwargs = dict(dual_averaging_kwargs)
  dual_averaging_kwargs.setdefault('target_accept_prob', 0.8)
  dual_averaging_kwargs.setdefault('reduce_fn',
                                   math_generic.reduce_log_harmonic_mean_exp)

  kernel = SNAPERHamiltonianMonteCarlo(
      target_log_prob_fn=flat_target_log_prob_fn,
      step_size=init_step_size,
      num_adaptation_steps=num_adaptation_steps,
      experimental_shard_axis_names=shard_axis_names_parts,
      experimental_reduce_chain_axis_names=(
          experimental_reduce_chain_axis_names),
      validate_args=validate_args,
      **snaper_kwargs,
  )

  kernel = dassa.DualAveragingStepSizeAdaptation(
      kernel,
      num_adaptation_steps=num_adaptation_steps,
      experimental_reduce_chain_axis_names=experimental_reduce_chain_axis_names,
      **dual_averaging_kwargs,
  )

  num_outer_kernels = 0
  if reducer is not None:
    kernel = with_reductions.WithReductions(
        kernel,
        reducer=_SNAPERReducer(
            reducer,
            num_burnin_steps=num_burnin_steps * (num_steps_between_results + 1),
            flat_event_space_bijector=flat_event_space_bijector))
    num_outer_kernels += 1

  if experimental_reduce_chain_axis_names is not None:
    kernel = sharded.Sharded(kernel, experimental_reduce_chain_axis_names)
    # Sharded doesn't add a wrapper over the results, so we don't increment the
    # num_outer_kernels.

  if discard_burnin_steps:
    kernel = sample_discarding_kernel.SampleDiscardingKernel(
        kernel,
        # This behavior is different than sample_chain because otherwise
        # discarding or keeping the burnin steps would affect the length of the
        # burnin, which is confusing. sample_chain doesn't have this problem
        # because it doesn't have an option to keep burnin.
        num_burnin_steps=num_burnin_steps * (num_steps_between_results + 1),
        num_steps_between_results=num_steps_between_results)
    num_outer_kernels += 1
  elif num_steps_between_results > 0:
    kernel = thinning_kernel.ThinningKernel(
        kernel, num_steps_to_skip=num_steps_between_results)
    # ThinningKernel doesn't add a wrapper over the results, so we don't
    # increment the num_outer_kernels.

  def get_inner_results(kernel_results):
    for _ in range(num_outer_kernels):
      kernel_results = kernel_results.inner_results
    return kernel_results

  return kernel, get_inner_results


def _sample_snaper_loop(
    unconstrained_state,
    kernel,
    trace_fn,
    reducer,
    flat_event_space_bijector,
    num_results,
    num_burnin_steps,
    discard_burnin_steps,
    get_inner_results,
    seed,
):
  """The sampling loop for sample_snaper_hmc."""
  with tf.name_scope('sample_snaper_loop'):
    # TODO(siege): Figure out if we can use sample_chain_with_burnin here:
    #
    # - We need to manually transform the state (could be addressed by fixing
    # the TransformedTransitionKernel).
    # - We want to use masking to deal with delayed reduction rather than what
    # sample_chain_with_burnin is doing.
    def loop_fn(all_state, _):
      unconstrained_state, kernel_results, seed = all_state
      seed, hmc_seed = samplers.split_seed(seed)

      unconstrained_state, kernel_results = kernel.one_step(
          unconstrained_state, kernel_results, seed=hmc_seed)

      return unconstrained_state, kernel_results, seed

    def outer_trace_fn(all_state):
      unconstrained_state, kernel_results, _ = all_state

      state = flat_event_space_bijector(unconstrained_state)
      reducer_state = unnest.get_innermost(kernel_results, 'reduction_results',
                                           None)
      inner_results = get_inner_results(kernel_results)

      return trace_fn(
          state=state,
          is_burnin=inner_results.step < num_burnin_steps,
          kernel_results=inner_results,
          reducer=reducer,
          reducer_state=reducer_state)

    output_size = num_results
    if not discard_burnin_steps:
      output_size += num_burnin_steps

    kernel_results = kernel.bootstrap_results(unconstrained_state)

    all_state, trace = loop_util.trace_scan(
        loop_fn=loop_fn,
        initial_state=(unconstrained_state, kernel_results,
                       samplers.sanitize_seed(seed)),
        elems=tf.range(output_size),
        trace_fn=outer_trace_fn,
    )

    unconstrained_state, kernel_results, _ = all_state

    return unconstrained_state, kernel_results, trace


def default_snaper_trace_fn(state, is_burnin, kernel_results, reducer,
                            reducer_state):
  del reducer, reducer_state
  kr = kernel_results
  energy_diff = unnest.get_innermost(kr, 'log_accept_ratio')
  # The ~ is here to catch NaNs.
  has_divergence = ~(tf.math.abs(energy_diff) < 500.)
  return state, {
      'step_size':
          unnest.get_innermost(kr, 'step_size'),
      'n_steps':
          unnest.get_innermost(kr, 'num_leapfrog_steps'),
      'tune':
          is_burnin,
      'max_trajectory_length':
          unnest.get_innermost(kr, 'max_trajectory_length'),
      'variance_scaling':
          tf.nest.map_structure(lambda x: 1. / x,
                                unnest.get_innermost(kr, 'ema_variance')),
      'diverging':
          has_divergence,
      'accept_ratio':
          tf.minimum(tf.ones_like(energy_diff), tf.exp(energy_diff)),
      'is_accepted':
          unnest.get_innermost(kr, 'is_accepted'),
  }


class SampleSNAPERHamiltonianMonteCarloResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('SampleSNAPERHamiltonianMonteCarloResults', [
        'trace',
        'reduction_results',
        'final_state',
        'final_kernel_results',
    ])):
  """Results of `sample_snaper_hmc`.

  Attributes:
    trace: Traced quantities defined by `trace_fn`.
    reduction_results: Finalized reducer results.
    final_state: Final state of the MCMC chain.
    final_kernel_results: The final results of `DualAveragingStepSizeAdaptation`
      wrapping `SNAPERHamiltonianMonteCarlo` kernels.
  """
  __slots__ = ()


def sample_snaper_hmc(model,
                      num_results,
                      reducer=None,
                      trace_fn=default_snaper_trace_fn,
                      num_burnin_steps=1000,
                      num_adaptation_steps=None,
                      num_chains=None,
                      discard_burnin_steps=True,
                      num_steps_between_results=0,
                      init_state=None,
                      init_step_size=None,
                      event_space_bijector=None,
                      event_dtype=None,
                      event_shape=None,
                      experimental_shard_axis_names=None,
                      experimental_reduce_chain_axis_names=None,
                      dual_averaging_kwargs=None,
                      snaper_kwargs=None,
                      seed=None,
                      validate_args=False,
                      name='snaper_hmc'):
  """Generates samples using SNAPER HMC [1] with step size adaptation.

  This utility function generates samples from a probabilistic model using
  `SNAPERHamiltonianMonteCarlo` kernel combined with
  `DualAveragingStepSizeAdaptation` kernel. The `model` argument can either be
  an instance of `tfp.distributions.Distribution` or a callable that computes
  the target log-density. In the latter case, it is also necessary to specify
  `event_space_bijector`, `event_dtype` and `event_shape` (these are inferred if
  `model` is a distribution instance).

  This function can accept a structure of `tfp.experimental.mcmc.Reducer`s,
  which allow computing streaming statitics with minimal memory usage. The
  reducers only incorporate samples after the burnin period.

  By default, this function traces the following quantities:

  - The chain state.
  - A dict of auxiliary information, using keys from ArviZ [2].
    - step_size: Float scalar `Tensor`. HMC step size.
    - n_steps: Int `Tensor`. Number of HMC leapfrog steps.
    - tune: Bool `Tensor`. Whether this step is part of the burnin.
    - max_trajectory_length: Float `Tensor`. Maximum HMC trajectory length.
    - variance_scaling: List of float `Tensor`s. The diagonal variance of the
      unconstrained state, used as the mass matrix.
    - diverging: Bool `Tensor`. Whether the sampler is divering.
    - accept_ratio: Float `Tensor`. Probability of acceptance of the proposal
      for this step.
    - is_accepted: Bool `Tensor. Whether this step is a result of an accepted
      proposal.

  It is possible to trace nothing at all, and rely on the reducers to compute
  the necessary statitiscs.

  Args:
    model: Either an instance of `tfp.distributions.Distribution` or a callable
      that evaluates the target log-density at a batch of chain states.
    num_results: Number of MCMC results to return after burnin.
    reducer: A structure of reducers.
    trace_fn: A callable with signature: `(state, is_burnin, kernel_results,
      reducer, reducer_state) -> structure` which defines what quantities to
      trace.
    num_burnin_steps: Python `int`. Number of burnin steps.
    num_adaptation_steps: Python `int`. Number of adaptation steps. Default:
      `0.9 * num_burnin_steps`.
    num_chains: Python `int`. Number of chains. This can be inferred from
      `init_state`. Otherwise, this is 64 by default.
    discard_burnin_steps: Python `bool`. Whether to discard the burnin steps
      when returning the trace. Burning steps are never used for the reducers.
    num_steps_between_results: Python `int`. Number of steps to take between
      MCMC results. This acts as a multiplier on the total number of steps taken
      by the MCMC (burnin included). The size of the output trace tensors is not
      affected, but each element is produced by this many sub-steps.
    init_state: Structure of `Tensor`s. Initial state of the chain. Default:
      `num_chains` worth of zeros in unconstrained space.
    init_step_size: Scalar float `Tensor`. Initial step size. Default: `1e-2 *
      total_num_dims ** -0.25`,
    event_space_bijector: Bijector or a list of bijectors used to go from
      unconstrained to constrained space to improve MCMC mixing. Default: Either
      inferred from `model` or an identity.
    event_dtype: Structure of dtypes. The event dtype. Default: Inferred from
      `model` or `init_state`.
    event_shape: Structure of tuples. The event shape. Default: Inferred from
      `model` or `init_state`.
    experimental_shard_axis_names: A structure of string names indicating how
      members of the state are sharded.
    experimental_reduce_chain_axis_names: A string or list of string names
      indicating which named axes to average cross-chain statistics over.
    dual_averaging_kwargs: Keyword arguments passed into
      `DualAveragingStepSizeAdaptation` kernel. Default: `{'target_accept_prob':
      0.8}`.
    snaper_kwargs: Keyword arguments passed into `SNAPERHamiltonianMonteCarlo`
      kernel. Default: `{}`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    validate_args: Python `bool`. When `True`, kernel parameters are checked
      for validity. When `False`, invalid inputs may silently render incorrect
      outputs.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
    results: `SampleSNAPERHamiltonianMonteCarloResults`.

  #### Tuning

  The defaults for this function should function well for many models, but it
  does provide a number of arguments for verifying sampler behavior. If there's
  a question of efficiency, the first thing to do is to set
  `discard_burnin_steps=False` and examine the `step_size` and
  `max_trajectory_length` and `variance_scaling` traces. A well-functioning
  sampler will have these quantities converge before sampling begins. If they
  are not converged, consider increasing `num_burnin_steps`, or adjusting the
  `snaper_kwargs` to tune SNAPER more.

  #### Examples

  Here we sample from a simple model while performing a reduction.

  ```
  num_dims = 8

  eigenvalues = np.exp(np.linspace(0., 3., num_dims))
  q, r = np.linalg.qr(np.random.randn(num_dims, num_dims))
  q *= np.sign(np.diag(r))
  covariance = (q * eigenvalues).dot(q.T).astype(self.dtype)

  gaussian = tfd.MultivariateNormalTriL(
      loc=tf.zeros(num_dims, self.dtype),
      scale_tril=tf.linalg.cholesky(covariance),
  )

  @tf.function(jit_compile=True)
  def run():
    results = tfp.experimental.mcmc.sample_snaper_hmc(
        model=gaussian,
        num_results=500,
        reducer=tfp.experimental.mcmc.PotentialScaleReductionReducer(),
    )

    return results.trace, results.reduction_results

  (chain, trace), potential_scale_reduction = run(tfp.random.sanitize_seed(0))

  # Compute sampler diagnostics.

  # Should be high (at least 100-1000).
  tfp.mcmc.effective_sample_size(chain, cross_chain_dims=1)
  # Should be close to 1.
  potential_scale_reduction

  # Compute downstream statistics.

  # Should be close to np.diag(covariance)
  tf.math.reduce_variance(chain, [0, 1])
  ```

  #### References

  [1]: Sountsov, P. & Hoffman, M. (2021). Focusing on Difficult Directions for
       Learning HMC Trajectory Lengths. <https://arxiv.org/abs/2110.11576>

  [2]: Kumar, R., Carroll, C., Hartikainen, A., & Martin, O. (2019). ArviZ a
       unified library for exploratory analysis of Bayesian models in Python.
       Journal of Open Source Software, 4(33), 1143.

  """
  with tf.name_scope(name):
    (unconstrained_state, flat_event_space_bijector,
     shard_axis_names_parts) = _init_chain_state(
         model,
         event_space_bijector=event_space_bijector,
         num_chains=num_chains,
         event_dtype=event_dtype,
         event_shape=event_shape,
         init_state=init_state,
         experimental_shard_axis_names=experimental_shard_axis_names,
     )

    if num_adaptation_steps is None:
      num_adaptation_steps = int(0.9 * num_burnin_steps)

    kernel, get_inner_results = _make_snaper_kernel(
        model=model,
        unconstrained_state=unconstrained_state,
        flat_event_space_bijector=flat_event_space_bijector,
        init_step_size=init_step_size,
        num_burnin_steps=num_burnin_steps,
        num_adaptation_steps=num_adaptation_steps,
        num_steps_between_results=num_steps_between_results,
        shard_axis_names_parts=shard_axis_names_parts,
        experimental_reduce_chain_axis_names=experimental_reduce_chain_axis_names,
        validate_args=validate_args,
        snaper_kwargs=snaper_kwargs,
        dual_averaging_kwargs=dual_averaging_kwargs,
        reducer=reducer,
        discard_burnin_steps=discard_burnin_steps,
    )

    unconstrained_state, kernel_results, trace = _sample_snaper_loop(
        unconstrained_state=unconstrained_state,
        kernel=kernel,
        reducer=reducer,
        flat_event_space_bijector=flat_event_space_bijector,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        discard_burnin_steps=discard_burnin_steps,
        get_inner_results=get_inner_results,
        trace_fn=trace_fn,
        seed=seed,
    )

    if reducer is None:
      reduction_results = None
    else:
      reduction_results = nest.map_structure_up_to(
          reducer, lambda r, rs: r.finalize(rs), reducer,
          unnest.get_innermost(kernel_results, 'reduction_results'))

    final_state = flat_event_space_bijector(unconstrained_state)

    return SampleSNAPERHamiltonianMonteCarloResults(
        trace=trace,
        reduction_results=reduction_results,
        final_state=final_state,
        final_kernel_results=get_inner_results(kernel_results),
    )


class _SNAPERReducer(reducer_lib.Reducer):
  """A Reducer utility wrapper for `snaper_hmc`.

  This does two things:
  - Pre-transforms the chain state.
  - Prevents reduction before num_burnin_steps.
  """

  def __init__(self, reducer, num_burnin_steps, flat_event_space_bijector):
    self._reducer = reducer
    self._num_burnin_steps = num_burnin_steps
    self._flat_event_space_bijector = flat_event_space_bijector

  def initialize(self, initial_chain_state, initial_inner_kernel_results):
    initial_chain_state = self._flat_event_space_bijector(initial_chain_state)
    return tf.nest.map_structure(lambda r: r.initialize(initial_chain_state),
                                 self._reducer)

  def one_step(self, new_chain_state, current_reducer_state,
               previous_kernel_results):
    new_chain_state = self._flat_event_space_bijector(new_chain_state)
    new_reducer_state = nest.map_structure_up_to(
        self._reducer,
        lambda r, rs: r.one_step(new_chain_state, rs, previous_kernel_results),
        self._reducer, current_reducer_state)
    return mcmc_util.choose(
        previous_kernel_results.step > self._num_burnin_steps,
        new_reducer_state, current_reducer_state)

  def finalize(self, final_reducer_state):
    return nest.map_structure_up_to(self._reducer, lambda r, rs: r.finalize(rs),
                                    self._reducer, final_reducer_state)


def _dot_product(x, y, axis, named_axis):

  def _dot_product_part(x, y, axis, named_axis):
    return distribute_lib.reduce_sum(x * y, axis, named_axis)

  dot_products = nest.map_structure_up_to(x, _dot_product_part, x, y, axis,
                                          named_axis)
  return sum(tf.nest.flatten(dot_products))


def _normalize(x, axis, named_axis):
  norm = tf.sqrt(_dot_product(x, x, axis, named_axis)) + 1e-20
  return tf.nest.map_structure(
      lambda x: x / bu.left_justified_expand_dims_like(norm, x), x)
