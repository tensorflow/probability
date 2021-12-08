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

from tensorflow_probability.python.experimental.mcmc import gradient_based_trajectory_length_adaptation as gbtla
from tensorflow_probability.python.experimental.mcmc import preconditioned_hmc
from tensorflow_probability.python.experimental.mcmc import preconditioning_utils
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'SNAPERHamiltonianMonteCarlo',
    'SNAPERHamiltonianMonteCarloResults',
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
      local_axis_size = ps.maximum(ps.size(tlp), 1)
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
