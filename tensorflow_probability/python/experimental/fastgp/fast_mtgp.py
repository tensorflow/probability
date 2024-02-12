# Copyright 2024 The TensorFlow Probability Authors.
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
"""Fast likelihoods etc. for Multi-task Gaussian Processeses."""

import jax
import jax.numpy as jnp
from tensorflow_probability.python.experimental.fastgp import fast_gp
from tensorflow_probability.python.experimental.fastgp import fast_log_det
from tensorflow_probability.python.experimental.fastgp import linear_operator_sum
from tensorflow_probability.python.experimental.fastgp import preconditioners
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal.backend import jax as tf2jax
from tensorflow_probability.substrates.jax.distributions import distribution
from tensorflow_probability.substrates.jax.distributions.internal import stochastic_process_util
from tensorflow_probability.substrates.jax.experimental.psd_kernels import multitask_kernel
from tensorflow_probability.substrates.jax.internal import dtype_util
from tensorflow_probability.substrates.jax.internal import prefer_static as ps
from tensorflow_probability.substrates.jax.internal import tensor_util


Array = jnp.ndarray


def _vec(x):
  # Vec takes in a (batch) of matrices of shape B1 + [n, k] and returns
  # a (batch) of vectors of shape B1 + [n * k].
  return jnp.reshape(x, ps.concat([ps.shape(x)[:-2], [-1]], axis=0))


def _unvec(x, matrix_shape):
  # Unvec takes in a (batch) of matrices of shape B1 + [n * k] and returns
  # a (batch) of vectors of shape B1 + [n, k], where n and k are specified
  # by matrix_shape.
  return jnp.reshape(x, ps.concat([ps.shape(x)[:-1], matrix_shape], axis=0))


class MultiTaskGaussianProcess(distribution.AutoCompositeTensorDistribution):
  """Fast, JAX-only implementation of a MTGP distribution class.

  See tfp.experimental.distributions.MultiTaskGaussianProcess for a description
  and parameter documentation.
  """

  def __init__(
      self,
      kernel,
      index_points=None,
      mean_fn=None,
      observation_noise_variance=0.0,
      config=fast_gp.GaussianProcessConfig(),
      validate_args=False,
      allow_nan_stats=False):
    """Instantiate a fast GaussianProcess distribution.

    Args:
      kernel: A `PositiveSemidefiniteKernel`-like instance representing the GP's
        covariance function.
      index_points: Tensor specifying the points over which the GP is defined.
      mean_fn: Python callable that acts on index_points.  Default `None`
        implies a constant zero mean function.
      observation_noise_variance: `float` `Tensor` representing the scalar
        variance of the noise in the Normal likelihood distribution of the
        model.
      config: `GaussianProcessConfig` to control speed and quality of GP
        approximations.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs. Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined. Default value: `False`.

    """
    parameters = dict(locals())
    if jax.tree_util.treedef_is_leaf(
        jax.tree_util.tree_structure(kernel.feature_ndims)):
      # If the index points are not nested, we assume they are of the same
      # float dtype as the GP.
      dtype = dtype_util.common_dtype(
          {
              'index_points': index_points,
              'observation_noise_variance': observation_noise_variance,
          },
          jnp.float32,
      )
    else:
      dtype = dtype_util.common_dtype(
          {'observation_noise_variance': observation_noise_variance},
          jnp.float32,
      )

    self._kernel = kernel
    self._index_points = index_points
    self._mean_fn = stochastic_process_util.maybe_create_multitask_mean_fn(
        mean_fn, kernel, dtype
    )
    self._observation_noise_variance = tensor_util.convert_nonref_to_tensor(
        observation_noise_variance
    )
    self._config = config
    self._probe_vector_type = fast_log_det.ProbeVectorType[
        config.probe_vector_type.upper()]
    self._log_det_fn = fast_log_det.get_log_det_algorithm(
        config.log_det_algorithm)

    super(MultiTaskGaussianProcess, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name='MultiTaskGaussianProcess',
    )

  @property
  def kernel(self):
    return self._kernel

  @property
  def index_points(self):
    return self._index_points

  @property
  def mean_fn(self):
    return self._mean_fn

  @property
  def observation_noise_variance(self):
    return self._observation_noise_variance

  @property
  def event_shape(self):
    return stochastic_process_util.multitask_event_shape(
        self._kernel, self.index_points
    )

  def _mean(self):
    loc = self._mean_fn(self._index_points)
    return jnp.broadcast_to(loc, self.event_shape)

  def _variance(self):
    index_points = self._index_points
    kernel_matrix = self.kernel.matrix_over_all_tasks(
        index_points, index_points)
    observation_noise_variance = self.observation_noise_variance
    # We can add the observation noise to each block.
    if isinstance(self.kernel, multitask_kernel.Independent):
      single_task_variance = kernel_matrix.operators[0].diag_part()
      if observation_noise_variance is not None:
        single_task_variance = (
            single_task_variance + observation_noise_variance[..., jnp.newaxis])
      # Each task has the same variance, so shape this in to an `[..., e, t]`
      # shaped tensor and broadcast to batch shape
      variance = jnp.stack(
          [single_task_variance] * self.kernel.num_tasks, axis=-1)
      return variance

    # If `kernel_matrix` has structure, `diag_part` will try to take advantage
    # of that structure. In the case of a `Separable` kernel, `diag_part` will
    # efficiently compute the diagonal of a kronecker product.
    variance = kernel_matrix.diag_part()
    if observation_noise_variance is not None:
      variance = (
          variance +
          observation_noise_variance[..., jnp.newaxis])

    variance = _unvec(variance, [-1, self.kernel.num_tasks])

    # Finally broadcast with batch shape.
    batch_shape = self._batch_shape_tensor(index_points=index_points)
    event_shape = self._event_shape_tensor(index_points=index_points)

    variance = jnp.broadcast_to(
        variance, ps.concat([batch_shape, event_shape], axis=0))
    return variance

  @jax.named_call
  def log_prob(self, value, key) -> Array:
    """log P(value | GP)."""
    empty_sample_batch_shape = value.ndim == 2
    if empty_sample_batch_shape:
      value = value[jnp.newaxis]
    if value.ndim != 3:
      raise ValueError(
          'fast_mtgp.MultiTaskGaussianProcess.log_prob only supports values '
          f'of rank 2 or 3, got rank {value.ndim} instead.'
      )
    index_points = self._index_points
    loc = self.mean()
    loc = _vec(loc)
    covariance = self.kernel.matrix_over_all_tasks(index_points, index_points)

    centered_value = _vec(value) - loc
    key1, key2 = jax.random.split(key)

    is_scaling_preconditioner = self._config.preconditioner.endswith('scaling')

    def get_preconditioner(cov):
      scaling = None
      if is_scaling_preconditioner:
        scaling = self.observation_noise_variance
      return preconditioners.get_preconditioner(
          self._config.preconditioner,
          cov,
          key=key1,
          rank=self._config.preconditioner_rank,
          num_iters=self._config.preconditioner_num_iters,
          scaling=scaling)

    if is_scaling_preconditioner:
      preconditioner = get_preconditioner(covariance)

    covariance = linear_operator_sum.LinearOperatorSum([
        covariance,
        tf2jax.linalg.LinearOperatorScaledIdentity(
            num_rows=covariance.range_dimension,
            multiplier=self._observation_noise_variance,
        ),
    ])

    if not is_scaling_preconditioner:
      preconditioner = get_preconditioner(covariance)

    # TODO(srvasude): Specialize for Independent and Separable kernels.
    # In particular, we should be able to take advantage of the kronecker
    # structure, and construct a kronecker preconditioner.

    det_term = self._log_det_fn(
        covariance,
        preconditioner,
        key=key2,
        num_probe_vectors=self._config.num_probe_vectors,
        probe_vector_type=self._probe_vector_type,
        num_iters=self._config.log_det_iters,
    )

    exp_term = fast_gp.yt_inv_y(
        covariance,
        preconditioner.full_preconditioner(),
        jnp.transpose(centered_value),
        max_iters=self._config.cg_iters
    )

    lp = -0.5 * (
        fast_gp.LOG_TWO_PI * value.shape[-1] * value.shape[-2] +
        det_term + exp_term)
    if empty_sample_batch_shape:
      return jnp.squeeze(lp, axis=0)

    return lp
