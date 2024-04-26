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
"""Fast likelihoods etc. for Gaussian Processes.

It's recommended to use `GaussianProcess` in `float64` mode only.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from tensorflow_probability.python.experimental.fastgp import fast_log_det
from tensorflow_probability.python.experimental.fastgp import mbcg
from tensorflow_probability.python.experimental.fastgp import preconditioners
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal.backend import jax as tf2jax
from tensorflow_probability.substrates.jax.bijectors import softplus
from tensorflow_probability.substrates.jax.distributions import distribution
from tensorflow_probability.substrates.jax.distributions import gaussian_process_regression_model
from tensorflow_probability.substrates.jax.distributions.internal import stochastic_process_util
from tensorflow_probability.substrates.jax.internal import dtype_util
from tensorflow_probability.substrates.jax.internal import parameter_properties
from tensorflow_probability.substrates.jax.internal import tensor_util

Array = jnp.ndarray

LOG_TWO_PI = 1.8378770664093453


@dataclasses.dataclass
class GaussianProcessConfig:
  """Configuration for distributions in the FastGP family."""

  # The maximum number of iterations to run conjugate gradients
  # for when calculating the yt_inv_y part of the log prob.
  cg_iters: int = 25
  # The name of a preconditioner in the preconditioner.PRECONDITIONER_REGISTRY
  # or 'auto' which will used truncated_randomized_svd_plus_scaling when n is
  # large and partial_cholesky_split when is small.
  preconditioner: str = 'auto'
  # Use a preconditioner based on a low rank
  # approximation of this rank.  Note that not all preconditioners have
  # adjustable ranks.
  preconditioner_rank: int = 25
  # Some preconditioners (like `truncated_svd`) can
  # get better approximation accuracy for running for more iterations (even
  # at a fixed rank size).  This parameter controls that.  Note that the
  # cost of computing the preconditioner can be as much as
  # O(preconditioner_rank * preconditioner_num_iters * n^2)
  preconditioner_num_iters: int = 5
  # If "true", compute the preconditioner before any diagonal terms are added
  # to the covariance.  If "false", compute the preconditioner on the sum of
  # the original covariance plus the diagonal terms.  If "auto", compute the
  # preconditioner before the diagonal terms for scaling preconditioners,
  # and after the diagonal terms for all other preconditioners.
  precondition_before_jitter: str = 'auto'
  # Either `normal`, `normal_qmc`, `normal_orthogonal` or
  # `rademacher`.  `normal_qmc` is only valid for n <= 10000
  probe_vector_type: str = 'rademacher'
  # The number of probe vectors to use when estimating the log det.
  num_probe_vectors: int = 35
  # One of 'slq' (for stochastic Lanczos quadrature) or
  # 'r1', 'r2', 'r3', 'r4', 'r5', or 'r6' for the rational function
  # approximation of the given order.
  log_det_algorithm: str = 'r3'
  # The number of iterations to use when doing solves inside the log det
  # algorithm.
  log_det_iters: int = 20


class GaussianProcess(distribution.Distribution):
  """Fast, JAX-only implementation of a GP distribution class.

  See tfd.distributions.GaussianProcess for a description and parameter
  documentation, but note that not all of that class's methods are supported.

  The default parameters are tuned to give a good time / error trade-off
  in the n > 15,000 regime where this class gives a substantial speed-up
  over tfd.distributions.GaussianProcess.
  """

  def __init__(
      self,
      kernel,
      index_points=None,
      mean_fn=None,
      observation_noise_variance=0.0,
      jitter=1e-6,
      config=GaussianProcessConfig(),
      validate_args=False,
      allow_nan_stats=False,
  ):
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
      jitter: `float` scalar `Tensor` that gets added to the diagonal of the
        GP's covariance matrix to ensure it is positive definite.
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
              'jitter': jitter,
          },
          jnp.float32,
      )
    else:
      dtype = dtype_util.common_dtype(
          {
              'observation_noise_variance': observation_noise_variance,
              'jitter': jitter,
          },
          jnp.float32,
      )

    self._kernel = kernel
    self._index_points = index_points
    self._mean_fn = stochastic_process_util.maybe_create_mean_fn(mean_fn, dtype)
    self._observation_noise_variance = tensor_util.convert_nonref_to_tensor(
        observation_noise_variance
    )
    self._jitter = jitter
    self._config = config
    self._probe_vector_type = fast_log_det.ProbeVectorType[
        config.probe_vector_type.upper()]
    self._log_det_fn = fast_log_det.get_log_det_algorithm(
        config.log_det_algorithm
    )

    super(GaussianProcess, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name='GaussianProcess',
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
  def jitter(self):
    return self._jitter

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        index_points=parameter_properties.ParameterProperties(
            event_ndims=lambda self: jax.tree_util.tree_map(  # pylint: disable=g-long-lambda
                lambda nd: nd + 1, self.kernel.feature_ndims
            ),
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        kernel=parameter_properties.BatchedComponentProperties(),
        observation_noise_variance=(
            parameter_properties.ParameterProperties(
                event_ndims=0,
                shape_fn=lambda sample_shape: sample_shape[:-1],
                default_constraining_bijector_fn=(
                    lambda: softplus.Softplus(  # pylint: disable=g-long-lambda
                        low=dtype_util.eps(dtype)
                    )
                ),
            )
        ),
    )

  @property
  def event_shape(self):
    return stochastic_process_util.event_shape(self._kernel, self.index_points)

  def _mean(self):
    mean = self._mean_fn(self._index_points)
    mean = jnp.broadcast_to(mean, self.event_shape)
    return mean

  def _covariance(self):
    index_points = self._index_points
    _, covariance = stochastic_process_util.get_loc_and_kernel_matrix(
        kernel=self.kernel,
        mean_fn=self._mean_fn,
        observation_noise_variance=self.observation_noise_variance,
        index_points=index_points,
    )
    return covariance

  def _variance(self):
    index_points = self._index_points
    kernel_diag = self.kernel.apply(index_points, index_points, example_ndims=1)
    return kernel_diag + self.observation_noise_variance[..., jnp.newaxis]

  def _log_det(self, key, is_missing=None):
    """Returns log_det, loc, covariance and preconditioner."""
    key1, key2 = jax.random.split(key)

    # TODO(thomaswc): Considering caching loc and covariance for a given
    # is_missing.
    loc, covariance = stochastic_process_util.get_loc_and_kernel_matrix(
        kernel=self._kernel,
        mean_fn=self._mean_fn,
        observation_noise_variance=self.dtype(0.0),
        index_points=self._index_points,
        is_missing=is_missing,
        mask_loc=False,
    )

    is_scaling_preconditioner = preconditioners.resolve_preconditioner(
        self._config.preconditioner, covariance,
        self._config.preconditioner_rank).endswith('scaling')
    precondition_before_jitter = (
        self._config.precondition_before_jitter == 'true'
        or (
            self._config.precondition_before_jitter == 'auto'
            and is_scaling_preconditioner
        )
    )

    def get_preconditioner(cov):
      scaling = None
      if is_scaling_preconditioner:
        scaling = self._observation_noise_variance + self._jitter
      return preconditioners.get_preconditioner(
          self._config.preconditioner,
          cov,
          key=key1,
          rank=self._config.preconditioner_rank,
          num_iters=self._config.preconditioner_num_iters,
          scaling=scaling)

    if precondition_before_jitter:
      preconditioner = get_preconditioner(covariance)

    updated_diagonal = (
        jnp.diag(covariance) + self._jitter + self.observation_noise_variance)
    if is_missing is not None:
      updated_diagonal = jnp.where(
          is_missing, self.dtype(1. + self._jitter), updated_diagonal)

    covariance = (covariance * (
        1 - jnp.eye(
            updated_diagonal.shape[-1], dtype=updated_diagonal.dtype)
    ) + jnp.diag(updated_diagonal))

    if not precondition_before_jitter:
      preconditioner = get_preconditioner(covariance)

    det_term = self._log_det_fn(
        covariance,
        preconditioner,
        key=key2,
        num_probe_vectors=self._config.num_probe_vectors,
        probe_vector_type=self._probe_vector_type,
        num_iters=self._config.log_det_iters,
    )

    return det_term, loc, covariance, preconditioner

  @jax.named_call
  def log_prob(self, value, key, is_missing=None) -> Array:
    """log P(value | GP).

    Args:
      value: `float` or `double` jax.Array.
      key: A jax KeyArray.  This method uses stochastic methods to quickly
        estimate the log probability of `value`, and `key` is needed to
        generate the stochasticity.  `key` is also used when computing the
        derivative of this function.  In some circumstances it is acceptable
        and in fact even necessary to pass the same value of `key` to multiple
        invocations of log_prob; for example if the log_prob is being
        optimized by an algorithm that assumes it is deterministic.
      is_missing: Optional `bool` jax.Array of shape `[..., e]` where `e` is
        the number of index points in each batch.  Represents a batch of
        Boolean masks.  When not `None`, the returned log_prob is for the
        *marginal* distribution in which all dimensions with `is_missing==True`
        have been marginalized out.

    Returns:
      A stochastic approximation to log P(value | GP).
    """
    empty_sample_batch_shape = value.ndim == 1
    if empty_sample_batch_shape:
      value = value[jnp.newaxis]
    if value.ndim != 2:
      raise ValueError(
          'fast_gp.GaussianProcess.log_prob only supports values of rank 1 or '
          f'2, got rank {value.ndim} instead.'
      )

    num_unmasked_dims = value.shape[-1]

    det_term, loc, covariance, preconditioner = self._log_det(key, is_missing)

    centered_value = value - loc
    if is_missing is not None:
      centered_value = jnp.where(is_missing, 0.0, centered_value)
      num_unmasked_dims = num_unmasked_dims - jnp.count_nonzero(
          is_missing, axis=-1
      )

    exp_term = yt_inv_y(
        covariance,
        preconditioner.full_preconditioner(),
        jnp.transpose(centered_value),
        max_iters=self._config.cg_iters,
    )

    lp = -0.5 * (LOG_TWO_PI * num_unmasked_dims + det_term + exp_term)
    if empty_sample_batch_shape:
      return jnp.squeeze(lp, axis=0)

    return lp

  def posterior_predictive(
      self,
      observations,
      predictive_index_points=None,
      observations_is_missing=None,
      **kwargs
  ):
    # TODO(thomaswc): Speed this up, if possible.
    return gaussian_process_regression_model.GaussianProcessRegressionModel.precompute_regression_model(
        kernel=self._kernel,
        observation_index_points=self._index_points,
        observations=observations,
        observations_is_missing=observations_is_missing,
        index_points=predictive_index_points,
        observation_noise_variance=self._observation_noise_variance,
        mean_fn=self._mean_fn,
        cholesky_fn=None,
        jitter=self._jitter,
        **kwargs,
    )


# pylint: disable=invalid-name


@functools.partial(jax.custom_jvp, nondiff_argnums=(3,))
def yt_inv_y(
    kernel: tf2jax.linalg.LinearOperator,
    preconditioner: tf2jax.linalg.LinearOperator,
    y: Array,
    max_iters: int = 20,
) -> Array:
  """Compute y^t (kernel)^(-1) y.

  Args:
    kernel: A matrix or linalg.LinearOperator representing a linear map from R^n
      to itself.
    preconditioner: An operator on R^n that when applied before kernel, reduces
      the condition number of the system.
    y: A matrix of shape (n, m).
    max_iters: The maximum number of iterations to perform the modified batched
      conjugate gradients algorithm for.

  Returns:
    y's inner product with itself, with respect to the inverse of the kernel.
  """
  def multiplier(B):
    return kernel @ B

  inv_y, _ = mbcg.modified_batched_conjugate_gradients(
      multiplier, y, preconditioner.solve, max_iters
  )
  return jnp.einsum('ij,ij->j', y, inv_y)


@yt_inv_y.defjvp
def yt_inv_y_jvp(max_iters, primals, tangents):
  """Jacobian-Vector product for yt_inv_y."""
  # According to 2.3.3 of
  # https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf,
  # d(y^t A^(-1) y) = dy^t A^(-1) y - y^t A^(-1) dA A^(-1) y + y^t A^(-1) dy
  kernel = primals[0]
  preconditioner = primals[1]
  y = primals[2]

  dkernel = tangents[0]
  dy = tangents[2]

  def multiplier(B):
    return kernel @ B

  inv_y, _ = mbcg.modified_batched_conjugate_gradients(
      multiplier, y, preconditioner.solve, max_iters
  )

  primal_out = jnp.einsum('ij,ij->j', y, inv_y)
  tangent_out = 2.0 * jnp.einsum('ik,ik->k', inv_y, dy)

  if isinstance(dkernel, tf2jax.linalg.LinearOperator):
    tangent_out = tangent_out - jnp.einsum('ik,ik->k', inv_y, dkernel @ inv_y)
  else:
    tangent_out = tangent_out - jnp.einsum('ik,ij,jk->k', inv_y, dkernel, inv_y)

  return primal_out, tangent_out
