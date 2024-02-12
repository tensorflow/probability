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
"""Fast GPRM."""

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.experimental.fastgp import fast_gp
from tensorflow_probability.python.experimental.fastgp import mbcg
from tensorflow_probability.python.experimental.fastgp import preconditioners
from tensorflow_probability.python.experimental.fastgp import schur_complement
from tensorflow_probability.substrates.jax.bijectors import softplus
from tensorflow_probability.substrates.jax.distributions.internal import stochastic_process_util
from tensorflow_probability.substrates.jax.internal import dtype_util
from tensorflow_probability.substrates.jax.internal import nest_util
from tensorflow_probability.substrates.jax.internal import parameter_properties

__all__ = [
    'GaussianProcessRegressionModel',
]


class GaussianProcessRegressionModel(fast_gp.GaussianProcess):
  """Fast, JAX-only implementation of a GP distribution class.

   See tfd.distributions.GaussianProcessRegressionModel for a description and
   parameter documentation. Note: We assume that the observation index points
   and observations are fixed, and so precompute quantities associated with
   them.
  """

  def __init__(
      self,
      kernel,
      key: jax.random.PRNGKey,
      index_points=None,
      observation_index_points=None,
      observations=None,
      observation_noise_variance=0.0,
      predictive_noise_variance=None,
      mean_fn=None,
      jitter=1e-6,
      config=fast_gp.GaussianProcessConfig(),
  ):
    """Instantiate a fast GaussianProcessRegressionModel instance.

    Args:
      kernel: `PositiveSemidefiniteKernel`-like instance representing the GP's
        covariance function.
      key: `jax.random.PRNGKey` to use when computing the preconditioner.
      index_points: (nested) `Tensor` representing finite collection, or batch
        of collections, of points in the index set over which the GP is defined.
        Shape (of each nested component) has the form `[b1, ..., bB, e, f1, ...,
        fF]` where `F` is the number of feature dimensions and must equal
        `kernel.feature_ndims` (or its corresponding nested component) and `e`
        is the number (size) of index points in each batch. Ultimately this
        distribution corresponds to an `e`-dimensional multivariate normal. The
        batch shape must be broadcastable with `kernel.batch_shape` and any
        batch dims yielded by `mean_fn`.
      observation_index_points: (nested) `Tensor` representing finite
        collection, or batch of collections, of points in the index set for
        which some data has been observed. Shape (of each nested component) has
        the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of
        feature dimensions and must equal `kernel.feature_ndims` (or its
        corresponding nested component), and `e` is the number (size) of index
        points in each batch. `[b1, ..., bB, e]` must be broadcastable with the
        shape of `observations`, and `[b1, ..., bB]` must be broadcastable with
        the shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc). The default value is `None`, which corresponds to
        the empty set of observations, and simply results in the prior
        predictive model (a GP with noise of variance
        `predictive_noise_variance`).
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
        must be brodcastable with the batch and example shapes of
        `observation_index_points`. The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc.). The default value is
        `None`, which corresponds to the empty set of observations, and simply
        results in the prior predictive model (a GP with noise of variance
        `predictive_noise_variance`).
      observation_noise_variance: `float` `Tensor` representing the variance of
        the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.). Default value: `0.`
      predictive_noise_variance: `float` `Tensor` representing the variance in
        the posterior predictive model. If `None`, we simply re-use
        `observation_noise_variance` for the posterior predictive noise. If set
        explicitly, however, we use this value. This allows us, for example, to
        omit predictive noise variance (by setting this to zero) to obtain
        noiseless posterior predictions of function values, conditioned on noisy
        observations.
      mean_fn: Python `callable` that acts on `index_points` to produce a
        collection, or batch of collections, of mean values at `index_points`.
        Takes a (nested) `Tensor` of shape `[b1, ..., bB, e, f1, ..., fF]` and
        returns a `Tensor` whose shape is broadcastable with `[b1, ..., bB, e]`.
        Default value: `None` implies the constant zero function.
      jitter: `float` scalar `Tensor` that gets added to the diagonal of the
        GP's covariance matrix to ensure it is positive definite.
      config: `GaussianProcessConfig` to control speed and quality of GP
        approximations.

    Raises:
      ValueError: if either
        - only one of `observations` and `observation_index_points` is given, or
        - `mean_fn` is not `None` and not callable.
    """
    # TODO(srvasude): Add support for masking observations. In addition, cache
    # the observation matrix so that it isn't recomputed every iteration.
    parameters = dict(locals())
    input_dtype = dtype_util.common_dtype(
        dict(
            kernel=kernel,
            index_points=index_points,
            observation_index_points=observation_index_points,
        ),
        dtype_hint=nest_util.broadcast_structure(
            kernel.feature_ndims, np.float32
        ),
    )

    # If the input dtype is non-nested float, we infer a single dtype for the
    # input and the float parameters, which is also the dtype of the GP's
    # samples, log_prob, etc. If the input dtype is nested (or not float), we
    # do not use it to infer the GP's float dtype.
    if not jax.tree_util.treedef_is_leaf(
        jax.tree_util.tree_structure(input_dtype)
    ) and dtype_util.is_floating(input_dtype):
      dtype = dtype_util.common_dtype(
          dict(
              kernel=kernel,
              index_points=index_points,
              observations=observations,
              observation_index_points=observation_index_points,
              observation_noise_variance=observation_noise_variance,
              predictive_noise_variance=predictive_noise_variance,
              jitter=jitter,
          ),
          dtype_hint=np.float32,
      )
      input_dtype = dtype
    else:
      dtype = dtype_util.common_dtype(
          dict(
              observations=observations,
              observation_noise_variance=observation_noise_variance,
              predictive_noise_variance=predictive_noise_variance,
              jitter=jitter,
          ),
          dtype_hint=np.float32,
      )

    if predictive_noise_variance is None:
      predictive_noise_variance = observation_noise_variance
    if (observation_index_points is None) != (observations is None):
      raise ValueError(
          '`observations` and `observation_index_points` must both be given '
          'or None. Got {} and {}, respectively.'.format(
              observations, observation_index_points))
    # Default to a constant zero function, borrowing the dtype from
    # index_points to ensure consistency.
    mean_fn = stochastic_process_util.maybe_create_mean_fn(mean_fn, dtype)

    self._observation_index_points = observation_index_points
    self._observations = observations
    self._observation_noise_variance = observation_noise_variance
    self._predictive_noise_variance = predictive_noise_variance
    self._jitter = jitter

    covariance = kernel.matrix(
        observation_index_points, observation_index_points)

    is_scaling_preconditioner = config.preconditioner.endswith('scaling')

    def get_preconditioner(cov):
      scaling = None
      if is_scaling_preconditioner:
        scaling = self._observation_noise_variance + self._jitter
      return preconditioners.get_preconditioner(
          config.preconditioner,
          cov,
          key=key,
          rank=config.preconditioner_rank,
          num_iters=config.preconditioner_num_iters,
          scaling=scaling)

    if is_scaling_preconditioner:
      schur_preconditioner = get_preconditioner(covariance)

    updated_diagonal = jnp.diag(covariance) + (
        self._observation_noise_variance + self._jitter)

    covariance = (covariance * (
        1 - jnp.eye(
            updated_diagonal.shape[-1], dtype=updated_diagonal.dtype)
    ) + jnp.diag(updated_diagonal))

    if not is_scaling_preconditioner:
      schur_preconditioner = get_preconditioner(covariance)

    conditional_kernel = schur_complement.SchurComplement(
        base_kernel=kernel,
        preconditioner_fn=schur_preconditioner.full_preconditioner().solve,
        fixed_inputs=observation_index_points,
        diag_shift=self._observation_noise_variance + self._jitter)

    def conditional_mean_fn(x):
      """Conditional mean."""
      k_x_obs = kernel.matrix(x, observation_index_points)
      diff = observations - mean_fn(observation_index_points)
      k_obs_inv_diff, _ = mbcg.modified_batched_conjugate_gradients(
          lambda x: covariance @ x,
          diff[..., jnp.newaxis],
          preconditioner_fn=schur_preconditioner.full_preconditioner().solve,
          max_iters=config.cg_iters,
      )

      return mean_fn(x) + jnp.squeeze(k_x_obs @ k_obs_inv_diff, axis=-1)

    # Special logic for mean_fn only; SchurComplement already handles the
    # case of empty observations (ie, falls back to base_kernel).
    if not stochastic_process_util.is_empty_observation_data(
        feature_ndims=kernel.feature_ndims,
        observation_index_points=observation_index_points,
        observations=observations,
    ):
      stochastic_process_util.validate_observation_data(
          kernel=kernel,
          observation_index_points=observation_index_points,
          observations=observations,
      )

    super(GaussianProcessRegressionModel, self).__init__(
        index_points=index_points,
        jitter=jitter,
        kernel=conditional_kernel,
        mean_fn=conditional_mean_fn,
        # What the GP super class calls "observation noise variance" we call
        # here the "predictive noise variance". We use the observation noise
        # variance for the fit/solve process above, and predictive for
        # downstream computations like sampling.
        observation_noise_variance=predictive_noise_variance,
        config=config,
    )
    self._parameters = parameters

  @property
  def observation_index_points(self):
    return self._observation_index_points

  @property
  def observations(self):
    return self._observations

  @property
  def predictive_noise_variance(self):
    return self._predictive_noise_variance

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    def _event_ndims_fn(self):
      return jax.tree_util.treep_map(
          lambda nd: nd + 1, self.kernel.feature_ndims)
    return dict(
        index_points=parameter_properties.ParameterProperties(
            event_ndims=_event_ndims_fn,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        observations=parameter_properties.ParameterProperties(
            event_ndims=1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        observation_index_points=parameter_properties.ParameterProperties(
            event_ndims=_event_ndims_fn,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        observations_is_missing=parameter_properties.ParameterProperties(
            event_ndims=1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        kernel=parameter_properties.BatchedComponentProperties(),
        observation_noise_variance=parameter_properties.ParameterProperties(
            event_ndims=0,
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(  # pylint:disable=g-long-lambda
                    low=dtype_util.eps(dtype)
                )
            ),
        ),
        predictive_noise_variance=parameter_properties.ParameterProperties(
            event_ndims=0,
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(  # pylint:disable=g-long-lambda
                    low=dtype_util.eps(dtype)
                )
            ),
        ),
    )
