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
"""The MultiTaskGaussianProcessRegressionModel distribution class."""

# Dependency imports

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import cholesky_util
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.experimental.distributions import multitask_gaussian_process as mtgp
from tensorflow_probability.python.experimental.linalg import linear_operator_unitary
from tensorflow_probability.python.experimental.psd_kernels import multitask_kernel
from tensorflow_probability.python.internal import batch_shape_lib
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.psd_kernels.internal import util as psd_kernels_util


def _vec(x):
  # Vec takes in a (batch) of matrices of shape B1 + [n, k] and returns
  # a (batch) of vectors of shape B1 + [n * k].
  return tf.reshape(x, ps.concat([ps.shape(x)[:-2], [-1]], axis=0))


def _unvec(x, matrix_shape):
  # Unvec takes in a (batch) of matrices of shape B1 + [n * k] and returns
  # a (batch) of vectors of shape B1 + [n, k], where n and k are specified
  # by matrix_shape.
  return tf.reshape(x, ps.concat([ps.shape(x)[:-1], matrix_shape], axis=0))


def _add_diagonal_shift(m, c):
  return tf.linalg.set_diag(m, tf.linalg.diag_part(m) + c[..., tf.newaxis])


def _flattened_conditional_mean_fn_helper(
    x,
    kernel,
    observations,
    observation_index_points,
    observations_is_missing,
    observation_scale,
    mean_fn,
    solve_on_observations=None):
  """Flattened Conditional mean helper."""
  observations = tf.convert_to_tensor(observations)
  observation_index_points = tf.convert_to_tensor(
      observation_index_points)

  k_x_obs_linop = kernel.matrix_over_all_tasks(x, observation_index_points)
  if solve_on_observations is None:
    vec_diff = _vec(observations - mean_fn(observation_index_points))

  if observations_is_missing is not None:
    k_x_obs_linop = tf.linalg.LinearOperatorFullMatrix(
        tf.where(_vec(observations_is_missing)[..., tf.newaxis, :],
                 tf.zeros([], dtype=k_x_obs_linop.dtype),
                 k_x_obs_linop.to_dense()))
    if solve_on_observations is None:
      vec_diff = tf.where(_vec(observations_is_missing),
                          tf.zeros([], dtype=vec_diff.dtype),
                          vec_diff)
  if solve_on_observations is None:
    solve_on_observations = observation_scale.solvevec(
        observation_scale.solvevec(vec_diff), adjoint=True)

  flattened_mean = k_x_obs_linop.matvec(solve_on_observations)
  return _vec(mean_fn(x) + _unvec(
      flattened_mean, [-1, kernel.num_tasks]))


def _compute_observation_scale(
    kernel,
    observation_index_points,
    cholesky_fn,
    observation_noise_variance=None,
    observations_is_missing=None):
  """Compute matrix square root of the kernel on observation index points."""
  if observations_is_missing is not None:
    observations_is_missing = tf.convert_to_tensor(observations_is_missing)
    # If observations are missing, there's nothing we can do to preserve the
    # operator structure, so densify.

    observation_covariance = kernel.matrix_over_all_tasks(
        observation_index_points, observation_index_points).to_dense()

    if observation_noise_variance is not None:
      broadcast_shape = distribution_util.get_broadcast_shape(
          observation_covariance,
          observation_noise_variance[..., tf.newaxis, tf.newaxis])
      observation_covariance = tf.broadcast_to(
          observation_covariance, broadcast_shape)
      observation_covariance = _add_diagonal_shift(
          observation_covariance, observation_noise_variance)
    vec_observations_is_missing = _vec(observations_is_missing)
    observation_covariance = tf.linalg.LinearOperatorFullMatrix(
        psd_kernels_util.mask_matrix(
            observation_covariance,
            is_missing=vec_observations_is_missing),
        is_non_singular=True,
        is_positive_definite=True)
    observation_scale = cholesky_util.cholesky_from_fn(
        observation_covariance, cholesky_fn)
  else:
    observation_scale = mtgp._compute_flattened_scale(  # pylint:disable=protected-access
        kernel=kernel,
        index_points=observation_index_points,
        cholesky_fn=cholesky_fn,
        observation_noise_variance=observation_noise_variance)

  return observation_scale


def _scale_from_precomputed(precomputed_cholesky, kernel):
  """Rebuilds `observation_scale` from precomputed values."""
  params, = tuple(precomputed_cholesky.values())
  if 'tril' in precomputed_cholesky:
    return tf.linalg.LinearOperatorLowerTriangular(
        params['chol_tril'], is_non_singular=True)
  if 'multitask' in precomputed_cholesky:
    return tf.linalg.LinearOperatorKronecker(
        [tf.linalg.LinearOperatorLowerTriangular(
            params['chol_tril'], is_non_singular=True),
         tf.linalg.LinearOperatorIdentity(
             kernel.num_tasks, dtype=kernel.dtype)],
        is_square=True,
        is_non_singular=True)
  if 'separable' in precomputed_cholesky:
    diag_op = tf.linalg.LinearOperatorDiag(
        params['diag'],
        is_square=True,
        is_non_singular=True,
        is_positive_definite=True)
    orthogonal_op = tf.linalg.LinearOperatorKronecker(
        [linear_operator_unitary.LinearOperatorUnitary(orth)
         for orth in params['kronecker_orths']],
        is_square=True, is_non_singular=True)
    return orthogonal_op.matmul(diag_op)
  # This should not happen.
  raise ValueError(
      f'Unexpected value for `precompute_cholesky`: {precomputed_cholesky}.')


def _precomputed_from_scale(observation_scale, kernel):
  """Extracts expensive precomputed values."""
  if isinstance(observation_scale, tf.linalg.LinearOperatorLowerTriangular):
    return {'tril': {'chol_tril': observation_scale.tril}}
  if isinstance(kernel, multitask_kernel.Independent):
    base_kernel_chol_op = observation_scale.operators[0]
    return {'multitask': {'chol_tril': base_kernel_chol_op.tril}}
  if isinstance(kernel, multitask_kernel.Separable):
    kronecker_op, diag_op = observation_scale.operators
    kronecker_orths = [k.matrix for k in kronecker_op.operators]
    return {'separable': {'kronecker_orths': kronecker_orths,
                          'diag': diag_op.diag}}
  # This should not happen.
  raise ValueError('Unexpected values for kernel and observation_scale.')


class MultiTaskGaussianProcessRegressionModel(
    distribution.AutoCompositeTensorDistribution):
  """Posterior predictive in a conjugate Multi-task GP regression model."""

  # pylint:disable=invalid-name

  def __init__(self,
               kernel,
               observation_index_points,
               observations,
               observations_is_missing=None,
               index_points=None,
               mean_fn=None,
               observation_noise_variance=None,
               predictive_noise_variance=None,
               cholesky_fn=None,
               validate_args=False,
               allow_nan_stats=False,
               name='MultiTaskGaussianProcessRegressionModelWithCholesky',
               _flattened_conditional_mean_fn=None,
               _observation_scale=None):
    """Construct a MultiTaskGaussianProcessRegressionModelWithCholesky instance.

    Args:
      kernel: `MultiTaskKernel`-like instance representing the GP's covariance
        function.
      observation_index_points: `float` `Tensor` representing finite collection,
        or batch of collections, of points in the index set for which some data
        has been observed. Shape has the form `[b1, ..., bB, e, f1, ..., fF]`
        where `F` is the number of feature dimensions and must equal
        `kernel.feature_ndims`, and `e` is the number (size) of index points in
        each batch. `[b1, ..., bB, e]` must be broadcastable with the shape of
        `observations`, and `[b1, ..., bB]` must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc).
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `observation_index_points`. Shape has the form `[b1, ..., bB, e, t]`,
        which must be broadcastable with the batch and example shapes of
        `observation_index_points`. The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc.).
      observations_is_missing:  `bool` `Tensor` of shape `[..., e, t]`,
        representing a batch of boolean masks.  When
        `observations_is_missing` is not `None`, this distribution is
        conditioned only on the observations for which the
        corresponding elements of `observations_is_missing` are `False`.
      index_points: `float` `Tensor` representing finite collection, or batch of
        collections, of points in the index set over which the GP is defined.
        Shape has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the
        number of feature dimensions and must equal `kernel.feature_ndims` and
        `e` is the number (size) of index points in each batch. Ultimately this
        distribution corresponds to an `e`-dimensional multivariate normal. The
        batch shape must be broadcastable with `kernel.batch_shape`.
      mean_fn: Python `callable` that acts on `index_points` to produce a (batch
        of) collection of mean values at `index_points`. Takes a `Tensor` of
        shape `[b1, ..., bB, e, f1, ..., fF]` and returns a `Tensor` whose shape
        is broadcastable with `[b1, ..., bB, e, t]`, where `t` is the number of
        tasks.
      observation_noise_variance: `float` `Tensor` representing the variance of
        the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
        Default value: `None`
      predictive_noise_variance: `float` `Tensor` representing the variance in
        the posterior predictive model. If `None`, we simply re-use
        `observation_noise_variance` for the posterior predictive noise. If set
        explicitly, however, we use this value. This allows us, for example, to
        omit predictive noise variance (by setting this to zero) to obtain
        noiseless posterior predictions of function values, conditioned on noisy
        observations.
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
          in which case `make_cholesky_with_jitter_fn(1e-6)` is used.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value `NaN` to indicate the result
        is undefined. When `False`, an exception is raised if one or more of the
        statistic's batch members are undefined.
        Default value: `False`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'MultiTaskGaussianProcessRegressionModel'.
      _flattened_conditional_mean_fn: Internal parameter -- do not use.
      _observation_scale: Internal parameter -- do not use.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:

      if not isinstance(kernel, multitask_kernel.MultiTaskKernel):
        raise ValueError('`kernel` must be a `MultiTaskKernel`.')

      dtype = dtype_util.common_dtype([
          index_points, observation_index_points, observations,
          observation_noise_variance, predictive_noise_variance
      ], tf.float32)
      index_points = tensor_util.convert_nonref_to_tensor(
          index_points, dtype=dtype, name='index_points')
      observation_index_points = tensor_util.convert_nonref_to_tensor(
          observation_index_points,
          dtype=dtype,
          name='observation_index_points')
      observations = tensor_util.convert_nonref_to_tensor(
          observations, dtype=dtype, name='observations')
      if observations_is_missing is not None:
        observations_is_missing = tensor_util.convert_nonref_to_tensor(
            observations_is_missing, dtype=tf.bool)
      if observation_noise_variance is not None:
        observation_noise_variance = tensor_util.convert_nonref_to_tensor(
            observation_noise_variance,
            dtype=dtype,
            name='observation_noise_variance')
      predictive_noise_variance = tensor_util.convert_nonref_to_tensor(
          predictive_noise_variance,
          dtype=dtype,
          name='predictive_noise_variance')
      if predictive_noise_variance is None:
        predictive_noise_variance = observation_noise_variance
      if cholesky_fn is None:
        self._cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn()
      else:
        if not callable(cholesky_fn):
          raise ValueError('`cholesky_fn` must be a Python callable')
        self._cholesky_fn = cholesky_fn

      self._kernel = kernel
      self._index_points = index_points

      # Scalar or vector the size of the number of tasks.
      if mean_fn is None:
        def _mean_fn(x):
          # Shape B1 + [E, N], where E is the number of index points, and N is
          # the number of tasks.
          return tf.zeros(
              ps.concat([
                  ps.shape(x)[:-self.kernel.feature_ndims],
                  [self.kernel.num_tasks]], axis=0), dtype=self.dtype)
        mean_fn = _mean_fn
      else:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')
      self._mean_fn = mean_fn
      self._observation_noise_variance = observation_noise_variance
      self._predictive_noise_variance = predictive_noise_variance
      self._index_ponts = index_points
      self._observation_index_points = observation_index_points
      self._observations = observations
      self._observations_is_missing = observations_is_missing

      self._check_observations_valid(observations)

      if _flattened_conditional_mean_fn is None:

        def flattened_conditional_mean_fn(x):
          """Flattened Conditional mean."""
          observation_scale = _compute_observation_scale(
              kernel,
              observation_index_points,
              self._cholesky_fn,
              observation_noise_variance=self.observation_noise_variance,
              observations_is_missing=observations_is_missing)

          return _flattened_conditional_mean_fn_helper(
              x,
              self.kernel,
              self._observations,
              self._observation_index_points,
              observations_is_missing,
              observation_scale,
              mean_fn)

        _flattened_conditional_mean_fn = flattened_conditional_mean_fn

      self._flattened_conditional_mean_fn = _flattened_conditional_mean_fn
      self._observation_scale = _observation_scale

      super(MultiTaskGaussianProcessRegressionModel, self).__init__(
          dtype=dtype,
          reparameterization_type=(reparameterization.FULLY_REPARAMETERIZED),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  def _check_observations_valid(self, observations):
    observation_rank = tensorshape_util.rank(observations.shape)

    if observation_rank is None:
      return

    if observation_rank >= 1:
      # Check that the last dimension of observations matches the number of
      # tasks.
      num_observations = tf.compat.dimension_value(observations.shape[-1])
      if (num_observations is not None and
          num_observations != 1 and
          num_observations != self.kernel.num_tasks):
        raise ValueError(
            f'Expected the number of observations {num_observations} '
            f'to broadcast / match the number of tasks '
            f'{self.kernel.num_tasks}')

    if observation_rank >= 2:
      num_index_points = tf.compat.dimension_value(observations.shape[-2])

      expected_num_index_points = self.observation_index_points.shape[
          -(self.kernel.feature_ndims + 1)]
      if (num_index_points is not None and
          expected_num_index_points is not None and
          num_index_points != 1 and
          num_index_points != expected_num_index_points):
        raise ValueError(
            f'Expected number of observation index points '
            f'{expected_num_index_points} to broadcast / match the second '
            f'to last dimension of `observations` {num_index_points}')

  @staticmethod
  def precompute_regression_model(
      kernel,
      observation_index_points,
      observations,
      observations_is_missing=None,
      index_points=None,
      observation_noise_variance=None,
      predictive_noise_variance=None,
      mean_fn=None,
      cholesky_fn=None,
      validate_args=False,
      allow_nan_stats=False,
      name='PrecomputedMultiTaskGaussianProcessRegressionModel',
      _precomputed_divisor_matrix_cholesky=None,
      _precomputed_solve_on_observation=None):
    """Returns a MTGaussianProcessRegressionModel with precomputed quantities.

    This differs from the constructor by precomputing quantities associated with
    observations in a non-tape safe way. `index_points` is the only parameter
    that is allowed to vary (i.e. is a `Variable` / changes after
    initialization).

    Specifically:

    * We make `observation_index_points` and `observations` mandatory
      parameters.
    * We precompute `kernel(observation_index_points, observation_index_points)`
      along with any other associated quantities relating to the `kernel`,
      `observations` and `observation_index_points`.

    A typical usecase would be optimizing kernel hyperparameters for a
    `MultiTaskGaussianProcess`, and computing the posterior predictive with
    respect to those optimized hyperparameters and observation / index-points
    pairs.

    WARNING: This method assumes `index_points` is the only varying parameter
    (i.e. is a `Variable` / changes after initialization) and hence is not
    tape-safe.

    Args:
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        GP's covariance function.
      observation_index_points: `float` `Tensor` representing finite collection,
        or batch of collections, of points in the index set for which some data
        has been observed. Shape has the form `[b1, ..., bB, e, f1, ..., fF]`
        where `F` is the number of feature dimensions and must equal
        `kernel.feature_ndims`, and `e` is the number (size) of index points in
        each batch. `[b1, ..., bB, e]` must be broadcastable with the shape of
        `observations`, and `[b1, ..., bB]` must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc). The default value is `None`, which corresponds to
        the empty set of observations, and simply results in the prior
        predictive model (a GP with noise of variance
        `predictive_noise_variance`).
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `observation_index_points`. Shape has the form `[b1, ..., bB, e, t]`
        The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc.). The default value is
        `None`, which corresponds to the empty set of observations, and simply
        results in the prior predictive model (a GP with noise of variance
        `predictive_noise_variance`).
      observations_is_missing:  `bool` `Tensor` of shape `[..., e]`,
        representing a batch of boolean masks.  When `observations_is_missing`
        is not `None`, the returned distribution is conditioned only on the
        observations for which the corresponding elements of
        `observations_is_missing` are `True`.
      index_points: `float` `Tensor` representing finite collection, or batch of
        collections, of points in the index set over which the GP is defined.
        Shape has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the
        number of feature dimensions and must equal `kernel.feature_ndims` and
        `e` is the number (size) of index points in each batch. Ultimately this
        distribution corresponds to an `e`-dimensional multivariate normal. The
        batch shape must be broadcastable with `kernel.batch_shape` and any
        batch dims yielded by `mean_fn`.
      observation_noise_variance: `float` `Tensor` representing the variance
        of the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
        Default value: `None`
      predictive_noise_variance: `float` `Tensor` representing the variance in
        the posterior predictive model. If `None`, we simply re-use
        `observation_noise_variance` for the posterior predictive noise. If set
        explicitly, however, we use this value. This allows us, for example, to
        omit predictive noise variance (by setting this to zero) to obtain
        noiseless posterior predictions of function values, conditioned on noisy
        observations.
      mean_fn: Python `callable` that acts on `index_points` to produce a
        collection, or batch of collections, of mean values at `index_points`.
        Takes a `Tensor` of shape `[b1, ..., bB, f1, ..., fF]` and returns a
        `Tensor` whose shape is broadcastable with `[b1, ..., bB, t]`.
        Default value: `None` implies the constant zero function.
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
        in which case `make_cholesky_with_jitter_fn` is used with the `jitter`
        parameter.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value `NaN` to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
        Default value: `False`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'PrecomputedGaussianProcessRegressionModel'.
      _precomputed_divisor_matrix_cholesky: Internal parameter -- do not use.
      _precomputed_solve_on_observation: Internal parameter -- do not use.
    Returns
      An instance of `MultiTaskGaussianProcessRegressionModel` with precomputed
      quantities associated with observations.
    """

    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([
          index_points, observation_index_points, observations,
          observation_noise_variance, predictive_noise_variance,
      ], tf.float32)

      # Convert-to-tensor arguments that are expected to not be Variables / not
      # going to change.
      observation_index_points = tf.convert_to_tensor(
          observation_index_points, dtype=dtype)
      if observation_noise_variance is not None:
        observation_noise_variance = tf.convert_to_tensor(
            observation_noise_variance, dtype=dtype)
      observations = tf.convert_to_tensor(observations, dtype=dtype)

      if observations_is_missing is not None:
        observations_is_missing = tf.convert_to_tensor(observations_is_missing)

      if cholesky_fn is None:
        cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn()
      else:
        if not callable(cholesky_fn):
          raise ValueError('`cholesky_fn` must be a Python callable')

      if mean_fn is None:
        mean_fn = lambda x: tf.zeros([1], dtype=dtype)
      else:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')

      if _precomputed_divisor_matrix_cholesky is not None:
        observation_scale = _scale_from_precomputed(
            _precomputed_divisor_matrix_cholesky, kernel)
      elif observations_is_missing is not None:
        # If observations are missing, there's nothing we can do to preserve the
        # operator structure, so densify.

        observation_covariance = kernel.matrix_over_all_tasks(
            observation_index_points, observation_index_points).to_dense()

        if observation_noise_variance is not None:
          broadcast_shape = distribution_util.get_broadcast_shape(
              observation_covariance, observation_noise_variance[
                  ..., tf.newaxis, tf.newaxis])
          observation_covariance = tf.broadcast_to(observation_covariance,
                                                   broadcast_shape)
          observation_covariance = _add_diagonal_shift(
              observation_covariance, observation_noise_variance)
        vec_observations_is_missing = _vec(observations_is_missing)
        observation_covariance = tf.linalg.LinearOperatorFullMatrix(
            psd_kernels_util.mask_matrix(
                observation_covariance,
                is_missing=vec_observations_is_missing),
            is_non_singular=True,
            is_positive_definite=True)
        observation_scale = cholesky_util.cholesky_from_fn(
            observation_covariance, cholesky_fn)
      else:
        observation_scale = mtgp._compute_flattened_scale(  # pylint:disable=protected-access
            kernel=kernel,
            index_points=observation_index_points,
            cholesky_fn=cholesky_fn,
            observation_noise_variance=observation_noise_variance)

      # Note that the conditional mean is
      # k(x, o) @ (k(o, o) + sigma**2)^-1 obs. We can precompute the latter
      # term since it won't change per iteration.
      vec_diff = _vec(observations - mean_fn(observation_index_points))

      if observations_is_missing is not None:
        vec_diff = tf.where(vec_observations_is_missing,
                            tf.zeros([], dtype=vec_diff.dtype),
                            vec_diff)
      solve_on_observations = _precomputed_solve_on_observation
      if solve_on_observations is None:
        solve_on_observations = observation_scale.solvevec(
            observation_scale.solvevec(vec_diff), adjoint=True)

      def flattened_conditional_mean_fn(x):

        return _flattened_conditional_mean_fn_helper(
            x,
            kernel,
            observations,
            observation_index_points,
            observations_is_missing,
            observation_scale,
            mean_fn,
            solve_on_observations=solve_on_observations)

      mtgprm = MultiTaskGaussianProcessRegressionModel(
          kernel=kernel,
          observation_index_points=observation_index_points,
          observations=observations,
          index_points=index_points,
          observation_noise_variance=observation_noise_variance,
          predictive_noise_variance=predictive_noise_variance,
          cholesky_fn=cholesky_fn,
          _flattened_conditional_mean_fn=flattened_conditional_mean_fn,
          _observation_scale=observation_scale,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)

      # pylint: disable=protected-access
      mtgprm._precomputed_divisor_matrix_cholesky = (
          _precomputed_from_scale(observation_scale, kernel))
      mtgprm._precomputed_solve_on_observation = solve_on_observations
      # pylint: enable=protected-access

    return mtgprm

  @property
  def mean_fn(self):
    return self._mean_fn

  @property
  def kernel(self):
    return self._kernel

  @property
  def observation_index_points(self):
    return self._observation_index_points

  @property
  def observation_scale(self):
    return self._observation_scale

  @property
  def observations(self):
    return self._observations

  @property
  def observations_is_missing(self):
    return self._observations_is_missing

  @property
  def index_points(self):
    return self._index_points

  @property
  def observation_noise_variance(self):
    return self._observation_noise_variance

  @property
  def predictive_noise_variance(self):
    return self._predictive_noise_variance

  @property
  def cholesky_fn(self):
    return self._cholesky_fn

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        index_points=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims + 1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        observations=parameter_properties.ParameterProperties(
            event_ndims=2,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
        observation_index_points=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims + 1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        observations_is_missing=parameter_properties.ParameterProperties(
            event_ndims=2,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        kernel=parameter_properties.BatchedComponentProperties(),
        observation_noise_variance=parameter_properties.ParameterProperties(
            event_ndims=0,
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        predictive_noise_variance=parameter_properties.ParameterProperties(
            event_ndims=0,
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        _observation_scale=parameter_properties.BatchedComponentProperties())

  def _event_shape(self):
    # The examples index is one position to the left of the feature dims.
    index_points = self.index_points

    if index_points is None:
      return tf.TensorShape([None, self.kernel.num_tasks])
    examples_index = -(self.kernel.feature_ndims + 1)
    shape = tensorshape_util.concatenate(
        index_points.shape[examples_index:examples_index + 1],
        (self.kernel.num_tasks,))
    if tensorshape_util.rank(shape) is None:
      return tensorshape_util.concatenate(
          [index_points.shape[examples_index:examples_index + 1]],
          [self.kernel.num_tasks])
    return shape

  def _event_shape_tensor(self, index_points=None):
    index_points = self._get_index_points(index_points)
    return ps.concat(
        [[ps.shape(index_points)[-(self.kernel.feature_ndims + 1)]],
         [self.kernel.num_tasks]],
        axis=0)

  def _batch_shape(self, index_points=None):
    kwargs = {}
    if index_points is not None:
      kwargs = {'index_points': index_points}
    return batch_shape_lib.inferred_batch_shape(self, **kwargs)

  def _batch_shape_tensor(self, index_points=None):
    kwargs = {}
    if index_points is not None:
      kwargs = {'index_points': index_points}
    return batch_shape_lib.inferred_batch_shape_tensor(self, **kwargs)

  def _compute_flattened_covariance(self, index_points=None):
    # This is of shape KN x KN, where K is the number of outputs
    # Compute this explicitly via the Schur Complement of the vector kernel.
    # The reason this is written explicitly as opposed to using a GPRM
    # internally for reshaping is there is potential for efficiency gains when
    # `observation_noise_variance = 0.`.
    index_points = self._get_index_points(index_points)
    kxx = self.kernel.matrix_over_all_tasks(index_points, index_points)

    kxz = self.kernel.matrix_over_all_tasks(
        index_points, self.observation_index_points).to_dense()
    if self._observations_is_missing is not None:
      kxz = tf.where(_vec(self._observations_is_missing)[..., tf.newaxis, :],
                     tf.zeros([], dtype=kxz.dtype),
                     kxz)
    if self._observation_scale is not None:
      observation_scale = self._observation_scale
    else:
      observation_scale = _compute_observation_scale(
          self.kernel,
          self.observation_index_points,
          self.cholesky_fn,
          observation_noise_variance=self.observation_noise_variance,
          observations_is_missing=self.observations_is_missing)

    cholinv_kzx = observation_scale.solve(kxz, adjoint_arg=True)
    kxz_kzzinv_kzx = tf.linalg.matmul(
        cholinv_kzx, cholinv_kzx, transpose_a=True)

    flattened_covariance = kxx.to_dense() - kxz_kzzinv_kzx
    if self.predictive_noise_variance is None:
      return flattened_covariance
    broadcast_shape = distribution_util.get_broadcast_shape(
        flattened_covariance, self.predictive_noise_variance[..., tf.newaxis,
                                                             tf.newaxis])
    flattened_covariance = tf.broadcast_to(flattened_covariance,
                                           broadcast_shape)
    return _add_diagonal_shift(flattened_covariance,
                               self.predictive_noise_variance)

  def _get_flattened_marginal_distribution(self, index_points=None):
    # This returns a MVN of event size [N * E], where N is the number of tasks
    # and E is the number of index points.
    with self._name_and_control_scope('get_flattened_marginal_distribution'):
      index_points = self._get_index_points(index_points)
      covariance = self._compute_flattened_covariance(index_points)
      loc = self._flattened_conditional_mean_fn(index_points)
      scale = tf.linalg.LinearOperatorLowerTriangular(
          self._cholesky_fn(covariance),
          is_non_singular=True,
          name='GaussianProcessScaleLinearOperator')
      return mvn_linear_operator.MultivariateNormalLinearOperator(
          loc=loc,
          scale=scale,
          validate_args=self._validate_args,
          allow_nan_stats=self._allow_nan_stats,
          name='marginal_distribution')

  def _log_prob(self, value, index_points=None):
    return self._get_flattened_marginal_distribution(
        index_points=index_points).log_prob(_vec(value))

  def _mean(self, index_points=None):
    # The mean is of shape B1 + [E, N], where E is the number of index points,
    # and N is the number of tasks.
    return _unvec(
        self._get_flattened_marginal_distribution(
            index_points=index_points).mean(), [-1, self.kernel.num_tasks])

  def _variance(self, index_points=None):
    # This is of shape KN x KN, where K is the number of outputs
    # Compute this explicitly via the Schur Complement of the vector kernel.
    # The reason this is written explicitly as opposed to using a GPRM
    # internally for reshaping is there is potential for efficiency gains when
    # `observation_noise_variance = 0.`.
    index_points = self._get_index_points(index_points)
    kxx_diag = self.kernel.matrix_over_all_tasks(
        index_points, index_points).diag_part()

    kxz = self.kernel.matrix_over_all_tasks(
        index_points, self.observation_index_points).to_dense()
    if self.observations_is_missing is not None:
      kxz = tf.where(_vec(self.observations_is_missing)[..., tf.newaxis, :],
                     tf.zeros([], dtype=kxz.dtype), kxz)
    if self._observation_scale is not None:
      observation_scale = self._observation_scale
    else:
      observation_scale = _compute_observation_scale(
          self.kernel,
          self.observation_index_points,
          self._cholesky_fn,
          observations_is_missing=self.observations_is_missing)

    cholinv_kzx = observation_scale.solve(kxz, adjoint_arg=True)
    kxz_kzzinv_kzx_diag = tf.linalg.diag_part(tf.linalg.matmul(
        cholinv_kzx, cholinv_kzx, transpose_a=True))

    flattened_variance = kxx_diag - kxz_kzzinv_kzx_diag
    if self.predictive_noise_variance is not None:
      flattened_variance = (
          flattened_variance + self.predictive_noise_variance[..., tf.newaxis])

    variance = _unvec(flattened_variance, [-1, self.kernel.num_tasks])

    # Finally broadcast with batch shape.
    batch_shape = self._batch_shape_tensor(index_points=index_points)
    event_shape = self._event_shape_tensor(index_points=index_points)

    return tf.broadcast_to(
        variance, ps.concat([batch_shape, event_shape], axis=0))

  def _sample_n(self, n, seed=None, index_points=None):
    # Samples is of shape [n] + B1 + [E, N], where E is the number of index
    # points, and N is the number of tasks.
    samples = self._get_flattened_marginal_distribution(
        index_points=index_points).sample(
            n, seed=seed)
    return _unvec(samples, [-1, self.kernel.num_tasks])

  def _get_index_points(self, index_points=None):
    """Return `index_points` if not None, else `self._index_points`.

    Args:
      index_points: if given, this is what is returned; else,
        `self._index_points`

    Returns:
      index_points: the given arg, if not None, else the class member
      `self._index_points`.

    Rases:
      ValueError: if `index_points` and `self._index_points` are both `None`.
    """
    if self._index_points is None and index_points is None:
      raise ValueError(
          'This MultiTaskGaussianProcessRegressionModel instance was not '
          'instantiated with a value for index_points. One must therefore be '
          'provided when calling sample, log_prob, and other such methods. In '
          'particular, one can\'t  compute KL divergences to/from an instance '
          'of `MultiTaskGaussianProcessRegressionModel` with unspecified '
          '`index_points` directly. Instead, use the '
          '`get_marginal_distribution` function, which takes `index_points` as '
          'an argument and returns a `Normal` or '
          '`MultivariateNormalLinearOperator` instance, whose KL can be '
          'computed.')
    return tf.convert_to_tensor(
        index_points if index_points is not None else self._index_points)
