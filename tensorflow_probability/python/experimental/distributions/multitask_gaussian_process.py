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
"""The MultiTaskGaussianProcess distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import cholesky_util
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import mvn_linear_operator
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


def _vec(x):
  # Vec takes in a (batch) of matrices of shape B1 + [n, k] and returns
  # a (batch) of vectors of shape B1 + [n * k].
  return tf.reshape(
      x, ps.concat([ps.shape(x)[:-2], [-1]], axis=0))


def _unvec(x, matrix_shape):
  # Unvec takes in a (batch) of matrices of shape B1 + [n * k] and returns
  # a (batch) of vectors of shape B1 + [n, k], where n and k are specified
  # by matrix_shape.
  return tf.reshape(x, ps.concat(
      [ps.shape(x)[:-1], matrix_shape], axis=0))


def _compute_flattened_scale(
    kernel,
    index_points,
    cholesky_fn,
    observation_noise_variance=None):
  """Computes a matrix square root of the flattened covariance matrix.

  Given a multi-task kernel `k`, computes a matrix square root of the
  matrix over all tasks of `index_points`. That is, compute `S` such that
  `S^T @ S = k.matrix_over_all_tasks(index_points, index_points)`.

  In the case of a `Separable` or `Independent` kernel, this function tries to
  do this efficiently in O(N^3 + T^3) time where `N` is the number of
  `index_points` and `T` is the number of tasks.

  Args:
    kernel: `MultiTaskKernel`-like instance representing the GP's covariance
      function.
    index_points: `float` `Tensor` representing finite collection, or batch of
      collections, of points in the index set over which the GP is defined.
      Shape has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the
      number of feature dimensions and must equal `kernel.feature_ndims` and
      `e` is the number (size) of index points in each batch. Ultimately this
      distribution corresponds to an `e`-dimensional multivariate normal. The
      batch shape must be broadcastable with `kernel.batch_shape`.
    cholesky_fn: Callable which takes a single (batch) matrix argument and
      returns a Cholesky-like lower triangular factor.  Default value: `None`,
      in which case `make_cholesky_with_jitter_fn(1e-6)` is used.
    observation_noise_variance: `float` `Tensor` representing the variance
      of the noise in the Normal likelihood distribution of the model. May be
      batched, in which case the batch shape must be broadcastable with the
      shapes of all other batched parameters (`kernel.batch_shape`,
      `index_points`, etc.).
      Default value: `None`
  Returns:
    scale_operator: `LinearOperator` representing a matrix square root of
    the flattened kernel matrix over all tasks.

  """
  # This is of shape KN x KN, where K is the number of outputs
  kernel_matrix = kernel.matrix_over_all_tasks(index_points, index_points)
  if observation_noise_variance is None:
    return cholesky_util.cholesky_from_fn(kernel_matrix, cholesky_fn)

  observation_noise_variance = tf.convert_to_tensor(observation_noise_variance)

  # We can add the observation noise to each block.
  if isinstance(kernel, multitask_kernel.Independent):
    # The Independent kernel matrix is realized as a kronecker product of the
    # kernel over inputs, and an identity matrix per task (representing
    # independent tasks). Update the diagonal of the first matrix and take the
    # cholesky of it (since the cholesky of the second matrix will remain the
    # identity matrix.)
    base_kernel_matrix = kernel_matrix.operators[0].to_dense()

    broadcast_shape = distribution_util.get_broadcast_shape(
        base_kernel_matrix,
        observation_noise_variance[..., tf.newaxis, tf.newaxis])
    base_kernel_matrix = tf.broadcast_to(base_kernel_matrix, broadcast_shape)
    base_kernel_matrix = tf.linalg.set_diag(
        base_kernel_matrix,
        tf.linalg.diag_part(base_kernel_matrix) +
        observation_noise_variance[..., tf.newaxis])
    base_kernel_matrix = tf.linalg.LinearOperatorFullMatrix(
        base_kernel_matrix)
    kernel_matrix = tf.linalg.LinearOperatorKronecker(
        operators=[base_kernel_matrix] + kernel_matrix.operators[1:])
    return cholesky_util.cholesky_from_fn(kernel_matrix, cholesky_fn)

  if isinstance(kernel, multitask_kernel.Separable):
    # When `kernel_matrix` is a kronecker product, we can compute
    # an eigenvalue decomposition to get a matrix square-root, which will
    # be faster than densifying the kronecker product.

    # Let K = A X B. Let A (and B) have an eigenvalue decomposition of
    # U @ D @ U^T, where U is an orthogonal matrix. Then,
    # K = (U_A @ D_A @ U_A^T) X (U_B @ D_B @ U_B^T) =
    # (U_A X U_B) @ (D_A X D_B) @ (U_A X U_B)^T
    # Thus, a matrix square root of K would be
    # (U_A X U_B) @ (sqrt(D_A) X sqrt(D_B)) which offers
    # efficient matmul and solves.

    # Now, if we update the diagonal by `v * I`, we have
    # (U_A X U_B) @ (sqrt((D_A X D_B + vI)) @ (U_A X U_B)^T
    # which still admits an efficient matmul and solve.

    kronecker_diags = []
    kronecker_orths = []
    for block in kernel_matrix.operators:
      diag, orth = tf.linalg.eigh(block.to_dense())
      kronecker_diags.append(tf.linalg.LinearOperatorDiag(diag))
      kronecker_orths.append(
          linear_operator_unitary.LinearOperatorUnitary(orth))

    full_diag = tf.linalg.LinearOperatorKronecker(kronecker_diags).diag_part()
    full_diag = full_diag + observation_noise_variance[..., tf.newaxis]
    scale_diag = tf.math.sqrt(full_diag)
    diag_operator = tf.linalg.LinearOperatorDiag(
        scale_diag,
        is_square=True,
        is_non_singular=True,
        is_positive_definite=True)

    orthogonal_operator = tf.linalg.LinearOperatorKronecker(
        kronecker_orths, is_square=True, is_non_singular=True)
    # This is efficient as a scale matrix. When used for matmuls, we take
    # advantage of the kronecker product and diagonal operator. When used for
    # solves, we take advantage of the orthogonal and diagonal structure,
    # which essentially reduces to the matmul case.
    return orthogonal_operator.matmul(diag_operator)

  # By default densify the kernel matrix and add noise.

  kernel_matrix = kernel_matrix.to_dense()
  broadcast_shape = distribution_util.get_broadcast_shape(
      kernel_matrix,
      observation_noise_variance[..., tf.newaxis, tf.newaxis])
  kernel_matrix = tf.broadcast_to(kernel_matrix, broadcast_shape)
  kernel_matrix = tf.linalg.set_diag(
      kernel_matrix,
      tf.linalg.diag_part(kernel_matrix) +
      observation_noise_variance[..., tf.newaxis])
  kernel_matrix = tf.linalg.LinearOperatorFullMatrix(kernel_matrix)
  kernel_cholesky = cholesky_util.cholesky_from_fn(
      kernel_matrix, cholesky_fn)
  return kernel_cholesky


class MultiTaskGaussianProcess(distribution.AutoCompositeTensorDistribution):
  """Marginal distribution of a Multitask GP at finitely many points."""

  def __init__(self,
               kernel,
               index_points=None,
               mean_fn=None,
               observation_noise_variance=None,
               cholesky_fn=None,
               validate_args=False,
               allow_nan_stats=False,
               name='MultiTaskGaussianProcess'):
    """Constructs a MultiTaskGaussianProcess instance.

    Args:
      kernel: `MultiTaskKernel`-like instance representing the
        GP's covariance function.
      index_points: `float` `Tensor` representing finite collection, or batch of
        collections, of points in the index set over which the GP is defined.
        Shape has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the
        number of feature dimensions and must equal `kernel.feature_ndims` and
        `e` is the number (size) of index points in each batch. Ultimately this
        distribution corresponds to an `e`-dimensional multivariate normal. The
        batch shape must be broadcastable with `kernel.batch_shape`.
      mean_fn: Python `callable` that acts on `index_points` to produce a
        (batch of) collection of mean values at `index_points`. Takes a `Tensor`
        of shape `[b1, ..., bB, e, f1, ..., fF]` and returns a `Tensor` whose
        shape is broadcastable with `[b1, ..., bB, e, t]`, where `t` is the
        number of tasks.
      observation_noise_variance: `float` `Tensor` representing the variance
        of the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
        Default value: `0.`
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
        in which case `make_cholesky_with_jitter_fn(1e-6)` is used.
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
        Default value: 'MultiTaskGaussianProcess'.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [index_points, observation_noise_variance], tf.float32)
      index_points = tensor_util.convert_nonref_to_tensor(
          index_points, dtype=dtype, name='index_points')
      observation_noise_variance = tensor_util.convert_nonref_to_tensor(
          observation_noise_variance,
          dtype=dtype,
          name='observation_noise_variance')

      if not isinstance(kernel, multitask_kernel.MultiTaskKernel):
        raise ValueError('`kernel` must be a `MultiTaskKernel`.')
      self._kernel = kernel
      self._index_points = index_points

      if mean_fn is None:
        def _mean_fn(x):
          # Shape B1 + [E, N], where E is the number of index points, and N is
          # the number of tasks.
          return tf.zeros(ps.concat(
              [ps.shape(x)[:-self.kernel.feature_ndims],
               [self.kernel.num_tasks]], axis=0), dtype=dtype)
        mean_fn = _mean_fn
      else:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')
      self._mean_fn = mean_fn
      # Scalar or vector the size of the number of tasks.
      self._observation_noise_variance = observation_noise_variance

      if cholesky_fn is None:
        self._cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn()
      else:
        if not callable(cholesky_fn):
          raise ValueError('`cholesky_fn` must be a Python callable')
        self._cholesky_fn = cholesky_fn

      with tf.name_scope('init'):
        super(MultiTaskGaussianProcess, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name)

  def posterior_predictive(
      self,
      observations,
      observations_is_missing=None,
      predictive_index_points=None,
      **kwargs):
    """Return the posterior predictive distribution associated with this distribution.

    Returns the posterior predictive distribution `p(Y' | X, Y, X')` where:
      * `X'` is `predictive_index_points`
      * `X` is `self.index_points`.
      * `Y` is `observations`.

    This is equivalent to using the
    `MultiTaskGaussianProcessRegressionModel.precompute_regression_model`
    method.

    WARNING: This method assumes `predictive_index_points` is the only varying
    parameter (i.e. is a `Variable` / changes after initialization) and hence
    is not tape-safe.

    Args:
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `self.index_points`. Shape has the form `[b1, ..., bB, t, e]`, where
        `t` is the number of tasks. The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
      observations_is_missing:  `bool` `Tensor` of shape `[..., e, t]`,
        representing a batch of boolean masks.  When
        `observations_is_missing` is not `None`, this distribution is
        conditioned only on the observations for which the
        corresponding elements of `observations_is_missing` are `False`.
      predictive_index_points: `float` `Tensor` representing finite collection,
        or batch of collections, of points in the index set over which the GP
        is defined.
        Shape has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the
        number of feature dimensions and must equal `kernel.feature_ndims` and
        `e` is the number (size) of predictive index points in each batch.
        The batch shape must be broadcastable with this distributions
        `batch_shape`.
        Default value: `None`.
      **kwargs: Any other keyword arguments to pass / override.

    Returns:
      mtgprm: An instance of `Distribution` that represents the posterior
        predictive.
    """
    from tensorflow_probability.python.experimental.distributions import multitask_gaussian_process_regression_model as mtgprm  # pylint:disable=g-import-not-at-top
    if self.index_points is None:
      raise ValueError(
          'Expected that `self.index_points` is not `None`. Using '
          '`self.index_points=None` is equivalent to using a `GaussianProcess` '
          'prior, which this class encapsulates.')
    argument_dict = {
        'kernel': self.kernel,
        'observation_index_points': self.index_points,
        'observations_is_missing': observations_is_missing,
        'observations': observations,
        'index_points': predictive_index_points,
        'observation_noise_variance': self.observation_noise_variance,
        'cholesky_fn': self.cholesky_fn,
        'mean_fn': self.mean_fn,
        'validate_args': self.validate_args,
        'allow_nan_stats': self.allow_nan_stats
    }
    argument_dict.update(**kwargs)

    return mtgprm.MultiTaskGaussianProcessRegressionModel.precompute_regression_model(
        **argument_dict)

  @property
  def mean_fn(self):
    return self._mean_fn

  @property
  def kernel(self):
    return self._kernel

  @property
  def index_points(self):
    return self._index_points

  @property
  def observation_noise_variance(self):
    return self._observation_noise_variance

  @property
  def cholesky_fn(self):
    return self._cholesky_fn

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    from tensorflow_probability.python.bijectors import softplus as softplus_bijector  # pylint:disable=g-import-not-at-top
    return dict(
        index_points=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims + 1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
        kernel=parameter_properties.BatchedComponentProperties(),
        observation_noise_variance=parameter_properties.ParameterProperties(
            event_ndims=0,
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  def _event_shape(self, index_points=None):
    index_points = self._get_index_points(index_points)
    return tf.TensorShape([
        index_points.shape[-(self.kernel.feature_ndims + 1)],
        self.kernel.num_tasks])

  def _event_shape_tensor(self, index_points=None):
    index_points = self._get_index_points(index_points)
    return ps.concat([
        [ps.shape(index_points)[-(self.kernel.feature_ndims + 1)]],
        [self.kernel.num_tasks]], axis=0)

  def _batch_shape(self, index_points=None):
    # TODO(b/249858459): Update `batch_shape_lib` so it can take override
    # parameters.
    result = batch_shape_lib.inferred_batch_shape(self)
    if index_points is not None:
      return ps.broadcast_shape(
          result,
          index_points.shape[:-(self.kernel.feature_ndims + 1)])
    return result

  def _batch_shape_tensor(self, index_points=None):
    kwargs = {}
    if index_points is not None:
      kwargs = {'index_points': index_points}
    return batch_shape_lib.inferred_batch_shape_tensor(self, **kwargs)

  def _get_flattened_marginal_distribution(self, index_points=None):
    # This returns a MVN of event size [N * E], where N is the number of tasks
    # and E is the number of index points.
    with self._name_and_control_scope('get_flattened_marginal_distribution'):
      index_points = self._get_index_points(index_points)
      scale = _compute_flattened_scale(
          kernel=self.kernel,
          index_points=index_points,
          cholesky_fn=self._cholesky_fn,
          observation_noise_variance=self.observation_noise_variance)

      batch_shape = self._batch_shape_tensor(index_points=index_points)
      event_shape = self._event_shape_tensor(index_points=index_points)

      loc = self._mean_fn(index_points)
      # Ensure that we broadcast the mean function result to ensure we support
      # constant mean functions (constant over all tasks, and a constant
      # per-task)
      loc = ps.broadcast_to(
          loc, ps.concat([batch_shape, event_shape], axis=0))
      loc = _vec(loc)
      return mvn_linear_operator.MultivariateNormalLinearOperator(
          loc=loc,
          scale=scale,
          validate_args=self._validate_args,
          allow_nan_stats=self._allow_nan_stats,
          name='marginal_distribution')

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
          'This MultiTaskGaussianProcess instance was not instantiated with a '
          'value for index_points. One must therefore be provided when calling '
          'sample, log_prob, and other such methods. In particular, one can\'t '
          ' compute KL divergences to/from an instance of '
          '`MultiTaskGaussianProccess` with unspecified `index_points` '
          'directly. Instead, use the `get_marginal_distribution` function, '
          'which takes `index_points` as an argument and returns a `Normal` or '
          '`MultivariateNormalLinearOperator` instance, whose KL can be '
          'computed.')
    return tf.convert_to_tensor(
        index_points if index_points is not None else self._index_points)

  def _check_observations_valid(self, observations, index_points):
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

      expected_num_index_points = index_points.shape[
          -(self.kernel.feature_ndims + 1)]
      if (num_index_points is not None and
          expected_num_index_points is not None and
          num_index_points != 1 and
          num_index_points != expected_num_index_points):
        raise ValueError(
            f'Expected number of index points '
            f'{expected_num_index_points} to broadcast / match the second '
            f'to last dimension of `observations` {num_index_points}')

  def _log_prob(self, value, index_points=None):
    # Check that observations with at least 2 dimensions have
    # shape that's broadcastable to `[N, T]`, where `N` is the number
    # of index points, and T the number of tasks.
    index_points = self._get_index_points(index_points)
    self._check_observations_valid(value, index_points)

    return self._get_flattened_marginal_distribution(
        index_points=index_points).log_prob(_vec(value))

  def _mean(self, index_points=None):
    # The mean is of shape B1 + [E, N], where E is the number of index points,
    # and N is the number of tasks.
    return _unvec(
        self._get_flattened_marginal_distribution(
            index_points=index_points).mean(),
        [-1, self.kernel.num_tasks])

  def _variance(self, index_points=None):
    index_points = self._get_index_points(index_points)
    kernel_matrix = self.kernel.matrix_over_all_tasks(
        index_points, index_points)
    observation_noise_variance = None
    if self.observation_noise_variance is not None:
      observation_noise_variance = tf.convert_to_tensor(
          self.observation_noise_variance)

    # We can add the observation noise to each block.
    if isinstance(self.kernel, multitask_kernel.Independent):
      single_task_variance = kernel_matrix.operators[0].diag_part()
      if observation_noise_variance is not None:
        single_task_variance = (
            single_task_variance + observation_noise_variance[..., tf.newaxis])
      # Each task has the same variance, so shape this in to an `[..., e, t]`
      # shaped tensor and broadcast to batch shape
      variance = tf.stack(
          [single_task_variance] * self.kernel.num_tasks, axis=-1)
      # Finally broadcast with batch shape.
      batch_shape = self._batch_shape_tensor(index_points=index_points)
      event_shape = self._event_shape_tensor(index_points=index_points)

      variance = tf.broadcast_to(
          variance, ps.concat([batch_shape, event_shape], axis=0))
      return variance

    # If `kernel_matrix` has structure, `diag_part` will try to take advantage
    # of that structure. In the case of a `Separable` kernel, `diag_part` will
    # efficiently compute the diagonal of a kronecker product.
    variance = kernel_matrix.diag_part()
    if observation_noise_variance is not None:
      variance = (
          variance +
          observation_noise_variance[..., tf.newaxis])

    variance = _unvec(variance, [-1, self.kernel.num_tasks])

    # Finally broadcast with batch shape.
    batch_shape = self._batch_shape_tensor(index_points=index_points)
    event_shape = self._event_shape_tensor(index_points=index_points)

    variance = tf.broadcast_to(
        variance, ps.concat([batch_shape, event_shape], axis=0))
    return variance

  def _sample_n(self, n, index_points=None, seed=None):
    # Samples is of shape [n] + B1 + [E, N], where E is the number of index
    # points, and N is the number of tasks.
    samples = self._get_flattened_marginal_distribution(
        index_points=index_points).sample(n, seed=seed)
    return _unvec(samples, [-1, self.kernel.num_tasks])

  # Override to incorporate `index_points`
  def _set_sample_static_shape(self, x, sample_shape, index_points=None):
    """Helper to `sample`; sets static shape info."""
    batch_shape = self._batch_shape(index_points=index_points)
    event_shape = tf.TensorShape(self._event_shape(index_points=index_points))
    return distribution._set_sample_static_shape_for_tensor(  # pylint:disable=protected-access
        x,
        sample_shape=sample_shape,
        event_shape=event_shape,
        batch_shape=batch_shape)
