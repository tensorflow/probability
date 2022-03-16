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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import cholesky_util
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.experimental.psd_kernels import multitask_kernel
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
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


class MultiTaskGaussianProcessRegressionModel(distribution.Distribution):
  """Posterior predictive in a conjugate Multi-task GP regression model."""

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
               name='MultiTaskGaussianProcessRegressionModelWithCholesky'):
    """Construct a MultiTaskGaussianProcessRegressionModelWithCholesky instance.

    WARNING: This method assumes `index_points` is the only varying parameter
    (i.e. is a `Variable` / changes after initialization) and hence is not
    tape-safe.

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
      observation_index_points = tf.convert_to_tensor(
          observation_index_points,
          dtype=dtype,
          name='observation_index_points')
      observations = tf.convert_to_tensor(
          observations, dtype=dtype, name='observations')
      if observations_is_missing is not None:
        observations_is_missing = tf.convert_to_tensor(
            observations_is_missing, dtype=tf.bool)
      if observation_noise_variance is not None:
        observation_noise_variance = tf.convert_to_tensor(
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
      if mean_fn is not None:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')
      self._mean_fn = mean_fn
      self._observation_noise_variance = observation_noise_variance
      self._predictive_noise_variance = predictive_noise_variance
      self._index_ponts = index_points
      self._observation_index_points = observation_index_points
      self._observations = observations
      self._observations_is_missing = observations_is_missing

      observation_covariance = self.kernel.matrix_over_all_tasks(
          observation_index_points, observation_index_points)

      if observation_noise_variance is not None:
        observation_covariance = observation_covariance.to_dense()
        broadcast_shape = distribution_util.get_broadcast_shape(
            observation_covariance, observation_noise_variance[..., tf.newaxis,
                                                               tf.newaxis])
        observation_covariance = tf.broadcast_to(observation_covariance,
                                                 broadcast_shape)
        observation_covariance = _add_diagonal_shift(observation_covariance,
                                                     observation_noise_variance)
        observation_covariance = tf.linalg.LinearOperatorFullMatrix(
            observation_covariance,
            is_non_singular=True,
            is_positive_definite=True)

      if observations_is_missing is not None:
        vec_observations_is_missing = _vec(observations_is_missing)
        observation_covariance = tf.linalg.LinearOperatorFullMatrix(
            psd_kernels_util.mask_matrix(
                observation_covariance.to_dense(),
                is_missing=vec_observations_is_missing),
            is_non_singular=True,
            is_positive_definite=True)

      self._observation_cholesky = cholesky_util.cholesky_from_fn(
          observation_covariance, self._cholesky_fn)

      # Note that the conditional mean is
      # k(x, o) @ (k(o, o) + sigma**2)^-1 obs. We can precompute the latter
      # term since it won't change per iteration.
      if mean_fn:
        vec_observations = _vec(observations -
                                mean_fn(observation_index_points))
      else:
        vec_observations = _vec(observations)
      if observations_is_missing is not None:
        vec_observations = tf.where(~vec_observations_is_missing,
                                    vec_observations,
                                    tf.zeros([], dtype=vec_observations.dtype))
      self._solve_on_obs = self._observation_cholesky.solvevec(
          self._observation_cholesky.solvevec(vec_observations), adjoint=True)
      super(MultiTaskGaussianProcessRegressionModel, self).__init__(
          dtype=dtype,
          reparameterization_type=(reparameterization.FULLY_REPARAMETERIZED),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @property
  def mean_fn(self):
    # Default to a constant zero function, borrowing the dtype from
    # the class for consisency.
    if self._mean_fn is not None:
      return self._mean_fn

    def _mean_fn(x):
      # Shape B1 + [E, N], where E is the number of index points, and N is the
      # number of tasks.
      res = tf.zeros(
          tf.concat([
              tf.shape(x)[:-self.kernel.feature_ndims], [self.kernel.num_tasks]
          ],
                    axis=0),
          dtype=self.dtype)
      return res

    return _mean_fn

  def _conditional_mean_fn(self, x):
    """Conditional mean."""
    k_x_obs_linop = self.kernel.matrix_over_all_tasks(
        x, self._observation_index_points)
    if self._observations_is_missing is not None:
      k_x_obs_linop = tf.linalg.LinearOperatorFullMatrix(
          tf.where(_vec(tf.math.logical_not(
              self._observations_is_missing))[..., tf.newaxis, :],
                   k_x_obs_linop.to_dense(),
                   tf.zeros([], dtype=k_x_obs_linop.dtype)))

    mean_x = self.mean_fn(x)  # pylint:disable=not-callable
    batch_shape = self._batch_shape_tensor(index_points=x)
    event_shape = self._event_shape_tensor(index_points=x)
    mean_x = ps.broadcast_to(mean_x,
                             ps.concat([batch_shape, event_shape], axis=0))
    mean_x = _vec(mean_x)
    return mean_x + k_x_obs_linop.matvec(self._solve_on_obs)

  @property
  def kernel(self):
    return self._kernel

  @property
  def observation_index_points(self):
    return self._observation_index_points

  @property
  def observation_cholesky(self):
    return self._observation_cholesky

  @property
  def observations(self):
    return self._observations

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

  def _batch_shape_tensor(self, index_points=None):
    index_points = self._get_index_points(index_points)
    return functools.reduce(ps.broadcast_shape, [
        ps.shape(
            self.observation_index_points)[:-(self.kernel.feature_ndims + 1)],
        ps.shape(index_points)[:-(self.kernel.feature_ndims + 1)],
        self.kernel.batch_shape_tensor(),
        ps.shape(self.observations)[:-2],
        ps.shape(self.observation_noise_variance)
    ])

  def _event_shape_tensor(self, index_points=None):
    index_points = self._get_index_points(index_points)
    return tf.concat(
        [[tf.shape(index_points)[-(self.kernel.feature_ndims + 1)]],
         [self.kernel.num_tasks]],
        axis=0)

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
      kxz = tf.where(_vec(tf.math.logical_not(
          self._observations_is_missing))[..., tf.newaxis, :],
                     kxz,
                     tf.zeros([], dtype=kxz.dtype))
    cholinv_kzx = self.observation_cholesky.solve(kxz, adjoint_arg=True)
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
      loc = self._conditional_mean_fn(index_points)
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
