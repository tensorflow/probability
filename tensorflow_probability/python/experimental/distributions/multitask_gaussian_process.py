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
from tensorflow_probability.python.experimental.psd_kernels import multitask_kernel
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util


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


class MultiTaskGaussianProcess(distribution.Distribution):
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

  def _compute_flattened_covariance(self, index_points=None):
    # This is of shape KN x KN, where K is the number of outputs
    index_points = self._get_index_points(index_points)
    kernel_matrix = self.kernel.matrix_over_all_tasks(
        index_points, index_points)
    if self.observation_noise_variance is None:
      return kernel_matrix
    kernel_matrix = kernel_matrix.to_dense()
    broadcast_shape = distribution_util.get_broadcast_shape(
        kernel_matrix,
        self.observation_noise_variance[..., tf.newaxis, tf.newaxis])
    kernel_matrix = tf.broadcast_to(kernel_matrix, broadcast_shape)
    kernel_matrix = tf.linalg.set_diag(
        kernel_matrix,
        tf.linalg.diag_part(kernel_matrix) +
        self.observation_noise_variance[..., tf.newaxis])
    kernel_matrix = tf.linalg.LinearOperatorFullMatrix(
        kernel_matrix,
        is_non_singular=True,
        is_positive_definite=True)
    return kernel_matrix

  def _get_flattened_marginal_distribution(self, index_points=None):
    # This returns a MVN of event size [N * E], where N is the number of tasks
    # and E is the number of index points.
    with self._name_and_control_scope('get_flattened_marginal_distribution'):
      index_points = self._get_index_points(index_points)
      covariance = self._compute_flattened_covariance(index_points)

      batch_shape = self._batch_shape_tensor(index_points=index_points)
      event_shape = self._event_shape_tensor(index_points=index_points)

      # Now take the cholesky but specialize to cases where we have block-diag
      # and kronecker.
      covariance_cholesky = cholesky_util.cholesky_from_fn(
          covariance, self._cholesky_fn)
      loc = self._mean_fn(index_points)
      # Ensure that we broadcast the mean function result to ensure we support
      # constant mean functions (constant over all tasks, and a constant
      # per-task)
      loc = ps.broadcast_to(
          loc, ps.concat([batch_shape, event_shape], axis=0))
      loc = _vec(loc)
      return mvn_linear_operator.MultivariateNormalLinearOperator(
          loc=loc,
          scale=covariance_cholesky,
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

  def _log_prob(self, value, index_points=None):
    return self._get_flattened_marginal_distribution(
        index_points=index_points).log_prob(_vec(value))

  def _mean(self, index_points=None):
    # The mean is of shape B1 + [E, N], where E is the number of index points,
    # and N is the number of tasks.
    return _unvec(
        self._get_flattened_marginal_distribution(
            index_points=index_points).mean(),
        [-1, self.kernel.num_tasks])

  def _sample_n(self, n, index_points=None, seed=None):
    # Samples is of shape [n] + B1 + [E, N], where E is the number of index
    # points, and N is the number of tasks.
    samples = self._get_flattened_marginal_distribution(
        index_points=index_points).sample(n, seed=seed)
    return _unvec(samples, [-1, self.kernel.num_tasks])
