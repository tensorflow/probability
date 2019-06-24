# Copyright 2018 The TensorFlow Probability Authors.
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
"""The GaussianProcess distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import warnings

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization

__all__ = [
    'GaussianProcess',
]


def _add_diagonal_shift(matrix, shift):
  return tf.linalg.set_diag(
      matrix, tf.linalg.diag_part(matrix) + shift, name='add_diagonal_shift')


class GaussianProcess(distribution.Distribution):
  """Marginal distribution of a Gaussian process at finitely many points.

  A Gaussian process (GP) is an indexed collection of random variables, any
  finite collection of which are jointly Gaussian. While this definition applies
  to finite index sets, it is typically implicit that the index set is infinite;
  in applications, it is often some finite dimensional real or complex vector
  space. In such cases, the GP may be thought of as a distribution over
  (real- or complex-valued) functions defined over the index set.

  Just as Gaussian distributions are fully specified by their first and second
  moments, a Gaussian process can be completely specified by a mean and
  covariance function. Let `S` denote the index set and `K` the space in which
  each indexed random variable takes its values (again, often R or C). The mean
  function is then a map `m: S -> K`, and the covariance function, or kernel, is
  a positive-definite function `k: (S x S) -> K`. The properties of functions
  drawn from a GP are entirely dictated (up to translation) by the form of the
  kernel function.

  This `Distribution` represents the marginal joint distribution over function
  values at a given finite collection of points `[x[1], ..., x[N]]` from the
  index set `S`. By definition, this marginal distribution is just a
  multivariate normal distribution, whose mean is given by the vector
  `[ m(x[1]), ..., m(x[N]) ]` and whose covariance matrix is constructed from
  pairwise applications of the kernel function to the given inputs:

  ```none
      | k(x[1], x[1])    k(x[1], x[2])  ...  k(x[1], x[N]) |
      | k(x[2], x[1])    k(x[2], x[2])  ...  k(x[2], x[N]) |
      |      ...              ...                 ...      |
      | k(x[N], x[1])    k(x[N], x[2])  ...  k(x[N], x[N]) |
  ```

  For this to be a valid covariance matrix, it must be symmetric and positive
  definite; hence the requirement that `k` be a positive definite function
  (which, by definition, says that the above procedure will yield PD matrices).

  We also support the inclusion of zero-mean Gaussian noise in the model, via
  the `observation_noise_variance` parameter. This augments the generative model
  to

  ```none
  f ~ GP(m, k)
  (y[i] | f, x[i]) ~ Normal(f(x[i]), s)
  ```

  where

    * `m` is the mean function
    * `k` is the covariance kernel function
    * `f` is the function drawn from the GP
    * `x[i]` are the index points at which the function is observed
    * `y[i]` are the observed values at the index points
    * `s` is the scale of the observation noise.

  Note that this class represents an *unconditional* Gaussian process; it does
  not implement posterior inference conditional on observed function
  evaluations. This class is useful, for example, if one wishes to combine a GP
  prior with a non-conjugate likelihood using MCMC to sample from the posterior.

  #### Mathematical Details

  The probability density function (pdf) is a multivariate normal whose
  parameters are derived from the GP's properties:

  ```none
  pdf(x; index_points, mean_fn, kernel) = exp(-0.5 * y) / Z
  K = (kernel.matrix(index_points, index_points) +
       (observation_noise_variance + jitter) * eye(N))
  y = (x - mean_fn(index_points))^T @ K @ (x - mean_fn(index_points))
  Z = (2 * pi)**(.5 * N) |det(K)|**(.5)
  ```

  where:

  * `index_points` are points in the index set over which the GP is defined,
  * `mean_fn` is a callable mapping the index set to the GP's mean values,
  * `kernel` is `PositiveSemidefiniteKernel`-like and represents the covariance
    function of the GP,
  * `observation_noise_variance` represents (optional) observation noise.
  * `jitter` is added to the diagonal to ensure positive definiteness up to
     machine precision (otherwise Cholesky-decomposition is prone to failure),
  * `eye(N)` is an N-by-N identity matrix.

  #### Examples

  ##### Draw joint samples from a GP prior

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp

  tfd = tfp.distributions
  psd_kernels = tfp.positive_semidefinite_kernels

  num_points = 100
  # Index points should be a collection (100, here) of feature vectors. In this
  # example, we're using 1-d vectors, so we just need to reshape the output from
  # np.linspace, to give a shape of (100, 1).
  index_points = np.expand_dims(np.linspace(-1., 1., num_points), -1)

  # Define a kernel with default parameters.
  kernel = psd_kernels.ExponentiatedQuadratic()

  gp = tfd.GaussianProcess(kernel, index_points)

  samples = gp.sample(10)
  # ==> 10 independently drawn, joint samples at `index_points`

  noisy_gp = tfd.GaussianProcess(
      kernel=kernel,
      index_points=index_points,
      observation_noise_variance=.05)
  noisy_samples = noisy_gp.sample(10)
  # ==> 10 independently drawn, noisy joint samples at `index_points`
  ```

  ##### Optimize kernel parameters via maximum marginal likelihood.

  ```python
  # Suppose we have some data from a known function. Note the index points in
  # general have shape `[b1, ..., bB, f1, ..., fF]` (here we assume `F == 1`),
  # so we need to explicitly consume the feature dimensions (just the last one
  # here).
  f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
  observed_index_points = np.expand_dims(np.random.uniform(-1., 1., 50), -1)
  # Squeeze to take the shape from [50, 1] to [50].
  observed_values = f(observed_index_points)

  # Define a kernel with trainable parameters.
  kernel = psd_kernels.ExponentiatedQuadratic(
      amplitude=tf.get_variable('amplitude', shape=(), dtype=np.float64),
      length_scale=tf.get_variable('length_scale', shape=(), dtype=np.float64))

  gp = tfd.GaussianProcess(kernel, observed_index_points)
  neg_log_likelihood = -gp.log_prob(observed_values)

  optimize = tf.train.AdamOptimizer().minimize(neg_log_likelihood)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
      _, neg_log_likelihood_ = sess.run([optimize, neg_log_likelihood])
      if i % 100 == 0:
        print("Step {}: NLL = {}".format(i, neg_log_likelihood_))
    print("Final NLL = {}".format(neg_log_likelihood_))
  ```

  """

  def __init__(self,
               kernel,
               index_points=None,
               mean_fn=None,
               observation_noise_variance=0.,
               jitter=1e-6,
               validate_args=False,
               allow_nan_stats=False,
               name='GaussianProcess'):
    """Instantiate a GaussianProcess Distribution.

    Args:
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        GP's covariance function.
      index_points: `float` `Tensor` representing finite (batch of) vector(s) of
        points in the index set over which the GP is defined. Shape has the form
        `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` and `e` is the number
        (size) of index points in each batch. Ultimately this distribution
        corresponds to a `e`-dimensional multivariate normal. The batch shape
        must be broadcastable with `kernel.batch_shape` and any batch dims
        yielded by `mean_fn`.
      mean_fn: Python `callable` that acts on `index_points` to produce a (batch
        of) vector(s) of mean values at `index_points`. Takes a `Tensor` of
        shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
        broadcastable with `[b1, ..., bB]`. Default value: `None` implies
        constant zero function.
      observation_noise_variance: `float` `Tensor` representing (batch of)
        scalar variance(s) of the noise in the Normal likelihood
        distribution of the model. If batched, the batch shape must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc.).
        Default value: `0.`
      jitter: `float` scalar `Tensor` added to the diagonal of the covariance
        matrix to ensure positive definiteness of the covariance matrix.
        Default value: `1e-6`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
        Default value: `False`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "GaussianProcess".

    Raises:
      ValueError: if `mean_fn` is not `None` and is not callable.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [index_points, observation_noise_variance, jitter], tf.float32)
      if index_points is not None:
        index_points = tf.convert_to_tensor(
            index_points, dtype=dtype, name='index_points')
      jitter = tf.convert_to_tensor(jitter, dtype=dtype, name='jitter')
      observation_noise_variance = tf.convert_to_tensor(
          observation_noise_variance,
          dtype=dtype,
          name='observation_noise_variance')

      self._kernel = kernel
      self._index_points = index_points
      # Default to a constant zero function, borrowing the dtype from
      # index_points to ensure consistency.
      if mean_fn is None:
        mean_fn = lambda x: tf.zeros([1], dtype=dtype)
      else:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')
      self._mean_fn = mean_fn
      self._observation_noise_variance = observation_noise_variance
      self._jitter = jitter

      graph_parents = [observation_noise_variance, jitter]
      if index_points is not None: graph_parents.append(index_points)

      with tf.name_scope('init'):
        super(GaussianProcess, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=graph_parents,
            name=name)

  def _is_univariate_marginal(self, index_points):
    """True if the given index_points would yield a univariate marginal.

    Args:
      index_points: the set of index set locations at which to compute the
      marginal Gaussian distribution. If this set is of size 1, the marginal is
      univariate.

    Returns:
      is_univariate: Boolean indicating whether the marginal is univariate or
      multivariate. In the case of dynamic shape in the number of index points,
      defaults to "multivariate" since that's the best we can do.
    """
    num_index_points = tf.compat.dimension_value(
        index_points.shape[-(self.kernel.feature_ndims + 1)])
    if num_index_points is None:
      warnings.warn(
          'Unable to detect statically whether the number of index_points is '
          '1. As a result, defaulting to treating the marginal GP at '
          '`index_points` as a multivariate Gaussian. This makes some methods, '
          'like `cdf` unavailable.')
    return num_index_points == 1

  def _compute_covariance(self, index_points):
    kernel_matrix = self.kernel.matrix(index_points, index_points)
    if self._is_univariate_marginal(index_points):
      # kernel_matrix thus has shape [..., 1, 1]; squeeze off the last dims and
      # tack on the observation noise variance.
      return (tf.squeeze(kernel_matrix, axis=[-2, -1]) +
              self.observation_noise_variance)
    else:
      # We are compute K + obs_noise_variance * I. The shape of this matrix
      # is going to be a broadcast of the shapes of K and obs_noise_variance *
      # I.
      broadcast_shape = distribution_util.get_broadcast_shape(
          kernel_matrix,
          # We pad with two single dimension since this represents a batch of
          # scaled identity matrices.
          self.observation_noise_variance[..., tf.newaxis, tf.newaxis])

      kernel_matrix = tf.broadcast_to(kernel_matrix, broadcast_shape)
      return _add_diagonal_shift(
          kernel_matrix, self.observation_noise_variance[..., tf.newaxis])

  def get_marginal_distribution(self, index_points=None):
    """Compute the marginal of this GP over function values at `index_points`.

    Args:
      index_points: `float` `Tensor` representing finite (batch of) vector(s) of
        points in the index set over which the GP is defined. Shape has the form
        `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` and `e` is the number
        (size) of index points in each batch. Ultimately this distribution
        corresponds to a `e`-dimensional multivariate normal. The batch shape
        must be broadcastable with `kernel.batch_shape` and any batch dims
        yielded by `mean_fn`.

    Returns:
      marginal: a `Normal` or `MultivariateNormalLinearOperator` distribution,
        according to whether `index_points` consists of one or many index
        points, respectively.
    """
    with self._name_and_control_scope('get_marginal_distribution'):
      # TODO(cgs): consider caching the result here, keyed on `index_points`.
      index_points = self._get_index_points(index_points)
      covariance = self._compute_covariance(index_points)
      loc = self._mean_fn(index_points)
      # If we're sure the number of index points is 1, we can just construct a
      # scalar Normal. This has computational benefits and supports things like
      # CDF that aren't otherwise straightforward to provide.
      if self._is_univariate_marginal(index_points):
        scale = tf.sqrt(covariance)
        # `loc` has a trailing 1 in the shape; squeeze it.
        loc = tf.squeeze(loc, axis=-1)
        return normal.Normal(
            loc=loc,
            scale=scale,
            validate_args=self._validate_args,
            allow_nan_stats=self._allow_nan_stats,
            name='marginal_distribution')
      else:
        scale = tf.linalg.LinearOperatorLowerTriangular(
            tf.linalg.cholesky(_add_diagonal_shift(covariance, self.jitter)),
            is_non_singular=True,
            name='GaussianProcessScaleLinearOperator')
        return mvn_linear_operator.MultivariateNormalLinearOperator(
            loc=loc,
            scale=scale,
            validate_args=self._validate_args,
            allow_nan_stats=self._allow_nan_stats,
            name='marginal_distribution')

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
  def jitter(self):
    return self._jitter

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
          'This GaussianProcess instance was not instantiated with a value for '
          'index_points. One must therefore be provided when calling sample, '
          'log_prob, and other such methods. In particular, one can\'t compute '
          'KL divergences to/from an instance of `GaussianProccess` with '
          'unspecified `index_points` directly. Instead, use the '
          '`get_marginal_distribution` function, which takes `index_points` as '
          'an argument and returns a `Normal` or '
          '`MultivariateNormalLinearOperator` instance, whose KL can be '
          'computed.')
    return index_points if index_points is not None else self._index_points

  def _log_prob(self, value, index_points=None):
    return self.get_marginal_distribution(index_points).log_prob(value)

  def _batch_shape_tensor(self, index_points=None):
    index_points = self._get_index_points(index_points)
    return functools.reduce(tf.broadcast_dynamic_shape, [
        tf.shape(index_points)[:-(self.kernel.feature_ndims + 1)],
        self.kernel.batch_shape_tensor(),
        tf.shape(self.observation_noise_variance)
    ])

  def _batch_shape(self, index_points=None):
    index_points = self._get_index_points(index_points)
    return functools.reduce(
        tf.broadcast_static_shape,
        [index_points.shape[:-(self.kernel.feature_ndims + 1)],
         self.kernel.batch_shape,
         self.observation_noise_variance.shape])

  def _event_shape_tensor(self, index_points=None):
    index_points = self._get_index_points(index_points)
    if self._is_univariate_marginal(index_points):
      return tf.constant([], dtype=tf.int32)
    else:
      # The examples index is one position to the left of the feature dims.
      examples_index = -(self.kernel.feature_ndims + 1)
      return tf.shape(index_points)[examples_index:examples_index + 1]

  def _event_shape(self, index_points=None):
    index_points = self._get_index_points(index_points)
    if self._is_univariate_marginal(index_points):
      return tf.TensorShape([])
    else:
      # The examples index is one position to the left of the feature dims.
      examples_index = -(self.kernel.feature_ndims + 1)
      shape = index_points.shape[examples_index:examples_index + 1]
      if shape.rank is None:
        return tf.TensorShape([None])
      return shape

  def _sample_n(self, n, seed=None, index_points=None):
    return self.get_marginal_distribution(index_points).sample(n, seed=seed)

  def _log_survival_function(self, value, index_points=None):
    return self.get_marginal_distribution(
        index_points).log_survival_function(value)

  def _survival_function(self, value, index_points=None):
    return self.get_marginal_distribution(index_points).survival_function(value)

  def _log_cdf(self, value, index_points=None):
    return self.get_marginal_distribution(index_points).log_cdf(value)

  def _entropy(self, index_points=None):
    return self.get_marginal_distribution(index_points).entropy()

  def _mean(self, index_points=None):
    return self.get_marginal_distribution(index_points).mean()

  def _quantile(self, value, index_points=None):
    return self.get_marginal_distribution(index_points).quantile(value)

  def _stddev(self, index_points=None):
    return tf.sqrt(self._variance(index_points=index_points))

  def _variance(self, index_points=None):
    index_points = self._get_index_points(index_points)

    kernel_diag = self.kernel.apply(index_points, index_points, example_ndims=1)
    if self._is_univariate_marginal(index_points):
      return (tf.squeeze(kernel_diag, axis=[-1]) +
              self.observation_noise_variance)
    else:
      # We are computing diag(K + obs_noise_variance * I) = diag(K) +
      # obs_noise_variance. We pad obs_noise_variance with a dimension in order
      # to broadcast batch shapes of kernel_diag and obs_noise_variance (since
      # kernel_diag has an extra dimension corresponding to the number of index
      # points).
      return kernel_diag + self.observation_noise_variance[..., tf.newaxis]

  def _covariance(self, index_points=None):
    # Using the result of get_marginal_distribution would involve an extra
    # matmul, and possibly even an unneceesary cholesky first. We can avoid that
    # by going straight through the kernel function.
    return self._compute_covariance(self._get_index_points(index_points))

  def _mode(self, index_points=None):
    return self.get_marginal_distribution(index_points).mode()


def _assert_kl_compatible(marginal, other):
  if ((isinstance(marginal, normal.Normal) and
       isinstance(other, normal.Normal)) or
      (isinstance(marginal,
                  mvn_linear_operator.MultivariateNormalLinearOperator) and
       isinstance(other,
                  mvn_linear_operator.MultivariateNormalLinearOperator))):
    return
  raise ValueError(
      'Attempting to compute KL between a GP marginal and a distribution of '
      'incompatible type. GP marginal has type {} and other distribution has '
      'type {}.'.format(type(marginal), type(other)))


def _kl_gp_compatible(gp, compatible, name):
  with tf.name_scope(name):
    marginal = gp.get_marginal_distribution()
    _assert_kl_compatible(marginal, compatible)
    return kullback_leibler.kl_divergence(marginal, compatible)


def _kl_compatible_gp(compatible, gp, name):
  with tf.name_scope(name):
    marginal = gp.get_marginal_distribution()
    _assert_kl_compatible(marginal, compatible)
    return kullback_leibler.kl_divergence(compatible, marginal)


@kullback_leibler.RegisterKL(GaussianProcess, normal.Normal)
def _kl_gp_normal(gp, n, name=None):
  """Calculate the batched KL divergence KL(gp || n).

  Args:
    gp: instance of a GaussianProcess distribution object.
    n: instance of a Normal distribution object.
    name: (optional) Name to use for created operations.
      default is 'kl_gp_normal'.

  Returns:
    Batchwise KL(gp || n)
  """
  return _kl_gp_compatible(gp, n, name or 'kl_gp_normal')


@kullback_leibler.RegisterKL(
    GaussianProcess, mvn_linear_operator.MultivariateNormalLinearOperator)
def _kl_gp_mvn(gp, mvn, name=None):
  """Calculate the batched KL divergence KL(gp || mvn).

  Args:
    gp: instance of a GaussianProcess distribution object.
    mvn: instance of a multivariate Normal distribution object (any subclass of
      MultivariateNormalLinearOperator)
    name: (optional) Name to use for created operations.
      default is 'kl_gp_mvn'.

  Returns:
    Batchwise KL(gp || mvn)
  """
  return _kl_gp_compatible(gp, mvn, name or 'kl_gp_mvn')


@kullback_leibler.RegisterKL(normal.Normal, GaussianProcess)
def _kl_normal_gp(n, gp, name=None):
  """Calculate the batched KL divergence KL(gp || n).

  Args:
    n: instance of a Normal distribution object.
    gp: instance of a GaussianProcess distribution object.
    name: (optional) Name to use for created operations.
      default is 'kl_normal_gp'.

  Returns:
    Batchwise KL(n || gp)
  """
  return _kl_compatible_gp(n, gp, name or 'kl_normal_gp')


@kullback_leibler.RegisterKL(
    mvn_linear_operator.MultivariateNormalLinearOperator, GaussianProcess)
def _kl_mvn_gp(mvn, gp, name=None):
  """Calculate the batched KL divergence KL(mvn || gp).

  Args:
    mvn: instance of a multivariate Normal distribution object (any subclass of
      MultivariateNormalLinearOperator)
    gp: instance of a GaussianProcess distribution object.
    name: (optional) Name to use for created operations.
      default is 'kl_mvn_gp'.

  Returns:
    Batchwise KL(mvn || gp)
  """
  return _kl_compatible_gp(mvn, gp, name or 'kl_mvn_gp')
