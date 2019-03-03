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

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'GaussianProcess',
]


def _add_diagonal_shift(matrix, shift):
  return tf.linalg.set_diag(
      matrix, tf.linalg.diag_part(matrix) + shift, name='add_diagonal_shift')


class GaussianProcess(mvn_linear_operator.MultivariateNormalLinearOperator):
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
  noisy_samples = noise_gp.sample(10)
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
      amplitude=tf.get_variable('amplitude', np.float32),
      length_scale=tf.get_variable('length_scale', np.float32))

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
               index_points,
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
      observation_noise_variance: `float` `Tensor` representing the variance
        of the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
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
    with tf.compat.v1.name_scope(
        name, values=[index_points, observation_noise_variance,
                      jitter]) as name:
      dtype = dtype_util.common_dtype(
          [index_points, observation_noise_variance, jitter], tf.float32)
      index_points = tf.convert_to_tensor(
          value=index_points, dtype=dtype, name='index_points')
      jitter = tf.convert_to_tensor(value=jitter, dtype=dtype, name='jitter')
      observation_noise_variance = tf.convert_to_tensor(
          value=observation_noise_variance,
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

      with tf.compat.v1.name_scope('init', values=[index_points, jitter]):
        kernel_matrix = _add_diagonal_shift(
            kernel.matrix(self.index_points, self.index_points),
            jitter + observation_noise_variance)
        self._covariance_matrix = kernel_matrix

        scale = tf.linalg.LinearOperatorLowerTriangular(
            tf.linalg.cholesky(kernel_matrix),
            is_non_singular=True,
            name='GaussianProcessScaleLinearOperator')

        super(GaussianProcess, self).__init__(
            loc=mean_fn(index_points),
            scale=scale,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)
        self._parameters = parameters
        self._graph_parents = [index_points, jitter]

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

  def _covariance(self):
    return self._covariance_matrix
