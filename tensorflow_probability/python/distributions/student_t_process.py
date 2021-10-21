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
"""The StudentTProcess distribution class."""

import warnings

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import cholesky_util
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import multivariate_student_t
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'StudentTProcess',
]


def _add_diagonal_shift(matrix, shift):
  return tf.linalg.set_diag(
      matrix, tf.linalg.diag_part(matrix) + shift, name='add_diagonal_shift')


def make_cholesky_factored_marginal_fn(cholesky_fn):
  """Construct a `marginal_fn` for use with `tfd.StudentTProcess`.

  The returned function computes the Cholesky factorization of the input
  covariance plus a diagonal jitter, and uses that for the `scale` of a
  `tfd.MultivariateNormalLinearOperator`.

  Args:
    cholesky_fn: Callable which takes a single (batch) matrix argument and
      returns a Cholesky-like lower triangular factor.

  Returns:
    marginal_fn: A Python function that takes a location, covariance matrix,
      optional `validate_args`, `allow_nan_stats` and `name` arguments, and
      returns a `tfd.MultivariateNormalLinearOperator`.
  """
  def marginal_fn(
      df,
      loc,
      covariance,
      validate_args=False,
      allow_nan_stats=False,
      name='marginal_distribution'):
    squared_scale = ((df - 2.) / df)[
        ..., tf.newaxis, tf.newaxis] * covariance
    scale = tf.linalg.LinearOperatorLowerTriangular(
        cholesky_fn(squared_scale),
        is_non_singular=True,
        name='StudentTProcessScaleLinearOperator')
    return multivariate_student_t.MultivariateStudentTLinearOperator(
        df=df,
        loc=loc,
        scale=scale,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)

  return marginal_fn


class StudentTProcess(distribution.AutoCompositeTensorDistribution):
  """Marginal distribution of a Student's T process at finitely many points.

  A Student's T process (TP) is an indexed collection of random variables, any
  finite collection of which are jointly Multivariate Student's T. While this
  definition applies to finite index sets, it is typically implicit that the
  index set is infinite; in applications, it is often some finite dimensional
  real or complex vector space. In such cases, the TP may be thought of as a
  distribution over (real- or complex-valued) functions defined over the index
  set.

  Just as Student's T distributions are fully specified by their degrees of
  freedom, location and scale, a Student's T process can be completely specified
  by a degrees of freedom parameter, mean function and covariance function.
  Let `S` denote the index set and `K` the space in
  which each indexed random variable takes its values (again, often R or C).
  The mean function is then a map `m: S -> K`, and the covariance function,
  or kernel, is a positive-definite function `k: (S x S) -> K`. The properties
  of functions drawn from a TP are entirely dictated (up to translation) by
  the form of the kernel function.

  This `Distribution` represents the marginal joint distribution over function
  values at a given finite collection of points `[x[1], ..., x[N]]` from the
  index set `S`. By definition, this marginal distribution is just a
  multivariate Student's T distribution, whose mean is given by the vector
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

  Note also we use a parameterization as suggested in [1], which requires `df`
  to be greater than 2. This allows for the covariance for any finite
  dimensional marginal of the TP (a multivariate Student's T distribution) to
  just be the PD matrix generated by the kernel.


  #### Mathematical Details

  The probability density function (pdf) is a multivariate Student's T whose
  parameters are derived from the TP's properties:

  ```none
  pdf(x; df, index_points, mean_fn, kernel) = MultivariateStudentT(df, loc, K)
  K = (df - 2) / df  * (kernel.matrix(index_points, index_points) +
       observation_noise_variance * eye(N))
  loc = (x - mean_fn(index_points))^T @ K @ (x - mean_fn(index_points))
  ```

  where:

  * `df` is the degrees of freedom parameter for the TP.
  * `index_points` are points in the index set over which the TP is defined,
  * `mean_fn` is a callable mapping the index set to the TP's mean values,
  * `kernel` is `PositiveSemidefiniteKernel`-like and represents the covariance
    function of the TP,
  * `observation_noise_variance` is a term added to the diagonal of the kernel
    matrix. In the limit of `df` to `inf`, this represents the observation noise
    of a gaussian likelihood.
  * `eye(N)` is an N-by-N identity matrix.

  #### Examples

  ##### Draw joint samples from a TP prior

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp

  tfd = tfp.distributions
  psd_kernels = tfp.math.psd_kernels

  num_points = 100
  # Index points should be a collection (100, here) of feature vectors. In this
  # example, we're using 1-d vectors, so we just need to reshape the output from
  # np.linspace, to give a shape of (100, 1).
  index_points = np.expand_dims(np.linspace(-1., 1., num_points), -1)

  # Define a kernel with default parameters.
  kernel = psd_kernels.ExponentiatedQuadratic()

  tp = tfd.StudentTProcess(3., kernel, index_points)

  samples = tp.sample(10)
  # ==> 10 independently drawn, joint samples at `index_points`

  noisy_tp = tfd.StudentTProcess(
      df=3.,
      kernel=kernel,
      index_points=index_points)
  noisy_samples = noisy_tp.sample(10)
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

  amplitude = tfp.util.TransformedVariable(
      1., tfp.bijectors.Softplus(), dtype=np.float64, name='amplitude')
  length_scale = tfp.util.TransformedVariable(
      1., tfp.bijectors.Softplus(), dtype=np.float64, name='length_scale')

  # Define a kernel with trainable parameters.
  kernel = psd_kernels.ExponentiatedQuadratic(
      amplitude=amplitude,
      length_scale=length_scale)

  tp = tfd.StudentTProcess(3., kernel, observed_index_points)

  optimizer = tf.optimizers.Adam()

  @tf.function
  def optimize():
    with tf.GradientTape() as tape:
      loss = -tp.log_prob(observed_values)
    grads = tape.gradient(loss, tp.trainable_variables)
    optimizer.apply_gradients(zip(grads, tp.trainable_variables))
    return loss

  for i in range(1000):
    nll = optimize()
    if i % 100 == 0:
      print("Step {}: NLL = {}".format(i, nll))
  print("Final NLL = {}".format(nll))
  ```

  #### References

  [1]: Amar Shah, Andrew Gordon Wilson, and Zoubin Ghahramani. Student-t
       Processes as Alternatives to Gaussian Processes. In _Artificial
       Intelligence and Statistics_, 2014.
       https://www.cs.cmu.edu/~andrewgw/tprocess.pdf
  """

  @deprecation.deprecated_args(
      '2021-06-26',
      '`jitter` is deprecated; please use `marginal_fn` directly.',
      'jitter')
  def __init__(self,
               df,
               kernel,
               index_points=None,
               mean_fn=None,
               observation_noise_variance=0.,
               marginal_fn=None,
               cholesky_fn=None,
               jitter=1e-6,
               validate_args=False,
               allow_nan_stats=False,
               name='StudentTProcess'):
    """Instantiate a StudentTProcess Distribution.

    Args:
      df: Positive Floating-point `Tensor` representing the degrees of freedom.
        Must be greater than 2.
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        TP's covariance function.
      index_points: `float` `Tensor` representing finite (batch of) vector(s) of
        points in the index set over which the TP is defined. Shape has the form
        `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` and `e` is the number
        (size) of index points in each batch. Ultimately this distribution
        corresponds to a `e`-dimensional multivariate Student's T. The batch
        shape must be broadcastable with `kernel.batch_shape` and any batch dims
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
      marginal_fn: A Python callable that takes a location, covariance matrix,
        optional `validate_args`, `allow_nan_stats` and `name` arguments, and
        returns a multivariate normal subclass of `tfd.Distribution`.
        Default value: `None`, in which case a Cholesky-factorizing function is
        is created using `make_cholesky_factored_marginal_fn` and the
        `jitter` argument.
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
        in which case `make_cholesky_with_jitter_fn` is used with the `jitter`
        parameter. At most one of `cholesky_fn` and `marginal_fn` should be set.
      jitter: `float` scalar `Tensor` added to the diagonal of the covariance
        matrix to ensure positive definiteness of the covariance matrix.
        This argument is ignored if `cholesky_fn` is set.
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
        Default value: "StudentTProcess".

    Raises:
      ValueError: if `mean_fn` is not `None` and is not callable.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [df, kernel, index_points, observation_noise_variance, jitter],
          tf.float32)
      df = tensor_util.convert_nonref_to_tensor(df, dtype=dtype, name='df')
      observation_noise_variance = tensor_util.convert_nonref_to_tensor(
          observation_noise_variance,
          dtype=dtype,
          name='observation_noise_variance')
      index_points = tensor_util.convert_nonref_to_tensor(
          index_points, dtype=dtype, name='index_points')
      jitter = tensor_util.convert_nonref_to_tensor(
          jitter, dtype=dtype, name='jitter')

      self._kernel = kernel
      self._index_points = index_points
      # Default to a constant zero function, borrowing the dtype from
      # index_points to ensure consistency.
      if mean_fn is None:
        mean_fn = lambda x: tf.zeros([1], dtype=dtype)
      else:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')
      self._df = df
      self._observation_noise_variance = observation_noise_variance
      self._mean_fn = mean_fn
      self._jitter = jitter
      self._cholesky_fn = cholesky_fn
      if marginal_fn is not None and cholesky_fn is not None:
        raise ValueError(
            'At most one of `marginal_fn` and `cholesky_fn` should be set.')
      if marginal_fn is None:
        if self._cholesky_fn is None:
          self._cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn(
              jitter)
        self._marginal_fn = make_cholesky_factored_marginal_fn(
            self._cholesky_fn)
      else:
        self._marginal_fn = marginal_fn

      with tf.name_scope('init'):
        super(StudentTProcess, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(  # pylint: disable=g-long-lambda
                    low=dtype_util.as_numpy_dtype(dtype)(2.)))),
        index_points=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims + 1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
        kernel=parameter_properties.BatchedComponentProperties(),
        observation_noise_variance=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED))

  def _is_univariate_marginal(self, index_points):
    """True if the given index_points would yield a univariate marginal.

    Args:
      index_points: the set of index set locations at which to compute the
      marginal Student T distribution. If this set is of size 1, the marginal is
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
          '1. As a result, defaulting to treating the marginal Student T '
          'Process at `index_points` as a multivariate Student T. This makes '
          'some methods, like `cdf` unavailable.')
    return num_index_points == 1

  def _compute_covariance(self, index_points):
    kernel_matrix = self.kernel.matrix(index_points, index_points)
    if self._is_univariate_marginal(index_points):
      # kernel_matrix thus has shape [..., 1, 1]; squeeze off the last dims and
      # tack on the observation noise variance.
      return (tf.squeeze(kernel_matrix, axis=[-2, -1]) +
              self.observation_noise_variance)
    else:
      observation_noise_variance = tf.convert_to_tensor(
          self.observation_noise_variance)
      # We are compute K + obs_noise_variance * I. The shape of this matrix
      # is going to be a broadcast of the shapes of K and obs_noise_variance *
      # I.
      broadcast_shape = distribution_util.get_broadcast_shape(
          kernel_matrix,
          # We pad with two single dimension since this represents a batch of
          # scaled identity matrices.
          observation_noise_variance[..., tf.newaxis, tf.newaxis])

      kernel_matrix = tf.broadcast_to(kernel_matrix, broadcast_shape)
      return _add_diagonal_shift(
          kernel_matrix, observation_noise_variance[..., tf.newaxis])

  def get_marginal_distribution(self, index_points=None):
    """Compute the marginal over function values at `index_points`.

    Args:
      index_points: `float` `Tensor` representing finite (batch of) vector(s) of
        points in the index set over which the TP is defined. Shape has the form
        `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` and `e` is the number
        (size) of index points in each batch. Ultimately this distribution
        corresponds to a `e`-dimensional multivariate student t. The batch shape
        must be broadcastable with `kernel.batch_shape` and any batch dims
        yielded by `mean_fn`.

    Returns:
      marginal: a `StudentT` or `MultivariateStudentT` distribution,
        according to whether `index_points` consists of one or many index
        points, respectively.
    """
    with self._name_and_control_scope('get_marginal_distribution'):
      df = tf.convert_to_tensor(self.df)
      index_points = self._get_index_points(index_points)
      covariance = self._compute_covariance(index_points)
      loc = self._mean_fn(index_points)

      # If we're sure the number of index points is 1, we can just construct a
      # scalar Normal. This has computational benefits and supports things like
      # CDF that aren't otherwise straightforward to provide.
      if self._is_univariate_marginal(index_points):
        squared_scale = (df - 2.) / df * covariance
        scale = tf.sqrt(squared_scale)
        # `loc` has a trailing 1 in the shape; squeeze it.
        loc = tf.squeeze(loc, axis=-1)
        return student_t.StudentT(
            df=df,
            loc=loc,
            scale=scale,
            validate_args=self.validate_args,
            allow_nan_stats=self.allow_nan_stats,
            name='marginal_distribution')
      else:
        return self._marginal_fn(
            df=df,
            loc=loc,
            covariance=covariance,
            validate_args=self.validate_args,
            allow_nan_stats=self.allow_nan_stats,
            name='marginal_distribution')

  @property
  def df(self):
    return self._df

  @property
  def observation_noise_variance(self):
    return self._observation_noise_variance

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
  def marginal_fn(self):
    return self._marginal_fn

  @property
  def cholesky_fn(self):
    return self._cholesky_fn

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
          'This StudentTProcess instance was not instantiated with a value for '
          'index_points. One must therefore be provided when calling sample, '
          'log_prob, and other such methods.')
    return (index_points if index_points is not None
            else tf.convert_to_tensor(self._index_points))

  def _log_prob(self, value, index_points=None):
    return self.get_marginal_distribution(index_points).log_prob(value)

  def _event_shape_tensor(self, index_points=None):
    index_points = self._get_index_points(index_points)
    if self._is_univariate_marginal(index_points):
      return tf.constant([], dtype=tf.int32)
    else:
      # The examples index is one position to the left of the feature dims.
      examples_index = -(self.kernel.feature_ndims + 1)
      return tf.shape(index_points)[examples_index:examples_index + 1]

  def _event_shape(self, index_points=None):
    index_points = (
        index_points if index_points is not None else self._index_points)
    if self._is_univariate_marginal(index_points):
      return tf.TensorShape([])
    else:
      # The examples index is one position to the left of the feature dims.
      examples_index = -(self.kernel.feature_ndims + 1)
      shape = index_points.shape[examples_index:examples_index + 1]
      if tensorshape_util.rank(shape) is None:
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
    # matmul, and possibly even an unnecessary cholesky first. We can avoid that
    # by going straight through the kernel function.
    return self._compute_covariance(self._get_index_points(index_points))

  def _mode(self, index_points=None):
    return self.get_marginal_distribution(index_points).mode()

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.df):
      assertions.append(
          assert_util.assert_greater(
              self.df, dtype_util.as_numpy_dtype(self.df.dtype)(2.),
              message='`df` must be greater than 2.'))
    return assertions

  def posterior_predictive(
      self, observations, predictive_index_points=None, **kwargs):
    """Return the posterior predictive distribution associated with this distribution.

    Returns the posterior predictive distribution `p(Y' | X, Y, X')` where:
      * `X'` is `predictive_index_points`
      * `X` is `self.index_points`.
      * `Y` is `observations`.

    This is equivalent to using the
    `StudentTProcessRegressionModel.precompute_regression_model` method.

    WARNING: This method assumes `index_points` is the only varying parameter
    (i.e. is a `Variable` / changes after initialization) and hence is not
    tape-safe.

    Args:
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `self.index_points`. Shape has the form `[b1, ..., bB, e]`, which
        must be broadcastable with the batch and example shapes of
        `self.index_points`. The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
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
      stprm: An instance of `Distribution` that represents the posterior
        predictive.
    """
    from tensorflow_probability.python.distributions import student_t_process_regression_model as stprm  # pylint:disable=g-import-not-at-top
    if self.index_points is None:
      raise ValueError(
          'Expected that `self.index_points` is not `None`. Using '
          '`self.index_points=None` is equivalent to using a `StudentTProcess` '
          'prior, which this class encapsulates.')
    argument_dict = {
        'df': self.df,
        'kernel': self.kernel,
        'observation_index_points': self.index_points,
        'observations': observations,
        'index_points': predictive_index_points,
        'observation_noise_variance': self.observation_noise_variance,
        'cholesky_fn': self.cholesky_fn,
        'mean_fn': self.mean_fn,
        'validate_args': self.validate_args,
        'allow_nan_stats': self.allow_nan_stats
    }
    argument_dict.update(**kwargs)

    return stprm.StudentTProcessRegressionModel.precompute_regression_model(
        **argument_dict)

