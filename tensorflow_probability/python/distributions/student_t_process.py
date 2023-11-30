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

import functools
import warnings

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import cholesky_util
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import multivariate_student_t
from tensorflow_probability.python.distributions.internal import stochastic_process_util
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import batch_shape_lib
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import linalg
from tensorflow_probability.python.math import special
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'StudentTProcess',
]


_ALWAYS_YIELD_MVST_DEPRECATION_WARNING = (
    '`always_yield_multivariate_student_t` is deprecated. This arg is now '
    'ignored and will be removed after 2023-11-15. A `StudentTProcess` '
    'evaluated at a single index point now always has event shape `[1]` (the '
    'previous behavior for `always_yield_multivariate_student_t=True`). To '
    'reproduce the previous behavior of '
    '`always_yield_multivariate_student_t=False`, squeeze the rightmost '
    'singleton dimension from the output of `mean`, `sample`, etc.')


_GET_MARGINAL_DISTRIBUTION_ALREADY_WARNED = False


def make_cholesky_factored_marginal_fn(cholesky_fn):
  """Construct a `marginal_fn` for use with `tfd.StudentTProcess`.

  The returned function computes the Cholesky factorization of the input
  covariance plus a diagonal jitter, and uses that for the `scale` of a
  `tfd.MultivariateStudentTLinearOperator`.

  Args:
    cholesky_fn: Callable which takes a single (batch) matrix argument and
      returns a Cholesky-like lower triangular factor.

  Returns:
    marginal_fn: A Python function that takes a location, covariance matrix,
      optional `validate_args`, `allow_nan_stats` and `name` arguments, and
      returns a `tfd.MultivariateStudentTLinearOperator`.
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

  optimizer = tf_keras.optimizers.Adam()

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
  @deprecation.deprecated_args(
      '2023-11-15',
      _ALWAYS_YIELD_MVST_DEPRECATION_WARNING,
      'always_yield_multivariate_student_t')
  def __init__(self,
               df,
               kernel,
               index_points=None,
               mean_fn=None,
               observation_noise_variance=0.,
               marginal_fn=None,
               cholesky_fn=None,
               jitter=1e-6,
               always_yield_multivariate_student_t=None,
               validate_args=False,
               allow_nan_stats=False,
               name='StudentTProcess'):
    """Instantiate a StudentTProcess Distribution.

    Args:
      df: Positive Floating-point `Tensor` representing the degrees of freedom.
        Must be greater than 2.
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        TP's covariance function.
      index_points: (Nested) `float` `Tensor` representing finite (batch of)
        vector(s) of points in the index set over which the TP is defined. Shape
        (or shape of each nested component) has the form
        `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` (or its corresponding
        nested component) and `e` is the number (size) of index points in each
        batch. Ultimately this distribution corresponds to a `e`-dimensional
        multivariate Student's T. The batch shape must be broadcastable with
        `kernel.batch_shape` and any batch dims yielded by `mean_fn`.
      mean_fn: Python `callable` that acts on `index_points` to produce a (batch
        of) vector(s) of mean values at `index_points`. Takes a (nested)
        `Tensor` of shape `[b1, ..., bB, e, f1, ..., fF]` and returns a `Tensor`
        whose shape is broadcastable with `[b1, ..., bB, e]`.
        Default value: `None` implies constant zero function.
      observation_noise_variance: `float` `Tensor` representing (batch of)
        scalar variance(s) of the noise in the Normal likelihood
        distribution of the model. If batched, the batch shape must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc.).
        Default value: `0.`
      marginal_fn: A Python callable that takes a location, covariance matrix,
        optional `validate_args`, `allow_nan_stats` and `name` arguments, and
        returns a multivariate Student T subclass of `tfd.Distribution`.
        Default value: `None`, in which case a Cholesky-factorizing function
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
      always_yield_multivariate_student_t: Deprecated and ignored.
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
      input_dtype = dtype_util.common_dtype(
          dict(
              kernel=kernel,
              index_points=index_points,
          ),
          dtype_hint=nest_util.broadcast_structure(
              kernel.feature_ndims, tf.float32))

      # If the input dtype is non-nested float, we infer a single dtype for the
      # input and the float parameters, which is also the dtype of the STP's
      # samples, log_prob, etc. If the input dtype is nested (or not float), we
      # do not use it to infer the STP's float dtype.
      if (not tf.nest.is_nested(input_dtype) and
          dtype_util.is_floating(input_dtype)):
        dtype = dtype_util.common_dtype(
            dict(
                kernel=kernel,
                index_points=index_points,
                observation_noise_variance=observation_noise_variance,
                jitter=jitter,
                df=df,
            ),
            dtype_hint=tf.float32,
        )
        input_dtype = dtype
      else:
        dtype = dtype_util.common_dtype(
            dict(
                df=df,
                observation_noise_variance=observation_noise_variance,
                jitter=jitter,
            ), dtype_hint=tf.float32)

      if index_points is not None:
        index_points = nest_util.convert_to_nested_tensor(
            index_points, dtype=input_dtype, name='index_points',
            convert_ref=False, allow_packing=True)
      df = tensor_util.convert_nonref_to_tensor(df, dtype=dtype, name='df')
      observation_noise_variance = tensor_util.convert_nonref_to_tensor(
          observation_noise_variance,
          dtype=dtype,
          name='observation_noise_variance')
      jitter = tensor_util.convert_nonref_to_tensor(
          jitter, dtype=dtype, name='jitter')

      self._kernel = kernel
      self._index_points = index_points
      # Default to a constant zero function, borrowing the dtype from
      # index_points to ensure consistency.
      mean_fn = stochastic_process_util.maybe_create_mean_fn(mean_fn, dtype)
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

      self._always_yield_multivariate_student_t = (
          always_yield_multivariate_student_t)
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
            event_ndims=lambda self: tf.nest.map_structure(  # pylint: disable=g-long-lambda
                lambda nd: nd + 1, self.kernel.feature_ndims),
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        kernel=parameter_properties.BatchedComponentProperties(),
        observation_noise_variance=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED))

  def get_marginal_distribution(self, index_points=None):
    """Compute the marginal over function values at `index_points`.

    Args:
      index_points: `float` (nested) `Tensor` representing finite (batch of)
        vector(s) of points in the index set over which the STP is defined.
        Shape (or the shape of each nested component) has the form
        `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` (or its corresponding
        nested component) and `e` is the number (size) of index points in each
        batch. Ultimately this distribution corresponds to a `e`-dimensional
        multivariate student t. The batch shape must be broadcastable with
        `kernel.batch_shape` and any batch dims yielded by `mean_fn`.

    Returns:
      marginal: a Student T distribution with vector event shape.
    """
    with self._name_and_control_scope('get_marginal_distribution'):
      global _GET_MARGINAL_DISTRIBUTION_ALREADY_WARNED
      if (not _GET_MARGINAL_DISTRIBUTION_ALREADY_WARNED and  # pylint: disable=protected-access
          self._always_yield_multivariate_student_t is not None):  # pylint: disable=protected-access
        warnings.warn(
            'The `always_yield_multivariate_student_t` arg to '
            '`StudentTProcess.__init__` is now ignored and '
            '`get_marginal_distribution` always returns a Student T '
            'distribution with vector event shape. This was the previous '
            'behavior of `always_yield_multivariate_student_t=True`. To '
            'recover the behavior of '
            '`always_yield_multivariate_student_t=False` when `index_points`'
            'contains a single index point, build a scalar `StudentT`'
            'distribution as follows: '
            '`mvst = get_marginal_distribution(index_points); `'
            '`dist = tfd.StudentT(mvst.loc[..., 0], `'
            '`scale=mvst.stddev()[..., 0], mvst.df)`. To suppress these '
            'warnings, build the `StudentTProcess` with '
            '`always_yield_multivariate_student_t=True`.',
            FutureWarning)
        _GET_MARGINAL_DISTRIBUTION_ALREADY_WARNED = True  # pylint: disable=protected-access
      df = tf.convert_to_tensor(self.df)
      index_points = self._get_index_points(index_points)
      covariance = stochastic_process_util.compute_kernel_matrix(
          self.kernel, index_points, self.observation_noise_variance)
      loc = self._mean_fn(index_points)
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
    return nest_util.convert_to_nested_tensor(
        index_points if index_points is not None else self._index_points,
        dtype_hint=self.kernel.dtype, allow_packing=True)

  @distribution_util.AppendDocstring(kwargs_dict={
      'index_points':
          'optional `float` `Tensor` representing a finite (batch of) of '
          'points in the index set over which this STP is defined. The shape '
          '(or shape of each nested component) has the form `[b1, ..., bB, e,'
          'f1, ..., fF]` where `F` is the ' 'number of feature dimensions and '
          'must equal ' '`self.kernel.feature_ndims` (or its corresponding '
          'nested component) and `e` is the number of index points in each '
          'batch. Ultimately, this distribution corresponds to an '
          '`e`-dimensional multivariate Student T. The batch shape must be '
          'broadcastable with `kernel.batch_shape` and any batch dims yielded'
          'by `mean_fn`. If not specified, `self.index_points` is used. '
          'Default value: `None`.',
      'is_missing':
          'optional `bool` `Tensor` of shape `[..., e]`, where `e` is the '
          'number of index points in each batch.  Represents a batch of '
          'Boolean masks.  When `is_missing` is not `None`, the returned '
          'log-prob is for the *marginal* distribution, in which all '
          'dimensions for which `is_missing` is `True` have been marginalized '
          'out.  The batch dimensions of `is_missing` must broadcast with the '
          'sample and batch dimensions of `value` and of this `Distribution`. '
          'Default value: `None`.'
  })
  def _log_prob(self, value, index_points=None, is_missing=None):
    if is_missing is not None:
      is_missing = tf.convert_to_tensor(is_missing)
    value = tf.convert_to_tensor(value, dtype=self.dtype)
    index_points = self._get_index_points(index_points)
    observation_noise_variance = tf.convert_to_tensor(
        self.observation_noise_variance)
    loc, covariance = stochastic_process_util.get_loc_and_kernel_matrix(
        kernel=self.kernel,
        index_points=index_points,
        mean_fn=self._mean_fn,
        observation_noise_variance=observation_noise_variance,
        is_missing=is_missing,
        mask_loc=False)
    event_shape = self._event_shape_tensor(index_points=index_points)
    # Use marginal_fn if cholesky_fn doesn't exist.
    if self.cholesky_fn is None:
      # TODO(b/280821509): Add support for `is_missing` with `marginal_fn`.
      if is_missing is not None:
        raise ValueError(
            '`is_missing` can not be used with `marginal_fn`. '
            'If you need this functionality, please contact '
            '`tfprobability@tensorflow.org`.')
      return self.get_marginal_distribution(index_points).log_prob(value)
    df = tf.convert_to_tensor(self.df)
    value = value - loc
    num_masked_dims = 0.
    if is_missing is not None:
      value = tf.where(is_missing, 0., value)
      num_masked_dims = tf.cast(
          tf.math.count_nonzero(is_missing, axis=-1), self.dtype)
    num_dims = tf.cast(event_shape[-1], self.dtype)

    chol_covariance = self.cholesky_fn(covariance)  # pylint: disable=not-callable
    lp = -(df + num_dims - num_masked_dims) / 2. * tf.math.log1p(
        linalg.hpsd_quadratic_form_solvevec(
            covariance, value, cholesky_matrix=chol_covariance) / (df - 2.))
    lp = lp - 0.5 * linalg.hpsd_logdet(
        covariance, cholesky_matrix=chol_covariance)

    lp = lp - special.log_gamma_difference(
        (num_dims - num_masked_dims) / 2., df / 2.)
    lp = lp - (num_dims - num_masked_dims) / 2. * (
        tf.math.log(df - 2.) + tf.cast(np.log(np.pi), self.dtype))
    return lp

  def _event_shape_tensor(self, index_points=None):
    index_points = self._get_index_points(index_points)
    return stochastic_process_util.event_shape_tensor(self.kernel, index_points)

  def _event_shape(self, index_points=None):
    index_points = (
        index_points if index_points is not None else self._index_points)
    return stochastic_process_util.event_shape(self.kernel, index_points)

  def _batch_shape(self, index_points=None):
    # TODO(b/249858459): Update `batch_shape_lib` so it can take override
    # parameters.
    result = batch_shape_lib.inferred_batch_shape(self)
    if index_points is not None:
      shapes = tf.nest.map_structure(
          lambda t, nd: t.shape[:-(nd + 1)],
          index_points, self.kernel.feature_ndims)
      flat_shapes = nest.flatten_up_to(self.kernel.feature_ndims, shapes)
      return functools.reduce(ps.broadcast_shape, flat_shapes, result)
    return result

  def _sample_n(self, n, seed=None, index_points=None):
    return self.get_marginal_distribution(index_points).sample(n, seed=seed)

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
    # We are computing diag(K + obs_noise_variance * I) = diag(K) +
    # obs_noise_variance. We pad obs_noise_variance with a dimension in order
    # to broadcast batch shapes of kernel_diag and obs_noise_variance (since
    # kernel_diag has an extra dimension corresponding to the number of index
    # points).
    return kernel_diag + self.observation_noise_variance[..., tf.newaxis]

  def _covariance(self, index_points=None):
    observation_noise_variance = tf.convert_to_tensor(
        self.observation_noise_variance)
    index_points = self._get_index_points(index_points)
    return stochastic_process_util.compute_kernel_matrix(
        kernel=self.kernel,
        index_points=index_points,
        observation_noise_variance=observation_noise_variance)

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
      predictive_index_points: `float` (nested) `Tensor` representing finite
        collection, or batch of collections, of points in the index set over
        which the TP is defined. Shape (or shape of each nested component) has
        the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of
        feature dimensions and must equal `kernel.feature_ndims` (or its
        corresponding nested component) and `e` is the number (size) of
        predictive index points in each batch. The batch shape must be
        broadcastable with this distributions `batch_shape`.
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

