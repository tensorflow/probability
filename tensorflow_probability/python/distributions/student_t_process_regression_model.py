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
"""The StudentTProcessRegressionModel distribution class."""

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import cholesky_util
from tensorflow_probability.python.distributions import student_t_process
from tensorflow_probability.python.distributions.internal import stochastic_process_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels import schur_complement as schur_complement_lib
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'StudentTProcessRegressionModel',
]


_ALWAYS_YIELD_MVST_DEPRECATION_WARNING = (
    '`always_yield_multivariate_student_t` is deprecated. This arg is now '
    'ignored and will be removed after 2023-11-15. A '
    '`StudentTProcessRegressionModel` evaluated at a single index point now '
    'always has event shape `[1]` (the previous behavior for '
    '`always_yield_multivariate_student_t=True`). To reproduce the previous '
    'behavior of `always_yield_multivariate_student_t=False`, squeeze the '
    'rightmost singleton dimension from the output of `mean`, `sample`, etc.')


class DampedSchurComplement(psd_kernel.AutoCompositeTensorPsdKernel):
  """Schur complement kernel, damped by scalar factors.

  This kernel is the same as the SchurComplement kernel, except we multiply by
  the following factor:

  `(df + b - 2) / (df + n - 2)`, where:

    * `df` is the degrees of freedom parameter.
    * `n` is the number of observations for `fixed_inputs`.
    * `b` is `||divisor_matrix_cholesky^-1 fixed_inputs_observations|| ** 2`.
  """

  def __init__(self,
               df,
               schur_complement,
               fixed_inputs_observations,
               observations_is_missing=None,
               validate_args=False,
               name='DampedSchurComplement'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      kernel_dtype = schur_complement.dtype

      # If the input dtype is non-nested float, we infer a single dtype for the
      # input and the float parameters, which is also the dtype of the STP's
      # samples, log_prob, etc. If the input dtype is nested (or not float), we
      # do not use it to infer the STP's float dtype.
      if (not tf.nest.is_nested(kernel_dtype) and
          dtype_util.is_floating(kernel_dtype)):
        dtype = dtype_util.common_dtype(
            dict(
                schur_complement=schur_complement,
                fixed_inputs_observations=fixed_inputs_observations,
                df=df,
            ),
            dtype_hint=tf.float32,
        )
        kernel_dtype = dtype
      else:
        dtype = dtype_util.common_dtype(
            dict(
                fixed_inputs_observations=fixed_inputs_observations,
                df=df,
            ),
            dtype_hint=tf.float32,
        )
      self._schur_complement = schur_complement
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      self._observations_is_missing = tensor_util.convert_nonref_to_tensor(
          observations_is_missing,
          name='observations_is_missing', dtype=tf.bool)
      self._fixed_inputs_observations = tensor_util.convert_nonref_to_tensor(
          fixed_inputs_observations,
          name='fixed_inputs_observations', dtype=dtype)

    super(DampedSchurComplement, self).__init__(
        feature_ndims=schur_complement.feature_ndims,
        dtype=kernel_dtype,
        name=name,
        parameters=parameters)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(  # pylint:disable=g-long-lambda
                    low=dtype_util.as_numpy_dtype(dtype)(2.)))),
        schur_complement=parameter_properties.BatchedComponentProperties(),
        observations_is_missing=parameter_properties.ParameterProperties(
            event_ndims=1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
        fixed_inputs_observations=parameter_properties.ParameterProperties(
            event_ndims=1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED))

  @property
  def df(self):
    return self._df

  @property
  def schur_complement(self):
    return self._schur_complement

  @property
  def fixed_inputs_observations(self):
    return self._fixed_inputs_observations

  @property
  def observations_is_missing(self):
    return self._observations_is_missing

  def divisor_matrix_cholesky(self, **kwargs):
    return self.schur_complement.divisor_matrix_cholesky(**kwargs)

  def _compute_scaling(self):
    if (self.fixed_inputs_observations is None or
        self.schur_complement._is_fixed_inputs_empty()):  # pylint:disable=protected-access
      return dtype_util.as_numpy_dtype(self.df.dtype)(1.)

    df = tf.convert_to_tensor(self.df)
    numerator = df - 2.
    denominator = df - 2.

    fixed_inputs_observations = tf.convert_to_tensor(
        self.fixed_inputs_observations)
    if self.observations_is_missing is not None:
      fixed_inputs_observations = tf.where(
          self.observations_is_missing,
          tf.zeros([], df.dtype),
          fixed_inputs_observations)

    b = tf.linalg.LinearOperatorLowerTriangular(
        self.divisor_matrix_cholesky()).solvevec(
            fixed_inputs_observations)
    b = tf.reduce_sum(b**2, axis=-1)
    numerator = numerator + b
    n = tf.cast(ps.shape(fixed_inputs_observations)[-1], b.dtype)
    denominator = denominator + n
    if self.observations_is_missing is not None:
      denominator = denominator - tf.cast(
          tf.math.count_nonzero(
              self.observations_is_missing, axis=-1), self.dtype)

    return numerator / denominator

  def _apply(self, x1, x2, example_ndims):
    undamped = self.schur_complement.apply(x1, x2, example_ndims=example_ndims)
    scaling_factor = self._compute_scaling()
    scaling_factor = tf.reshape(
        scaling_factor,
        ps.concat([ps.shape(scaling_factor), [1] * example_ndims], axis=0))

    return scaling_factor * undamped

  def _matrix(self, x1, x2):
    undamped = self.schur_complement.matrix(x1, x2)
    scaling_factor = self._compute_scaling()
    return scaling_factor[..., tf.newaxis, tf.newaxis] * undamped


class StudentTProcessRegressionModel(student_t_process.StudentTProcess):
  """StudentTProcessRegressionModel.

  This is an analogue of the GaussianProcessRegressionModel, except
  we use a Student-T process here (i.e. this represents the posterior predictive
  of a Student-T process, where we assume that the kernel hyperparameters and
  degrees of freedom are constants, and hence do optimizations in the
  constructor).

  Specifically, we assume a Student-T Process prior, `f ~ StP(df, m, k)` along
  with a noise model whose mathematical implementation is similar to a Gaussian
  Process albeit whose interpretation is very different.

  In particular, this noise model takes the following form:

  ```none
    e ~ MVT(df + n, 0, 1 + v / df * (1 + f^T k(t, t)^-1 f^T))
  ```
  which can be interpreted as Multivariate T noise whose covariance depends on
  the data fit term.

  Using the conventions in the `GaussianProcessRegressionModel` class, we have
  the posterior predictive distribution as:

  ```none
    (f(t) | t, x, f(x)) ~ MVT(df + n, loc, cov)

    where

    n is the number of observation points
    b = (y - loc)^T @ inv(k(x, x) + v * I) @ (y - loc)
    v = observation noise variance
    loc = k(t, x) @ inv(k(x, x) + v * I) @ (y - loc)
    cov = (df + b - 2) / (df + n - 2) * (
      k(t, t) - k(t, x) @ inv(k(x, x) + v * I) @ k(x, t))
  ```

  Note that the posterior predictive mean is the same as a Gaussian Process,
  but the covariance is multiplied by a term that takes in to account
  observations.

  This distribution does precomputaiton in the constructor by assuming
  `observation_index_points`, `observations`, `kernel` and
  `observation_noise_variance` are fixed, and that `mean_fn` is the zero
  function. We do these precomputations in a non-tape safe way.

  #### References
  [1]: Amar Shah, Andrew Gordon Wilson, Zoubin Ghahramani.
       Student-t Processes as Alternatives to Gaussian Processes
       https://arxiv.org/abs/1402.4306

  [2]: Qingtao Tang, Yisen Wang, Shu-Tao Xia
       Student-T Process Regression with Dependent Student-t Noise
       https://www.ijcai.org/proceedings/2017/393
  """
  # pylint:disable=invalid-name

  @deprecation.deprecated_args(
      '2023-11-15',
      _ALWAYS_YIELD_MVST_DEPRECATION_WARNING,
      'always_yield_multivariate_student_t')
  def __init__(
      self,
      df,
      kernel,
      index_points=None,
      observation_index_points=None,
      observations=None,
      observation_noise_variance=0.,
      predictive_noise_variance=None,
      mean_fn=None,
      cholesky_fn=None,
      marginal_fn=None,
      always_yield_multivariate_student_t=None,
      validate_args=False,
      allow_nan_stats=False,
      name='StudentTProcessRegressionModel',
      _conditional_kernel=None,
      _conditional_mean_fn=None):
    """Construct a StudentTProcessRegressionModel instance.

    Args:
      df: Positive Floating-point `Tensor` representing the degrees of freedom.
        Must be greather than 2.
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        StP's covariance function.
      index_points: (Nested) `Tensor` representing finite collection, or batch
        of collections, of points in the index set over which the STP is
        defined. Shape (of each nested component) has the form `[b1, ..., bB,
        e, f1, ..., fF]` where `F` is the number of feature dimensions and
        must equal `kernel.feature_ndims` (or its corresponding nested
        component) and `e` is the number (size) of index points in each
        batch. Ultimately this distribution corresponds to an `e`-dimensional
        multivariate Student T. The batch shape must be broadcastable with
        `kernel.batch_shape` and any batch dims yielded by `mean_fn`.
      observation_index_points: (Nested) `Tensor` representing finite
        collection, or batch of collections, of points in the index set for
        which some data has been observed. Shape (of each nested component)
        has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number
        of feature dimensions and must equal `kernel.feature_ndims` (or its
        corresponding nested component), and `e` is the number (size) of
        index points in each batch. `[b1, ..., bB, e]` must be broadcastable
        with the shape of `observations`, and `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc). The default value is
        `None`, which corresponds to the empty set of observations, and
        simply results in the prior predictive model (a STP with noise of
        variance `predictive_noise_variance`).
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
        must be brodcastable with the batch and example shapes of
        `observation_index_points`. The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc.).
      observation_noise_variance: `float` `Tensor` representing the variance
        of the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
        Default value: `0.`
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
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
        in which case `make_cholesky_with_jitter_fn`.
      marginal_fn: A Python callable that takes a location, covariance matrix,
        optional `validate_args`, `allow_nan_stats` and `name` arguments, and
        returns a multivariate Student-T subclass of `tfd.Distribution`.
        Default value: `None`, in which case a Cholesky-factorizing function
        is created using `make_cholesky_with_jitter_fn`.
      always_yield_multivariate_student_t: Deprecated and ignored.
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
        Default value: 'StudentTProcessRegressionModel'.
      _conditional_kernel: Internal parameter -- do not use.
      _conditional_mean_fn: Internal parameter -- do not use.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      if tf.nest.is_nested(kernel.feature_ndims):
        input_dtype = dtype_util.common_dtype(
            [kernel, index_points, observation_index_points],
            dtype_hint=nest_util.broadcast_structure(
                kernel.feature_ndims, tf.float32))
        dtype = dtype_util.common_dtype(
            [observations, observation_noise_variance,
             predictive_noise_variance, df], tf.float32)
      else:
        # If the index points are not nested, we assume they are of the same
        # dtype as the STPRM.
        dtype = dtype_util.common_dtype([
            kernel, index_points, observation_index_points, observations,
            observation_noise_variance, predictive_noise_variance, df
        ], tf.float32)
        input_dtype = dtype

      if index_points is not None:
        index_points = nest_util.convert_to_nested_tensor(
            index_points, dtype=input_dtype, convert_ref=False,
            name='index_points', allow_packing=True)
      if observation_index_points is not None:
        observation_index_points = nest_util.convert_to_nested_tensor(
            observation_index_points, dtype=input_dtype, convert_ref=False,
            name='observation_index_points', allow_packing=True)
      df = tensor_util.convert_nonref_to_tensor(
          df, dtype=dtype, name='df')
      observations = tensor_util.convert_nonref_to_tensor(
          observations, dtype=dtype, name='observations')
      observation_noise_variance = tensor_util.convert_nonref_to_tensor(
          observation_noise_variance, dtype=dtype,
          name='observation_noise_variance')
      predictive_noise_variance = tensor_util.convert_nonref_to_tensor(
          predictive_noise_variance,
          dtype=dtype,
          name='predictive_noise_variance')
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
      if cholesky_fn is None:
        cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn()

      self._observation_index_points = observation_index_points
      self._observations = observations
      self._observation_noise_variance = observation_noise_variance
      self._predictive_noise_variance = predictive_noise_variance

      with tf.name_scope('init'):
        if _conditional_kernel is None:
          _conditional_kernel = DampedSchurComplement(
              df=df,
              schur_complement=schur_complement_lib.SchurComplement(
                  base_kernel=kernel,
                  fixed_inputs=self._observation_index_points,
                  cholesky_fn=cholesky_fn,
                  diag_shift=observation_noise_variance),
              fixed_inputs_observations=self._observations,
              validate_args=validate_args)

        # Special logic for mean_fn only; SchurComplement already handles the
        # case of empty observations (ie, falls back to base_kernel).
        if stochastic_process_util.is_empty_observation_data(
            feature_ndims=kernel.feature_ndims,
            observation_index_points=observation_index_points,
            observations=observations):
          if _conditional_mean_fn is None:
            _conditional_mean_fn = mean_fn
        else:
          stochastic_process_util.validate_observation_data(
              kernel=kernel,
              observation_index_points=observation_index_points,
              observations=observations)

          # If `__init__` is called for CompositeTensor/Pytree unflattening,
          # `df` is already the `DeferredTensor`.
          if _conditional_mean_fn is None:
            n = tf.cast(ps.shape(observations)[-1], dtype=dtype)
            df = tfp_util.DeferredTensor(df, lambda x: x + n)

          if _conditional_mean_fn is None:

            def conditional_mean_fn(x):
              """Conditional mean."""
              observations = tf.convert_to_tensor(self._observations)
              observation_index_points = nest_util.convert_to_nested_tensor(
                  self._observation_index_points, dtype_hint=self.kernel.dtype,
                  allow_packing=True)
              k_x_obs_linop = tf.linalg.LinearOperatorFullMatrix(
                  kernel.matrix(x, observation_index_points))
              chol_linop = tf.linalg.LinearOperatorLowerTriangular(
                  _conditional_kernel.divisor_matrix_cholesky(
                      fixed_inputs=observation_index_points))
              diff = observations - mean_fn(observation_index_points)
              return mean_fn(x) + k_x_obs_linop.matvec(
                  chol_linop.solvevec(chol_linop.solvevec(diff), adjoint=True))
            _conditional_mean_fn = conditional_mean_fn

        # Store `_conditional_kernel` and `_conditional_mean_fn` as attributes
        # for `AutoCompositeTensor`.
        self._conditional_kernel = _conditional_kernel
        self._conditional_mean_fn = _conditional_mean_fn
        super(StudentTProcessRegressionModel, self).__init__(
            df=df,
            kernel=_conditional_kernel,
            mean_fn=_conditional_mean_fn,
            cholesky_fn=cholesky_fn,
            index_points=index_points,
            observation_noise_variance=predictive_noise_variance,
            always_yield_multivariate_student_t=(
                always_yield_multivariate_student_t),
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats, name=name)
        self._parameters = parameters

  @staticmethod
  @deprecation.deprecated_args(
      '2023-11-15',
      _ALWAYS_YIELD_MVST_DEPRECATION_WARNING,
      'always_yield_multivariate_student_t')
  def precompute_regression_model(
      df,
      kernel,
      observation_index_points,
      observations,
      observations_is_missing=None,
      index_points=None,
      observation_noise_variance=0.,
      predictive_noise_variance=None,
      mean_fn=None,
      cholesky_fn=None,
      always_yield_multivariate_student_t=None,
      validate_args=False,
      allow_nan_stats=False,
      name='PrecomputedStudentTProcessRegressionModel',
      _precomputed_divisor_matrix_cholesky=None,
      _precomputed_solve_on_observation=None):
    """Returns a StudentTProcessRegressionModel with precomputed quantities.

    This differs from the constructor by precomputing quantities associated with
    observations in a non-tape safe way. `index_points` is the only parameter
    that is allowed to vary (i.e. is a `Variable` / changes after
    initialization).

    Specifically:

    * We make `observation_index_points` and `observations` mandatory
      parameters.
    * We precompute `kernel(observation_index_points, observation_index_points)`
      along with any other associated quantities relating to `df`, `kernel`,
      `observations` and `observation_index_points`.

    A typical usecase would be optimizing kernel hyperparameters for a
    `StudenTProcess`, and computing the posterior predictive with respect to
    those optimized hyperparameters and observation / index-points pairs.

    WARNING: This method assumes `index_points` is the only varying parameter
    (i.e. is a `Variable` / changes after initialization) and hence is not
    tape-safe.

    Args:
      df: Positive Floating-point `Tensor` representing the degrees of freedom.
        Must be greather than 2.
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        StP's covariance function.
      observation_index_points: (Nested) `float` `Tensor` representing finite
        collection, or batch of collections, of points in the index set for
        which some data has been observed. Shape (or shape of each nested
        component) has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is
        the number of feature dimensions and must equal
        `kernel.feature_ndims` (or its corresponding nested component), and
        `e` is the number (size) of index points in each batch. `[b1, ...,
        bB, e]` must be broadcastable with the shape of `observations`, and
        `[b1, ..., bB]` must be broadcastable with the shapes of all other
        batched parameters (`kernel.batch_shape`, `index_points`, etc). The
        default value is `None`, which corresponds to the empty set of
        observations, and simply results in the prior predictive model (a StP
        with noise of variance `predictive_noise_variance`).
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
        must be brodcastable with the batch and example shapes of
        `observation_index_points`. The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc.). The default value is
        `None`, which corresponds to the empty set of observations, and simply
        results in the prior predictive model (a StP with noise of variance
        `predictive_noise_variance`).
      observations_is_missing:  `bool` `Tensor` of shape `[..., e]`,
        representing a batch of boolean masks.  When `observations_is_missing`
        is not `None`, the returned distribution is conditioned only on the
        observations for which the corresponding elements of
        `observations_is_missing` are `True`.
      index_points: (Nested) `float` `Tensor` representing finite collection, or
        batch of collections, of points in the index set over which the StP is
        defined.  Shape (or shape of each nested component) has the form
        `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` (or its corresponding
        nested component) and `e` is the number (size) of index points in each
        batch. Ultimately this distribution corresponds to an `e`-dimensional
        multivariate Student T. The batch shape must be broadcastable with
        `kernel.batch_shape` and any batch dims yielded by `mean_fn`.
      observation_noise_variance: `float` `Tensor` representing the variance
        of the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
        Default value: `0.`
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
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
        in which case `make_cholesky_with_jitter_fn` is used with the `jitter`
        parameter.
      always_yield_multivariate_student_t: Deprecated and ignored.
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
        Default value: 'PrecomputedStudentTProcessRegressionModel'.
      _precomputed_divisor_matrix_cholesky: Internal parameter -- do not use.
      _precomputed_solve_on_observation: Internal parameter -- do not use.
    Returns
      An instance of `StudentTProcessRegressionModel` with precomputed
      quantities associated with observations.
    """

    with tf.name_scope(name) as name:
      if tf.nest.is_nested(kernel.feature_ndims):
        input_dtype = dtype_util.common_dtype(
            [kernel, index_points, observation_index_points],
            dtype_hint=nest_util.broadcast_structure(
                kernel.feature_ndims, tf.float32))
        dtype = dtype_util.common_dtype(
            [observations, observation_noise_variance,
             predictive_noise_variance, df], tf.float32)
      else:
        # If the index points are not nested, we assume they are of the same
        # dtype as the STPRM.
        dtype = dtype_util.common_dtype([
            index_points, observation_index_points, observations,
            observation_noise_variance, predictive_noise_variance, df
        ], tf.float32)
        input_dtype = dtype

      # Convert to tensor arguments that are expected to not be Variables / not
      # going to change.
      df = tf.convert_to_tensor(df, dtype=dtype)
      observation_index_points = nest_util.convert_to_nested_tensor(
          observation_index_points, dtype=input_dtype, allow_packing=True)
      observation_noise_variance = tf.convert_to_tensor(
          observation_noise_variance, dtype=dtype)
      observations = tf.convert_to_tensor(observations, dtype=dtype)

      if observations_is_missing is not None:
        observations_is_missing = tf.convert_to_tensor(observations_is_missing)

      if cholesky_fn is None:
        cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn()

      conditional_kernel = DampedSchurComplement(
          df=df,
          schur_complement=schur_complement_lib.SchurComplement.with_precomputed_divisor(
              base_kernel=kernel,
              fixed_inputs=observation_index_points,
              fixed_inputs_is_missing=observations_is_missing,
              diag_shift=observation_noise_variance,
              cholesky_fn=cholesky_fn,
              _precomputed_divisor_matrix_cholesky=(
                  _precomputed_divisor_matrix_cholesky)),
          fixed_inputs_observations=observations,
          observations_is_missing=observations_is_missing,
          validate_args=validate_args)

      mean_fn = stochastic_process_util.maybe_create_mean_fn(mean_fn, dtype)

      solve_on_observation = _precomputed_solve_on_observation
      if solve_on_observation is None:
        observation_cholesky_operator = tf.linalg.LinearOperatorLowerTriangular(
            conditional_kernel.divisor_matrix_cholesky())
        diff = observations - mean_fn(observation_index_points)
        if observations_is_missing is not None:
          diff = tf.where(
              observations_is_missing, tf.zeros([], dtype=diff.dtype), diff)
        solve_on_observation = observation_cholesky_operator.solvevec(
            observation_cholesky_operator.solvevec(diff), adjoint=True)

      def conditional_mean_fn(x):
        k_x_obs = kernel.matrix(x, observation_index_points)
        if observations_is_missing is not None:
          k_x_obs = tf.where(observations_is_missing[..., tf.newaxis, :],
                             tf.zeros([], dtype=k_x_obs.dtype),
                             k_x_obs)
        return mean_fn(x) + tf.linalg.matvec(k_x_obs, solve_on_observation)

      stprm = StudentTProcessRegressionModel(
          df=df,
          kernel=kernel,
          observation_index_points=observation_index_points,
          observations=observations,
          index_points=index_points,
          observation_noise_variance=observation_noise_variance,
          predictive_noise_variance=predictive_noise_variance,
          cholesky_fn=cholesky_fn,
          always_yield_multivariate_student_t=(
              always_yield_multivariate_student_t),
          _conditional_kernel=conditional_kernel,
          _conditional_mean_fn=conditional_mean_fn,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)

      # pylint: disable=protected-access
      stprm._precomputed_divisor_matrix_cholesky = (
          conditional_kernel.schur_complement
          ._precomputed_divisor_matrix_cholesky)
      stprm._precomputed_solve_on_observation = solve_on_observation
      # pylint: enable=protected-access

    return stprm

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    def _event_ndims_fn(self):
      return tf.nest.map_structure(lambda nd: nd + 1, self.kernel.feature_ndims)
    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(  # pylint:disable=g-long-lambda
                    low=dtype_util.as_numpy_dtype(dtype)(2.)))),
        index_points=parameter_properties.ParameterProperties(
            event_ndims=_event_ndims_fn,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        observations=parameter_properties.ParameterProperties(
            event_ndims=1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
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
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        predictive_noise_variance=parameter_properties.ParameterProperties(
            event_ndims=0,
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

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
  def predictive_noise_variance(self):
    return self._predictive_noise_variance
