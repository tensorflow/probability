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
from tensorflow_probability.python.internal import batch_shape_lib
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import psd_kernels as tfpk


__all__ = [
    'StudentTProcessRegressionModel',
]


def _is_empty_observation_data(
    feature_ndims, observation_index_points, observations):
  """Returns `True` if given observation data is empty.

  Emptiness means either
    1. Both `observation_index_points` and `observations` are `None`, or
    2. the "number of observations" shape is 0. The shape of
    `observation_index_points` is `[..., N, f1, ..., fF]`, where `N` is the
    number of observations and the `f`s are feature dims. Thus, we look at the
    shape element just to the left of the leftmost feature dim. If that shape is
    zero, we consider the data empty.

  We don't check the shape of observations; validations are checked elsewhere in
  the calling code, to ensure these shapes are consistent.

  Args:
    feature_ndims: the number of feature dims, as reported by the StP kernel.
    observation_index_points: the observation data locations in the index set.
    observations: the observation data.

  Returns:
    is_empty: True if the data were deemed to be empty.
  """
  # If both input locations and observations are `None`, we consider this
  # "empty" observation data.
  if observation_index_points is None and observations is None:
    return True
  num_obs = tf.compat.dimension_value(
      observation_index_points.shape[-(feature_ndims + 1)])
  if num_obs is not None and num_obs == 0:
    return True
  return False


def _validate_observation_data(
    kernel, observation_index_points, observations):
  """Ensure that observation data and locations have consistent shapes.

  This basically means that the batch shapes are broadcastable. We can only
  ensure this when those shapes are fully statically defined.

  Args:
    kernel: The StP kernel.
    observation_index_points: the observation data locations in the index set.
    observations: the observation data.

  Raises:
    ValueError: if the observations' batch shapes are not broadcastable.
  """
  # Check that observation index points and observation counts broadcast.
  ndims = kernel.feature_ndims
  if (tensorshape_util.is_fully_defined(
      observation_index_points.shape[:-ndims]) and
      tensorshape_util.is_fully_defined(observations.shape)):
    index_point_count = observation_index_points.shape[:-ndims]
    observation_count = observations.shape
    try:
      tf.broadcast_static_shape(index_point_count, observation_count)
    except ValueError:
      # Re-raise with our own more contextual error message.
      raise ValueError(
          'Observation index point and observation counts are not '
          'broadcastable: {} and {}, respectively.'.format(
              index_point_count, observation_count))


class DampedSchurComplement(tfpk.PositiveSemidefiniteKernel):
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
               validate_args=False,
               name='DampedSchurComplement'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([
          df,
          schur_complement,
          fixed_inputs_observations], tf.float32)
      self._schur_complement = schur_complement
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      self._fixed_inputs_observations = tensor_util.convert_nonref_to_tensor(
          fixed_inputs_observations,
          name='fixed_inputs_observations', dtype=dtype)

    super(DampedSchurComplement, self).__init__(
        feature_ndims=schur_complement.feature_ndims,
        dtype=dtype,
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

  def divisor_matrix_cholesky(self, **kwargs):
    return self.schur_complement.divisor_matrix_cholesky(**kwargs)

  def _apply(self, x1, x2, example_ndims):
    undamped = self.schur_complement.apply(x1, x2, example_ndims=example_ndims)
    numerator = self.df - 2.
    denominator = self.df - 2.
    if (self.fixed_inputs_observations is not None and
        not self.schur_complement._is_fixed_inputs_empty()):  # pylint:disable=protected-access
      b = tf.linalg.LinearOperatorLowerTriangular(
          self.divisor_matrix_cholesky()).solvevec(
              self.fixed_inputs_observations)
      b = tf.reduce_sum(b**2, axis=-1)
      numerator = numerator + b
      n = tf.cast(ps.shape(self.fixed_inputs_observations)[-1], self.dtype)
      denominator = denominator + n

    scaling_factor = numerator / denominator
    scaling_factor = tf.reshape(
        scaling_factor,
        ps.concat([ps.shape(scaling_factor), [1] * example_ndims], axis=0))

    return scaling_factor * undamped

  def _matrix(self, x1, x2):
    undamped = self.schur_complement.matrix(x1, x2)
    numerator = self.df - 2.
    denominator = self.df - 2.
    if (self.fixed_inputs_observations is not None and
        not self.schur_complement._is_fixed_inputs_empty()):  # pylint:disable=protected-access
      b = tf.linalg.LinearOperatorLowerTriangular(
          self.divisor_matrix_cholesky()).solvevec(
              self.fixed_inputs_observations)
      b = tf.reduce_sum(b**2, axis=-1)
      numerator = numerator + b
      n = tf.cast(ps.shape(self.fixed_inputs_observations)[-1], self.dtype)
      denominator = denominator + n
    return (numerator / denominator)[..., tf.newaxis, tf.newaxis] * undamped


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
      index_points: `float` `Tensor` representing finite collection, or batch of
        collections, of points in the index set over which the STP is defined.
        Shape has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the
        number of feature dimensions and must equal `kernel.feature_ndims` and
        `e` is the number (size) of index points in each batch. Ultimately this
        distribution corresponds to an `e`-dimensional multivariate normal. The
        batch shape must be broadcastable with `kernel.batch_shape`.
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
        Takes a `Tensor` of shape `[b1, ..., bB, f1, ..., fF]` and returns a
        `Tensor` whose shape is broadcastable with `[b1, ..., bB]`.
        Default value: `None` implies the constant zero function.
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
        in which case `make_cholesky_with_jitter_fn`.
      marginal_fn: A Python callable that takes a location, covariance matrix,
        optional `validate_args`, `allow_nan_stats` and `name` arguments, and
        returns a multivariate Student-T subclass of `tfd.Distribution`.
        Default value: `None`, in which case a Cholesky-factorizing function is
        is created using `make_cholesky_with_jitter_fn`.
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
      dtype = dtype_util.common_dtype(
          [df, kernel, index_points, observation_noise_variance, observations],
          tf.float32)
      df = tensor_util.convert_nonref_to_tensor(
          df, dtype=dtype, name='df')
      index_points = tensor_util.convert_nonref_to_tensor(
          index_points, dtype=dtype, name='index_points')
      observation_index_points = tensor_util.convert_nonref_to_tensor(
          observation_index_points,
          dtype=dtype,
          name='observation_index_points')
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
      if mean_fn is None:
        mean_fn = lambda x: tf.zeros([1], dtype=dtype)
      else:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')

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
              schur_complement=tfpk.SchurComplement(
                  base_kernel=kernel,
                  fixed_inputs=self._observation_index_points,
                  diag_shift=observation_noise_variance),
              fixed_inputs_observations=self._observations,
              validate_args=validate_args)

        # Special logic for mean_fn only; SchurComplement already handles the
        # case of empty observations (ie, falls back to base_kernel).
        if _is_empty_observation_data(
            feature_ndims=kernel.feature_ndims,
            observation_index_points=observation_index_points,
            observations=observations):
          if _conditional_mean_fn is None:
            _conditional_mean_fn = mean_fn
        else:
          _validate_observation_data(
              kernel=kernel,
              observation_index_points=observation_index_points,
              observations=observations)
          n = tf.cast(ps.shape(observations)[-1], dtype=dtype)
          df = tfp_util.DeferredTensor(df, lambda x: x + n)

          if _conditional_mean_fn is None:

            def conditional_mean_fn(x):
              """Conditional mean."""
              observations = tf.convert_to_tensor(self._observations)
              observation_index_points = tf.convert_to_tensor(
                  self._observation_index_points)
              k_x_obs_linop = tf.linalg.LinearOperatorFullMatrix(
                  kernel.matrix(x, observation_index_points))
              chol_linop = tf.linalg.LinearOperatorLowerTriangular(
                  _conditional_kernel.divisor_matrix_cholesky(
                      fixed_inputs=observation_index_points))
              diff = observations - mean_fn(observation_index_points)
              return mean_fn(x) + k_x_obs_linop.matvec(
                  chol_linop.solvevec(chol_linop.solvevec(diff), adjoint=True))
            _conditional_mean_fn = conditional_mean_fn

        super(StudentTProcessRegressionModel, self).__init__(
            df=df,
            kernel=_conditional_kernel,
            mean_fn=_conditional_mean_fn,
            cholesky_fn=cholesky_fn,
            index_points=index_points,
            observation_noise_variance=predictive_noise_variance,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats, name=name)
        self._parameters = parameters

  @staticmethod
  def precompute_regression_model(
      df,
      kernel,
      observation_index_points,
      observations,
      index_points=None,
      observation_noise_variance=0.,
      predictive_noise_variance=None,
      mean_fn=None,
      cholesky_fn=None,
      validate_args=False,
      allow_nan_stats=False,
      name='PrecomputedStudentTProcessRegressionModel'):
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
        predictive model (a StP with noise of variance
        `predictive_noise_variance`).
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
      index_points: `float` `Tensor` representing finite collection, or batch of
        collections, of points in the index set over which the StP is defined.
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
        Takes a `Tensor` of shape `[b1, ..., bB, f1, ..., fF]` and returns a
        `Tensor` whose shape is broadcastable with `[b1, ..., bB]`.
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
        Default value: 'PrecomputedStudentTProcessRegressionModel'.
    Returns
      An instance of `StudentTProcessRegressionModel` with precomputed
      quantities associated with observations.
    """

    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([
          df, index_points, observation_index_points, observations,
          observation_noise_variance, predictive_noise_variance,
      ], tf.float32)

      # Convert to tensor arguments that are expected to not be Variables / not
      # going to change.
      df = tf.convert_to_tensor(df, dtype=dtype)
      observation_index_points = tf.convert_to_tensor(
          observation_index_points, dtype=dtype)
      observation_noise_variance = tf.convert_to_tensor(
          observation_noise_variance, dtype=dtype)
      observations = tf.convert_to_tensor(observations, dtype=dtype)

      observation_cholesky = kernel.matrix(
          observation_index_points, observation_index_points)

      broadcast_shape = distribution_util.get_broadcast_shape(
          observation_cholesky,
          observation_noise_variance[..., tf.newaxis, tf.newaxis])

      observation_cholesky = tf.broadcast_to(
          observation_cholesky, broadcast_shape)

      observation_cholesky = tf.linalg.set_diag(
          observation_cholesky,
          tf.linalg.diag_part(observation_cholesky) +
          observation_noise_variance[..., tf.newaxis])
      if cholesky_fn is None:
        cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn()

      observation_cholesky = cholesky_fn(observation_cholesky)
      observation_cholesky_operator = tf.linalg.LinearOperatorLowerTriangular(
          observation_cholesky)

      conditional_kernel = DampedSchurComplement(
          df=df,
          schur_complement=tfpk.SchurComplement(
              base_kernel=kernel,
              fixed_inputs=observation_index_points,
              diag_shift=observation_noise_variance),
          fixed_inputs_observations=observations,
          validate_args=validate_args)

      if mean_fn is None:
        mean_fn = lambda x: tf.zeros([1], dtype=dtype)
      else:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')

      diff = observations - mean_fn(observation_index_points)
      solve_on_observation = observation_cholesky_operator.solvevec(
          observation_cholesky_operator.solvevec(diff), adjoint=True)

      def conditional_mean_fn(x):
        k_x_obs = kernel.matrix(x, observation_index_points)
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
          _conditional_kernel=conditional_kernel,
          _conditional_mean_fn=conditional_mean_fn,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)

    return stprm

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(  # pylint:disable=g-long-lambda
                    low=dtype_util.as_numpy_dtype(dtype)(2.)))),
        index_points=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims + 1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        observations=parameter_properties.ParameterProperties(
            event_ndims=1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
        observation_index_points=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims + 1,
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

  def _batch_shape_tensor(self, index_points=None):
    kwargs = {}
    if index_points is not None:
      kwargs = {'index_points': index_points}
    return batch_shape_lib.inferred_batch_shape_tensor(self, **kwargs)
