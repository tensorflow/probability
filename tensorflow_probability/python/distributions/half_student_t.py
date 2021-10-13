# Copyright 2020 The TensorFlow Probability Authors.
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
"""Half-Student's T Distribution Class."""

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import shift as shift_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'HalfStudentT',
]


class HalfStudentT(distribution.AutoCompositeTensorDistribution):
  """Half-Student's t distribution.

  The half-Student's t distribution has three parameters: degree of freedom
  `df`, location `loc`, and scale `scale`. It represents the right half of the
  two symmetric halves in a [Student's t
  distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution).

  #### Mathematical Details
  The probability density function (pdf) for the half-Student's t distribution
  is given by

  ```none
  pdf(x; df, loc, scale) = (1 + y**2 / df)**(-0.5 (df + 1)) / Z,
  where
  y = (x - loc) / scale
  Z = 2 * scale * sqrt(df * pi) * gamma(0.5 * df) / gamma(0.5 * (df + 1))

  ```

  where:
  * `df` is a positive scalar in `R`,
  * `loc` is a scalar in `R`,
  * `scale` is a positive scalar in `R`,
  * `Z` is the normalization constant, and
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The support of the distribution is given by the interval `[loc, infinity)`.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in the paper

  [Michael Figurnov, Shakir Mohamed, Andriy Mnih.
  Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)


  #### Examples
  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Student t distribution.
  single_dist = tfd.HalfStudentT(df=3, loc=0, scale=1)

  # Evaluate the pdf at 1, returning a scalar Tensor.
  single_dist.prob(1.)

  # Define a batch of two scalar valued half Student t's.
  # The first has degrees of freedom 2, mean 1, and scale 11.
  # The second 3, 2 and 22.
  multi_dist = tfd.HalfStudentT(df=[2, 3], loc=[1, 2], scale=[11, 22])

  # Evaluate the pdf of the first distribution at 1.5, and the second on 2.5,
  # returning a length two tensor.
  multi_dist.prob([1.5, 2.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  multi_dist.sample(3)
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two half Student's t distributions.
  # Both have df 2 and mean 1, but different scales.
  dist = tfd.HalfStudentT(df=2, loc=1, scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  Compute the gradients of samples w.r.t. the parameters via implicit
  reparameterization through the gamma:

  ```python
  df = tf.constant(2.0)
  loc = tf.constant(2.0)
  scale = tf.constant(11.0)
  dist = tfd.HalfStudentT(df=df, loc=loc, scale=scale)
  with tf.GradientTape() as tape:
    tape.watch((df, loc, scale))
    loss = tf.reduce_mean(dist.sample(5))
    # Unbiased stochastic gradients of the loss function
    grads = tape.gradient(loss, (df, loc, scale))
  ```

  """

  def __init__(self,
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='HalfStudentT'):
    """Construct a half-Student's t distribution.

    Args:
      df: Floating-point `Tensor`. The degrees of freedom of the
        distribution(s). `df` must contain only positive values.
      loc: Floating-point `Tensor`; the location(s) of the distribution(s).
      scale: Floating-point `Tensor`; the scale(s) of the distribution(s). Must
        contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs. Default value: `False` (i.e. do not validate args).
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'HalfStudentT'.

    Raises:
      TypeError: if `loc` and `scale` have different `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([df, loc, scale], dtype_hint=tf.float32)
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype((self._df, self._loc, self._scale))
      super(HalfStudentT, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def df(self):
    """Distribution parameter for the degrees of freedom."""
    return self._df

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for the scale."""
    return self._scale

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(df=df, loc=loc, scale=scale)
    samples = student_t.sample_n(
        n,
        df=df,
        loc=tf.zeros_like(loc),
        scale=scale,
        batch_shape=batch_shape,
        dtype=self.dtype,
        seed=seed)
    return tf.math.abs(samples) + self.loc

  def _log_prob(self, x):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    safe_x = tf.where(x < loc, 0.5 * scale + loc, x)  # avoid NaNs below
    # Where defined, log prob is twice StudentT log prob.
    log_prob = student_t.log_prob(
        safe_x, df=df, loc=loc, scale=scale) + np.log(2.)
    return tf.where(x < loc,
                    dtype_util.as_numpy_dtype(self.dtype)(-np.inf), log_prob)

  def _cdf(self, x):
    # If F(t) is the cdf of a symmetric f,
    # 2 * F(t) - 1 is the cdf of abs(f) for t > loc
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    safe_x = tf.where(x < loc, 0.5 * scale + loc, x)
    cdf = student_t.cdf(safe_x, df, loc, scale)
    return tf.where(x < loc,
                    dtype_util.as_numpy_dtype(self.dtype)(0.),
                    2. * cdf - 1)

  def _entropy(self):
    # Symmetric half-P entropy is
    # entropy(P) - log(2)
    df = tf.convert_to_tensor(self.df)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(df=df, scale=scale)
    return student_t.entropy(df, scale, batch_shape, self.dtype) - np.log(2.)

  @distribution_util.AppendDocstring(
      """The mean of a half-Student's t is defined if `df > 1`, otherwise it is
      `NaN`. If `self.allow_nan_stats=False`, then an exception will be raised
      rather than returning `NaN`.""")
  def _mean(self):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    log_correction = (
        tf.math.log(scale) + np.log(2.) + 0.5 *
        (tf.math.log(df) - np.log(np.pi)) -
        tfp_math.log_gamma_difference(0.5, 0.5 * df) -
        tf.math.log(df - 1))
    mean = tf.math.exp(log_correction) + loc
    if self.allow_nan_stats:
      return tf.where(df > 1., mean,
                      dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.ones([], dtype=self.dtype),
              df,
              message='mean not defined for components of df <= 1'),
      ], mean)

  @distribution_util.AppendDocstring("""
      The variance for half-Student's t is

      ```
      defined, when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      ```
      """)
  def _variance(self):
    df = tf.convert_to_tensor(self.df)
    scale = tf.convert_to_tensor(self.scale)
    # We need to put the tf.where inside the outer tf.where to ensure we never
    # hit a NaN in the gradient.
    first_denom = tf.where(df > 2., df - 2., 1.)
    second_denom = tf.where(df > 1., df - 1., 1.)
    var = (
        tf.ones(self._batch_shape_tensor(df=df, scale=scale),
                dtype=self.dtype) * tf.square(scale) * df / first_denom -
        tf.math.exp(2. * tf.math.log(scale) + np.log(4.) + tf.math.log(df) -
                    np.log(np.pi) - 2. * tf.math.log(second_denom) -
                    2. * tfp_math.log_gamma_difference(0.5, 0.5 * df)))
    # When 1 < df <= 2, variance is infinite.
    result_where_defined = tf.where(
        df > 2., var,
        dtype_util.as_numpy_dtype(self.dtype)(np.inf))

    if self.allow_nan_stats:
      return tf.where(df > 1., result_where_defined,
                      dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.ones([], dtype=self.dtype),
              df,
              message='variance not defined for components of df <= 1'),
      ], result_where_defined)

  def _default_event_space_bijector(self):
    return chain_bijector.Chain([
        shift_bijector.Shift(shift=self.loc, validate_args=self.validate_args),
        exp_bijector.Exp(validate_args=self.validate_args)
    ],
                                validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    """Checks the validity of a sample."""
    assertions = []
    if not self.validate_args:
      return assertions
    loc = tf.convert_to_tensor(self.loc)
    assertions.append(
        assert_util.assert_greater_equal(
            x, loc, message='Sample must be greater than or equal to `loc`.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(
          assert_util.assert_positive(
              self.scale, message='Argument `scale` must be positive.'))
    if is_init != tensor_util.is_ref(self.df):
      assertions.append(
          assert_util.assert_positive(
              self.df, message='Argument `df` must be positive.'))
    return assertions
