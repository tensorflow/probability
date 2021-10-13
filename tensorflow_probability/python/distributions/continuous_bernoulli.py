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
"""The continuous Bernoulli distribution class."""

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


def _log_xexp_ratio(x):
  """Compute log(x * (exp(x) + 1) / (exp(x) - 1)) in a numerically stable way."""
  with tf.name_scope('log_xexp_ratio'):
    x = tf.convert_to_tensor(x)
    dtype = dtype_util.as_numpy_dtype(x.dtype)
    eps = np.finfo(dtype).eps

    # This function is even, hence we use abs(x) everywhere.
    x = tf.math.abs(x)

    # For x near zero, we have the Taylor series:
    # log(2) + x**2 / 12 - 7 x**4 / 1440 + 31 x**6 / 90720 + O(x**8)
    # whose coefficients decrease in magnitude monotonically

    # For x large in magnitude, the ratio (exp(x) + 1) / (exp(x) - 1)) tends to
    # sign(x), so thus this function should tend to log(abs(x))

    # Finally for x medium in magnitude, we can use the naive expression. Thus,
    # we generate 2 cutofs.

    # Use the first 3 non-zero terms of the Taylor series when
    # |x| < small_cutoff.
    small_cutoff = np.power(eps * 90720. / 31, 1 / 6.)

    # Use log(abs(x)) when |x| > large_cutoff
    large_cutoff = -np.log(eps)

    x_squared = tf.math.square(x)

    result = (dtype(np.log(2.)) + x_squared / 112. -
              7 * tf.math.square(x_squared) / 1440.)
    middle_region = (x > small_cutoff) & (x < large_cutoff)
    safe_x_medium = tf.where(middle_region, x, dtype(1.))
    result = tf.where(
        middle_region,
        (tf.math.log(safe_x_medium) + tf.math.softplus(safe_x_medium) -
         tf.math.log(tf.math.expm1(safe_x_medium))),
        result)

    # We can do this by choosing a cutoff when x > log(1 / machine eps)
    safe_x_large = tf.where(x >= large_cutoff, x, dtype(1.))
    result = tf.where(x >= large_cutoff, tf.math.log(safe_x_large), result)
    return result


class ContinuousBernoulli(distribution.AutoCompositeTensorDistribution):
  """Continuous Bernoulli distribution.

  This distribution is parameterized by `probs`, a (batch of) parameters
  taking values in `(0, 1)`. Note that, unlike in the Bernoulli case, `probs`
  does not correspond to a probability, but the same name is used due to the
  similarity with the Bernoulli.

  #### Mathematical Details

  The continuous Bernoulli is a distribution over the interval `[0, 1]`,
  parameterized by `probs` in `(0, 1)`.

  The probability density function (pdf) is,

  ```none
  pdf(x; probs) = probs**x * (1 - probs)**(1 - x) * C(probs)
  C(probs) = (2 * atanh(1 - 2 * probs) / (1 - 2 * probs) if probs != 0.5
              else 2.)
  ```

  While the normalizing constant `C(probs)` is a continuous function of `probs`
  (even at `probs = 0.5`), computing it at values close to 0.5 can result in
  numerical instabilities due to 0/0 errors. A Taylor approximation of
  `C(probs)` is thus used for values of `probs`
  in a small interval around 0.5.

  NOTE: Unlike the Bernoulli, numerical instabilities can happen for `probs`
  very close to 0 or 1. Current implementation allows any value in `(0, 1)`,
  but this could be changed to `(1e-6, 1-1e-6)` to avoid these issues.

  #### References

  [1] Loaiza-Ganem G and Cunningham JP. The continuous Bernoulli: fixing a
      pervasive error in variational autoencoders. NeurIPS2019.
      https://arxiv.org/abs/1907.06845
  """

  def __init__(
      self,
      logits=None,
      probs=None,
      dtype=tf.float32,
      validate_args=False,
      allow_nan_stats=True,
      name='ContinuousBernoulli'):
    """Construct Bernoulli distributions.

    Args:
      logits: An N-D `Tensor`. Each entry in the `Tensor` parameterizes
       an independent continuous Bernoulli distribution with parameter
       sigmoid(logits). Only one of `logits` or `probs` should be passed
       in. Note that this does not correspond to the log-odds as in the
       Bernoulli case.
      probs: An N-D `Tensor` representing the parameter of a continuous
       Bernoulli. Each entry in the `Tensor` parameterizes an independent
       continuous Bernoulli distribution. Only one of `logits` or `probs`
       should be passed in. Note that this also does not correspond to a
       probability as in the Bernoulli case.
      dtype: The type of the event samples. Default: `float32`.
       validate_args: Python `bool`, default `False`. When `True`
       distribution parameters are checked for validity despite possibly
       degrading runtime performance. When `False` invalid inputs may
       silently render incorrect outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is
        raised if one or more of the statistic's batch members are
        undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: If probs and logits are passed, or if neither are passed.
    """
    parameters = dict(locals())
    if (probs is None) == (logits is None):
      raise ValueError('Must pass `probs` or `logits`, but not both.')
    with tf.name_scope(name) as name:
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype_hint=tf.float32, name='probs')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype_hint=tf.float32, name='logits')
    super(ContinuousBernoulli, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        logits=parameter_properties.ParameterProperties(),
        probs=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=sigmoid_bijector.Sigmoid,
            is_preferred=False))

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    logits = self._logits_parameter_no_checks()
    new_shape = ps.concat([[n], ps.shape(logits)], axis=0)
    uniform = samplers.uniform(new_shape, seed=seed, dtype=logits.dtype)
    sample = self._quantile(uniform, logits)
    return tf.cast(sample, self.dtype)

  def _log_normalizer(self, logits=None):
    # The normalizer is 2 * atanh(1 - 2 * probs) / (1 - 2 * probs), with the
    # removable singularity at probs = 0.5 removed (and replaced with 2).
    # We do this computation in logit space to be more numerically stable.
    # Note that 2 * atanh(1 - 2 / (1 + exp(-logits))) = logits.
    # Thus we end up with
    # logits / (1 - 2 / (1 + exp(-logits))) =
    # logits / ((-exp(-logits) + 1) / (exp(-logits) + 1)) =
    # (exp(-logits) + 1) * logits / (-exp(-logits) + 1) =
    # (1 + exp(logits)) * logits / (exp(logits) - 1)

    if logits is None:
      logits = self._logits_parameter_no_checks()
    return _log_xexp_ratio(logits)

  def _log_prob(self, event):
    log_probs0, log_probs1, _ = self._outcome_log_probs()
    event = tf.cast(event, log_probs0.dtype)
    tentative_log_pdf = (tf.math.multiply_no_nan(log_probs0, 1.0 - event)
                         + tf.math.multiply_no_nan(log_probs1, event)
                         + self._log_normalizer())
    return tf.where(
        (event < 0) | (event > 1),
        dtype_util.as_numpy_dtype(log_probs0.dtype)(-np.inf),
        tentative_log_pdf)

  def _log_cdf(self, x):
    # The CDF is (p**x * (1 - p)**(1 - x) + p - 1) / (2 * p - 1).
    # We do this computation in logit space to be more numerically stable.
    # p**x * (1- p)**(1 - x) becomes
    # 1 / (1 + exp(-logits))**x *
    # exp(-logits * (1 - x)) / (1 + exp(-logits)) ** (1 - x) =
    # exp(-logits * (1 - x)) / (1 + exp(-logits))
    # p - 1 becomes -exp(-logits) / (1 + exp(-logits))
    # Thus the whole numerator is
    # (exp(-logits * (1 - x)) - exp(-logits)) / (1 + exp(-logits))
    # The denominator is (1 - exp(-logits)) / (1 + exp(-logits))
    # Putting it all together, this gives:
    # (exp(-logits * (1 - x)) - exp(-logits)) / (1 - exp(-logits)) =
    # (exp(logits * x) - 1) / (exp(logits) - 1)
    logits = self._logits_parameter_no_checks()

    # For logits < 0, we can directly use the expression.
    safe_logits = tf.where(logits < 0., logits, -1.)
    result_negative_logits = (
        tfp_math.log1mexp(
            tf.math.multiply_no_nan(safe_logits, x)) -
        tfp_math.log1mexp(safe_logits))
    # For logits > 0, to avoid infs with large arguments we rewrite the
    # expression. Let z = log(exp(logits) - 1)
    # log_cdf = log((exp(logits * x) - 1) / (exp(logits) - 1))
    #         = log(exp(logits * x) - 1) - log(exp(logits) - 1)
    #         = log(exp(logits * x) - 1) - log(exp(z))
    #         = log(exp(logits * x - z) - exp(-z))
    # Because logits > 0, logits * x - z > -z, so we can pull it out to get
    #         = log(exp(logits * x - z) * (1 - exp(-logits * x)))
    #         = logits * x - z + tf.math.log(1 - exp(-logits * x))
    dtype = dtype_util.as_numpy_dtype(x.dtype)
    eps = np.finfo(dtype).eps
    # log(exp(logits) - 1)
    safe_logits = tf.where(logits > 0., logits, 1.)
    z = tf.where(
        safe_logits > -np.log(eps),
        safe_logits, tf.math.log(tf.math.expm1(safe_logits)))
    result_positive_logits = tf.math.multiply_no_nan(
        safe_logits, x) - z + tfp_math.log1mexp(
            -tf.math.multiply_no_nan(safe_logits, x))

    result = tf.where(
        logits < 0., result_negative_logits, result_positive_logits)

    # Finally, handle the case where `logits` and `p` are on the boundary,
    # as the above expressions can result in ratio of `infs` in that case as
    # well.
    result = tf.where(
        tf.math.equal(logits, np.inf), dtype(-np.inf), result)
    result = tf.where(
        (tf.math.equal(logits, -np.inf) & tf.math.not_equal(x, 0.)) | (
            tf.math.equal(logits, np.inf) & tf.math.equal(x, 1.)),
        tf.zeros_like(logits), result)

    result = tf.where(
        x < 0.,
        dtype(-np.inf),
        tf.where(x > 1., tf.zeros_like(x), result))

    return result

  def _outcome_log_probs(self):
    """Returns log(1-probs), log(probs) and logits."""
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      logits = tf.math.log(probs) - tf.math.log1p(-probs)
      return tf.math.log1p(-probs), tf.math.log(probs), logits
    s = tf.convert_to_tensor(self._logits)
    # softplus(s) = -Log[1 - p]
    # -softplus(-s) = Log[p]
    # softplus(+inf) = +inf, softplus(-inf) = 0, so...
    #  logits = -inf ==> log_probs0 = 0, log_probs1 = -inf (as desired)
    #  logits = +inf ==> log_probs0 = -inf, log_probs1 = 0 (as desired)
    return -tf.math.softplus(s), -tf.math.softplus(-s), s

  def _entropy(self):
    log_probs0, log_probs1, logits = self._outcome_log_probs()
    return (self._mean(logits) * (log_probs0 - log_probs1)
            - self._log_normalizer(logits) - log_probs0)

  def _mean(self, logits=None):
    # The mean is probs / (2 * probs - 1) + 1 / (2 * arctanh(1 - 2 * probs))
    # with the removable singularity at 0.5 removed.
    # We write this in logits space.
    # The first term becomes
    # 1 / (1 + exp(-logits)) / (2 / (1 + exp(-logits)) - 1) =
    # 1 / (2 - 1 - exp(-logits)) =
    # 1 / (1 - exp(-logits))
    # The second term becomes 1 / logits.
    # Thus we have mean = 1 / (1 - exp(-logits)) - 1 / logits.

    # When logits is close to zero, we can compute the Laurent series for the
    # first term as:
    # 1 / x + 1 / 2 + x / 12 - x**3 / 720 + x**5 / 30240 + O(x**7).
    # Thus we get the pole at zero canceling out with the second term.

    dtype = dtype_util.as_numpy_dtype(self.dtype)
    eps = np.finfo(dtype).eps

    if logits is None:
      logits = self._logits_parameter_no_checks()

    small_cutoff = np.power(eps * 30240, 1 / 5.)
    result = dtype(0.5) + logits / 12. - logits * tf.math.square(logits) / 720

    safe_logits_large = tf.where(
        tf.math.abs(logits) > small_cutoff, logits, dtype(1.))
    return tf.where(
        tf.math.abs(logits) > small_cutoff,
        -(tf.math.reciprocal(
            tf.math.expm1(-safe_logits_large)) +
          tf.math.reciprocal(safe_logits_large)),
        result)

  def _variance(self):
    # The variance is var = probs (probs - 1) / (2 * probs - 1)**2 +
    # 1 / (2 * arctanh(1 - 2 * probs))**2
    # with the removable singularity at 0.5 removed.
    # We write this in logits space.
    # Let v = 1 + exp(-logits) = 1 / probs
    # The first term becomes
    # probs * (probs - 1) / (2 * probs - 1)**2 turns in to:
    # 1 / v * (1 / v - 1) / (2 / v - 1) ** 2 =
    # (1 / v ** 2) * (1 - v) / (2 / v - 1) ** 2 =
    # (1 - v) / (2 - v) ** 2 =
    # -exp(-logits) / (1 - exp(-logits))**2 =
    # -exp(-logits) / (1 + exp(-2 * logits) - 2 * exp(-logits)) =
    # -1 / (exp(logits) + exp(-logits) - 2) =
    # 1 / (2 - 2 * cosh(logits))
    # For the second term, we have
    # 1 / (2 * arctanh(1 - 2 * probs))**2 =
    # 1 / (2 * 0.5 * log((1 + 1 - 2 * probs) / (1 - (1 - 2 * probs))))**2 =
    # 1 / (log(2 * (1 - probs) / (2 * probs)))**2 =
    # 1 / (log((1 - probs) / probs))**2 =
    # 1 / (log(1 / probs - 1))**2 =
    # 1 / (log(1 + exp(-logits) - 1))**2 =
    # 1 / (-logits)**2 =
    # 1 / logits**2

    # Thus we have var = 1 / (2 - 2 * cosh(logits)) + 1 / logits**2

    # For the function f(x) = exp(-x) / (1 - exp(-x)) ** 2 + 1 / x ** 2, when
    # logits is close to zero, we can compute the Laurent series for the first
    # term as:
    # -1 / x**2 + 1 / 12 - x**2 / 240 + x**4 / 6048 + x**6 / 172800 + O(x**8).
    # Thus we get the pole at zero canceling out with the second term.

    dtype = dtype_util.as_numpy_dtype(self.dtype)
    eps = np.finfo(dtype).eps

    logits = self._logits_parameter_no_checks()

    small_cutoff = np.power(eps * 172800, 1 / 6.)
    logits_sq = tf.math.square(logits)
    small_result = (dtype(1 / 12.) - logits_sq / 240. +
                    tf.math.square(logits_sq) / 6048)

    safe_logits_large = tf.where(
        tf.math.abs(logits) > small_cutoff, logits, dtype(1.))
    return tf.where(
        tf.math.abs(logits) > small_cutoff,
        (tf.math.reciprocal(2 * (1. - tf.math.cosh(safe_logits_large))) +
         tf.math.reciprocal(tf.math.square(safe_logits_large))),
        small_result)

  def _quantile(self, p, logits=None):
    if logits is None:
      logits = self._logits_parameter_no_checks()
    logp = tf.math.log(p)
    # The expression for the quantile function is:
    # log(1 + (e^s - 1) * p) / s, where s is `logits`. When s is large,
    # the e^s sub-term becomes increasingly ill-conditioned.  However,
    # since the numerator tends to s, we can reformulate the s > 0 case
    # as a offset from 1, which is more accurate.  Coincidentally,
    # this eliminates a ratio of infinities problem when `s == +inf`.

    safe_negative_logits = tf.where(logits < 0., logits, -1.)
    safe_positive_logits = tf.where(logits > 0., logits, 1.)
    result = tf.where(
        logits > 0.,
        1. + tfp_math.log_add_exp(
            logp + tfp_math.log1mexp(safe_positive_logits),
            tf.math.negative(safe_positive_logits)) / safe_positive_logits,
        tf.math.log1p(
            tf.math.expm1(safe_negative_logits) * p) / safe_negative_logits)

    # When logits is zero, we can simplify
    # log(1 + (e^s - 1) * p) / s ~= log(1 + s * p) / s ~= s * p / s = p
    # Specifically, when logits is zero, the naive computation produces a NaN.
    result = tf.where(tf.math.equal(logits, 0.), p, result)

    # Finally, handle the case where `logits` and `p` are on the boundary,
    # as the above expressions can result in ratio of `infs` in that case as
    # well.
    return tf.where(
        (tf.math.equal(logits, -np.inf) & tf.math.equal(logp, 0.)) |
        (tf.math.equal(logits, np.inf) & tf.math.is_inf(logp)),
        tf.ones_like(logits),
        result)

  def _mode(self):
    """Returns `1` if `prob > 0.5` and `0` otherwise."""
    return tf.cast(self._probs_parameter_no_checks() > 0.5, self.dtype)

  def logits_parameter(self, name=None):
    """Logits computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      return self._logits_parameter_no_checks()

  def _logits_parameter_no_checks(self):
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      return tf.math.log(probs) - tf.math.log1p(-probs)
    return tensor_util.identity_as_tensor(self._logits)

  def probs_parameter(self, name=None):
    """probs computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self):
    if self._logits is None:
      return tensor_util.identity_as_tensor(self._probs)
    return tf.math.sigmoid(self._logits)

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    return maybe_assert_continuous_bernoulli_param_correctness(
        is_init, self.validate_args, self._probs, self._logits)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(
        assert_util.assert_non_negative(
            x, message='Sample must be non-negative.'))
    assertions.append(
        assert_util.assert_less_equal(
            x,
            tf.ones([], dtype=x.dtype),
            message='Sample must be less than or equal to `1`.'))
    return assertions


def maybe_assert_continuous_bernoulli_param_correctness(
    is_init, validate_args, probs, logits):
  """Return assertions for `Bernoulli`-type distributions."""
  if is_init:
    x, name = (probs, 'probs') if logits is None else (logits, 'logits')
    if not dtype_util.is_floating(x.dtype):
      raise TypeError('Argument `{}` must having floating type.'.format(name))

  if not validate_args:
    return []

  assertions = []

  if probs is not None:
    if is_init != tensor_util.is_ref(probs):
      probs = tf.convert_to_tensor(probs)
      one = tf.constant(1.0, probs.dtype)
      assertions += [
          assert_util.assert_non_negative(
              probs,
              message='probs has components less than 0.'),
          assert_util.assert_less_equal(
              probs,
              one,
              message='probs has components greater than 1.')]
  return assertions


@kullback_leibler.RegisterKL(ContinuousBernoulli, ContinuousBernoulli)
def _kl_bernoulli_bernoulli(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) for Continuous Bernoullis.

  Args:
    a: instance of a continuous Bernoulli distribution object.
    b: instance of a continuous Bernoulli distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None`
      (i.e., `'kl_continuous_bernoulli_continuous_bernoulli'`).

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_continuous_bernoulli_continuous_bernoulli'):
    a_log_probs0, a_log_probs1, a_logits = a._outcome_log_probs()  # pylint:disable=protected-access
    b_log_probs0, b_log_probs1, b_probs = b._outcome_log_probs()  # pylint:disable=protected-access
    a_mean = a._mean(a_logits)  # pylint:disable=protected-access
    a_log_norm = a._log_normalizer(a_logits)  # pylint:disable=protected-access
    b_log_norm = b._log_normalizer(b_probs)  # pylint:disable=protected-access

    return (
        a_mean * (a_log_probs1 + b_log_probs0 - a_log_probs0 - b_log_probs1)
        + a_log_norm - b_log_norm + a_log_probs0 - b_log_probs0)
