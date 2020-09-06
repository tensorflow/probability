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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


class ContinuousBernoulli(distribution.Distribution):
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
  in a small interval `[lims[0], lims[1]]` around 0.5. For more details,
  see [Loaiza-Ganem and Cunningham (2019)][1].

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
      lims=(0.499, 0.501),
      dtype=tf.float32,
      validate_args=False,
      allow_nan_stats=True,
      name="ContinuousBernoulli"):
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
      lims: A list with two floats containing the lower and upper limits
       used to approximate the continuous Bernoulli around 0.5 for
       numerical stability purposes.
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
      raise ValueError("Must pass `probs` or `logits`, but not both.")
    self._lims = lims
    with tf.name_scope(name) as name:
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype_hint=tf.float32, name="probs")
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype_hint=tf.float32, name="logits")
    super(ContinuousBernoulli, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"logits": tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

  @classmethod
  def _params_event_ndims(cls):
    return dict(logits=0, probs=0)

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def _batch_shape_tensor(self):
    x = self._probs if self._logits is None else self._logits
    return ps.shape(x)

  def _batch_shape(self):
    x = self._probs if self._logits is None else self._logits
    return x.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _cut_probs(self, probs=None):
    if probs is None:
      probs = self._probs_parameter_no_checks()
    return tf.where(
        (probs < self._lims[0]) | (probs > self._lims[1]),
        probs,
        self._lims[0] * tf.ones_like(probs))

  def _sample_n(self, n, seed=None):
    probs = self._probs_parameter_no_checks()
    cut_probs = self._cut_probs(probs)
    new_shape = ps.concat([[n], ps.shape(cut_probs)], axis=0)
    uniform = samplers.uniform(new_shape, seed=seed, dtype=cut_probs.dtype)
    sample = self._quantile(uniform, probs)
    return tf.cast(sample, self.dtype)

  def _log_normalizer(self, probs=None):
    if probs is None:
      probs = self._probs_parameter_no_checks()
    cut_probs = self._cut_probs(probs)
    cut_probs_below_half = tf.where(
        cut_probs < 0.5, cut_probs, tf.zeros_like(cut_probs))
    cut_probs_above_half = tf.where(
        cut_probs > 0.5, cut_probs, tf.ones_like(cut_probs))
    log_norm = tf.math.log(
        tf.math.abs(tf.math.log1p(-cut_probs) - tf.math.log(cut_probs))
    ) - tf.where(
        cut_probs < 0.5,
        tf.math.log1p(-2.0 * cut_probs_below_half),
        tf.math.log(2.0 * cut_probs_above_half - 1.0))
    x = tf.math.square(probs - 0.5)
    taylor = np.log(2.).astype(dtype=dtype_util.as_numpy_dtype(x.dtype)) + (
        4.0 / 3.0 + 104.0 / 45.0 * x) * x
    return tf.where(
        (probs < self._lims[0]) | (probs > self._lims[1]), log_norm, taylor)

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

  def _cdf(self, x):
    probs = self._probs_parameter_no_checks()
    cut_probs = self._cut_probs(probs)
    cdfs = (
        tf.math.pow(cut_probs, x) * tf.math.exp(
            (1. - x) * tf.math.log1p(-cut_probs))
        + cut_probs - 1.) / (2.0 * cut_probs - 1.0)
    unbounded_cdfs = tf.where(
        (probs < self._lims[0]) | (probs > self._lims[1]), cdfs, x)
    return tf.where(
        x < 0.,
        tf.zeros_like(x),
        tf.where(x > 1., tf.ones_like(x), unbounded_cdfs))

  def _outcome_log_probs(self):
    """Returns log(1-probs) and log(probs)."""
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      return tf.math.log1p(-probs), tf.math.log(probs), probs
    s = tf.convert_to_tensor(self._logits)
    # softplus(s) = -Log[1 - p]
    # -softplus(-s) = Log[p]
    # softplus(+inf) = +inf, softplus(-inf) = 0, so...
    #  logits = -inf ==> log_probs0 = 0, log_probs1 = -inf (as desired)
    #  logits = +inf ==> log_probs0 = -inf, log_probs1 = 0 (as desired)
    return -tf.math.softplus(s), -tf.math.softplus(-s), tf.math.sigmoid(s)

  def _entropy(self):
    log_probs0, log_probs1, probs = self._outcome_log_probs()
    return (self._mean(probs) * (log_probs0 - log_probs1)
            - self._log_normalizer(probs) - log_probs0)

  def _mean(self, probs=None):
    if probs is None:
      probs = self._probs_parameter_no_checks()
    cut_probs = self._cut_probs(probs)
    mus = cut_probs / (2. * cut_probs - 1.0) + 1. / (
        tf.math.log1p(-cut_probs) - tf.math.log(cut_probs))
    x = probs - 0.5
    taylor = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * tf.math.square(x)) * x
    return tf.where(
        (probs < self._lims[0]) | (probs > self._lims[1]), mus, taylor)

  def _variance(self):
    probs = self._probs_parameter_no_checks()
    cut_probs = self._cut_probs(probs)
    variance = cut_probs * (cut_probs - 1.0) / tf.math.square(
        1.0 - 2.0 * cut_probs) + 1.0 / tf.math.square(
            tf.math.log1p(-cut_probs) - tf.math.log(cut_probs))
    x = tf.math.square(probs - 0.5)
    taylor = 1.0 / 12.0 - (1.0 / 15.0 - 128. / 945.0 * x) * x
    return tf.where(
        (probs < self._lims[0]) | (probs > self._lims[1]), variance, taylor)

  def _quantile(self, p, probs=None):
    if probs is None:
      probs = self._probs_parameter_no_checks()
    cut_probs = self._cut_probs(probs)
    return tf.where(
        (probs < self._lims[0]) | (probs > self._lims[1]),
        (tf.math.log1p(-cut_probs + p * (2.0 * cut_probs - 1.0))
         - tf.math.log1p(-cut_probs))
        / (tf.math.log(cut_probs) - tf.math.log1p(-cut_probs)), p)

  def _mode(self):
    """Returns `1` if `prob > 0.5` and `0` otherwise."""
    return tf.cast(self._probs_parameter_no_checks() > 0.5, self.dtype)

  def logits_parameter(self, name=None):
    """Logits computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or "logits_parameter"):
      return self._logits_parameter_no_checks()

  def _logits_parameter_no_checks(self):
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      return tf.math.log(probs) - tf.math.log1p(-probs)
    return tf.identity(self._logits)

  def probs_parameter(self, name=None):
    """probs computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or "probs_parameter"):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self):
    if self._logits is None:
      return tf.identity(self._probs)
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
            x, message="Sample must be non-negative."))
    assertions.append(
        assert_util.assert_less_equal(
            x,
            tf.ones([], dtype=x.dtype),
            message="Sample must be less than or equal to `1`."))
    return assertions


def maybe_assert_continuous_bernoulli_param_correctness(
    is_init, validate_args, probs, logits):
  """Return assertions for `Bernoulli`-type distributions."""
  if is_init:
    x, name = (probs, "probs") if logits is None else (logits, "logits")
    if not dtype_util.is_floating(x.dtype):
      raise TypeError("Argument `{}` must having floating type.".format(name))

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
              message="probs has components less than 0."),
          assert_util.assert_less_equal(
              probs,
              one,
              message="probs has components greater than 1.")]
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
  with tf.name_scope(name or "kl_continuous_bernoulli_continuous_bernoulli"):
    a_log_probs0, a_log_probs1, a_probs = a._outcome_log_probs()  # pylint:disable=protected-access
    b_log_probs0, b_log_probs1, b_probs = b._outcome_log_probs()  # pylint:disable=protected-access
    a_mean = a._mean(a_probs)  # pylint:disable=protected-access
    a_log_norm = a._log_normalizer(a_probs)  # pylint:disable=protected-access
    b_log_norm = b._log_normalizer(b_probs)  # pylint:disable=protected-access

    return (
        a_mean * (a_log_probs1 + b_log_probs0 - a_log_probs0 - b_log_probs1)
        + a_log_norm - b_log_norm + a_log_probs0 - b_log_probs0)
