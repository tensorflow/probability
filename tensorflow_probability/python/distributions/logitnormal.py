# Copyright 2019 The TensorFlow Probability Authors.
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
"""LogitNormal distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties


__all__ = [
    'LogitNormal',
]


class LogitNormal(transformed_distribution.TransformedDistribution):
  """The logit-normal distribution."""

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='LogitNormal'):
    """Construct a logit-normal distribution.

    The LogitNormal distribution models random variables between 0 and 1 whose
    logit (i.e., sigmoid_inverse, i.e., `log(p) - log1p(-p)`) is normally
    distributed with mean `loc` and standard deviation `scale`. It is
    constructed as the sigmoid transformation, (i.e., `1 / (1 + exp(-x))`) of a
    Normal distribution.

    Args:
      loc: Floating-point `Tensor`; the mean of the underlying
        Normal distribution(s). Must broadcast with `scale`.
      scale: Floating-point `Tensor`; the stddev of the underlying
        Normal distribution(s). Must broadcast with `loc`.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(LogitNormal, self).__init__(
          distribution=normal_lib.Normal(loc=loc, scale=scale),
          bijector=sigmoid_bijector.Sigmoid(),
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the pre-transformed mean."""
    return self.distribution.loc

  @property
  def scale(self):
    """Distribution parameter for the pre-transformed standard deviation."""
    return self.distribution.scale

  def mean_log_prob_approx(self, y=None, name='mean_log_prob_approx'):
    """Approximates `E_Normal(m,s)[ Bernoulli(sigmoid(X)).log_prob(Y) ]`.

    Warning: usual numerical guarantees are not offered for this function as it
    attempts to strike a balance between computational cost, implementation
    simplicity and numerical accuracy.

    Args:
      y: The events over which to compute the Bernoulli log prob.
        Default value: `None` (i.e., `1`).
      name: Python `str` prepended to names of ops created by this function.
        Default value: `'mean_log_prob_approx'`.

    Returns:
      mean_log_prob_approx: An approximation of the mean of the Bernoulli
        likelihood.
    """
    with self._name_and_control_scope(name):
      return approx_expected_log_prob_sigmoid(self.loc, self.scale, y)

  def mean_approx(self, name='mean_approx'):
    """Approximate the mean of a LogitNormal.

    Warning: usual numerical guarantees are not offered for this function as it
    attempts to strike a balance between computational cost, implementation
    simplicity and numerical accuracy.

    Args:
      name: Python `str` prepended to names of ops created by this function.
        Default value: `'mean_approx'`.

    Returns:
      mean_approx: An approximation of the mean of a LogitNormal.
    """
    with self._name_and_control_scope(name):
      return approx_expected_sigmoid(self.loc, self.scale)

  def variance_approx(self, name='variance_approx'):
    """Approximate the variance of a LogitNormal.

    Warning: usual numerical guarantees are not offered for this function as it
    attempts to strike a balance between computational cost, implementation
    simplicity and numerical accuracy.

    Args:
      name: Python `str` prepended to names of ops created by this function.
        Default value: `'variance_approx'`.

    Returns:
      variance_approx: An approximation of the variance of a LogitNormal.
    """
    with self._name_and_control_scope(name):
      return approx_var_sigmoid(self.loc, self.scale)

  def stddev_approx(self, name='stddev_approx'):
    """Approximate the stdandard deviation of a LogitNormal.

    Warning: usual numerical guarantees are not offered for this function as it
    attempts to strike a balance between computational cost, implementation
    simplicity and numerical accuracy.

    Args:
      name: Python `str` prepended to names of ops created by this function.
        Default value: `'stddev_approx'`.

    Returns:
      stddev_approx: An approximation of the variance of a LogitNormal.
    """
    with self._name_and_control_scope(name):
      return tf.math.sqrt(self.variance_approx())

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    assertions.append(assert_util.assert_less_equal(
        x, tf.ones([], x.dtype),
        message='Sample must be less than or equal to `1`.'))
    return assertions


@kullback_leibler.RegisterKL(LogitNormal, LogitNormal)
def _kl_logitnormal_logitnormal(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b LogitNormal.

  This is the same as the KL divergence between the underlying Normal
  distributions.

  Args:
    a: instance of a LogitNormal distribution object.
    b: instance of a LogitNormal distribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e., `'kl_logitnormal_logitnormal'`).

  Returns:
    kl_div: Batchwise KL(a || b)
  """
  return kullback_leibler.kl_divergence(
      a.distribution,
      b.distribution,
      name=(name or 'kl_logitnormal_logitnormal'))


# TODO(jvdillon): Submit an ArXiv whitepaper which explains the following magic
# (or better, see if we can find it published somewhere).

DEFAULT_ALPHA_ = (
    np.array([1.], np.float64),
    np.array([0.56503847370701580, 0.43496152629298430], np.float64),
    np.array([0.58313582494960810, 0.25578908169973547, 0.16107509335065637],
             np.float64),
)


DEFAULT_C_ = (
    np.array([1.6989831990656630], np.float64),
    np.array([1.3020384430143920, 2.2974030643118337], np.float64),
    np.array([1.7350019936800363, 1.1044586742992297, 2.7501402258610304],
             np.float64),
)


DEFAULT_ORDER = 2
DEFAULT_ALPHA = DEFAULT_ALPHA_[DEFAULT_ORDER - 1]
DEFAULT_C = DEFAULT_C_[DEFAULT_ORDER - 1]


def _prepare_args(m, s, alpha, c):
  dtype = dtype_util.common_dtype([m, s], dtype_hint=tf.float32)
  m = tf.convert_to_tensor(m, dtype, name='m')
  s = tf.convert_to_tensor(s, dtype, name='s')
  c = tf.cast(c, dtype=dtype, name='c')
  alpha = tf.cast(alpha, dtype=dtype, name='alpha')
  return m, s, alpha, c


def _get_cdf_pdf(c):
  dtype = dtype_util.as_numpy_dtype(c.dtype)
  d = normal_lib.Normal(dtype(0), 1)
  return d.cdf, d.prob
  # Could also try back-substituting the approximation, i.e.,
  # return lambda x: tf.math.sigmoid(c * x), d.prob


def _common(m, s, alpha, c):
  m, s, alpha, c = _prepare_args(m, s, alpha, c)
  m_ = m[..., tf.newaxis]
  s_ = s[..., tf.newaxis]
  one_over_rho = tf.math.rsqrt(c**2 + s_**2)
  m_over_rho = m_ * one_over_rho
  cdf, pdf = _get_cdf_pdf(c)
  return alpha, m_over_rho, cdf, pdf, one_over_rho, m_, s_, c


def approx_expected_log_prob_sigmoid(
    m, s, y=None, alpha=DEFAULT_ALPHA, c=DEFAULT_C, name=None):
  """Approximates `E_{N(m,s)}[Bernoulli(sigmoid(X)).log_prob(Y)]`."""
  with tf.name_scope(name or 'approx_expected_log_prob_sigmoid'):
    m, s, alpha, c = _prepare_args(m, s, alpha, c)
    ym = m if y is None else tf.cast(y, m.dtype, name='y') * m
    return ym - approx_expected_softplus(m, s, alpha, c)


def approx_expected_softplus(m, s, alpha=DEFAULT_ALPHA, c=DEFAULT_C, name=None):
  """Approximates `E_{N(m,s)}[softplus(X)]`."""
  with tf.name_scope(name or 'approx_expected_softplus'):
    alpha, m_over_rho, cdf, pdf, one_over_rho, m_, s_, c = _common(
        m, s, alpha, c)
    return tf.math.reduce_sum(
        alpha * (
            (c**2 + s_**2) * one_over_rho * pdf(m_over_rho) +
            m_ * cdf(m_over_rho)),
        axis=-1)


def approx_expected_sigmoid(m, s, alpha=DEFAULT_ALPHA, c=DEFAULT_C, name=None):
  """Approximates `E_{N(m,s)}[sigmoid(X)]`."""
  with tf.name_scope(name or 'approx_expected_sigmoid'):
    alpha, m_over_rho, cdf = _common(m, s, alpha, c)[:3]
    return tf.math.reduce_sum(alpha * cdf(m_over_rho), axis=-1)


def approx_var_sigmoid(m, s, alpha=DEFAULT_ALPHA, c=DEFAULT_C, name=None):
  """Approxmates `Var_{N(m,s)}[sigmoid(X)]`."""
  # TODO(jvdillon): See if we can rederive things so we can avoid catastrophic
  # cancellation which might be present in the following calculation. One idea
  # might be to apply the law of total variance by leveraging the fact that each
  # of the calculations below are subdivided into three segments.
  with tf.name_scope(name or 'approx_var_sigmoid'):
    alpha, m_over_rho, cdf, _, _, _, s_, c = _common(m, s, alpha, c)
    c2 = c**2
    c2s2_ = c2 * s_**2
    c2_over_big_rho_ = c2[:, tf.newaxis] * tf.math.rsqrt(
        c2[tf.newaxis, :] * c2[:, tf.newaxis] +
        c2s2_[..., tf.newaxis, :] +
        c2s2_[..., :, tf.newaxis])
    m_over_rho_ = m_over_rho[..., tf.newaxis]
    b = 0.5 * cdf(m_over_rho_) - tfp_math.owens_t(m_over_rho_, c2_over_big_rho_)
    bt = tf.linalg.matrix_transpose(b)
    mom2 = tf.math.reduce_sum(
        alpha[tf.newaxis, :] * alpha[:, tf.newaxis] * (b + bt),
        axis=[-2, -1])
    return mom2 - approx_expected_sigmoid(m, s, alpha, c)**2.
