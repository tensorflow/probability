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

import numpy as onp

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties


__all__ = [
    'LogitNormal',
]


# TODO(b/182603117): Remove `AutoCompositeTensor` subclass when
# `TransformedDistribution` is converted to `CompositeTensor`.
class LogitNormal(transformed_distribution.TransformedDistribution,
                  distribution.AutoCompositeTensorDistribution):
  """The logit-normal distribution."""

  def __init__(self,
               loc,
               scale,
               num_probit_terms_approx=2,
               gauss_hermite_scale_limit=None,
               gauss_hermite_degree=20,
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
      num_probit_terms_approx: The `k` used in the approximation,
        `sigmoid(x) approx= sum_i^k p[k,i] Normal(0, c[k, i]).cdf(x)`
        where `sum_i^k p[k,i]=1` and `p[k,i],c[k,i] > 0`
        [(Monahan and Stefanski, 1989)][1] and used in `mean_*_approx` functions
        [(Owen, 1980)][2]. Must be a python scalar integer between `1` and `8`
        (inclusive). Using `num_probit_terms_approx=2` should result in
        `mean_approx` error not exceeding `10**-4`.
        Default value: `2`.
      gauss_hermite_scale_limit: Floating-point `Tensor` or `None`.
        The (batch-wise) maximum scale at which to compute statistics
        with Gauss-Hermite quadrature instead of the Monahan-Stefanski
        approximation [1].  Default: `None`, which recovers the legacy
        behavior of using Monahan-Stefanski everywhere and does not
        add TF ops for Gauss-Hermite.  The best value depends on the
        working precision and the number of terms in the Gauss-Hermite
        or Monahan-Stefanski approximations being switched between,
        as well as the expected range of `loc` parameters; but `1` is
        not unreasonable.
      gauss_hermite_degree: Python integer giving the number of
        sample points to use for Gauss-Hermite quadrature.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    #### References

    [1]: Monahan, John H., and Leonard A. Stefanski. Normal scale mixture
         approximations to the logistic distribution with applications. North
         Carolina State University. Dept. of Statistics, 1989.
         http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.154.5032
    [2]: Owen, Donald Bruce. "A table of normal integrals: A table."
         Communications in Statistics-Simulation and Computation 9.4 (1980):
         389-419.
         https://www.tandfonline.com/doi/abs/10.1080/03610918008812164

    """
    parameters = dict(locals())
    num_probit_terms_approx = int(num_probit_terms_approx)
    if num_probit_terms_approx < 1 or num_probit_terms_approx > 8:
      raise ValueError(
          'Argument `num_probit_terms_approx` must be an integer between '
          '`1` and `8` (inclusive).')
    self._num_probit_terms_approx = num_probit_terms_approx
    self._gauss_hermite_scale_limit = gauss_hermite_scale_limit
    self._gauss_hermite_degree = gauss_hermite_degree
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

  @property
  def num_probit_terms_approx(self):
    """Number of `Normal(0, 1).cdf` terms using in `mean_*_approx` functions."""
    return self._num_probit_terms_approx

  @property
  def gauss_hermite_scale_limit(self):
    """Largest scale using Gauss-Hermite quadrature in `*_approx` functions."""
    return self._gauss_hermite_scale_limit

  @property
  def gauss_hermite_degree(self):
    """Number of points for Gauss-Hermite quadrature in `*_approx` functions."""
    return self._gauss_hermite_degree

  experimental_is_sharded = False

  def mean_log_prob_approx(self, y=None, name='mean_log_prob_approx'):
    """Approximates `E_Normal(m,s)[ Bernoulli(sigmoid(X)).log_prob(Y) ]`.

    This approximation is based on combining ideas from
    [(Monahan and Stefanski, 1989)][1] and [(Owen, 1980)][2].

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

    #### References

    [1]: Monahan, John H., and Leonard A. Stefanski. Normal scale mixture
         approximations to the logistic distribution with applications. North
         Carolina State University. Dept. of Statistics, 1989.
         http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.154.5032
    [2]: Owen, Donald Bruce. "A table of normal integrals: A table."
         Communications in Statistics-Simulation and Computation 9.4 (1980):
         389-419.
         https://www.tandfonline.com/doi/abs/10.1080/03610918008812164
    """
    with self._name_and_control_scope(name):
      return approx_expected_log_prob_sigmoid(
          self.loc, self.scale, y,
          MONAHAN_MIX_PROB[self.num_probit_terms_approx],
          MONAHAN_INVERSE_SCALE[self.num_probit_terms_approx])

  def mean_approx(self, name='mean_approx'):
    """Approximate the mean of a LogitNormal.

    This approximation is based on combining ideas from
    [(Monahan and Stefanski, 1989)][1] and [(Owen, 1980)][2].

    Warning: usual numerical guarantees are not offered for this function as it
    attempts to strike a balance between computational cost, implementation
    simplicity and numerical accuracy.

    Args:
      name: Python `str` prepended to names of ops created by this function.
        Default value: `'mean_approx'`.

    Returns:
      mean_approx: An approximation of the mean of a LogitNormal.

    #### References

    [1]: Monahan, John H., and Leonard A. Stefanski. Normal scale mixture
         approximations to the logistic distribution with applications. North
         Carolina State University. Dept. of Statistics, 1989.
         http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.154.5032
    [2]: Owen, Donald Bruce. "A table of normal integrals: A table."
         Communications in Statistics-Simulation and Computation 9.4 (1980):
         389-419.
         https://www.tandfonline.com/doi/abs/10.1080/03610918008812164
    """
    with self._name_and_control_scope(name):
      loc = tf.convert_to_tensor(self.loc)
      scale = tf.convert_to_tensor(self.scale)
      monahan_stefanski_answer = approx_expected_sigmoid(
          loc, scale,
          MONAHAN_MIX_PROB[self.num_probit_terms_approx],
          MONAHAN_INVERSE_SCALE[self.num_probit_terms_approx])
      if self.gauss_hermite_scale_limit is None:
        return monahan_stefanski_answer
      else:
        gauss_hermite_answer = logit_normal_mean_gh(
            loc, scale, self.gauss_hermite_degree)
        return tf.where(scale < self.gauss_hermite_scale_limit,
                        gauss_hermite_answer, monahan_stefanski_answer)

  def variance_approx(self, name='variance_approx'):
    """Approximate the variance of a LogitNormal.

    This approximation is based on combining ideas from
    [(Monahan and Stefanski, 1989)][1] and [(Owen, 1980)][2].

    Warning: usual numerical guarantees are not offered for this function as it
    attempts to strike a balance between computational cost, implementation
    simplicity and numerical accuracy.

    Args:
      name: Python `str` prepended to names of ops created by this function.
        Default value: `'variance_approx'`.

    Returns:
      variance_approx: An approximation of the variance of a LogitNormal.

    #### References

    [1]: Monahan, John H., and Leonard A. Stefanski. Normal scale mixture
         approximations to the logistic distribution with applications. North
         Carolina State University. Dept. of Statistics, 1989.
         http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.154.5032
    [2]: Owen, Donald Bruce. "A table of normal integrals: A table."
         Communications in Statistics-Simulation and Computation 9.4 (1980):
         389-419.
         https://www.tandfonline.com/doi/abs/10.1080/03610918008812164
    """
    with self._name_and_control_scope(name):
      loc = tf.convert_to_tensor(self.loc)
      scale = tf.convert_to_tensor(self.scale)
      monahan_stefanski_answer = approx_variance_sigmoid(
          loc, scale,
          MONAHAN_MIX_PROB[self.num_probit_terms_approx],
          MONAHAN_INVERSE_SCALE[self.num_probit_terms_approx])
      if self.gauss_hermite_scale_limit is None:
        return monahan_stefanski_answer
      else:
        gauss_hermite_answer = logit_normal_variance_gh(
            loc, scale, self.gauss_hermite_degree)
        return tf.where(scale < self.gauss_hermite_scale_limit,
                        gauss_hermite_answer, monahan_stefanski_answer)

  def stddev_approx(self, name='stddev_approx'):
    """Approximate the stdandard deviation of a LogitNormal.

    This approximation is based on combining ideas from
    [(Monahan and Stefanski, 1989)][1] and [(Owen, 1980)][2].

    Warning: usual numerical guarantees are not offered for this function as it
    attempts to strike a balance between computational cost, implementation
    simplicity and numerical accuracy.

    Args:
      name: Python `str` prepended to names of ops created by this function.
        Default value: `'stddev_approx'`.

    Returns:
      stddev_approx: An approximation of the variance of a LogitNormal.

    #### References

    [1]: Monahan, John H., and Leonard A. Stefanski. Normal scale mixture
         approximations to the logistic distribution with applications. North
         Carolina State University. Dept. of Statistics, 1989.
         http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.154.5032
    [2]: Owen, Donald Bruce. "A table of normal integrals: A table."
         Communications in Statistics-Simulation and Computation 9.4 (1980):
         389-419.
         https://www.tandfonline.com/doi/abs/10.1080/03610918008812164
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


# Our approximations to various LogitNormal expectations are based on the
# approximation:
#
#   expit(x) approx= sum_i^k p[k,i] Normal(loc=0, scale=c[k,i]).cdf(x)
#
# where `sum_i^k p[k,i] = 1` and `p[k,i],c[k,i] > 0`. This approximation was
# first proposed by,
#
#   Monahan, John H., and Leonard A. Stefanski. Normal scale mixture
#   approximations to the logistic distribution with applications. North
#   Carolina State University. Dept. of Statistics, 1989.
#
# Note: we use the reciprocal of the values reported in Table 1, i.e.,
# `c := 1 / s`.
#
# The `k=1` approximation, `Normal(0, t).cdf(x)`, with
#
#   `t = sqrt(8 / pi) = 1.5957691216057308`
#
# was proposed by:
#
#   Cox, D. R. (1970). Binary Regression. Chapman and Hall, London.
#
# Seeking to reduce error, the following recommends:
#
#   `t = (15*np.pi)/(16 * sqrt(3)) = 1.7004369039695792`.
#
#   Johnson, N. L. and Kotz, S. (1970). Distributions in Statistic3, Continuous
#   Univariate Distributions, Vol. 2. Boston: Houghton-Mifflin.
#
# The Monahan & Stefanaski work is notable in that they devise an adaptation of
# the Remez Algorithm to identify the values of `p,c` which minimize the
# maximum absolute deviation.
#
# We extend the idea of Monahan & Stefanski by computing variation expectations
# over their approximation. These integrals are sometimes complex but can be
# found in:
#
#   Owen, Donald Bruce. "A table of normal integrals: A table." Communications
#   in Statistics-Simulation and Computation 9.4 (1980): 389-419.
#
# Specifically:
#   #110       pg 396 (9)
#   #1,000     pg 396 (9)
#   #10,010.8  pg 403 (16)
#   #10,011.3  pg 404 (17)
#   #20,010.3  pg 407 (20)

# We've added small corrections to ensure the probability vectors sum to 1.


_p1 = (1.000000000000000,)
_p2 = (0.564424843456014, 0.435575156943986 - 4e-10,)
_p3 = (0.252201578098282, 0.585225059235736, 0.162573362665982,)
_p4 = (0.106498992656952, 0.458361227014536, 0.374189066914829,
       0.060950713413683,)
_p5 = (0.044333151939163, 0.294973376977114, 0.429812481900555,
       0.207589505757111, 0.023291483426056 + 1e-15,)
_p6 = (0.018446105135654, 0.172681380923308, 0.373930796025243,
       0.316969955813251, 0.108897300053481, 0.009074462049063,)
_p7 = (0.007711833444756, 0.095865353040290, 0.281948308310964,
       0.345462848809089, 0.209848686083383, 0.055564617900566,
       0.003598352410953 - 1e-15,)
_p8 = (0.003246343272134, 0.051517477033972, 0.195077912673858,
       0.315569823632818, 0.274149576158423, 0.131076880695470,
       0.027912418727972, 0.001449567805354 - 1e-15,)


_c1 = (1.7017449253380041,)
_c2 = (1.3010310715970526, 2.2975028116638367,)
_c3 = (1.1014054801893525, 1.7307407783023951, 2.7469680173784603)
_c4 = (0.9773875711666805, 1.4310771302494774, 2.1046981034348815,
       3.1146374470920946,)
_c5 = (0.8906971667576289, 1.2439340916676265, 1.7355155899786117,
       2.4336004973448710, 3.430768355233885,)
_c6 = (0.8255852816374327, 1.1143934930722950, 1.4962580515068244,
       2.0180151926414407, 2.7271068221270283, 3.7110992347402076,)
_c7 = (0.7742712977850712, 1.0184098437059321, 1.3284183394336735,
       1.7376954617683864, 2.2798822466698083, 2.9930775267481997,
       3.9648520442506237,)
_c8 = (0.7324178662121884, 0.9438200808616102, 1.2036717084431239,
       1.5367305494737573, 1.9679793025230570, 2.5232559342160914,
       3.2372490590787770, 4.1979304667967790,)


MONAHAN_MIX_PROB = (_p1, _p2, _p3, _p4, _p5, _p6, _p7, _p8)
MONAHAN_INVERSE_SCALE = (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8)


DEFAULT_ORDER = 2
DEFAULT_MIX_PROB = MONAHAN_MIX_PROB[DEFAULT_ORDER - 1]
DEFAULT_SCALE = MONAHAN_INVERSE_SCALE[DEFAULT_ORDER - 1]


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
    m, s, y=None, alpha=DEFAULT_MIX_PROB, c=DEFAULT_SCALE, name=None):
  """Approximates `E_{N(m,s)}[Bernoulli(sigmoid(X)).log_prob(Y)]`."""
  with tf.name_scope(name or 'approx_expected_log_prob_sigmoid'):
    m, s, alpha, c = _prepare_args(m, s, alpha, c)
    ym = m if y is None else tf.cast(y, m.dtype, name='y') * m
    return ym - approx_expected_softplus(m, s, alpha, c)


def approx_expected_softplus(
    m, s, alpha=DEFAULT_MIX_PROB, c=DEFAULT_SCALE, name=None):
  """Approximates `E_{N(m,s)}[softplus(X)]`."""
  with tf.name_scope(name or 'approx_expected_softplus'):
    alpha, m_over_rho, cdf, pdf, one_over_rho, m_, s_, c = _common(
        m, s, alpha, c)
    return tf.math.reduce_sum(
        alpha * (
            (c**2 + s_**2) * one_over_rho * pdf(m_over_rho) +
            m_ * cdf(m_over_rho)),
        axis=-1)


def approx_expected_sigmoid(
    m, s, alpha=DEFAULT_MIX_PROB, c=DEFAULT_SCALE, name=None):
  """Approximates `E_{N(m,s)}[sigmoid(X)]`."""
  with tf.name_scope(name or 'approx_expected_sigmoid'):
    alpha, m_over_rho, cdf = _common(m, s, alpha, c)[:3]
    return tf.math.reduce_sum(alpha * cdf(m_over_rho), axis=-1)


def approx_variance_sigmoid(
    m, s, alpha=DEFAULT_MIX_PROB, c=DEFAULT_SCALE, name=None):
  """Approxmates `Var_{N(m,s)}[sigmoid(X)]`."""
  # TODO(jvdillon): See if we can rederive things so we can avoid catastrophic
  # cancellation which might be present in the following calculation. One idea
  # might be to apply the law of total variance by leveraging the fact that each
  # of the calculations below are subdivided into three segments.
  with tf.name_scope(name or 'approx_variance_sigmoid'):
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


# The above approximations fail for small scales.  We compute
# statistics for small scales with Gauss-Hermite quadrature.


def logit_normal_mean_gh(loc, scale, deg):
  """Approximates `E_{N(m,s)}[sigmoid(X)]` by Gauss-Hermite quadrature."""
  # We want to integrate
  # A = \int_-inf^inf sigmoid(x) * Normal(loc, scale).pdf(x) dx
  # To bring it into the right form for Gauss-Hermite quadrature,
  # we make the substitution y = (x - loc) / scale, to get
  # A = (1/sqrt(2*pi)) * \int_-inf^inf [
  #       sigmoid(y * scale + loc) * exp(-1/2 y**2) dy]
  grid, weights = onp.polynomial.hermite_e.hermegauss(deg)
  grid = tf.cast(grid, dtype=loc.dtype)
  weights = tf.cast(weights, dtype=loc.dtype)
  normalizer = tf.constant(onp.sqrt(2 * onp.pi), dtype=loc.dtype)
  values = tf.sigmoid(grid * scale[..., tf.newaxis] + loc[..., tf.newaxis])
  return tf.reduce_sum(values * weights, axis=-1) / normalizer


def logit_normal_variance_gh(loc, scale, deg):
  """Approxmates `Var_{N(m,s)}[sigmoid(X)]` by Gauss-Hermite quadrature."""
  # Since we have to compute sigmoids for variance anyway, we inline
  # computing the mean by Gauss-Hermite quadrature at the same grid of points.
  grid, weights = onp.polynomial.hermite_e.hermegauss(deg)
  grid = tf.cast(grid, dtype=loc.dtype)
  weights = tf.cast(weights, dtype=loc.dtype)
  normalizer = tf.constant(onp.sqrt(2 * onp.pi), dtype=loc.dtype)
  sigmoids = tf.sigmoid(grid * scale[..., tf.newaxis] + loc[..., tf.newaxis])
  mean = tf.reduce_sum(sigmoids * weights, axis=-1) / normalizer
  residuals = (sigmoids - mean[..., tf.newaxis])**2
  return tf.reduce_sum(residuals * weights, axis=-1) / normalizer
