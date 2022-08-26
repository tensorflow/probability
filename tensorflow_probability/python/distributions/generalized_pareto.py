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
"""Generalized Pareto distribution."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import generalized_pareto as generalized_pareto_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import generic


class GeneralizedPareto(distribution.AutoCompositeTensorDistribution):
  """The Generalized Pareto distribution.

  The Generalized Pareto distributions are a family of continuous distributions
  on the reals. Special cases include `Exponential` (when `loc = 0`,
  `concentration = 0`), `Pareto` (when `concentration > 0`,
  `loc = scale / concentration`), and `Uniform` (when `concentration = -1`).

  This distribution is often used to model the tails of other distributions.

  As a member of the location-scale family,
  `X ~ GeneralizedPareto(loc=loc, scale=scale, concentration=conc)` maps to
  `Y ~ GeneralizedPareto(loc=0, scale=1, concentration=conc)` via
  `Y = (X - loc) / scale`.

  For positive concentrations, the distribution is equivalent to a hierarchical
  Exponential-Gamma model with `X|rate ~ Exponential(rate)` and
  `rate ~ Gamma(concentration=1 / concentration, scale=scale / concentration)`.
  In the following, `samp1` and `samps2` are identically distributed:

  ```python
  genp = tfd.GeneralizedPareto(loc=0, scale=scale, concentration=conc)
  samps1 = genp.sample(1000)
  jd = tfd.JointDistributionNamed(dict(
      rate=tfd.Gamma(1 / genp.concentration, genp.scale / genp.concentration),
      x=lambda rate: tfd.Exponential(rate)))
  samps2 = jd.sample(1000)['x']
  ```

  The support of the distribution is always lower bounded by `loc`. When
  `concentration < 0`, the support is also upper bounded by
  `loc + scale / abs(concentration)`.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; mu, sigma, shp, x > mu) =
      (1 + shp * (x - mu) / sigma)**(-1 / shp - 1) / sigma
  ```

  where:

  * `concentration = shp`, any real value,
  * `scale = sigma`, `sigma > 0`,
  * `loc = mu`.

  The cumulative density function (cdf) is,

  ```none
  cdf(x; mu, sigma, shp, x > mu) = 1 - (1 + shp * (x - mu) / sigma)**(-1 / shp)
  ```

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Samples of this distribution are reparameterized (pathwise differentiable).

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dist = tfd.GeneralizedPareto(loc=1., scale=2., concentration=0.03)
  dist2 = tfd.GeneralizedPareto(loc=-2., scale=[3., 4.],
                                concentration=[[-.4], [0.2]])
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  loc = tf.Variable(3.0)
  scale = tf.Variable(2.0)
  conc = tf.Variable(0.1)
  dist = tfd.GeneralizedPareto(loc, scale, conc)
  with tf.GradientTape() as tape:
    samples = dist.sample(5)  # Shape [5]
    loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tape.gradient(loss, dist.variables)
  ```

  """

  def __init__(self,
               loc,
               scale,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Construct a Generalized Pareto distribution.

    Args:
      loc: The location / shift of the distribution. GeneralizedPareto is a
        location-scale distribution. This parameter lower bounds the
        distribution's support. Must broadcast with `scale`, `concentration`.
        Floating point `Tensor`.
      scale: The scale of the distribution. GeneralizedPareto is a
        location-scale distribution, so doubling the `scale` doubles a sample
        and halves the density. Strictly positive floating point `Tensor`. Must
        broadcast with `loc`, `concentration`.
      concentration: The shape parameter of the distribution. The larger the
        magnitude, the more the distribution concentrates near `loc` (for
        `concentration >= 0`) or near `loc - (scale/concentration)` (for
        `concentration < 0`). Floating point `Tensor`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, variance) use the value "`NaN`" to indicate the result is
        undefined. When `False`, an exception is raised if one or more of the
        statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc`, `scale`, or `concentration` have different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'GeneralizedPareto') as name:
      dtype = dtype_util.common_dtype([loc, scale, concentration],
                                      dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      super(GeneralizedPareto, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration=parameter_properties.ParameterProperties())
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    return self._loc

  @property
  def scale(self):
    return self._scale

  @property
  def concentration(self):
    return self._concentration

  def _event_shape(self):
    return []

  def _sample_n(self, n, seed=None):
    # Inversion samples via inverse CDF.
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    concentration = tf.convert_to_tensor(self.concentration)
    # Pre-broadcast to ensure we draw enough randomness.
    sample_shp = ps.concat(
        [[n],
         self._batch_shape_tensor(
             loc=loc, scale=scale, concentration=concentration)],
        axis=0)
    logu = tf.math.log1p(
        -samplers.uniform(sample_shp, dtype=self.dtype, seed=seed))
    eq_zero = tf.equal(concentration, 0)
    safe_conc = tf.where(eq_zero, tf.constant(1, dtype=self.dtype),
                         concentration)
    where_nonzero = loc + scale / safe_conc * tf.math.expm1(-safe_conc * logu)
    where_zero = loc - scale * logu
    return tf.where(eq_zero, where_zero, where_nonzero)

  def _log_prob(self, x):
    scale = tf.convert_to_tensor(self.scale)
    concentration = tf.convert_to_tensor(self.concentration)
    z = self._z(x, scale)
    eq_zero = tf.equal(concentration, 0)  # Concentration = 0 ==> Exponential.
    nonzero_conc = tf.where(eq_zero, tf.constant(1, self.dtype), concentration)
    y = 1 / nonzero_conc + tf.ones_like(z, self.dtype)
    where_nonzero = tf.where(
        tf.equal(y, 0), y, y * tf.math.log1p(nonzero_conc * z))
    return -tf.math.log(scale) - tf.where(eq_zero, z, where_nonzero)

  def _log_survival_function(self, x):
    scale = tf.convert_to_tensor(self.scale)
    concentration = tf.convert_to_tensor(self.concentration)
    z = self._z(x, scale)
    eq_zero = tf.equal(concentration, 0)  # Concentration = 0 ==> Exponential.
    nonzero_conc = tf.where(eq_zero, tf.constant(1, self.dtype), concentration)
    where_nonzero = -tf.math.log1p(nonzero_conc * z) / nonzero_conc
    return tf.where(eq_zero, -z, where_nonzero)

  def _log_cdf(self, x):
    # Going through the survival function is more accurate when conc is near
    # zero, because it amounts to computing the (1 + conc * z)**(-1 / conc)
    # term in log-space with log1p.
    # tfp_math.log1mexp(a) accurately computes log(1 - exp(-|a|)).  The negation
    # and the absolute value are fine here because the log survival function is
    # always non-positive.
    return generic.log1mexp(self._log_survival_function(x))

  def _z(self, x, scale):
    loc = tf.convert_to_tensor(self.loc)
    return (x - loc) / scale

  def _mean(self):
    concentration = tf.convert_to_tensor(self.concentration)
    lim = tf.ones([], dtype=self.dtype)
    valid = concentration < lim
    safe_conc = tf.where(valid, concentration, tf.constant(.5, self.dtype))
    result = lambda: self.loc + self.scale / (1 - safe_conc)
    if self.allow_nan_stats:
      return tf.where(valid, result(), tf.constant(float('nan'), self.dtype))
    with tf.control_dependencies([
        assert_util.assert_less(
            concentration,
            lim,
            message='`mean` is undefined when `concentration >= 1`')
    ]):
      return result()

  def _variance(self):
    concentration = tf.convert_to_tensor(self.concentration)
    lim = tf.constant(.5, self.dtype)
    valid = concentration < lim
    safe_conc = tf.where(valid, concentration, tf.constant(.25, self.dtype))
    def result():
      answer = self.scale**2 / ((1 - safe_conc)**2 * (1 - 2 * safe_conc))
      # Force broadcasting with self.loc to get the shape right, even though the
      # variance doesn't depend on the location.
      return answer + tf.zeros_like(self.loc)
    if self.allow_nan_stats:
      return tf.where(valid, result(), tf.constant(float('nan'), self.dtype))
    with tf.control_dependencies([
        assert_util.assert_less(
            concentration,
            lim,
            message='`variance` is undefined when `concentration >= 0.5`')
    ]):
      return result()

  def _entropy(self):
    ans = tf.math.log(self.scale) + self.concentration + 1
    return tf.broadcast_to(ans, self._batch_shape_tensor())

  # TODO(b/145620027): Finalize choice of bijector.
  def _default_event_space_bijector(self):
    return generalized_pareto_bijector.GeneralizedPareto(
        self.loc,
        scale=self.scale,
        concentration=self.concentration,
        validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(
          assert_util.assert_positive(
              self.scale, message='Argument `scale` must be positive.'))
    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    concentration = tf.convert_to_tensor(self.concentration)
    assertions.append(assert_util.assert_greater_equal(
        x, loc, message='Sample must be greater than or equal to `loc`.'))
    assertions.append(assert_util.assert_equal(
        tf.logical_or(tf.greater_equal(concentration, 0),
                      tf.less_equal(x, loc - scale / concentration)),
        True,
        message=('If `concentration < 0`, sample must be less than or '
                 'equal to `loc - scale / concentration`.'),
        summarize=100))
    return assertions

  def _quantile(self, p):
    k = tf.convert_to_tensor(self.concentration)
    m = tf.convert_to_tensor(self.loc)
    s = tf.convert_to_tensor(self.scale)
    is_k_zero = tf.equal(k, 0)
    # Use double where trick to ensure gradient correctness.
    safe_k = tf.where(is_k_zero, tf.ones([], k.dtype), k)
    neglog1mp = -tf.math.log1p(-p)
    return m + s * tf.where(is_k_zero,
                            neglog1mp,
                            tf.math.expm1(k * neglog1mp) / safe_k)
