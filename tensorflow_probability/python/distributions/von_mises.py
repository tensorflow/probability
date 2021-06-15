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
"""The von Mises distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.gradient import value_and_gradient

__all__ = ['VonMises']


class VonMises(distribution.AutoCompositeTensorDistribution):
  """The von Mises distribution over angles.

  The von Mises distribution is a univariate directional distribution.
  Similarly to Normal distribution, it is a maximum entropy distribution.
  The samples of this distribution are angles, measured in radians.
  They are 2 pi-periodic: x = 0 and x = 2pi are equivalent.
  This means that the density is also 2 pi-periodic.
  The generated samples, however, are guaranteed to be in [-pi, pi) range.

  When `concentration = 0`, this distribution becomes a Uniform distribuion on
  the [-pi, pi) domain.

  The von Mises distribution is a special case of von Mises-Fisher distribution
  for n=2. However, the TFP's VonMisesFisher implementation represents the
  samples and location as (x, y) points on a circle, while VonMises represents
  them as scalar angles.

  #### Mathematical details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; loc, concentration) = exp(concentration cos(x - loc)) / Z
  Z = 2 * pi * I_0 (concentration)
  ```

  where:
  * `I_0 (concentration)` is the modified Bessel function of order zero;
  * `loc` the circular mean of the distribution, a scalar. It can take arbitrary
    values, but it is 2pi-periodic: loc and loc + 2pi result in the same
    distribution.
  * `concentration >= 0` parameter is the concentration parameter. When
  `concentration = 0`,
    this distribution becomes a Uniform distribution on [-pi, pi).

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Create a batch of three von Mises distributions.
  loc = [1, 2, 3]
  concentration = [1, 2, 3]
  dist = tfp.distributions.VonMises(loc=[1.0, 2.0], concentration=[0.5, 2.0])

  dist.sample([3])  # Shape [3, 2]
  ```

  Arguments are broadcast when possible.

  ```python
  dist = tfp.distributions.VonMises(loc=1.0, concentration=[0.5, 2.0])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  """

  def __init__(self,
               loc,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name='VonMises'):
    """Construct von Mises distributions with given location and concentration.

    The parameters `loc` and `concentration` must be shaped in a way that
    supports broadcasting (e.g. `loc + concentration` is a valid operation).

    Args:
      loc: Floating point tensor, the circular means of the distribution(s).
      concentration: Floating point tensor, the level of concentration of the
        distribution(s) around `loc`. Must take non-negative values.
        `concentration = 0` defines a Uniform distribution, while
        `concentration = +inf` indicates a Deterministic distribution at `loc`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if loc and concentration are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, concentration],
                                      dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, name='concentration', dtype=dtype)
      dtype_util.assert_same_float_dtype([self._loc, self._concentration])
      super(VonMises, self).__init__(
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
        loc=parameter_properties.ParameterProperties(),
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the circular mean (loc)."""
    return self._loc

  @property
  def concentration(self):
    """Distribution parameter for the concentration."""
    return self._concentration

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    log_normalization = np.log(2. * np.pi) + tf.math.log(
        tf.math.bessel_i0e(concentration))
    return (self._log_unnormalized_prob(x, loc=self.loc,
                                        concentration=concentration) -
            log_normalization)

  def _log_unnormalized_prob(self, x, loc, concentration):
    z = self._z(x, loc=loc)
    return concentration * (tf.cos(z) - 1)

  def _prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    unnormalized_prob = tf.exp(self._log_unnormalized_prob(
        x, loc=self.loc, concentration=concentration))
    normalization = (2. * np.pi) * tf.math.bessel_i0e(concentration)
    return unnormalized_prob / normalization

  def _cdf(self, x):
    loc = tf.convert_to_tensor(self.loc)
    concentration = tf.convert_to_tensor(self.concentration)
    batch_shape = ps.broadcast_shape(
        self._batch_shape_tensor(loc=loc, concentration=concentration),
        ps.shape(x))
    z = tf.broadcast_to(self._z(x, loc=loc), batch_shape)
    concentration = tf.broadcast_to(concentration, batch_shape)
    return von_mises_cdf(z, concentration)

  def _entropy(self):
    concentration = tf.convert_to_tensor(self.concentration)
    i0e = tf.math.bessel_i0e(concentration)
    i1e = tf.math.bessel_i1e(concentration)
    entropy = (
        concentration * (1 - i1e / i0e) + tf.math.log(i0e) +
        np.log(2 * np.pi))
    return tf.broadcast_to(entropy, self._batch_shape_tensor(
        concentration=concentration))

  @distribution_util.AppendDocstring(
      """Note: This function computes the circular mean, defined as
      atan2(E sin(x), E cos(x)). The values are in the [-pi, pi] range.""")
  def _mean(self):
    loc = tf.convert_to_tensor(self.loc)
    return tf.broadcast_to(loc, self._batch_shape_tensor(loc=loc))

  def _mode(self):
    return self._mean()

  @distribution_util.AppendDocstring(
      """Note: This function computes the circular variance defined as
      1 - E cos(x - loc). The values are in the [0, 1] range.""")
  def _variance(self):
    concentration = tf.convert_to_tensor(self.concentration)
    concentration = tf.broadcast_to(
        concentration, self._batch_shape_tensor(concentration=concentration))
    return 1. - tf.math.bessel_i1e(concentration) / tf.math.bessel_i0e(
        concentration)

  def _z(self, x, loc):
    """Standardize input `x` to a zero-loc von Mises."""
    with tf.name_scope('standardize'):
      return x - loc

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    concentration = tf.convert_to_tensor(self.concentration)
    concentration = tf.broadcast_to(
        concentration, self._batch_shape_tensor(
            loc=loc, concentration=concentration))

    # random_von_mises does not work for zero concentration, so round it up to
    # something very small.
    tiny = np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny
    concentration = tf.maximum(concentration, tiny)

    sample_batch_shape = ps.concat([
        [n], ps.shape(concentration)], axis=0)
    samples = random_von_mises(
        sample_batch_shape, concentration, dtype=self.dtype, seed=seed)

    # vonMises(0, concentration) -> vonMises(loc, concentration)
    samples = samples + loc
    # Map the samples to [-pi, pi].
    samples = samples - 2. * np.pi * tf.round(samples / (2. * np.pi))
    return samples

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    return sigmoid_bijector.Sigmoid(
        low=tf.constant(-np.pi, dtype=self.dtype),
        high=tf.constant(np.pi, dtype=self.dtype),
        validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_non_negative(
          self.concentration,
          message='Argument `concentration` must be non-negative.'))
    return assertions


@kullback_leibler.RegisterKL(VonMises, VonMises)
def _kl_von_mises_von_mises(d1, d2, name=None):
  """Batchwise KL divergence KL(d1 || d2) with d1 and d2 von Mises.

  Args:
    d1: instance of a von Mises distribution object.
    d2: instance of a a von Mises distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_von_mises_von_mises".

  Returns:
    Batchwise KL(d1 || d2)
  """
  with tf.name_scope(name or 'kl_von_mises_von_mises'):
    # The density of von Mises is (abbreviating the concentration for conc):
    #   vonMises(x; loc, conc) = exp(conc cos(x - loc)) / (2 pi I_0 (conc) )
    # We need two properties:
    # 1. Standardization: if z ~ vonMises(0, conc), then
    #    z + loc ~ vonMises(loc, conc).
    # 2. Expectation of cosine:
    #    E_q(z | 0, conc) cos z = I_1 (conc) / I_0 (conc)
    # Now,
    # KL(d1 || d2)
    #   = E_vonMises(x; loc1, conc1) log vonMises(x; loc1, conc1)
    #                                     / vonMises(x; loc2, conc2)
    # Plugging the densities and rearranging, we have
    #   log I_0(conc2) / I_0(conc1)
    #     + E_vonMises(x; loc1, conc1) [ conc1 cos (z - loc1)
    #                                     - conc2 cos (z - loc2) ]
    # Let's transform the second term using the standardization property:
    #   E_vonMises(x; 0, conc1) [conc1 cos z - conc2 cos (z - (loc2 - loc1))]
    # Applying the cos (x - y) = cos x cos y + sin x sin y expansion, we get
    #   E_vonMises(x; 0, conc1) [conc1 cos z - conc2 cos (loc2 - loc1) cos z
    #     - conc2 sin(loc2 - loc1) sin z]
    # Because the distribution is symmetric around zero, the last term vanishes
    # in expectation. The remaining two terms are computed using the
    # "expectation of cosine" property:
    #   (conc1 - conc2 cos (loc2 - loc1) E_vonMises(x; 0, conc1) cos z
    #     = (conc1 - conc2 cos (loc2 - loc1)) I_1(conc1) / I_0(conc1)
    # In total, we have
    #   KL(d1 || d2) = log I_0(conc2) / I_0(conc1)
    #     + (conc1 - conc2 cos (loc2 - loc1)) I_1(conc1) / I_0(conc1)
    # To improve the numerical stability, we can replace I_j(k) functions with
    # the exponentially scaled versions using the equality
    # I_j(k) = I_j^E(k) exp(k) (which holds for k >= 0):
    #   KL(d1 || d2) = (conc2 - conc1) + log I_0^E(conc2) / I_0^E(conc1)
    #     + (conc1 - conc2 cos (loc2 - loc1)) I_1^E(conc1) / I_0^E(conc1)
    # Note that this formula is numerically stable for conc1 = 0 and/or
    # conc2 = 0 because I_0 (0) = I_0^E (0) = 1.
    concentration1 = tf.convert_to_tensor(d1.concentration)
    concentration2 = tf.convert_to_tensor(d2.concentration)

    i0e_concentration1 = tf.math.bessel_i0e(concentration1)
    i1e_concentration1 = tf.math.bessel_i1e(concentration1)
    i0e_concentration2 = tf.math.bessel_i0e(concentration2)
    return ((concentration2 - concentration1) +
            tf.math.log(i0e_concentration2 / i0e_concentration1) +
            (concentration1 - concentration2 * tf.cos(d1.loc - d2.loc)) *
            (i1e_concentration1 / i0e_concentration1))


@tf.custom_gradient
def von_mises_cdf(x, concentration):
  """Computes the cumulative density function (CDF) of von Mises distribution.

  Denote the density of vonMises(loc=0, concentration=concentration) by p(t).
  Note that p(t) is periodic, p(t) = p(t + 2 pi).
  The CDF at the point x is defined as int_{-pi}^x p(t) dt.
  Thus, when x in [-pi, pi], the CDF is in [0, 1]; when x is in [pi, 3pi], the
  CDF is in [1, 2], etc.

  The CDF is not available in closed form. Instead, we use the method [1]
  which uses either a series expansion or a Normal approximation, depending on
  the value of concentration.

  We also compute the derivative of the CDF w.r.t. both x and concentration.
  The derivative w.r.t. x is p(x), while the derivative w.r.t. concentration is
  computed
  using automatic differentiation. We use forward mode for the series case
  (which allows to save memory) and backward mode for the Normal approximation.

  Args:
    x: The point at which to evaluate the CDF.
    concentration: The concentration parameter of the von Mises distribution.

  Returns:
    The value of the CDF computed elementwise.

  References:
    [1] G. Hill "Algorithm 518: Incomplete Bessel Function I_0. The Von Mises
    Distribution." ACM Transactions on Mathematical Software, 1977
  """
  x = tf.convert_to_tensor(x)
  concentration = tf.convert_to_tensor(concentration)
  dtype = x.dtype

  # Map x to [-pi, pi].
  num_periods = tf.round(x / (2. * np.pi))
  x = x - (2. * np.pi) * num_periods

  # We take the hyperparameters from Table I of [1], the row for D=8
  # decimal digits of accuracy. ck is the cut-off for concentration:
  # if concentration < ck,  the series expansion is used;
  # otherwise, the Normal approximation is used.
  ck = 10.5
  # The number of terms in the series expansion. [1] chooses it as a function
  # of concentration, n(concentration). This is hard to implement in TF.
  # Instead, we upper bound it over concentrations:
  #   num_terms = ceil ( max_{concentration <= ck} n(concentration) ).
  # The maximum is achieved for concentration = ck.
  num_terms = 20

  cdf_series, dcdf_dconcentration_series = _von_mises_cdf_series(
      x, concentration, num_terms, dtype)
  cdf_normal, dcdf_dconcentration_normal = _von_mises_cdf_normal(
      x, concentration, dtype)

  use_series = concentration < ck
  cdf = tf.where(use_series, cdf_series, cdf_normal)
  cdf = cdf + num_periods
  dcdf_dconcentration = tf.where(use_series, dcdf_dconcentration_series,
                                 dcdf_dconcentration_normal)

  def grad(dy):
    prob = tf.exp(concentration * (tf.cos(x) - 1.)) / (
        (2. * np.pi) * tf.math.bessel_i0e(concentration))
    return dy * prob, dy * dcdf_dconcentration

  return cdf, grad


def _von_mises_cdf_series(x, concentration, num_terms, dtype):
  """Computes the von Mises CDF and its derivative via series expansion."""
  # Keep the number of terms as a float. It should be a small integer, so
  # exactly representable as a float.
  num_terms = tf.cast(num_terms, dtype=dtype)

  def loop_body(n, rn, drn_dconcentration, vn, dvn_dconcentration):
    """One iteration of the series loop."""

    denominator = 2. * n / concentration + rn
    ddenominator_dk = -2. * n / concentration ** 2 + drn_dconcentration
    rn = 1. / denominator
    drn_dconcentration = -ddenominator_dk / denominator ** 2

    multiplier = tf.sin(n * x) / n + vn
    vn = rn * multiplier
    dvn_dconcentration = (drn_dconcentration * multiplier +
                          rn * dvn_dconcentration)
    n = n - 1.

    return n, rn, drn_dconcentration, vn, dvn_dconcentration

  (_, _, _, vn, dvn_dconcentration) = tf.while_loop(
      cond=lambda n, *_: n > 0.,
      body=loop_body,
      loop_vars=(
          num_terms,  # n
          tf.zeros_like(x, name='rn'),
          tf.zeros_like(x, name='drn_dconcentration'),
          tf.zeros_like(x, name='vn'),
          tf.zeros_like(x, name='dvn_dconcentration'),
      ),
  )

  cdf = .5 + x / (2. * np.pi) + vn / np.pi
  dcdf_dconcentration = dvn_dconcentration / np.pi

  # Clip the result to [0, 1].
  cdf_clipped = tf.clip_by_value(cdf, 0., 1.)
  # The clipped values do not depend on concentration anymore, so set their
  # derivative to zero.
  dcdf_dconcentration = (
      dcdf_dconcentration * tf.cast((cdf >= 0.) & (cdf <= 1.), dtype))

  return cdf_clipped, dcdf_dconcentration


def _von_mises_cdf_normal(x, concentration, dtype):
  """Computes the von Mises CDF and its derivative via Normal approximation."""

  def cdf_func(concentration):
    """A helper function that is passed to value_and_gradient."""
    # z is an "almost Normally distributed" random variable.
    z = (tf.constant(np.sqrt(2. / np.pi), dtype=dtype)
         / tf.math.bessel_i0e(concentration)) * tf.sin(.5 * x)

    # This is the correction described in [1] which reduces the error
    # of the Normal approximation.
    z2 = z ** 2
    z3 = z2 * z
    z4 = z2 ** 2
    c = 24. * concentration
    c1 = 56.

    xi = z - z3 / ((c - 2. * z2 - 16.) / 3. -
                   (z4 + (7. / 4.) * z2 + 167. / 2.) / (c - c1 - z2 + 3.)) ** 2

    distrib = normal.Normal(tf.cast(0., dtype), tf.cast(1., dtype))

    return distrib.cdf(xi)

  return value_and_gradient(cdf_func, concentration)


def _von_mises_sample_no_gradient(shape, concentration, seed):
  """Performs rejection sampling for standardized von Mises.

  Args:
    shape: The output sample shape.
    concentration: The concentration parameter of the distribution.
    seed: The random seed.

  Returns:
    samples: Samples of standardized von Mises.
  """
  r = 1. + tf.sqrt(1. + 4. * concentration ** 2)
  rho = (r - tf.sqrt(2. * r)) / (2. * concentration)

  s_exact = (1. + rho ** 2) / (2. * rho)

  # For low concentration, s becomes numerically unstable.
  # To fix that, we use an approximation. Here is the derivation.
  # First-order Taylor expansion at conc = 0 gives
  #   sqrt(1 + 4 concentration^2) ~= 1 + (2 concentration)^2 / 2.
  # Therefore, r ~= 2 + 2 concentration. By plugging this into rho, we have
  #   rho ~= conc + 1 / conc - sqrt(1 + 1 / concentration^2).
  # Let's expand the last term at concentration=0 up to the linear term:
  #   sqrt(1 + 1 / concentration^2) ~= 1 / concentration + concentration / 2
  # Thus, rho ~= concentration / 2. Finally,
  #   s = 1 / (2 rho) + rho / 2 ~= 1 / concentration + concentration / 4.
  # Since concentration is small, we drop the second term and simply use
  #   s ~= 1 / concentration.
  s_approximate = 1. / concentration

  # To compute the cutoff, we compute s_exact using mpmath with 30 decimal
  # digits precision and compare that to the s_exact and s_approximate
  # computed with dtype. Then, the cutoff is the largest concentration for
  # which abs(s_exact - s_exact_mpmath) > abs(s_approximate - s_exact_mpmath).
  s_concentration_cutoff_dict = {
      tf.float16: 1.8e-1,
      np.float16: 1.8e-1,
      np.finfo(np.float16).dtype: 1.8e-1,

      tf.float32: 2e-2,
      np.float32: 2e-2,
      np.finfo(np.float32).dtype: 2e-2,

      tf.float64: 1.2e-4,
      np.float64: 1.2e-4,
      np.finfo(np.float64).dtype: 1.2e-4,
  }
  s_concentration_cutoff = s_concentration_cutoff_dict[concentration.dtype]

  s = tf.where(concentration > s_concentration_cutoff, s_exact, s_approximate)

  def loop_body(done, u_in, w, seed):
    """Resample the non-accepted points."""
    # We resample u each time completely. Only its sign is used outside the
    # loop, which is random.
    u_seed, v_seed, next_seed = samplers.split_seed(seed, n=3)
    u = samplers.uniform(
        shape, minval=-1., maxval=1., dtype=concentration.dtype, seed=u_seed)
    tensorshape_util.set_shape(u, u_in.shape)
    z = tf.cos(np.pi * u)
    # Update the non-accepted points.
    w = tf.where(done, w, (1. + s * z) / (s + z))
    y = concentration * (s - w)

    v = samplers.uniform(
        shape, minval=0., maxval=1., dtype=concentration.dtype, seed=v_seed)
    accept = (y * (2. - y) >= v) | (tf.math.log(y / v) + 1. >= y)

    return done | accept, u, w, next_seed

  _, u, w, _ = tf.while_loop(
      cond=lambda done, *_: ~tf.reduce_all(done),
      body=loop_body,
      loop_vars=(
          tf.zeros(shape, dtype=tf.bool, name='done'),
          tf.zeros(shape, dtype=concentration.dtype, name='u'),
          tf.zeros(shape, dtype=concentration.dtype, name='w'),
          seed,
      ),
      # The expected number of iterations depends on concentration.
      # It monotonically increases from one iteration for concentration = 0 to
      # sqrt(2 pi / e) ~= 1.52 iterations for concentration = +inf [1].
      # We use a limit of 100 iterations to avoid infinite loops
      # for very large / nan concentration.
      maximum_iterations=100,
  )

  return tf.sign(u) * tf.math.acos(w)


def _von_mises_sample_fwd(shape, concentration, seed):
  """Compute output, aux (collaborates with _von_mises_sample_bwd)."""
  samples = _von_mises_sample_no_gradient(shape, concentration, seed)
  return samples, (concentration, samples)


def _von_mises_sample_bwd(_, aux, dy):
  """The gradient of the von Mises samples w.r.t. concentration."""
  concentration, samples = aux
  broadcast_concentration = tf.broadcast_to(concentration, ps.shape(samples))
  _, dcdf_dconcentration = value_and_gradient(
      lambda conc: von_mises_cdf(samples, conc), broadcast_concentration)
  inv_prob = tf.exp(-broadcast_concentration * (tf.cos(samples) - 1.)) * (
      (2. * np.pi) * tf.math.bessel_i0e(broadcast_concentration))
  # Compute the implicit reparameterization gradient [2],
  # dz/dconc = -(dF(z; conc) / dconc) / p(z; conc)
  ret = dy * (-dcdf_dconcentration * inv_prob)
  # Sum over the sample dimensions. Assume that they are always the first
  # ones.
  num_sample_dimensions = (tf.rank(broadcast_concentration) -
                           tf.rank(concentration))

  # None gradients for seed
  return tf.reduce_sum(ret, axis=tf.range(num_sample_dimensions)), None


def _von_mises_sample_jvp(shape, primals, tangents):
  """Compute primals and tangents using implicit derivative."""
  concentration, seed = primals
  dconcentration, dseed = tangents
  del dseed

  dconcentration = tf.broadcast_to(dconcentration, shape)
  broadcast_concentration = tf.broadcast_to(concentration, shape)

  samples = _von_mises_sample_no_gradient(shape, concentration, seed)

  _, dcdf_dconcentration = value_and_gradient(
      lambda conc: von_mises_cdf(samples, conc), broadcast_concentration)
  inv_prob = tf.exp(-concentration * (tf.cos(samples) - 1.)) * (
      (2. * np.pi) * tf.math.bessel_i0e(concentration))
  # Compute the implicit derivative,
  #   dz = dconc * -(dF(z; conc) / dconc) / p(z; conc)
  dsamples = dconcentration * (-dcdf_dconcentration * inv_prob)

  return samples, dsamples


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_von_mises_sample_fwd,
    vjp_bwd=_von_mises_sample_bwd,
    jvp_fn=_von_mises_sample_jvp,
    nondiff_argnums=(0,))
def _von_mises_sample_with_gradient(shape, concentration, seed):
  """Performs rejection sampling for standardized von Mises.

  Args:
    shape: The output sample shape.
    concentration: The concentration parameter of the distribution.
    seed: (optional) The random seed.

  Returns:
    sample: Differentiable samples of standardized von Mises.
  """
  return _von_mises_sample_no_gradient(shape, concentration, seed)


def random_von_mises(shape, concentration, dtype=tf.float32, seed=None):
  """Samples from the standardized von Mises distribution.

  The distribution is vonMises(loc=0, concentration=concentration), so the mean
  is zero.
  The location can then be changed by adding it to the samples.

  The sampling algorithm is rejection sampling with wrapped Cauchy proposal [1].
  The samples are pathwise differentiable using the approach of [2].

  Args:
    shape: The output sample shape.
    concentration: The concentration parameter of the von Mises distribution.
    dtype: The data type of concentration and the outputs.
    seed: (optional) The random seed.

  Returns:
    Differentiable samples of standardized von Mises.

  References:
    [1] Luc Devroye "Non-Uniform Random Variate Generation", Springer-Verlag,
    1986; Chapter 9, p. 473-476.
    http://www.nrbook.com/devroye/Devroye_files/chapter_nine.pdf
    + corrections http://www.nrbook.com/devroye/Devroye_files/errors.pdf
    [2] Michael Figurnov, Shakir Mohamed, Andriy Mnih. "Implicit
    Reparameterization Gradients", 2018.
  """
  shape = ps.convert_to_shape_tensor(shape, dtype_hint=tf.int32, name='shape')
  seed = samplers.sanitize_seed(seed, salt='von_mises')
  concentration = tf.convert_to_tensor(
      concentration, dtype=dtype, name='concentration')

  return _von_mises_sample_with_gradient(shape, concentration, seed)
