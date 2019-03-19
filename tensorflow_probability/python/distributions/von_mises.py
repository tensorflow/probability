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
import tensorflow as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions.seed_stream import SeedStream
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.math.gradient import value_and_gradient


class VonMises(distribution.Distribution):
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
               name="VonMises"):
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
    with tf.compat.v1.name_scope(name, values=[loc, concentration]) as name:
      dtype = dtype_util.common_dtype([loc, concentration],
                                      preferred_dtype=tf.float32)
      loc = tf.convert_to_tensor(value=loc, name="loc", dtype=dtype)
      concentration = tf.convert_to_tensor(
          value=concentration, name="concentration", dtype=dtype)
      with tf.control_dependencies(
          [tf.compat.v1.assert_non_negative(concentration
                                           )] if validate_args else []):
        self._loc = tf.identity(loc, name="loc")
        self._concentration = tf.identity(concentration, name="concentration")
        tf.debugging.assert_same_float_dtype([self._loc, self._concentration])
    super(VonMises, self).__init__(
        dtype=self._concentration.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._loc, self._concentration],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("loc", "concentration"),
            ([tf.convert_to_tensor(value=sample_shape, dtype=tf.int32)] * 2)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, concentration=0)

  @property
  def loc(self):
    """Distribution parameter for the circular mean (loc)."""
    return self._loc

  @property
  def concentration(self):
    """Distribution parameter for the concentration."""
    return self._concentration

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(input=self.loc), tf.shape(input=self.concentration))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.loc.shape, self.concentration.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _log_unnormalized_prob(self, x):
    z = self._z(x)
    return self.concentration * (tf.cos(z) - 1)

  def _log_normalization(self):
    return np.log(2. * np.pi) + tf.math.log(
        tf.math.bessel_i0e(self.concentration))

  def _prob(self, x):
    unnormalized_prob = tf.exp(self._log_unnormalized_prob(x))
    normalization = (2. * np.pi) * tf.math.bessel_i0e(self.concentration)
    return unnormalized_prob / normalization

  def _cdf(self, x):
    z = self._z(x) + tf.zeros_like(self.concentration)
    concentration = self.concentration + tf.zeros_like(z)
    return von_mises_cdf(z, concentration)

  def _entropy(self):
    i0e = tf.math.bessel_i0e(self.concentration)
    i1e = tf.math.bessel_i1e(self.concentration)
    entropy = (
        self.concentration * (1 - i1e / i0e) + tf.math.log(i0e) +
        np.log(2 * np.pi))
    return entropy + tf.zeros_like(self.loc)

  @distribution_util.AppendDocstring(
      """Note: This function computes the circular mean, defined as
      atan2(E sin(x), E cos(x)). The values are in the [-pi, pi] range.""")
  def _mean(self):
    return self.loc + tf.zeros_like(self.concentration)

  def _mode(self):
    return self.mean()

  @distribution_util.AppendDocstring(
      """Note: This function computes the circular variance defined as
      1 - E cos(x - loc). The values are in the [0, 1] range.""")
  def _variance(self):
    concentration = self.concentration + tf.zeros_like(self.loc)
    return 1. - tf.math.bessel_i1e(concentration) / tf.math.bessel_i0e(
        concentration)

  def _z(self, x):
    """Standardize input `x` to a zero-loc von Mises."""
    with tf.compat.v1.name_scope("standardize", values=[x]):
      return x - self.loc

  def _sample_n(self, n, seed=None):
    # random_von_mises does not work for zero concentration, so round it up to
    # something very small.
    tiny = np.finfo(self.dtype.as_numpy_dtype).tiny
    concentration = tf.maximum(self.concentration, tiny)

    sample_batch_shape = tf.concat([[n], self._batch_shape_tensor()], axis=0)
    samples = random_von_mises(
        sample_batch_shape, concentration, dtype=self.dtype, seed=seed)

    # vonMises(0, concentration) -> vonMises(loc, concentration)
    samples += self.loc
    # Map the samples to [-pi, pi].
    samples -= 2. * np.pi * tf.round(samples / (2. * np.pi))
    return samples


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
  with tf.compat.v1.name_scope(
      name,
      "kl_von_mises_von_mises",
      values=[d1.loc, d1.concentration, d2.loc, d2.concentration]):
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
    i0e_concentration1 = tf.math.bessel_i0e(d1.concentration)
    i1e_concentration1 = tf.math.bessel_i1e(d1.concentration)
    i0e_concentration2 = tf.math.bessel_i0e(d2.concentration)
    return ((d2.concentration - d1.concentration) +
            tf.math.log(i0e_concentration2 / i0e_concentration1) +
            (d1.concentration - d2.concentration * tf.cos(d1.loc - d2.loc)) *
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

  Arguments:
    x: The point at which to evaluate the CDF.
    concentration: The concentration parameter of the von Mises distribution.

  Returns:
    The value of the CDF computed elementwise.

  References:
    [1] G. Hill "Algorithm 518: Incomplete Bessel Function I_0. The Von Mises
    Distribution." ACM Transactions on Mathematical Software, 1977
  """
  x = tf.convert_to_tensor(value=x)
  concentration = tf.convert_to_tensor(value=concentration)
  dtype = x.dtype

  # Map x to [-pi, pi].
  num_periods = tf.round(x / (2. * np.pi))
  x -= (2. * np.pi) * num_periods

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
  cdf += num_periods
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
    n -= 1.

    return n, rn, drn_dconcentration, vn, dvn_dconcentration

  (_, _, _, vn, dvn_dconcentration) = tf.while_loop(
      cond=lambda n, *_: n > 0.,
      body=loop_body,
      loop_vars=(
          num_terms,  # n
          tf.zeros_like(x, name="rn"),
          tf.zeros_like(x, name="drn_dconcentration"),
          tf.zeros_like(x, name="vn"),
          tf.zeros_like(x, name="dvn_dconcentration"),
      ),
  )

  cdf = .5 + x / (2. * np.pi) + vn / np.pi
  dcdf_dconcentration = dvn_dconcentration / np.pi

  # Clip the result to [0, 1].
  cdf_clipped = tf.clip_by_value(cdf, 0., 1.)
  # The clipped values do not depend on concentration anymore, so set their
  # derivative to zero.
  dcdf_dconcentration *= tf.cast((cdf >= 0.) & (cdf <= 1.), dtype)

  return cdf_clipped, dcdf_dconcentration


def _von_mises_cdf_normal(x, concentration, dtype):
  """Computes the von Mises CDF and its derivative via Normal approximation."""

  def cdf_func(concentration):
    """A helper function that is passed to value_and_gradient."""
    # z is an "almost Normally distributed" random variable.
    z = ((np.sqrt(2. / np.pi) / tf.math.bessel_i0e(concentration)) *
         tf.sin(.5 * x))

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


def random_von_mises(shape, concentration, dtype=tf.float32, seed=None):
  """Samples from the standardized von Mises distribution.

  The distribution is vonMises(loc=0, concentration=concentration), so the mean
  is zero.
  The location can then be changed by adding it to the samples.

  The sampling algorithm is rejection sampling with wrapped Cauchy proposal [1].
  The samples are pathwise differentiable using the approach of [2].

  Arguments:
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
  seed = SeedStream(seed, salt="von_mises")
  concentration = tf.convert_to_tensor(
      value=concentration, dtype=dtype, name="concentration")

  @tf.custom_gradient
  def rejection_sample_with_gradient(concentration):
    """Performs rejection sampling for standardized von Mises.

    A nested function is required because @tf.custom_gradient does not handle
    non-tensor inputs such as dtype. Instead, they are captured by the outer
    scope.

    Arguments:
      concentration: The concentration parameter of the distribution.

    Returns:
      Differentiable samples of standardized von Mises.
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
        tf.float32: 2e-2,
        tf.float64: 1.2e-4,
    }
    s_concentration_cutoff = s_concentration_cutoff_dict[dtype]

    s = tf.where(concentration > s_concentration_cutoff, s_exact, s_approximate)

    def loop_body(done, u, w):
      """Resample the non-accepted points."""
      # We resample u each time completely. Only its sign is used outside the
      # loop, which is random.
      u = tf.random.uniform(
          shape, minval=-1., maxval=1., dtype=dtype, seed=seed())
      z = tf.cos(np.pi * u)
      # Update the non-accepted points.
      w = tf.where(done, w, (1. + s * z) / (s + z))
      y = concentration * (s - w)

      v = tf.random.uniform(
          shape, minval=0., maxval=1., dtype=dtype, seed=seed())
      accept = (y * (2. - y) >= v) | (tf.math.log(y / v) + 1. >= y)

      return done | accept, u, w

    _, u, w = tf.while_loop(
        cond=lambda done, *_: ~tf.reduce_all(input_tensor=done),
        body=loop_body,
        loop_vars=(
            tf.zeros(shape, dtype=tf.bool, name="done"),
            tf.zeros(shape, dtype=dtype, name="u"),
            tf.zeros(shape, dtype=dtype, name="w"),
        ),
        # The expected number of iterations depends on concentration.
        # It monotonically increases from one iteration for concentration = 0 to
        # sqrt(2 pi / e) ~= 1.52 iterations for concentration = +inf [1].
        # We use a limit of 100 iterations to avoid infinite loops
        # for very large / nan concentration.
        maximum_iterations=100,
        parallel_iterations=1 if seed.original_seed is None else 10,
    )

    x = tf.sign(u) * tf.math.acos(w)

    def grad(dy):
      """The gradient of the von Mises samples w.r.t. concentration."""
      broadcast_concentration = concentration + tf.zeros_like(x)
      _, dcdf_dconcentration = value_and_gradient(
          lambda conc: von_mises_cdf(x, conc), broadcast_concentration)
      inv_prob = tf.exp(-broadcast_concentration * (tf.cos(x) - 1.)) * (
          (2. * np.pi) * tf.math.bessel_i0e(broadcast_concentration))
      # Compute the implicit reparameterization gradient [2],
      # dz/dconc = -(dF(z; conc) / dconc) / p(z; conc)
      ret = dy * (-inv_prob * dcdf_dconcentration)
      # Sum over the sample dimensions. Assume that they are always the first
      # ones.
      num_sample_dimensions = (tf.rank(broadcast_concentration) -
                               tf.rank(concentration))
      return tf.reduce_sum(
          input_tensor=ret, axis=tf.range(num_sample_dimensions))

    return x, grad

  return rejection_sample_with_gradient(concentration)
