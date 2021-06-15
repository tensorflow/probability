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
"""The InverseGaussian distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import scale as scale_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'InverseGaussian',
]


class InverseGaussian(distribution.AutoCompositeTensorDistribution):
  """Inverse Gaussian distribution.

  The [inverse Gaussian distribution]
  (https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)
  is parameterized by a `loc` and a `concentration` parameter. It's also known
  as the Wald distribution. Some, e.g., the Python scipy package, refer to the
  special case when `loc` is 1 as the Wald distribution.

  The "inverse" in the name does not refer to the distribution associated to
  the multiplicative inverse of a random variable. Rather, the cumulant
  generating function of this distribution is the inverse to that of a Gaussian
  random variable.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; mu, lambda) = [lambda / (2 pi x ** 3)] ** 0.5
                       exp{-lambda(x - mu) ** 2 / (2 mu ** 2 x)}
  ```

  where
  * `loc = mu`
  * `concentration = lambda`.

  The support of the distribution is defined on `(0, infinity)`.

  Mapping to R and Python scipy's parameterization:
  * R: statmod::invgauss
     - mean = loc
     - shape = concentration
     - dispersion = 1 / concentration. Used only if shape is NULL.
  * Python: scipy.stats.invgauss
     - mu = loc / concentration
     - scale = concentration
  """

  def __init__(self,
               loc,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name='InverseGaussian'):
    """Constructs inverse Gaussian distribution with `loc` and `concentration`.

    Args:
      loc: Floating-point `Tensor`, the loc params. Must contain only positive
        values.
      concentration: Floating-point `Tensor`, the concentration params.
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False` (i.e. do not validate args).
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'InverseGaussian'.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype([loc, concentration],
                                      dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')

      super(InverseGaussian, self).__init__(
          dtype=self._loc.dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Location parameter."""
    return self._loc

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    seed = samplers.sanitize_seed(seed, salt='inverse_gaussian')

    loc = tf.convert_to_tensor(self.loc)
    concentration = tf.convert_to_tensor(self.concentration)
    total_shape = ps.concat(
        [ps.convert_to_shape_tensor([n]),
         self._batch_shape_tensor(loc=loc, concentration=concentration)],
        axis=0)
    return _random_inverse_gaussian_gradient(
        total_shape, loc, concentration, seed)

  def _log_prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    loc = tf.convert_to_tensor(self.loc)
    return (0.5 * (tf.math.log(concentration) - np.log(2. * np.pi) -
                   3. * tf.math.log(x)) +
            (-concentration * tf.math.squared_difference(x, loc)) /
            (2. * tf.square(loc) * x))

  def _cdf(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    loc = tf.convert_to_tensor(self.loc)
    return (
        special_math.ndtr((tf.math.rsqrt(x / concentration) * (x / loc - 1.))) +
        tf.exp(2. * concentration / loc) *
        special_math.ndtr(-tf.math.rsqrt(x / concentration) * (x / loc + 1)))

  @distribution_util.AppendDocstring(
      """The mean of inverse Gaussian is the `loc` parameter.""")
  def _mean(self):
    # Shape is broadcasted with + tf.zeros_like().
    return self.loc + tf.zeros_like(self.concentration)

  @distribution_util.AppendDocstring(
      """The variance of inverse Gaussian is `loc` ** 3 / `concentration`.""")
  def _variance(self):
    return self.loc ** 3 / self.concentration

  def _default_event_space_bijector(self):
    return chain_bijector.Chain([
        softplus_bijector.Softplus(validate_args=self.validate_args),
        scale_bijector.Scale(scale=-1., validate_args=self.validate_args),
        exp_bijector.Log(validate_args=self.validate_args),
        softplus_bijector.Softplus(validate_args=self.validate_args)
    ], validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_positive(
          self.concentration,
          message='Argument `concentration` must be positive.'))
    if is_init != tensor_util.is_ref(self.loc):
      assertions.append(assert_util.assert_positive(
          self.loc,
          message='Argument `loc` must be positive.'))
    return assertions


def _random_inverse_gaussian_no_gradient(shape, loc, concentration, seed):
  """Sample from Inverse Gaussian distribution."""
  # See https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution or
  # https://www.jstor.org/stable/2683801
  dtype = dtype_util.common_dtype([loc, concentration], tf.float32)
  concentration = tf.convert_to_tensor(concentration)
  loc = tf.convert_to_tensor(loc)
  chi2_seed, unif_seed = samplers.split_seed(seed, salt='inverse_gaussian')
  sampled_chi2 = tf.square(samplers.normal(shape, seed=chi2_seed, dtype=dtype))
  sampled_uniform = samplers.uniform(shape, seed=unif_seed, dtype=dtype)
  # Wikipedia defines an intermediate x with the formula
  #   x = loc + loc ** 2 * y / (2 * conc)
  #       - loc / (2 * conc) * sqrt(4 * loc * conc * y + loc ** 2 * y ** 2)
  # where y ~ N(0, 1)**2 (sampled_chi2 above) and conc is the concentration.
  # Let us write
  #   w = loc * y / (2 * conc)
  # Then we can extract the common factor in the last two terms to obtain
  #   x = loc + loc * w * (1 - sqrt(2 / w + 1))
  # Now we see that the Wikipedia formula suffers from catastrphic
  # cancellation for large w (e.g., if conc << loc).
  #
  # Fortunately, we can fix this by multiplying both sides
  # by 1 + sqrt(2 / w + 1).  We get
  #   x * (1 + sqrt(2 / w + 1)) =
  #     = loc * (1 + sqrt(2 / w + 1)) + loc * w * (1 - (2 / w + 1))
  #     = loc * (sqrt(2 / w + 1) - 1)
  # The term sqrt(2 / w + 1) + 1 no longer presents numerical
  # difficulties for large w, and sqrt(2 / w + 1) - 1 is just
  # sqrt1pm1(2 / w), which we know how to compute accurately.
  # This just leaves the matter of small w, where 2 / w may
  # overflow.  In the limit a w -> 0, x -> loc, so we just mask
  # that case.
  sqrt1pm1_arg = 4 * concentration / (loc * sampled_chi2)  # 2 / w above
  safe_sqrt1pm1_arg = tf.where(sqrt1pm1_arg < np.inf, sqrt1pm1_arg, 1.0)
  denominator = 1.0 + tf.sqrt(safe_sqrt1pm1_arg + 1.0)
  ratio = tfp_math.sqrt1pm1(safe_sqrt1pm1_arg) / denominator
  sampled = loc * tf.where(sqrt1pm1_arg < np.inf, ratio, 1.0)  # x above
  return tf.where(sampled_uniform <= loc / (loc + sampled),
                  sampled, tf.square(loc) / sampled)


def _random_inverse_gaussian_fwd(shape, loc, concentration, seed):
  """Compute output, aux (collaborates with _random_inverse_gaussian_bwd)."""
  samples = _random_inverse_gaussian_no_gradient(
      shape, loc, concentration, seed)
  return samples, (samples, loc, concentration)


def _compute_partials(samples, loc, concentration):
  """Compute the Implicit Gradients for the samples."""
  # The implicit gradient here is derived in Appendix H.1 of [1].
  # Note that [1] uses the parametermization (1 / loc, concentration), and the
  # formula below are modified accordingly.
  # References
  # [1] W. Lin, M. Schmidt, M. Khan. Handling the Positive-Definite Constraint
  # in the Bayesian Learning Rule. https://arxiv.org/abs/2002.10060v8
  dtype = dtype_util.common_dtype([samples, loc, concentration], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  norm = normal.Normal(numpy_dtype(0.), numpy_dtype(1.))
  z = -tf.math.sqrt(concentration / samples) * (samples / loc + 1.)
  log_mills_ratio = norm.log_cdf(z) - norm.log_prob(z)
  partial_c = (
      numpy_dtype(np.log(2.)) - tf.math.log(loc) -
      0.5 * tf.math.log(concentration) + 1.5 * tf.math.log(samples) +
      log_mills_ratio)
  partial_c = samples / concentration - tf.math.exp(partial_c)

  partial_l = (
      numpy_dtype(np.log(2.)) + 0.5 * tf.math.log(concentration) +
      1.5 * tf.math.log(samples) + log_mills_ratio)
  # We divide by -loc**2, since we need to take in to account that [1] is
  # parameterized by 1 / loc (hence by the chain rule we incur a - 1 / loc**2
  # factor, where the negative sign is cancelled out.
  partial_l = tf.math.exp(partial_l) / tf.math.square(loc)
  return partial_c, partial_l


def _random_inverse_gaussian_bwd(shape, aux, g):
  """The gradient of the inverse gaussian samples."""
  samples, loc, concentration = aux
  partial_concentration, partial_loc = _compute_partials(
      samples, loc, concentration)
  dsamples = g

  # These will need to be shifted by the extra dimensions added from
  # `sample_shape`.
  reduce_dims = tf.range(
      tf.size(shape) - tf.maximum(tf.rank(concentration), tf.rank(loc)))
  grad_concentration = tf.math.reduce_sum(
      dsamples * partial_concentration, axis=reduce_dims)
  grad_loc = tf.math.reduce_sum(dsamples * partial_loc, axis=reduce_dims)

  if (tensorshape_util.is_fully_defined(concentration.shape) and
      tensorshape_util.is_fully_defined(loc.shape) and
      concentration.shape == loc.shape):
    return grad_concentration, grad_loc, None  # seed=None

  ax_loc, ax_conc = tf.raw_ops.BroadcastGradientArgs(
      s0=ps.shape(loc), s1=ps.shape(concentration))
  grad_concentration = tf.reshape(
      tf.math.reduce_sum(grad_concentration, axis=ax_conc),
      ps.shape(concentration))
  grad_loc = tf.reshape(
      tf.math.reduce_sum(grad_loc, axis=ax_loc), ps.shape(loc))

  return grad_loc, grad_concentration, None  # seed=None


def _random_inverse_gaussian_jvp(shape, primals, tangents):
  """Computes JVP for inverse_gaussian sample (supports JAX custom derivative)."""
  loc, concentration, seed = primals
  dloc, dconcentration, dseed = tangents
  del dseed
  # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
  dconcentration = tf.broadcast_to(dconcentration, shape)
  dloc = tf.broadcast_to(dloc, shape)

  samples = _random_inverse_gaussian_no_gradient(
      shape, loc, concentration, seed)

  partial_concentration, partial_loc = _compute_partials(
      samples, loc, concentration)

  dsamples = (partial_concentration * dconcentration + partial_loc * dloc)
  return samples, dsamples


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_random_inverse_gaussian_fwd,
    vjp_bwd=_random_inverse_gaussian_bwd,
    jvp_fn=_random_inverse_gaussian_jvp,
    nondiff_argnums=(0,))
def _random_inverse_gaussian_gradient(
    shape, loc, concentration, seed):
  return _random_inverse_gaussian_no_gradient(shape, loc, concentration, seed)

