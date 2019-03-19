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
"""The Pareto distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization


class Pareto(distribution.Distribution):
  """Pareto distribution.

  The Pareto distribution is parameterized by a `scale` and a
  `concentration` parameter.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, scale, x >= scale) = alpha * scale ** alpha / x ** (alpha + 1)
  ```

  where `concentration = alpha`.

  Note that `scale` acts as a scaling parameter, since
  `Pareto(c, scale).pdf(x) == Pareto(c, 1.).pdf(x / scale)`.

  The support of the distribution is defined on `[scale, infinity)`.

  """

  def __init__(self,
               concentration,
               scale=1.,
               validate_args=False,
               allow_nan_stats=True,
               name="Pareto"):
    """Construct Pareto distribution with `concentration` and `scale`.

    Args:
      concentration: Floating point tensor. Must contain only positive values.
      scale: Floating point tensor, equivalent to `mode`. `scale` also
        restricts the domain of this distribution to be in `[scale, inf)`.
        Must contain only positive values. Default value: `1`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs. Default value: `False` (i.e. do not validate args).
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'Pareto'.
    """
    parameters = dict(locals())
    with tf.compat.v1.name_scope(name, values=[concentration, scale]):
      dtype = dtype_util.common_dtype([concentration, scale], tf.float32)
      self._concentration = tf.convert_to_tensor(
          value=concentration, name="concentration", dtype=dtype)
      self._scale = tf.convert_to_tensor(value=scale, name="scale", dtype=dtype)
      with tf.control_dependencies([
          tf.compat.v1.assert_positive(self._concentration),
          tf.compat.v1.assert_positive(self._scale)
      ] if validate_args else []):
        self._concentration = tf.identity(
            self._concentration, name="concentration")
        self._scale = tf.identity(self._scale, name="scale")
    super(Pareto, self).__init__(
        dtype=self._concentration.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._concentration, self._scale],
        name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(concentration=0, scale=0)

  @property
  def scale(self):
    """Scale parameter and also the lower bound of the support."""
    return self._scale

  @property
  def concentration(self):
    """Concentration parameter for this distribution."""
    return self._concentration

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(input=self.concentration), tf.shape(input=self.scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.concentration.shape, self.scale.shape)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    sampled = tf.random.uniform(shape, maxval=1., seed=seed, dtype=self.dtype)
    log_sample = tf.math.log(
        self.scale) - tf.math.log1p(-sampled) / self.concentration
    return tf.exp(log_sample)

  def _log_prob(self, x):
    with tf.control_dependencies([
        tf.compat.v1.assert_greater_equal(
            x,
            self.scale,
            message="x is not in the support of the distribution.")
    ] if self.validate_args else []):

      def log_prob_on_support(z):
        return (tf.math.log(self.concentration) +
                self.concentration * tf.math.log(self.scale) -
                (self.concentration + 1.) * tf.math.log(z))

      return self._extend_support(x, log_prob_on_support, alt=-np.inf)

  def _prob(self, x):
    with tf.control_dependencies([
        tf.compat.v1.assert_greater_equal(
            x,
            self.scale,
            message="x is not in the support of the distribution.")
    ] if self.validate_args else []):

      def prob_on_support(z):
        return (self.concentration * (self.scale ** self.concentration) /
                (z ** (self.concentration + 1)))
      return self._extend_support(x, prob_on_support, alt=0.)

  def _log_cdf(self, x):
    return self._extend_support(
        x,
        lambda x: tf.math.log1p(-(self.scale / x)**self.concentration),
        alt=-np.inf)

  def _cdf(self, x):
    return self._extend_support(
        x,
        lambda x: -tf.math.expm1(self.concentration * tf.math.log(self.scale / x
                                                                 )),
        alt=0.)

  def _log_survival_function(self, x):
    return self._extend_support(
        x,
        lambda x: self.concentration * tf.math.log(self.scale / x),
        alt=np.inf)

  @distribution_util.AppendDocstring(
      """The mean of Pareto is defined` if `concentration > 1.`, otherwise it
      is `Inf`.""")
  def _mean(self):
    broadcasted_concentration = self.concentration + tf.zeros_like(
        self.scale)
    infs = tf.fill(
        dims=tf.shape(input=broadcasted_concentration),
        value=np.array(np.inf, dtype=self.dtype.as_numpy_dtype))

    return tf.where(
        broadcasted_concentration > 1.,
        self.concentration * self.scale / (self.concentration - 1),
        infs)

  @distribution_util.AppendDocstring(
      """The variance of Pareto is defined` if `concentration > 2.`, otherwise
      it is `Inf`.""")
  def _variance(self):
    broadcasted_concentration = self.concentration + tf.zeros_like(self.scale)
    infs = tf.fill(
        dims=tf.shape(input=broadcasted_concentration),
        value=np.array(np.inf, dtype=self.dtype.as_numpy_dtype))
    return tf.where(
        broadcasted_concentration > 2.,
        self.scale ** 2 * self.concentration / (
            (self.concentration - 1.) ** 2 * (self.concentration - 2.)),
        infs)

  def _mode(self):
    return self.scale + tf.zeros_like(self.concentration)

  def _extend_support(self, x, f, alt):
    """Returns `f(x)` if x is in the support, and `alt` otherwise.

    Given `f` which is defined on the support of this distribution
    (e.g. x > scale), extend the function definition to the real line
    by defining `f(x) = alt` for `x < scale`.

    Args:
      x: Floating-point Tensor to evaluate `f` at.
      f: Lambda that takes in a tensor and returns a tensor. This represents
        the function who we want to extend the domain of definition.
      alt: Python or numpy literal representing the value to use for extending
        the domain.
    Returns:
      Tensor representing an extension of `f(x)`.
    """
    # We need to do a series of broadcasts for the tf.where.
    scale = self.scale + tf.zeros_like(self.concentration)
    is_invalid = x < scale
    scale = scale + tf.zeros_like(x)
    x = x + tf.zeros_like(scale)
    # We need to do this to ensure gradients are sound.
    y = f(tf.where(is_invalid, scale, x))
    if alt == 0.:
      alt = tf.zeros_like(y)
    elif alt == 1.:
      alt = tf.ones_like(y)
    else:
      alt = tf.fill(
          dims=tf.shape(input=y),
          value=np.array(alt, dtype=self.dtype.as_numpy_dtype))
    return tf.where(is_invalid, alt, y)


@kullback_leibler.RegisterKL(Pareto, Pareto)
def _kl_pareto_pareto(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Pareto.

  Args:
    a: instance of a Pareto distribution object.
    b: instance of a Pareto distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_pareto_pareto".

  Returns:
    Batchwise KL(a || b)
  """
  with tf.compat.v1.name_scope(
      name, "kl_pareto_pareto",
      [a.concentration, b.concentration, a.scale, b.scale]):
    # Consistent with
    # http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf, page 55
    # Terminology is different from source to source for Pareto distributions.
    # The 'concentration' parameter corresponds to 'a' in that source, and the
    # 'scale' parameter corresponds to 'm'.
    final_batch_shape = distribution_util.get_broadcast_shape(
        a.concentration, b.concentration, a.scale, b.scale)
    common_type = dtype_util.common_dtype(
        [a.concentration, b.concentration, a.scale, b.scale], tf.float32)
    return tf.where(
        a.scale >= b.scale,
        b.concentration * (tf.math.log(a.scale) - tf.math.log(b.scale)) +
        tf.math.log(a.concentration) - tf.math.log(b.concentration) +
        b.concentration / a.concentration - 1.0,
        tf.broadcast_to(tf.cast(np.inf, common_type), final_batch_shape))
