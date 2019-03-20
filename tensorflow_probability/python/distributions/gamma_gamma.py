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
"""The GammaGamma distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import seed_stream

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization

__all__ = [
    "GammaGamma",
]


class GammaGamma(distribution.Distribution):
  """Gamma-Gamma distribution.

  Gamma-Gamma is a [compound
  distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution)
  defined over positive real numbers using parameters `concentration`,
  `mixing_concentration` and `mixing_rate`.

  This distribution is also referred to as the beta of the second kind (B2), and
  can be useful for transaction value modeling, as [(Fader and Hardi, 2013)][1].

  #### Mathematical Details

  It is derived from the following Gamma-Gamma hierarchical model by integrating
  out the random variable `beta`.

  ```none
      beta ~ Gamma(alpha0, beta0)
  X | beta ~ Gamma(alpha, beta)
  ```
  where
  * `concentration = alpha`
  * `mixing_concentration = alpha0`
  * `mixing_rate = beta0`

  The probability density function (pdf) is

  ```none
                                         x**(alpha - 1)
  pdf(x; alpha, alpha0, beta0) = ---------------------------------
                                 Z * (x + beta0)**(alpha + alpha0)
  ```
  where the normalizing constant `Z = Beta(alpha, alpha0) * beta0**(-alpha0)`.

  Samples of this distribution are reparameterized as samples of the Gamma
  distribution are reparameterized using the technique described in
  [(Figurnov et al., 2018)][2].

  #### References

  [1]: Peter S. Fader, Bruce G. S. Hardi. The Gamma-Gamma Model of Monetary
       Value. _Technical Report_, 2013.
       http://www.brucehardie.com/notes/025/gamma_gamma.pdf

  [2]: Michael Figurnov, Shakir Mohamed, Andriy Mnih.
       Implicit Reparameterization Gradients. _arXiv preprint arXiv:1805.08498_,
       2018. https://arxiv.org/abs/1805.08498
  """

  def __init__(self,
               concentration,
               mixing_concentration,
               mixing_rate,
               validate_args=False,
               allow_nan_stats=True,
               name="GammaGamma"):
    """Initializes a batch of Gamma-Gamma distributions.

    The parameters `concentration` and `rate` must be shaped in a way that
    supports broadcasting (e.g.
    `concentration + mixing_concentration + mixing_rate` is a valid operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      mixing_concentration: Floating point tensor, the concentration params of
        the mixing Gamma distribution(s). Must contain only positive values.
      mixing_rate: Floating point tensor, the rate params of the mixing Gamma
        distribution(s). Must contain only positive values.
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
      TypeError: if `concentration` and `rate` are different dtypes.
    """
    parameters = dict(locals())
    with tf.compat.v1.name_scope(
        name, values=[concentration, mixing_concentration, mixing_rate]):
      dtype = dtype_util.common_dtype(
          [concentration, mixing_concentration, mixing_rate],
          preferred_dtype=tf.float32)
      concentration = tf.convert_to_tensor(
          value=concentration, name="concentration", dtype=dtype)
      mixing_concentration = tf.convert_to_tensor(
          value=mixing_concentration, name="mixing_concentration", dtype=dtype)
      mixing_rate = tf.convert_to_tensor(
          value=mixing_rate, name="mixing_rate", dtype=dtype)
      with tf.control_dependencies([
          tf.compat.v1.assert_positive(concentration),
          tf.compat.v1.assert_positive(mixing_concentration),
          tf.compat.v1.assert_positive(mixing_rate),
      ] if validate_args else []):
        self._concentration = tf.identity(concentration, name="concentration")
        self._mixing_concentration = tf.identity(
            mixing_concentration, name="mixing_concentration")
        self._mixing_rate = tf.identity(mixing_rate, name="mixing_rate")

      tf.debugging.assert_same_float_dtype(
          [self._concentration, self._mixing_concentration, self._mixing_rate])

    super(GammaGamma, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[
            self._concentration, self._mixing_concentration, self._mixing_rate
        ],
        name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(concentration=0, mixing_concentration=0, mixing_rate=0)

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def mixing_concentration(self):
    """Concentration parameter for the mixing Gamma distribution."""
    return self._mixing_concentration

  @property
  def mixing_rate(self):
    """Rate parameter for the mixing Gamma distribution."""
    return self._mixing_rate

  def _batch_shape_tensor(self):
    tensors = [self.concentration, self.mixing_concentration, self.mixing_rate]
    return functools.reduce(tf.broadcast_dynamic_shape,
                            [tf.shape(input=tensor) for tensor in tensors])

  def _batch_shape(self):
    tensors = [self.concentration, self.mixing_concentration, self.mixing_rate]
    return functools.reduce(tf.broadcast_static_shape,
                            [tensor.shape for tensor in tensors])

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(
      """Note: See `tf.random.gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    seed = seed_stream.SeedStream(seed, "gamma_gamma")
    rate = tf.random.gamma(
        shape=[n],
        # Be sure to draw enough rates for the fully-broadcasted gamma-gamma.
        alpha=self.mixing_concentration + tf.zeros_like(self.concentration),
        beta=self.mixing_rate,
        dtype=self.dtype,
        seed=seed())
    return tf.random.gamma(
        shape=[],
        alpha=self.concentration,
        beta=rate,
        dtype=self.dtype,
        seed=seed())

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _log_unnormalized_prob(self, x):
    x = self._maybe_assert_valid_sample(x)
    return (tf.math.xlogy(self.concentration - 1., x) -
            (self.concentration + self.mixing_concentration) *
            tf.math.log(x + self.mixing_rate))

  def _log_normalization(self):
    return (tf.math.lgamma(self.concentration) +
            tf.math.lgamma(self.mixing_concentration) -
            tf.math.lgamma(self.concentration + self.mixing_concentration) -
            self.mixing_concentration * tf.math.log(self.mixing_rate))

  @distribution_util.AppendDocstring(
      """The mean of a Gamma-Gamma distribution is
      `concentration * mixing_rate / (mixing_concentration - 1)`, when
      `mixing_concentration > 1`, and `NaN` otherwise. If `self.allow_nan_stats`
      is `False`, an exception will be raised rather than returning `NaN`""")
  def _mean(self):
    mean = self.concentration * self.mixing_rate / (
        self.mixing_concentration - 1.)
    if self.allow_nan_stats:
      nan = tf.fill(
          self.batch_shape_tensor(),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
          name="nan")
      return tf.where(self.mixing_concentration > 1., mean, nan)
    else:
      return distribution_util.with_dependencies([
          tf.compat.v1.assert_less(
              tf.ones([], self.dtype),
              self.mixing_concentration,
              message="mean undefined when `mixing_concentration` <= 1"),
      ], mean)

  @distribution_util.AppendDocstring(
      """The variance of a Gamma-Gamma distribution is
      `concentration**2 * mixing_rate**2 / ((mixing_concentration - 1)**2 *
      (mixing_concentration - 2))`, when `mixing_concentration > 2`, and `NaN`
      otherwise. If `self.allow_nan_stats` is `False`, an exception will be
      raised rather than returning `NaN`""")
  def _variance(self):
    variance = (tf.square(self.concentration * self.mixing_rate /
                          (self.mixing_concentration - 1.)) /
                (self.mixing_concentration - 2.))
    if self.allow_nan_stats:
      nan = tf.fill(
          self.batch_shape_tensor(),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
          name="nan")
      return tf.where(self.mixing_concentration > 2., variance, nan)
    else:
      return distribution_util.with_dependencies([
          tf.compat.v1.assert_less(
              tf.ones([], self.dtype) * 2.,
              self.mixing_concentration,
              message="variance undefined when `mixing_concentration` <= 2"),
      ], variance)

  def _maybe_assert_valid_sample(self, x):
    tf.debugging.assert_same_float_dtype(tensors=[x], dtype=self.dtype)
    if not self.validate_args:
      return x
    return distribution_util.with_dependencies([
        tf.compat.v1.assert_positive(x),
    ], x)
