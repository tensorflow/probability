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

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import seed_stream

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.distributions import util as distribution_util

__all__ = [
    "GammaGamma",
]


def _static_broadcast_shape_from_tensors(*tensors):
  shape = tensors[0].get_shape()
  for t in tensors[1:]:
    shape = tf.broadcast_static_shape(shape, t.get_shape())
  return shape


def _dynamic_broadcast_shape_from_tensors(*tensors):
  shape = tf.shape(tensors[0])
  for t in tensors[1:]:
    shape = tf.broadcast_dynamic_shape(shape, tf.shape(t))
  return shape


class GammaGamma(tf.distributions.Distribution):
  """Gamma-Gamma distribution.

  Gamma-Gamma is a [compound
  distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution)
  defined over positive real numbers using parameters `concentration`,
  `mixing_concentration` and `mixing_rate`.

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

  See:
    http://www.brucehardie.com/notes/025/gamma_gamma.pdf

  Samples of this distribution are reparameterized as samples of the Gamma
  distribution are reparameterized using the technique described in the paper

  [Michael Figurnov, Shakir Mohamed, Andriy Mnih.
  Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)
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
    with tf.name_scope(
        name, values=[concentration, mixing_concentration, mixing_rate]):
      with tf.control_dependencies([
          tf.assert_positive(concentration),
          tf.assert_positive(mixing_concentration),
          tf.assert_positive(mixing_rate),
      ] if validate_args else []):
        self._concentration = tf.convert_to_tensor(
            concentration, name="concentration")
        self._mixing_concentration = tf.convert_to_tensor(
            mixing_concentration, name="mixing_concentration")
        self._mixing_rate = tf.convert_to_tensor(
            mixing_rate, name="mixing_rate")

        tf.assert_same_float_dtype([
            self._concentration, self._mixing_concentration, self._mixing_rate
        ])

    super(GammaGamma, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=tf.distributions.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[
            self._concentration, self._mixing_concentration, self._mixing_rate
        ],
        name=name)

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
    return _dynamic_broadcast_shape_from_tensors(
        self.concentration, self.mixing_concentration, self.mixing_rate)

  def _batch_shape(self):
    return _static_broadcast_shape_from_tensors(
        self.concentration, self.mixing_concentration, self.mixing_rate)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  @distribution_util.AppendDocstring(
      """Note: See `tf.random_gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    seed = seed_stream.SeedStream(seed, "gamma_gamma")
    rate = tf.random_gamma(
        shape=[n],
        alpha=self.mixing_concentration,
        beta=self.mixing_rate,
        dtype=self.dtype,
        seed=seed())
    return tf.random_gamma(
        shape=[],
        alpha=self.concentration,
        beta=rate,
        dtype=self.dtype,
        seed=seed())

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _log_unnormalized_prob(self, x):
    x = self._maybe_assert_valid_sample(x)
    return ((self.concentration - 1.) * tf.log(x) -
            (self.concentration + self.mixing_concentration) *
            tf.log(x + self.mixing_rate))

  def _log_normalization(self):
    return (tf.lgamma(self.concentration) + tf.lgamma(self.mixing_concentration)
            - tf.lgamma(self.concentration + self.mixing_concentration) -
            self.mixing_concentration * tf.log(self.mixing_rate))

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
      return control_flow_ops.with_dependencies([
          tf.assert_less(
              tf.ones([], self.dtype),
              self.mixing_concentration,
              message="mean undefined when any `mixing_concentration` <= 1"),
      ], mean)

  def _maybe_assert_valid_sample(self, x):
    tf.assert_same_float_dtype(tensors=[x], dtype=self.dtype)
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        tf.assert_positive(x),
    ], x)
