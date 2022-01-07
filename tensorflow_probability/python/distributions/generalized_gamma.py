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
"""The GeneralizedGamma distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import distribution

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.bijectors import softplus as softplus_bijector


__all__ = [
  'GeneralizedGamma',
]

tfd = tfp.distributions
tfb = tfp.bijectors


class GeneralizedGamma(distribution.Distribution):
  """Generalized Gamma distribution
  
  The Generalized Gamma generalizes the Gamma
  distribution with an additional exponent parameter. It is parameterized by
  location `loc`, scale `scale` and shape `power`.

  #### Mathematical details
  
  Following the wikipedia parameterization 
  https://en.wikipedia.org/wiki/Generalized_gamma_distribution
  f(x; a=scale, d=shape, p=exponent) = 
    \frac{(p/a^d) x^{d-1} e^{-(x/a)^p}}{\Gamma(d/p)}
  """

  def __init__(self,
         scale, shape, exponent,
         validate_args=False,
         allow_nan_stats=True,
         name='GeneralizedGamma'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
        [scale, shape, exponent], dtype_hint=tf.float32)
      self._scale = tensor_util.convert_nonref_to_tensor(
        scale, dtype=dtype, name='scale')
      self._shape = tensor_util.convert_nonref_to_tensor(
        shape, dtype=dtype, name='shape')
      self._exponent = tensor_util.convert_nonref_to_tensor(
        exponent, dtype=dtype, name='exponent')

      super(GeneralizedGamma, self).__init__(
        dtype=dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=(
          reparameterization.FULLY_REPARAMETERIZED
        ),
        parameters=parameters,
        name=name)

  def _mean(self):
    return self.scale * tf.math.exp(
      tf.math.lgamma((self.shape + 1.)/self.exponent)
      - tf.math.lgamma(self.shape/self.exponent)
    )

  def _variance(self):
    return self._scale**2 * (
      tf.math.exp(
        tf.math.lgamma((self.shape+2.)/self.exponent)
        - tf.math.lgamma(self.shape/self.exponent)
      )
      - tf.math.exp(
        2*(
          tf.math.lgamma((self.shape+1.)/self.exponent)
          - tf.math.lgamma(self.shape/self.exponent)
        )

      )
    )

  def _cdf(self, x):
    return tf.math.igamma(self.shape/self.exponent,
                (x/self.scale)**self.exponent) * tf.exp(
      -tf.math.lgamma(self.shape/self.exponent)
    )

  def _log_prob(self, x, scale=None, shape=None, exponent=None):
    scale = tensor_util.convert_nonref_to_tensor(
      self.scale if scale is None else scale)
    shape = tensor_util.convert_nonref_to_tensor(
      self.shape if shape is None else shape)
    exponent = tensor_util.convert_nonref_to_tensor(
      self.exponent if exponent is None else exponent)
    log_unnormalized_prob = (
      tf.math.xlogy(shape-1., x) - (x/scale)**exponent)
    log_prefactor = (
      tf.math.log(exponent) - tf.math.xlogy(shape, scale)
      - tf.math.lgamma(shape/exponent))
    return log_unnormalized_prob + log_prefactor

  def _entropy(self):
    scale = tf.convert_to_tensor(self.scale)
    shape = tf.convert_to_tensor(self.shape)
    exponent = tf.convert_to_tensor(self.exponent)
    return (
      tf.math.log(scale) + tf.math.lgamma(shape/exponent)
      - tf.math.log(exponent) + shape/exponent
      + (1.0 - shape)/exponent*tf.math.digamma(shape/exponent)
    )

  def _stddev(self):
    return tf.math.sqrt(self._variance())

  def _default_event_space_bijector(self):
    return softplus_bijector.Softplus(validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
      x, message='Sample must be non-negative.'))
    return assertions

  @property
  def scale(self):
    return self._scale

  @property
  def shape(self):
    return self._shape

  @property
  def exponent(self):
    return self._exponent

  def _batch_shape_tensor(self, scale=None, shape=None, exponent=None):
    return prefer_static.broadcast_shape(
      prefer_static.shape(
        self.scale if scale is None else scale),
      prefer_static.shape(self.shape if shape is None else shape),
      prefer_static.shape(
        self.exponent if exponent is None else exponent))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
      self.scale.shape,
      self.shape.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _sample_n(self, n, seed=None):
    """Sample based on transforming Gamma RVs
    Arguments:
      n {int} -- [description]
    Keyword Arguments:
      seed {int} -- [description] (default: {None})
    Returns:
      [type] -- [description]
    """
    gamma_samples = tf.random.gamma(
      shape=[n],
      alpha=self.shape/self.exponent,
      beta=1.,
      dtype=self.dtype,
      seed=seed
    )
    ggamma_samples = (
      self.scale*tf.math.exp(tf.math.log(gamma_samples)/self.exponent)
    )
    return ggamma_samples

  def _event_shape(self):
    return tf.TensorShape([])

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
        self.scale,
        message='Argument `scale` must be positive.'))
    if is_init != tensor_util.is_ref(self.shape):
      assertions.append(assert_util.assert_positive(
        self.shape,
        message='Argument `shape` must be positive.'))
    if is_init != tensor_util.is_ref(self.exponent):
      assertions.append(assert_util.assert_positive(
        self.exponent,
        message='Argument `exponent` must be positive.'))
    return assertions

if __name__ == "__main__":
  test=GeneralizedGamma(1,1,1)
  b=test.sample(10)
  c=test.mean()
  pass