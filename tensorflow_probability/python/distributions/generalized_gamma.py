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
from tensorflow_probability.python.distributions import kullback_leibler

import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb


__all__ = [
  'GeneralizedGamma',
]

class GeneralizedGamma(distribution.Distribution):
  """Generalized Gamma distribution
  
  The Generalized Gamma generalizes the Gamma
  distribution with an additional exponent parameter. It is parameterized by
  location `loc`, scale `scale` and shape `power`.

  #### Mathematical details
  
  Following the wikipedia parameterization 
  https://en.wikipedia.org/wiki/Generalized_gamma_distribution
  f(x; a=scale, d=concentration, p=exponent) = 
    \frac{(p/a^d) x^{d-1} e^{-(x/a)^p}}{\Gamma(d/p)}
  """

  def __init__(self,
         scale, concentration, exponent,
         validate_args=False,
         allow_nan_stats=True,
         name='GeneralizedGamma'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
        [scale, concentration, exponent], dtype_hint=tf.float32)
      self._scale = tensor_util.convert_nonref_to_tensor(
        scale, dtype=dtype, name='scale')
      self._concentration = tensor_util.convert_nonref_to_tensor(
        concentration, dtype=dtype, name='concentration')
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
      tf.math.lgamma((self.concentration + 1.)/self.exponent)
      - tf.math.lgamma(self.concentration/self.exponent)
    )

  def _variance(self):
    return self._scale**2 * (
      tf.math.exp(
        tf.math.lgamma((self.concentration+2.)/self.exponent)
        - tf.math.lgamma(self.concentration/self.exponent)
      )
      - tf.math.exp(
        2*(
          tf.math.lgamma((self.concentration+1.)/self.exponent)
          - tf.math.lgamma(self.concentration/self.exponent)
        )

      )
    )

  def _cdf(self, x):
    return tf.math.igamma(self.concentration/self.exponent,
                (x/self.scale)**self.exponent) * tf.exp(
      -tf.math.lgamma(self.concentration/self.exponent)
    )

  def _log_prob(self, x, scale=None, concentration=None, exponent=None):
    scale = tensor_util.convert_nonref_to_tensor(
      self.scale if scale is None else scale)
    concentration = tensor_util.convert_nonref_to_tensor(
      self.concentration if concentration is None else concentration)
    exponent = tensor_util.convert_nonref_to_tensor(
      self.exponent if exponent is None else exponent)
    log_unnormalized_prob = (
      tf.math.xlogy(concentration-1., x) - (x/scale)**exponent)
    log_prefactor = (
      tf.math.log(exponent) - tf.math.xlogy(concentration, scale)
      - tf.math.lgamma(concentration/exponent))
    return log_unnormalized_prob + log_prefactor

  def _entropy(self):
    scale = tf.convert_to_tensor(self.scale)
    concentration = tf.convert_to_tensor(self.concentration)
    exponent = tf.convert_to_tensor(self.exponent)
    return (
      tf.math.log(scale) + tf.math.lgamma(concentration/exponent)
      - tf.math.log(exponent) + concentration/exponent
      + (1.0 - concentration)/exponent*tf.math.digamma(concentration/exponent)
    )

  def _stddev(self):
    return tf.math.sqrt(self._variance())

  def _mode(self):
    concentration = tf.convert_to_tensor(self.concentration)
    exponent = tf.convert_to_tensor(self.exponent)
    scale = tf.convert_to_tensor(self.scale)
    mode = scale*tf.math.pow(
      (concentration - 1.)/exponent,
      1./exponent
    )
    mode = tf.where(
      concentration > 1.,
      mode,
      tf.zeros_like(mode)
      )
    return mode
      
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
  def concentration(self):
    return self._concentration

  @property
  def exponent(self):
    return self._exponent

  def _batch_shape_tensor(self, scale=None, concentration=None, exponent=None):
    return prefer_static.broadcast_shape(
      prefer_static.shape(
        self.scale if scale is None else scale),
      prefer_static.shape(self.concentration if concentration is None else concentration),
      prefer_static.shape(
        self.exponent if exponent is None else exponent))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
      self.scale.shape,
      self.concentration.shape,
      )

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
      alpha=self.concentration/self.exponent,
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
    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_positive(
        self.concentration,
        message='Argument `concentration` must be positive.'))
    if is_init != tensor_util.is_ref(self.exponent):
      assertions.append(assert_util.assert_positive(
        self.exponent,
        message='Argument `exponent` must be positive.'))
    return assertions


@kullback_leibler.RegisterKL(GeneralizedGamma, GeneralizedGamma)
def _kl_ggamma_ggamma(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b GeneralizedGamma.
  \begin{align}
  D_{KL} (f_1 \parallel f_2) 
  & = \int_{0}^{\infty} f_1(x; a_1, d_1, p_1) \, \ln \frac{f_1(x; a_1, d_1, p_1)}{f_2(x; a_2, d_2, p_2)} \, dx\\
  & = \ln \frac{p_1 \, a_2^{d_2} \, \Gamma\left(d_2 / p_2\right)}{p_2 \, a_1^{d_1} \, \Gamma\left(d_1 /p_1\right)} 
      + \left[ \frac{\psi\left( d_1 / p_1 \right)}{p_1} + \ln a_1 \right]  (d_1 - d_2) 
      + \frac{\Gamma\bigl((d_1+p_2) / p_1 \bigr)}{\Gamma\left(d_1 / p_1\right)} \left( \frac{a_1}{a_2} \right)^{p_2} 
      - \frac{d_1}{p_1}
  \end{align}
  Args:
    a: instance of a GeneralizedGamma distribution object.
    b: instance of a GeneralizedGamma distribution object.
    name: (optional) Name to use for created operations. Default is
      '_kl_ggamma_ggamma'.

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or '_kl_ggamma_ggamma'):
    # Result from https://arxiv.org/pdf/1310.3713.pdf
    a_concentration = tf.convert_to_tensor(a.concentration)
    b_concentration = tf.convert_to_tensor(b.concentration)
    a_scale = tf.convert_to_tensor(a.scale)
    b_scale = tf.convert_to_tensor(b.scale)
    a_exponent = tf.convert_to_tensor(a.exponent)
    b_exponent = tf.convert_to_tensor(b.exponent)

    return (
      tf.math.log(a_exponent) - tf.math.log(b_exponent) 
      + b_concentration*tf.math.log(b_scale) - a_concentration*tf.math.log(a_scale)
      + tf.math.lgamma(b_concentration/b_exponent) - tf.math.lgamma(a_concentration/a_exponent)
      + (a_concentration - b_concentration)*(tf.math.digamma(a_concentration/a_exponent)/a_exponent + tf.math.log(a_scale))
      + tf.math.exp(tf.math.lgamma(
        (a_concentration + b_exponent)/a_exponent
        )-tf.math.lgamma(a_concentration/a_exponent) * tf.math.pow(a_scale/b_scale, b_exponent))
      - a_concentration/a_exponent
    )
