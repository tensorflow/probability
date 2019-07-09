
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import special_math
from tensorflow.python.framework import tensor_shape
from tensorflow_probability.python.distributions.normal import Normal

__all__ = [
  "SkewGeneralizedNormal",
]


class SkewGeneralizedNormal(Normal):
  '''
  The skew-generalized normal distribution.
  Also known as the generalized Gaussian distribution of the second type.
  This implementation is based on the distribution
  described as the generalized normal version 2
  defined in the Wikipedia article:
  https://en.wikipedia.org/wiki/Generalized_normal_distribution
  accessed January 2019.

  Quantile, survival, log_survival, and all other essential functions
  that are not given in the Wikipedia article were defined by
  Daniel Luria, legally Daniel Maryanovsky, of vAIral, Kabbalah AI,
  and formerly of Lofty AI and Locbit Inc.

  Implemented by Daniel Luria

  The distribution returns NaN when evaluating
  probability of points outside its support
  '''
  def __init__(self,
      loc,
      scale,
      peak,
      validate_args = False,
      allow_nan_stats = True,
      name = "GeneralizedGaussian"):



   parameters =  parameters = dict(locals())


   with tf.name_scope(name, values=[loc, scale, peak]) as name:
    dtype = dtype_util.common_dtype([loc, scale], tf.float32)
    loc = tf.convert_to_tensor(loc, name="loc", dtype=dtype)
    scale = tf.convert_to_tensor(scale, name="scale", dtype=dtype)
    peak = tf.convert_to_tensor(peak, name="peak", dtype=dtype)

    with tf.control_dependencies([tf.assert_positive(scale)] if
            validate_args else []):
     self._loc = tf.identity(loc)
     self._scale = tf.identity(scale)
     self._peak = tf.identity(peak)

     tf.assert_same_float_dtype([self._loc, self._scale, self._peak])
   super(Normal, self).__init__(
    dtype=dtype,
    reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
    validate_args=validate_args,
    allow_nan_stats=allow_nan_stats,
    parameters=parameters,
    graph_parents=[self._loc, self._scale, self._peak],
    name=name)

  @staticmethod
  def _param_shapes(sample_shape):
   return dict(
    zip(("loc", "scale", "peak"), ([tf.convert_to_tensor(
     sample_shape, dtype=tf.int32)] * 2)))

  @property
  def loc(self):
   """Distribution parameter for the mean."""
   return self._loc

  @property
  def scale(self):
   """Distribution parameter for standard deviation."""
   return self._scale

  @property
  def peak(self):
   """Distribution parameter for mode."""
   return self._peak

  def _batch_shape_tensor(self):
   return tf.broadcast_dynamic_shape(
    tf.shape(self.loc),
    tf.shape(self.scale))

  def _batch_shape(self):
   assert tf.broadcast_static_shape(self.loc.shape, self.scale.shape) == tf.broadcast_static_shape(self.loc.shape, self.peak.shape) == tf.broadcast_static_shape(self.peak.shape, self.scale.shape)
   return tf.broadcast_static_shape(
    self.loc.shape,
    self.scale.shape)

#-
  def _event_shape_tensor(self):
   return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
   return tensor_shape.scalar()
#-
  def _sample_n(self, n, seed=None):
   shape = tf.concat([[n], self.batch_shape_tensor()], 0)
   sampled = tf.random_uniform(
    shape=shape, minval = 0., maxval=1., dtype=self.loc.dtype, seed=seed)
   return self._quantile(sampled)

  def _n_log_prob(self, x):
   return self._n_log_unnormalized_prob(x) - self._n_log_normalization()

  def _log_prob(self, x):
   return self._n_log_prob(self._y(x)) - tf.log(self.scale - (self.peak * (x - self.loc)))

  def _prob(self,x):
   nan_tensor = tf.exp(self._log_prob(x))
   return tf.where(tf.is_nan(nan_tensor), tf.zeros_like(nan_tensor), nan_tensor)

  def _log_cdf(self, x):
   return special_math.log_ndtr(self._y(x))

  def _y(self, x):
   return (-1./self.peak) * tf.log(1. - self.peak*(x - self.loc)/self.scale)

  def _cdf(self, x):
   return special_math.ndtr(self._y(x))

  def _log_survival_function(self, x):
   return special_math.log_ndtr(-self._y(x))

  def _survival_function(self, x):
   return special_math.ndtr(-self._y(x))

  def _n_log_unnormalized_prob(self, x):
   return -0.5 * tf.square(x)
  #
  def _n_log_normalization(self):
   return 0.5 * math.log(2. * math.pi) + 1.

  def _mean(self):
   return self._z((tf.exp(tf.square(self.peak)/2.) - 1.)/self.peak ) * tf.ones_like(self.scale)

  def _quantile(self, p):
   return self._inv_z((1. - tf.exp(-self.peak * special_math.ndtri(p)))/self.peak)

  def _stddev(self):
   return (self.scale/self.peak)*tf.exp(tf.square(self.peak)/2)*tf.sqrt(tf.exp(tf.square(self.peak)) - 1.) * tf.ones_like(self.loc)

  def _mode(self):
   return (((1. - tf.exp(-tf.square(self.peak)))*self.scale)/self.peak + self.loc) * tf.ones_like(self.scale)

  def _z(self, x):
   """Standardize input `x` to a unit normal."""
   with tf.name_scope("standardize", values=[x]):
    return (x - self.loc) / self.scale

  def _inv_z(self, z):
   """Reconstruct input `x` from a its normalized version."""
   with tf.name_scope("reconstruct", values=[z]):
    return z * self.scale + self.loc
