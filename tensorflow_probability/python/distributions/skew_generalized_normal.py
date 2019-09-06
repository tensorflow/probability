
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import special_math
from tensorflow.python.framework import tensor_shape
from tensorflow_probability.python.distributions.distribution import Distribution
from tensorflow_probability.python.distributions.normal import Normal

__all__ = [
    "SkewGeneralizedNormal",
]


class SkewGeneralizedNormal(Normal, Distribution):
  '''
  The skew-generalized normal distribution.
  Also known as the generalized Gaussian distribution of the second type.
  This implementation is based on the distribution
  described as the generalized normal version 2
  defined in the Wikipedia article:
  https://en.wikipedia.org/wiki/Generalized_normal_distribution
  accessed January 2019.

  Quantile, survival, log_survival, and all other essential functions
  were derived and defined by
  Daniel Luria, legally Daniel Maryanovsky, of vAIral, Kabbalah AI,
  and formerly of Lofty AI and Locbit Inc.

  The distribution returns NaN when evaluating
  probability of points outside its support
  '''

  def __init__(self,
               loc,
               scale,
               peak,
               validate_args=False,
               allow_nan_stats=True,
               name="GeneralizedGaussian"):

    parameters = dict(locals())


    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], tf.float32)
      loc = tf.convert_to_tensor(loc, name="loc", dtype=dtype)
      scale = tf.convert_to_tensor(scale, name="scale", dtype=dtype)
      peak = tf.convert_to_tensor(peak, name="peak", dtype=dtype)

      with tf.control_dependencies([tf.assert_positive(scale)] if
                                   validate_args else []):
        self._loc = tf.identity(loc)
        self._scale = tf.identity(scale)
        self._peak = tf.identity(peak)

        tf.debugging.assert_same_float_dtype(
            [self._loc, self._scale, self._peak]
        )

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
        zip(("loc", "scale", "peak"),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

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
    loc_scale_s = tf.broadcast_static_shape(self.loc.shape, self.scale.shape)
    loc_peak_s = tf.broadcast_static_shape(self.loc.shape, self.peak.shape)
    peak_scale_s = tf.broadcast_static_shape(self.peak.shape, self.scale.shape)
    assert loc_scale_s == loc_peak_s == peak_scale_s
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
    sampled = tf.random.uniform(
        shape=shape, minval=0., maxval=1., dtype=self.loc.dtype, seed=seed)
    return self._quantile(sampled)

  def _n_log_prob(self, x):
    return self._n_log_unnormalized_prob(x) - self._n_log_normalization()

  def _log_prob(self, x):
    log_smpe = tf.math.log(self.scale - (self.peak * (x - self.loc)))
    return self._n_log_prob(self._y(x)) - log_smpe

  def _prob(self, x):
    log_prob = tf.exp(self._log_prob(x))
    return tf.where(tf.math.is_nan(log_prob), tf.zeros_like(log_prob), log_prob
                    )

  def _log_cdf(self, x):
    return special_math.log_ndtr(self._y(x))

  def _y(self, x):
    inv_peak = (-1./self.peak)
    inv_offset = 1. - self.peak * (x - self.loc) / self.scale
    return inv_peak * tf.math.log(inv_offset)

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
    broadcast_ones = tf.ones_like(self.scale)
    esp = (tf.exp(tf.square(self.peak) / 2.) - 1.)
    mean = self.loc - (self.scale*esp/self.peak)

    return mean * broadcast_ones

  def _quantile(self, p):
    quantile_z = (1. - tf.exp(-self.peak * special_math.ndtri(p)))/self.peak
    return self._inv_z(quantile_z)

  def _stddev(self):
    broadcast_ones = tf.ones_like(self.loc)
    root_sq_offset = tf.sqrt(tf.exp(tf.square(self.peak)) - 1.)
    exp_square_peak = tf.exp(tf.square(self.peak)/2)
    scale_q = self.scale/tf.abs(self.peak)
    return scale_q * exp_square_peak * root_sq_offset * broadcast_ones

  def _mode(self):
    broad_ones = tf.ones_like(self.scale)
    unit_mode = ((1. - tf.exp(-tf.square(self.peak)))*self.scale)/self.peak
    return (unit_mode + self.loc) * broad_ones

  def _z(self, x):
    """Standardize input `x` to a unit normal."""
    with tf.name_scope("standardize"):
      return (x - self.loc) / self.scale

  def _inv_z(self, z):
    """Reconstruct input `x` from a its normalized version."""
    with tf.name_scope("reconstruct"):
      return z * self.scale + self.loc
