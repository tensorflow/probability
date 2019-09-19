"""The PERT distribution class"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import Beta
from tensorflow_probability.python.bijectors import AffineScalar
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensor_util # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'PERT',
]

class PERT(transformed_distribution.TransformedDistribution):
  """Modified PERT distribution:
  PERT distribution is a loc-scale family of Beta distribution intended to
  model expert prediction on arbitrary real interval bounded by MAX and MIN
  values set by the user, along with MODE to indicate the expert's
  most frequent prediction.

  The shift and scale factor is given by:

  ```
  shift = MIN
  scale = MAX - MIN
  ```

  Modified PERT is a generalized version of PERT that has an additional
  parameter, smoothness (g).

  Smoothness determines the properties of Beta distribution:

  ```
  concentration1 = 1 + g * (MODE - MIN)/(MAX - MIN)
  concentration0 = 1 + g * (MAX - MODE)/(MAX - MIN)
  mean = (MIN + g * MODE + MAX)/(g + 2)
  ```

  PERT distribution is obtained when g = 4.
  """
  def __init__(self,
               mini,
               mode,
               maxi,
               smoothness=4.,
               validate_args=False,
               allow_nan_stats=False,
               name="PERT"):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([mini, maxi, mode], tf.float32)
      self._min = tensor_util.convert_nonref_to_tensor(
          mini, name='mini', dtype=dtype)
      self._mod = tensor_util.convert_nonref_to_tensor(
          mode, name='mode', dtype=dtype)
      self._max = tensor_util.convert_nonref_to_tensor(
          maxi, name='maxi', dtype=dtype)
      self._smoothness = tensor_util.convert_nonref_to_tensor(
          smoothness, name='smoothness', dtype=dtype)
      self._concentration1 = 1. + self._smoothness \
        * (self._mod - self._min) / (self._max - self._min)
      self._concentration0 = 1. + self._smoothness \
        * (self._max - self._mod) / (self._max - self._min)

      self._scale = self._max - self._min

      # Broadcasting scale on affine bijector to match batch dimension.
      # This prevents dimension mismatch for operations like cdf.
      super(PERT, self).__init__(
          distribution=Beta(
              concentration1=self._concentration1,
              concentration0=self._concentration0,
              allow_nan_stats=allow_nan_stats),
          bijector=AffineScalar(
              shift=self._min,
              scale=tf.broadcast_to(
                  input=self._scale,
                  shape=self._batch_shape_tensor(
                      concentration1=self._concentration1, 
                      concentration0=self._concentration0))),
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  def _batch_shape_tensor(self, concentration1=None, concentration0=None):
    return prefer_static.broadcast_shape(
        prefer_static.shape(
            self.concentration1 if concentration1 is None else concentration1),
        prefer_static.shape(
            self.concentration0 if concentration0 is None else concentration0))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._mod):
      if is_init != tensor_util.is_ref(self._min):
        assertions.append(assert_util.assert_greater(
            self._mod,
            self._min,
            message="Mode must be greater than minimum."
        ))
      if is_init != tensor_util.is_ref(self._max):
        assertions.append(assert_util.assert_greater(
            self._max,
            self._mod,
            message="Maximum must be greater than mode."
        ))
    if is_init != tensor_util.is_ref(self._smoothness):
      assertions.append(assert_util.assert_positive(
          self._smoothness,
          message="Smoothness parameter must be positive."
      ))
    return assertions

  def _survivial_function(self, value, **kwargs):
    return 1. - self.cdf(value, kwargs)

  def _variance(self, **kwargs):
    mean = self.mean()
    return (mean - self._min) * (self._max - mean) / (self._smoothness + 3.)

  def _stddev(self, **kwargs):
    return tf.sqrt(self.variance(kwargs))

  def _mode(self, **kwargs):
    return self._mod

  # Distribution properties
  @property
  def minimum(self):
    return self._min

  @property
  def maximum(self):
    return self._max

  @property
  def smoothness(self):
    return self._smoothness

  @property
  def loc(self):
    return self._min

  @property
  def scale(self):
    return self._scale

  @property
  def concentration0(self):
    return self._concentration0

  @property
  def concentration1(self):
    return self._concentration1
