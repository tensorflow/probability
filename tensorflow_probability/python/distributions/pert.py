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
  """Modified PERT distribution for modeling expert predictions.

  PERT distribution is a loc-scale family of Beta distribution
  fit onto an arbitrary real interval set between LOW and HIGH
  values set by the user, along with PEAK to indicate the expert's
  most frequent prediction.
  [1](https://en.wikipedia.org/wiki/PERT_distribution).

  It's like a [Triangle distribution]
  (https://en.wikipedia.org/wiki/Triangular_distribution)
  but with a smooth peak.

  #### Mathematical Details

  In terms of a Beta distribution, PERT can be expressed as

  ```none
  PERT ~ loc + scale * Beta(concentration1, concentration0)
  ```
  where

  ```none
  loc = LOW
  scale = HIGH - LOW
                            PEAK - LOW
  concentration1 = 1 + g * ------------
                            HIGH - LOW

                            HIGH - PEAK
  concentration1 = 1 + g * ------------
                            HIGH - LOW
  g > 0
  ```
  over support [LOW, HIGH]. And some PEAK such that
  LOW < PEAK < HIGH. `g` is a positive parameter that controls
  the shape of the the distribution. Higher value forms a sharper
  peak.

  Standard PERT distribution is obtained when g = 4.

  #### Examples
  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Single PERT distribution
  dist = tfd.PERT(low=1., peak=7., high=11., temperature=4.)
  dist.sample(10)
  dist.prob(7.)
  dist.prob(0.) # Throws nan when input out of support.

  # Multiple PERT distributions with varying temperature (broadcasted)
  dist = tfd.PERT(low=1., peak=7., high=11., temperature=[1., 2., 3., 4.])
  dist.sample(10)
  dist.prob(7.)
  dist.prob([[7.],[5.]])

  # Multiple PERT distributions with varying peak
  dist = tfd.PERT(low=1., peak=[2., 5., 8.], high=11., temperature=4.)
  dist.sample(10)
  dist.sample([10,5])
  ```
  """
  def __init__(self,
               low,
               peak,
               high,
               temperature=4.,
               validate_args=False,
               allow_nan_stats=False,
               name="PERT"):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([low, high, peak], tf.float32)
      self._low = tensor_util.convert_nonref_to_tensor(
          low, name='low', dtype=dtype)
      self._peak = tensor_util.convert_nonref_to_tensor(
          peak, name='peak', dtype=dtype)
      self._high = tensor_util.convert_nonref_to_tensor(
          high, name='high', dtype=dtype)
      self._temperature = tensor_util.convert_nonref_to_tensor(
          temperature, name='temperature', dtype=dtype)
      self._concentration1 = (1. + self._temperature
              * (self._peak - self._low) / (self._high - self._low))
      self._concentration0 = (1. + self._temperature
              * (self._high - self._peak) / (self._high - self._low))

      self._scale = self._high - self._low

      # Broadcasting scale on affine bijector to match batch dimension.
      # This prevents dimension mismatch for operations like cdf.
      super(PERT, self).__init__(
          distribution=Beta(
              concentration1=self._concentration1,
              concentration0=self._concentration0,
              allow_nan_stats=allow_nan_stats),
          bijector=AffineScalar(
              shift=self._low,
              scale=tf.broadcast_to(
                  input=self._scale,
                  shape=self._batch_shape())),
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(low=0, high=0, peak=0, temperature=0)

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.concentration1.shape, self.concentration0.shape)

  def _batch_shape_tensor(self, concentration1=None, concentration0=None):
    return prefer_static.broadcast_shape(
        prefer_static.shape(
            self.concentration1 if concentration1 is None else concentration1),
        prefer_static.shape(
            self.concentration0 if concentration0 is None else concentration0))

  def _variance(self):
    mean = self.mean()
    return (mean - self._low) * (self._high - mean) / (self._temperature + 3.)

  def _stddev(self):
    return tf.sqrt(self._variance())

  def _mode(self):
    return self._peak

  # Distribution properties
  @property
  def low(self):
    return self._low

  @property
  def peak(self):
    return self._peak

  @property
  def high(self):
    return self._high

  @property
  def temperature(self):
    return self._temperature

  @property
  def loc(self):
    return self._low

  @property
  def scale(self):
    return self._scale

  @property
  def concentration0(self):
    return self._concentration0

  @property
  def concentration1(self):
    return self._concentration1

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._peak):
      if is_init != tensor_util.is_ref(self._low):
        assertions.append(assert_util.assert_greater(
            self._peak,
            self._low,
            message="`peak` must be greater than `low`."
        ))
      if is_init != tensor_util.is_ref(self._high):
        assertions.append(assert_util.assert_greater(
            self._high,
            self._peak,
            message="`high` must be greater than `peak`."
        ))
    if is_init != tensor_util.is_ref(self._temperature):
      assertions.append(assert_util.assert_positive(
          self._temperature,
          message="`temperature` must be positive."
      ))
    return assertions