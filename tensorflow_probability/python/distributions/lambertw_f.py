# Lint as: python3
# Copyright 2020 The TensorFlow Probability Authors.
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
"""The Lambert W x F distribution class."""

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    "LambertWDistribution",
    "LambertWNormal",
]


class LambertWDistribution(transformed_distribution.TransformedDistribution):
  """Implements a general heavy-tail Lambert W x F distribution.

  Lambert W x F random variables are a transformed version of a random variables
  with distribution F that have heavier tails. In particular, they are defined
  as a (non-linear) transformation of random variables X with distribution F.
  It therefore is straightforward to implement Lambert W x F distributions as a
  particular TransformedDistribution, where the input can be specified by user
  as any TensorFlow Distribution class.

  ### Mathematical Details

  Let X be a random variable following distribution F with mean mu
  and standard deviation sigma, define as U = (X-mu)/sigma its zero-mean,
  unit-variance version. Then

  Y = (U * exp (delta/2 * U^2)) * sigma + mu

  is a location-scale heavy-tailed Lambert W x F with parameters mu,
  sigma and delta, where delta can take any non-negative real value. In
  particular, for delta = 0, the Lambert W x F distribution reduces to the
  F distribution. That is F distributions are a subset of Lambert W x
  F distributions.

  See `tfp.bijectors.LambertWTail` for details on the transformation.

  ### References:
  [1]: Goerg, G.M., 2011. Lambert W random variables - a new family of
  generalized skewed distributions with applications to risk estimation.
  The Annals of Applied Statistics, 5(3), pp.2197-2230.
  [2]: Goerg, G.M., 2015. The Lambert way to Gaussianize heavy-tailed data with
  the inverse of Tukey's h transformation as a special case. The Scientific
  World Journal.
  """

  def __init__(self,
               distribution,
               shift,
               scale,
               tailweight=None,
               validate_args=False,
               allow_nan_stats=True,
               name="LambertWDistribution"):
    """Initializes the class.

    Args:
      distribution: `tf.Distribution`-like instance. Distribution F that is
        transformed to produce this Lambert W x F distribution.
      shift: shift that should be applied before & after tail transformation.
        For a location-scale family `distribution` (e.g., `Normal` or
        `StudentT`) this usually is set as the mean / location parameter. For a
        scale family `distribution` (e.g., `Gamma` or `Fisher`) this must be
        set to 0 to guarantee a proper transformation on the positive
        real-line.
      scale: scaling factor that should be applied before & after the tail
        trarnsformation.  Usually the standard deviation or scaling parameter
        of the `distribution`.
      tailweight: Tail parameter `delta` of the resulting Lambert W x F
        distribution(s).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: A name for the operation (optional).
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([tailweight, shift, scale], tf.float32)
      tailweight = 0. if tailweight is None else tailweight
      self._tailweight = tensor_util.convert_nonref_to_tensor(
          tailweight, name="tailweight", dtype=dtype)
      self._shift = tensor_util.convert_nonref_to_tensor(
          shift, name="shift", dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name="scale", dtype=dtype)
      dtype_util.assert_same_float_dtype((self.tailweight, self.shift,
                                          self.scale))
      self._allow_nan_stats = allow_nan_stats
      super(LambertWDistribution, self).__init__(
          distribution=distribution,
          bijector=tfb.LambertWTail(shift=shift, scale=scale,
                                    tailweight=tailweight,
                                    validate_args=validate_args),
          parameters=parameters,
          validate_args=validate_args,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        shift=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        tailweight=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def allow_nan_stats(self):
    return self._allow_nan_stats

  @property
  def shift(self):
    """Distribution parameter for the shift before & after transformation."""
    return self._shift

  @property
  def scale(self):
    """Distribution parameter for the scaling before & after transformation."""
    return self._scale

  @property
  def tailweight(self):
    """Distribution parameter for the tail parameter delta."""
    return self._tailweight

  experimental_is_sharded = False

  def _batch_shape_tensor(self, shift=None, scale=None, tailweight=None):
    """Returns the batch shape of tensor parameter broadcasting."""
    return ps.shape(
        ps.shape(self.tailweight if tailweight is None
                 else tailweight,
                 ps.shape(self.shift if shift is None else shift)),
        ps.shape(self.scale if scale is None else scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        tf.broadcast_static_shape(self.tailweight.shape,
                                  self.shift.shape),
        self.scale.shape)


class LambertWNormal(LambertWDistribution):
  """Implements a location-scale heavy-tail Lambert W x Normal distribution."""

  def __init__(self,
               loc,
               scale,
               tailweight=None,
               validate_args=False,
               allow_nan_stats=True,
               name="LambertWNormal"):
    """Initializes the class.

    See `tfp.distributions.LambertWDistribution` for details.

    Args:
      loc: location parameter `loc` of the Normal distribution(s). This
        coincides with the location parameter of the resulting LambertWNormal.
      scale: scale parameter `scale` of the Normal distribution(s).
      tailweight: Tail parameter `delta` of the distribution(s). If `None`, it
        defaults to 0, which implies LambertWNormal == Normal.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: A name for the operation (optional).
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([tailweight, loc, scale], tf.float32)
      super(LambertWNormal, self).__init__(
          distribution=normal.Normal(loc=loc, scale=scale),
          shift=loc,
          scale=scale,
          tailweight=tailweight,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
      self._parameters = parameters
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name="loc", dtype=dtype)
      dtype_util.assert_same_float_dtype((self.tailweight, self.loc,
                                          self.scale))

  @property
  def loc(self):
    """Location parameter of the Lambert W x Normal distribution."""
    return self._loc

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        tailweight=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @distribution_util.AppendDocstring(
      """The mean of Lambert W x Normal equals `loc` if `tailweight > 1`,
      otherwise it is `NaN`. If `self.allow_nan_stats=True`, then an exception
      will be raised rather than returning `NaN`.""")
  def _mean(self):
    tailweight = tf.convert_to_tensor(self.tailweight)
    loc = tf.convert_to_tensor(self.loc)
    mean = loc * tf.ones(self.batch_shape, dtype=self.dtype)
    if self.allow_nan_stats:
      return tf.where(
          tailweight < 1.,
          mean,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.ones([], dtype=self.dtype),
              tailweight,
              message="mean not defined for components of tailweight >= 1"),
      ], mean)

  @distribution_util.AppendDocstring("""
      The variance for Lambert W x Normal is finite if `tailweight < 0.5`. For
      `0.5 <= tailweight < 1` it is infinite, and for `tailweight > 1` it is
      undefined (since mean does not exist either).
      """)
  def _variance(self):
    tailweight = tf.convert_to_tensor(self.tailweight)
    scale = tf.convert_to_tensor(self.scale)
    # For tail < 0.5, the variance is finite. See Eq (18) in
    # https://www.hindawi.com/journals/tswj/2015/909231/
    var = (tf.cast(tf.pow(1. - 2. * tailweight, - 3. / 2.), dtype=self.dtype) *
           tf.math.square(scale))
    # We need to put the tf.where inside the outer tf.where to ensure we never
    # hit a NaN in the gradient.
    result_where_defined = tf.where(
        tailweight < 0.5,
        var,
        tf.convert_to_tensor(np.inf, dtype=self.dtype))

    if self.allow_nan_stats:
      return tf.where(
          tailweight < 1.0,
          result_where_defined,
          tf.convert_to_tensor(np.nan, self.dtype))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_greater_equal(
              tf.ones([], dtype=self.dtype),
              tailweight,
              message="variance not defined for components of tailweight >= 1"),
      ], result_where_defined)

  def _mode(self):
    # Mode always exists (for any tail parameter) and equals the location / mean
    # independent of the tail parameter.
    loc = tf.convert_to_tensor(self.loc)
    return tf.broadcast_to(loc, self.batch_shape)

  def _batch_shape_tensor(self, loc=None, scale=None, tailweight=None):
    """Returns the batch shape of tensor parameter broadcasting."""
    return ps.shape(
        ps.shape(self.tailweight if tailweight is None
                 else tailweight,
                 ps.shape(self.loc if loc is None else loc)),
        ps.shape(self.scale if scale is None else scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        tf.broadcast_static_shape(self.tailweight.shape,
                                  self.loc.shape),
        self.scale.shape)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._tailweight):
      assertions.append(assert_util.assert_greater_equal(
          self._tailweight, tf.zeros([], dtype=self.dtype),
          message="Argument `tailweight` must be non-negative."))
    return assertions

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    return identity_bijector.Identity(validate_args=self.validate_args)
