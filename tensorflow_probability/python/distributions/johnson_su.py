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
"""Johnson's SU distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.bijectors import scale as scale_bijector
from tensorflow_probability.python.bijectors import shift as shift_bijector
from tensorflow_probability.python.bijectors import sinh as sinh_bijector
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
    'JohnsonSU',
]


class JohnsonSU(transformed_distribution.TransformedDistribution):
  """Johnson's SU-distribution.

  This distribution has parameters: shape parameters `skewness` and
  `tailweight`, location `loc`, and `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; s, t, xi, sigma) = exp(-0.5 (s + t arcsinh(y))**2) / Z
  where,
  s = skewness
  t = tailweight
  y = (x - xi) / sigma
  Z = sigma sqrt(2 pi) sqrt(1 + y**2) / t
  ```

  where:
  * `loc = xi`,
  * `scale = sigma`, and,
  * `Z` is the normalization constant.

  The JohnsonSU distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ JohnsonSU(skewness, tailweight, loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Johnson's SU-distribution.
  single_dist = tfd.JohnsonSU(skewness=-2., tailweight=2., loc=1.1, scale=1.5)

  # Evaluate the pdf at 1, returning a scalar Tensor.
  single_dist.prob(1.)

  # Define a batch of two scalar valued Johnson SU's.
  # The first has shape parameters 1 and 2, mean 3, and scale 11.
  # The second 4, 5, 6 and 22.
  multi_dist = tfd.JohnsonSU(skewness=[1, 4], tailweight=[2, 5],
                             loc=[3, 6], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  multi_dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  multi_dist.sample(3)
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two Johnson's SU distributions.
  # Both have skewness 2, tailweight 3 and mean 1, but different scales.
  dist = tfd.JohnsonSU(skewness=2, tailweight=3, loc=1, scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  skewness = tf.Variable(2.0)
  tailweight = tf.Variable(3.0)
  loc = tf.Variable(2.0)
  scale = tf.Variable(11.0)
  dist = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=loc,
                       scale=scale)
  with tf.GradientTape() as tape:
    samples = dist.sample(5)  # Shape [5]
    loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tape.gradient(loss, dist.variables)
  ```

  """

  def __init__(self,
               skewness,
               tailweight,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Construct Johnson's SU distributions.

    The distributions have shape parameteres `tailweight` and `skewness`,
    mean `loc`, and scale `scale`.

    The parameters `tailweight`, `skewness`, `loc`, and `scale` must be shaped
    in a way that supports broadcasting
    (e.g. `skewness + tailweight + loc + scale` is a valid operation).

    Args:
      skewness: Floating-point `Tensor`. Skewness of the distribution(s).
      tailweight: Floating-point `Tensor`. Tail weight of the
        distribution(s). `tailweight` must contain only positive values.
      loc: Floating-point `Tensor`. The mean(s) of the distribution(s).
      scale: Floating-point `Tensor`. The scaling factor(s) for the
        distribution(s). Note that `scale` is not technically the standard
        deviation of this distribution but has semantics more similar to
        standard deviation than variance.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if any of skewness, tailweight, loc and scale are different
        dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'JohnsonSU') as name:
      dtype = dtype_util.common_dtype([skewness, tailweight, loc, scale],
                                      tf.float32)
      self._skewness = tensor_util.convert_nonref_to_tensor(
          skewness, name='skewness', dtype=dtype)
      self._tailweight = tensor_util.convert_nonref_to_tensor(
          tailweight, name='tailweight', dtype=dtype)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)

      norm_shift = invert_bijector.Invert(
          shift_bijector.Shift(shift=self._skewness,
                               validate_args=validate_args)
      )

      norm_scale = invert_bijector.Invert(
          scale_bijector.Scale(scale=self._tailweight,
                               validate_args=validate_args)
      )

      sinh = sinh_bijector.Sinh(validate_args=validate_args)

      scale = scale_bijector.Scale(scale=self._scale,
                                   validate_args=validate_args)

      shift = shift_bijector.Shift(shift=self._loc,
                                   validate_args=validate_args)

      bijector = shift(scale(sinh(norm_scale(norm_shift))))

      batch_rank = ps.reduce_max([
          distribution_util.prefer_static_rank(x)
          for x in (self._skewness, self._tailweight, self._loc, self._scale)])

      super(JohnsonSU, self).__init__(
          # TODO(b/160730249): Make `loc` a scalar `0.` and remove overridden
          # `batch_shape` and `batch_shape_tensor` when
          # TransformedDistribution's bijector can modify its `batch_shape`.
          distribution=normal.Normal(
              loc=tf.zeros(ps.ones(batch_rank, tf.int32), dtype=dtype),
              scale=tf.ones([], dtype=dtype),
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats),
          bijector=bijector,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        skewness=parameter_properties.ParameterProperties(),
        tailweight=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def skewness(self):
    """Skewness of these Johnson's SU distribution(s)."""
    return self._skewness

  @property
  def tailweight(self):
    """Tail weight of these Johnson's SU distribution(s)."""
    return self._tailweight

  @property
  def loc(self):
    """Locations of these Johnson's SU distribution(s)."""
    return self._loc

  @property
  def scale(self):
    """Scaling factors of these Johnson's SU distribution(s)."""
    return self._scale

  def _mean(self):
    skewness, tailweight, scale, loc = (
        [tf.convert_to_tensor(v)
         for v in (self.skewness, self.tailweight, self.scale, self.loc)])

    return (loc - scale * tf.math.exp(0.5 / tf.math.square(tailweight)) *
            tf.math.sinh(skewness / tailweight))

  def _variance(self):
    skewness, tailweight, scale = (
        [tf.convert_to_tensor(v)
         for v in (self.skewness, self.tailweight, self.scale)])

    variance = (0.5 * tf.math.square(scale) *
                tf.math.expm1(tf.math.reciprocal(tf.math.square(tailweight))) *
                (tf.math.exp(tf.math.reciprocal(tf.math.square(tailweight))) *
                 tf.math.cosh(2. * skewness / tailweight) + 1.))

    return tf.broadcast_to(variance, self.batch_shape_tensor())

  def _batch_shape(self):
    params = [self.skewness, self.tailweight, self.loc, self.scale]
    s_shape = params[0].shape
    for t in params[1:]:
      s_shape = tf.broadcast_static_shape(s_shape, t.shape)
    return s_shape

  def _batch_shape_tensor(self):
    return distribution_util.get_broadcast_shape(
        self.skewness, self.tailweight, self.loc, self.scale)

  def _parameter_control_dependencies(self, is_init):
    assertions = []
    if self.validate_args:
      if is_init != tensor_util.is_ref(self.tailweight):
        assertions.append(assert_util.assert_positive(
            self.tailweight, message='Argument `tailweight` must be positive.'))
      if is_init != tensor_util.is_ref(self.scale):
        assertions.append(assert_util.assert_positive(
            self.scale, message='Argument `scale` must be positive.'))

    return assertions
