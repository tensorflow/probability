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
"""SinhArcsinh transformation of a distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import affine_scalar as affine_scalar_bijector
from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import sinh_arcsinh as sinh_arcsinh_bijector
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'SinhArcsinh',
]


class SinhArcsinh(transformed_distribution.TransformedDistribution):
  """The SinhArcsinh transformation of a distribution on `(-inf, inf)`.

  This distribution models a random variable, making use of
  a `SinhArcsinh` transformation (which has adjustable tailweight and skew),
  a rescaling, and a shift.

  The `SinhArcsinh` transformation of the Normal is described in great depth in
  [Sinh-arcsinh distributions](https://www.jstor.org/stable/27798865).
  Here we use a slightly different parameterization, in terms of `tailweight`
  and `skewness`.  Additionally we allow for distributions other than Normal,
  and control over `scale` as well as a "shift" parameter `loc`.

  #### Mathematical Details

  Given random variable `Z`, we define the SinhArcsinh
  transformation of `Z`, `Y`, parameterized by
  `(loc, scale, skewness, tailweight)`, via the relation:

  ```
  Y := loc + scale * F(Z)
  F(Z) := Sinh( (Arcsinh(Z) + skewness) * tailweight ) * (2 / F_0(2))
  F_0(Z) := Sinh( Arcsinh(Z) * tailweight )
  ```

  This distribution is similar to the location-scale transformation
  `L(Z) := loc + scale * Z` in the following ways:

  * If `skewness = 0` and `tailweight = 1` (the defaults), `F(Z) = Z`, and then
    `Y = L(Z)` exactly.
  * `loc` is used in both to shift the result by a constant factor.
  * The multiplication of `scale` by `2 / F_0(2)` ensures that if `skewness = 0`
    `P[Y - loc <= 2 * scale] = P[L(Z) - loc <= 2 * scale]`.
    Thus it can be said that the weights in the tails of `Y` and `L(Z)` beyond
    `loc + 2 * scale` are the same.

  This distribution is different than `loc + scale * Z` due to the
  reshaping done by `F`:

  * Positive (negative) `skewness` leads to positive (negative) skew.
    * positive skew means, the mode of `F(Z)` is "tilted" to the right.
    * positive skew means positive values of `F(Z)` become more likely, and
      negative values become less likely.
  * Larger (smaller) `tailweight` leads to fatter (thinner) tails.
    * Fatter tails mean larger values of `|F(Z)|` become more likely.
    * `tailweight < 1` leads to a distribution that is "flat" around `Y = loc`,
      and a very steep drop-off in the tails.
    * `tailweight > 1` leads to a distribution more peaked at the mode with
      heavier tails.

  To see the argument about the tails, note that for `|Z| >> 1` and
  `|Z| >> (|skewness| * tailweight)**tailweight`, we have
  `Y approx 0.5 Z**tailweight e**(sign(Z) skewness * tailweight)`.

  To see the argument regarding multiplying `scale` by `2 / F_0(2)`,

  ```
  P[(Y - loc) / scale <= 2] = P[F(Z) * (2 / F_0(2)) <= 2]
                            = P[F(Z) <= F_0(2)]
                            = P[Z <= 2]  (if F = F_0).
  ```
  """

  @deprecation.deprecated(
      '2020-06-01', 'Previously, `distribution` was required to have scalar '
      'batch. Because batch shape overrides to `TransformedDistribution` are '
      'deprecated, `distribution` now must have a batch shape to which the '
      'shapes of `loc`, `scale`, `skewness`, and `tailweight` all broadcast.')
  def __init__(self,
               loc,
               scale,
               skewness=None,
               tailweight=None,
               distribution=None,
               validate_args=False,
               allow_nan_stats=True,
               name='SinhArcsinh'):
    """Construct SinhArcsinh distribution on `(-inf, inf)`.

    Arguments `(loc, scale, skewness, tailweight)` must have broadcastable shape
    (indexing batch dimensions).  They must all have the same `dtype`.

    Args:
      loc: Floating-point `Tensor`.
      scale:  `Tensor` of same `dtype` as `loc`.
      skewness:  Skewness parameter.  Default is `0.0` (no skew).
      tailweight:  Tailweight parameter. Default is `1.0` (unchanged tailweight)
      distribution: `tf.Distribution`-like instance. Distribution that is
        transformed to produce this distribution.
        Default is `tfd.Normal(0., 1.)`.
        Must be a scalar-batch, scalar-event distribution.  Typically
        `distribution.reparameterization_type = FULLY_REPARAMETERIZED` or it is
        a function of non-trainable parameters. WARNING: If you backprop through
        a `SinhArcsinh` sample and `distribution` is not
        `FULLY_REPARAMETERIZED` yet is a function of trainable variables, then
        the gradient will be incorrect!
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())

    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale, skewness, tailweight],
                                      tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      tailweight = 1. if tailweight is None else tailweight
      has_default_skewness = skewness is None
      skewness = 0. if has_default_skewness else skewness
      self._tailweight = tensor_util.convert_nonref_to_tensor(
          tailweight, name='tailweight', dtype=dtype)
      self._skewness = tensor_util.convert_nonref_to_tensor(
          skewness, name='skewness', dtype=dtype)

      batch_shape = distribution_util.get_broadcast_shape(
          self._loc, self._scale, self._tailweight, self._skewness)

      # Recall, with Z a random variable,
      #   Y := loc + scale * F(Z),
      #   F(Z) := Sinh( (Arcsinh(Z) + skewness) * tailweight ) * C
      #   C := 2 / F_0(2)
      #   F_0(Z) := Sinh( Arcsinh(Z) * tailweight )
      if distribution is None:
        # TODO(b/151180729): When `batch_shape` arg to `TransformedDistribution`
        # is deprecated, broadcast `loc` or `scale` parameter to `batch_shape`
        # and remove `else` condition.
        distribution = normal.Normal(
            loc=tf.zeros([], dtype=dtype),
            scale=tf.ones([], dtype=dtype),
            allow_nan_stats=allow_nan_stats,
            validate_args=validate_args)
      else:
        asserts = distribution_util.maybe_check_scalar_distribution(
            distribution, dtype, validate_args)
        if asserts:
          self._loc = distribution_util.with_dependencies(asserts, self._loc)

      # Make the SAS bijector, 'F'.
      f = sinh_arcsinh_bijector.SinhArcsinh(
          skewness=self._skewness, tailweight=self._tailweight,
          validate_args=validate_args)

      # Make the AffineScalar bijector, Z --> loc + scale * Z (2 / F_0(2))
      affine = affine_scalar_bijector.AffineScalar(
          shift=self._loc,
          scale=self._scale,
          validate_args=validate_args)

      bijector = chain_bijector.Chain([affine, f])

      super(SinhArcsinh, self).__init__(
          distribution=distribution,
          bijector=bijector,
          batch_shape=batch_shape,
          validate_args=validate_args,
          name=name)
      self._parameters = parameters

  @property
  def loc(self):
    """The `loc` in `Y := loc + scale @ F(Z)`."""
    return self._loc

  @property
  def scale(self):
    """The `LinearOperator` `scale` in `Y := loc + scale @ F(Z)`."""
    return self._scale

  @property
  def tailweight(self):
    """Controls the tail decay.  `tailweight > 1` means faster than Normal."""
    return self._tailweight

  @property
  def skewness(self):
    """Controls the skewness.  `Skewness > 0` means right skew."""
    return self._skewness

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    return identity_bijector.Identity(validate_args=self.validate_args)
