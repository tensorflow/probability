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
"""Multi-dimensional (Vector) SinhArcsinh transformation of a distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import affine as affine_bijector
from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import sinh_arcsinh as sinh_arcsinh_bijector
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'VectorSinhArcsinhDiag',
]


# Copied from distribution_util, where it is to be removed, and duplicated here
# to support VectorLaplaceDiag until the deprecation window is closed.
def _make_diag_scale(loc=None,
                     scale_diag=None,
                     scale_identity_multiplier=None,
                     shape_hint=None,
                     validate_args=False,
                     assert_positive=False,
                     name=None,
                     dtype=None):
  """Creates a LinearOperator representing a diagonal matrix.

  Args:
    loc: Floating-point `Tensor`. This is used for inferring shape in the case
      where only `scale_identity_multiplier` is set.
    scale_diag: Floating-point `Tensor` representing the diagonal matrix.
      `scale_diag` has shape [N1, N2, ...  k], which represents a k x k diagonal
      matrix. When `None` no diagonal term is added to the LinearOperator.
    scale_identity_multiplier: floating point rank 0 `Tensor` representing a
      scaling done to the identity matrix. When `scale_identity_multiplier =
      scale_diag = scale_tril = None` then `scale += IdentityMatrix`. Otherwise
      no scaled-identity-matrix is added to `scale`.
    shape_hint: scalar integer `Tensor` representing a hint at the dimension of
      the identity matrix when only `scale_identity_multiplier` is set.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness.
    assert_positive: Python `bool` indicating whether LinearOperator should be
      checked for being positive definite.
    name: Python `str` name given to ops managed by this object.
    dtype: TF `DType` to prefer when converting args to `Tensor`s. Else, we fall
      back to a compatible dtype across all of `loc`, `scale_diag`, and
      `scale_identity_multiplier`.

  Returns:
    `LinearOperator` representing a lower triangular matrix.

  Raises:
    ValueError:  If only `scale_identity_multiplier` is set and `loc` and
      `shape_hint` are both None.
  """

  with tf.name_scope(name or 'make_diag_scale'):
    if dtype is None:
      dtype = dtype_util.common_dtype(
          [loc, scale_diag, scale_identity_multiplier],
          dtype_hint=tf.float32)
    loc = tensor_util.convert_nonref_to_tensor(loc, name='loc', dtype=dtype)
    scale_diag = tensor_util.convert_nonref_to_tensor(
        scale_diag, name='scale_diag', dtype=dtype)
    scale_identity_multiplier = tensor_util.convert_nonref_to_tensor(
        scale_identity_multiplier,
        name='scale_identity_multiplier',
        dtype=dtype)

    if scale_diag is not None:
      if scale_identity_multiplier is not None:
        scale_diag = scale_diag + scale_identity_multiplier[..., tf.newaxis]
      return tf.linalg.LinearOperatorDiag(
          diag=scale_diag,
          is_non_singular=True,
          is_self_adjoint=True,
          is_positive_definite=assert_positive)

    if loc is None and shape_hint is None:
      raise ValueError('Cannot infer `event_shape` unless `loc` or '
                       '`shape_hint` is specified.')

    num_rows = shape_hint
    del shape_hint
    if num_rows is None:
      num_rows = tf.compat.dimension_value(loc.shape[-1])
      if num_rows is None:
        num_rows = tf.shape(loc)[-1]

    if scale_identity_multiplier is None:
      return tf.linalg.LinearOperatorIdentity(
          num_rows=num_rows,
          dtype=dtype,
          is_self_adjoint=True,
          is_positive_definite=True,
          assert_proper_shapes=validate_args)

    return tf.linalg.LinearOperatorScaledIdentity(
        num_rows=num_rows,
        multiplier=scale_identity_multiplier,
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=assert_positive,
        assert_proper_shapes=validate_args)


class VectorSinhArcsinhDiag(transformed_distribution.TransformedDistribution):
  """The (diagonal) SinhArcsinh transformation of a distribution on `R^k`.

  This distribution models a random vector `Y = (Y1,...,Yk)`, making use of
  a `SinhArcsinh` transformation (which has adjustable tailweight and skew),
  a rescaling, and a shift.

  The `SinhArcsinh` transformation of the Normal is described in great depth in
  [Sinh-arcsinh distributions](https://www.jstor.org/stable/27798865).
  Here we use a slightly different parameterization, in terms of `tailweight`
  and `skewness`.  Additionally we allow for distributions other than Normal,
  and control over `scale` as well as a 'shift' parameter `loc`.

  #### Mathematical Details

  Given iid random vector `Z = (Z1,...,Zk)`, we define the VectorSinhArcsinhDiag
  transformation of `Z`, `Y`, parameterized by
  `(loc, scale, skewness, tailweight)`, via the relation (with `@` denoting
  matrix multiplication):

  ```
  Y := loc + scale @ F(Z)
  F(Z) := Sinh( (Arcsinh(Z) + skewness) * tailweight ) * (2 / F_0(2))
  F_0(Z) := Sinh( Arcsinh(Z) * tailweight )
  ```

  This distribution is similar to the location-scale transformation
  `L(Z) := loc + scale @ Z` in the following ways:

  * If `skewness = 0` and `tailweight = 1` (the defaults), `F(Z) = Z`, and then
    `Y = L(Z)` exactly.
  * `loc` is used in both to shift the result by a constant factor.
  * The multiplication of `scale` by `2 / F_0(2)` ensures that if `skewness = 0`
    `P[Y - loc <= 2 * scale] = P[L(Z) - loc <= 2 * scale]`.
    Thus it can be said that the weights in the tails of `Y` and `L(Z)` beyond
    `loc + 2 * scale` are the same.

  This distribution is different than `loc + scale @ Z` due to the
  reshaping done by `F`:

  * Positive (negative) `skewness` leads to positive (negative) skew.
    * positive skew means, the mode of `F(Z)` is 'tilted' to the right.
    * positive skew means positive values of `F(Z)` become more likely, and
      negative values become less likely.
  * Larger (smaller) `tailweight` leads to fatter (thinner) tails.
    * Fatter tails mean larger values of `|F(Z)|` become more likely.
    * `tailweight < 1` leads to a distribution that is 'flat' around `Y = loc`,
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
      '2020-01-01',
      '`VectorSinhArcsinhDiag` is deprecated. If all you need is the diagonal '
      'scale, you can use '
      '`tfd.Independent(tfd.SinhArcsinh(loc=loc, scale=scale_diag, ...), 1)` '
      'instead.')
  def __init__(self,
               loc=None,
               scale_diag=None,
               scale_identity_multiplier=None,
               skewness=None,
               tailweight=None,
               distribution=None,
               validate_args=False,
               allow_nan_stats=True,
               name='VectorSinhArcsinhDiag'):
    """Construct VectorSinhArcsinhDiag distribution on `R^k`.

    The arguments `scale_diag` and `scale_identity_multiplier` combine to
    define the diagonal `scale` referred to in this class docstring:

    ```none
    scale = diag(scale_diag + scale_identity_multiplier * ones(k))
    ```

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` (if provided) must broadcast with this

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      scale_diag: Non-zero, floating-point `Tensor` representing a diagonal
        matrix added to `scale`. May have shape `[B1, ..., Bb, k]`, `b >= 0`,
        and characterizes `b`-batches of `k x k` diagonal matrices added to
        `scale`. When both `scale_identity_multiplier` and `scale_diag` are
        `None` then `scale` is the `Identity`.
      scale_identity_multiplier: Non-zero, floating-point `Tensor` representing
        a scale-identity-matrix added to `scale`. May have shape
        `[B1, ..., Bb]`, `b >= 0`, and characterizes `b`-batches of scale
        `k x k` identity matrices added to `scale`. When both
        `scale_identity_multiplier` and `scale_diag` are `None` then `scale`
        is the `Identity`.
      skewness:  Skewness parameter.  floating-point `Tensor` with shape
        broadcastable with `event_shape`.
      tailweight:  Tailweight parameter.  floating-point `Tensor` with shape
        broadcastable with `event_shape`.
      distribution: `tf.Distribution`-like instance. Distribution from which `k`
        iid samples are used as input to transformation `F`.  Default is
        `tfd.Normal(loc=0., scale=1.)`.
        Must be a scalar-batch, scalar-event distribution.  Typically
        `distribution.reparameterization_type = FULLY_REPARAMETERIZED` or it is
        a function of non-trainable parameters. WARNING: If you backprop through
        a VectorSinhArcsinhDiag sample and `distribution` is not
        `FULLY_REPARAMETERIZED` yet is a function of trainable variables, then
        the gradient will be incorrect!
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
      ValueError: if at most `scale_identity_multiplier` is specified.
    """
    parameters = dict(locals())

    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [loc, scale_diag, scale_identity_multiplier, skewness, tailweight],
          tf.float32)
      loc = loc if loc is None else tf.convert_to_tensor(
          loc, name='loc', dtype=dtype)
      tailweight = 1. if tailweight is None else tailweight
      skewness = 0. if skewness is None else skewness

      # Recall, with Z a random variable,
      #   Y := loc + C * F(Z),
      #   F(Z) := Sinh( (Arcsinh(Z) + skewness) * tailweight )
      #   F_0(Z) := Sinh( Arcsinh(Z) * tailweight )
      #   C := 2 * scale / F_0(2)

      # Construct shapes and 'scale' out of the scale_* and loc kwargs.
      # scale_linop is only an intermediary to:
      #  1. get shapes from looking at loc and the two scale args.
      #  2. combine scale_diag with scale_identity_multiplier, which gives us
      #     'scale', which in turn gives us 'C'.
      scale_linop = _make_diag_scale(
          loc=loc,
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          validate_args=False,
          assert_positive=False,
          dtype=dtype)
      batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
          loc, scale_linop)
      # scale_linop.diag_part() is efficient since it is a diag type linop.
      scale_diag_part = scale_linop.diag_part()
      dtype = scale_diag_part.dtype

      if distribution is None:
        distribution = normal.Normal(
            loc=tf.zeros([], dtype=dtype),
            scale=tf.ones([], dtype=dtype),
            allow_nan_stats=allow_nan_stats)
      else:
        asserts = distribution_util.maybe_check_scalar_distribution(
            distribution, dtype, validate_args)
        if asserts:
          scale_diag_part = distribution_util.with_dependencies(
              asserts, scale_diag_part)

      # Make the SAS bijector, 'F'.
      skewness = tf.convert_to_tensor(skewness, dtype=dtype, name='skewness')
      tailweight = tf.convert_to_tensor(
          tailweight, dtype=dtype, name='tailweight')
      f = sinh_arcsinh_bijector.SinhArcsinh(
          skewness=skewness, tailweight=tailweight)
      affine = affine_bijector.Affine(
          shift=loc, scale_diag=scale_diag_part, validate_args=validate_args)

      bijector = chain_bijector.Chain([affine, f])

      super(VectorSinhArcsinhDiag, self).__init__(
          distribution=distribution,
          bijector=bijector,
          batch_shape=batch_shape,
          event_shape=event_shape,
          validate_args=validate_args,
          name=name)
    self._parameters = parameters
    self._loc = loc
    self._scale = scale_linop
    self._tailweight = tailweight
    self._skewness = skewness

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
