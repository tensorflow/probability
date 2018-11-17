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
"""Affine bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow.python.ops import control_flow_ops


__all__ = [
    "Affine",
]


def _as_tensor(x, name, dtype):
  """Convenience to convert to `Tensor` or leave as `None`."""
  return None if x is None else tf.convert_to_tensor(x, name=name, dtype=dtype)


class Affine(bijector.Bijector):
  """Compute `Y = g(X; shift, scale) = scale @ X + shift`.

  Here `scale = c * I + diag(D1) + tril(L) + V @ diag(D2) @ V.T`.

  In TF parlance, the `scale` term is logically equivalent to:

  ```python
  scale = (
    scale_identity_multiplier * tf.diag(tf.ones(d)) +
    tf.diag(scale_diag) +
    scale_tril +
    scale_perturb_factor @ diag(scale_perturb_diag) @
      tf.transpose([scale_perturb_factor])
  )
  ```

  The `scale` term is applied without necessarily materializing constituent
  matrices, i.e., the matmul is [matrix-free](
  https://en.wikipedia.org/wiki/Matrix-free_methods) when possible.

  #### Examples

  ```python
  # Y = X
  b = Affine()

  # Y = X + shift
  b = Affine(shift=[1., 2, 3])

  # Y = 2 * I @ X.T + shift
  b = Affine(shift=[1., 2, 3],
             scale_identity_multiplier=2.)

  # Y = tf.diag(d1) @ X.T + shift
  b = Affine(shift=[1., 2, 3],
             scale_diag=[-1., 2, 1])         # Implicitly 3x3.

  # Y = (I + v * v.T) @ X.T + shift
  b = Affine(shift=[1., 2, 3],
             scale_perturb_factor=[[1., 0],
                                   [0, 1],
                                   [1, 1]])

  # Y = (diag(d1) + v * diag(d2) * v.T) @ X.T + shift
  b = Affine(shift=[1., 2, 3],
             scale_diag=[1., 3, 3],          # Implicitly 3x3.
             scale_perturb_diag=[2., 1],     # Implicitly 2x2.
             scale_perturb_factor=[[1., 0],
                                   [0, 1],
                                   [1, 1]])

  ```

  """

  def __init__(self,
               shift=None,
               scale_identity_multiplier=None,
               scale_diag=None,
               scale_tril=None,
               scale_perturb_factor=None,
               scale_perturb_diag=None,
               adjoint=False,
               validate_args=False,
               name="affine",
               dtype=None):
    """Instantiates the `Affine` bijector.

    This `Bijector` is initialized with `shift` `Tensor` and `scale` arguments,
    giving the forward operation:

    ```none
    Y = g(X) = scale @ X + shift
    ```

    where the `scale` term is logically equivalent to:

    ```python
    scale = (
      scale_identity_multiplier * tf.diag(tf.ones(d)) +
      tf.diag(scale_diag) +
      scale_tril +
      scale_perturb_factor @ diag(scale_perturb_diag) @
        tf.transpose([scale_perturb_factor])
    )
    ```

    If none of `scale_identity_multiplier`, `scale_diag`, or `scale_tril` are
    specified then `scale += IdentityMatrix`. Otherwise specifying a
    `scale` argument has the semantics of `scale += Expand(arg)`, i.e.,
    `scale_diag != None` means `scale += tf.diag(scale_diag)`.

    Args:
      shift: Floating-point `Tensor`. If this is set to `None`, no shift is
        applied.
      scale_identity_multiplier: floating point rank 0 `Tensor` representing a
        scaling done to the identity matrix.
        When `scale_identity_multiplier = scale_diag = scale_tril = None` then
        `scale += IdentityMatrix`. Otherwise no scaled-identity-matrix is added
        to `scale`.
      scale_diag: Floating-point `Tensor` representing the diagonal matrix.
        `scale_diag` has shape `[N1, N2, ...  k]`, which represents a k x k
        diagonal matrix.
        When `None` no diagonal term is added to `scale`.
      scale_tril: Floating-point `Tensor` representing the lower triangular
        matrix. `scale_tril` has shape `[N1, N2, ...  k, k]`, which represents a
        k x k lower triangular matrix.
        When `None` no `scale_tril` term is added to `scale`.
        The upper triangular elements above the diagonal are ignored.
      scale_perturb_factor: Floating-point `Tensor` representing factor matrix
        with last two dimensions of shape `(k, r)`. When `None`, no rank-r
        update is added to `scale`.
      scale_perturb_diag: Floating-point `Tensor` representing the diagonal
        matrix. `scale_perturb_diag` has shape `[N1, N2, ...  r]`, which
        represents an `r x r` diagonal matrix. When `None` low rank updates will
        take the form `scale_perturb_factor * scale_perturb_factor.T`.
      adjoint: Python `bool` indicating whether to use the `scale` matrix as
        specified or its adjoint.
        Default value: `False`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
      dtype: `tf.DType` to prefer when converting args to `Tensor`s. Else, we
        fall back to a common dtype inferred from the args, finally falling back
        to float32.

    Raises:
      ValueError: if `perturb_diag` is specified but not `perturb_factor`.
      TypeError: if `shift` has different `dtype` from `scale` arguments.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args

    # Ambiguous definition of low rank update.
    if scale_perturb_diag is not None and scale_perturb_factor is None:
      raise ValueError("When scale_perturb_diag is specified, "
                       "scale_perturb_factor must be specified.")

    # Special case, only handling a scaled identity matrix. We don't know its
    # dimensions, so this is special cased.
    # We don't check identity_multiplier, since below we set it to 1. if all
    # other scale args are None.
    self._is_only_identity_multiplier = (scale_tril is None and
                                         scale_diag is None and
                                         scale_perturb_factor is None)

    with self._name_scope("init", values=[
        shift, scale_identity_multiplier, scale_diag, scale_tril,
        scale_perturb_diag, scale_perturb_factor]):

      if dtype is None:
        dtype = dtype_util.common_dtype([
            shift, scale_identity_multiplier, scale_diag, scale_tril,
            scale_perturb_diag, scale_perturb_factor
        ], tf.float32)

      if shift is not None:
        shift = tf.convert_to_tensor(shift, name="shift", dtype=dtype)
      self._shift = shift

      # When no args are specified, pretend the scale matrix is the identity
      # matrix.
      if (self._is_only_identity_multiplier and
          scale_identity_multiplier is None):
        scale_identity_multiplier = tf.convert_to_tensor(1., dtype=dtype)

      # self._create_scale_operator returns a LinearOperator in all cases
      # except if self._is_only_identity_multiplier; in which case it
      # returns a scalar Tensor.
      scale = self._create_scale_operator(
          identity_multiplier=scale_identity_multiplier,
          diag=scale_diag,
          tril=scale_tril,
          perturb_diag=scale_perturb_diag,
          perturb_factor=scale_perturb_factor,
          shift=shift,
          validate_args=validate_args,
          dtype=dtype)

      if scale is not None and not self._is_only_identity_multiplier:
        if (shift is not None and
            shift.dtype.base_dtype != scale.dtype.base_dtype):
          raise TypeError(
              "shift.dtype({}) is incompatible with scale.dtype({}).".format(
                  shift.dtype, scale.dtype))

      self._scale = scale
      self._adjoint = adjoint
      super(Affine, self).__init__(
          forward_min_event_ndims=1,
          graph_parents=(
              [self._scale] if tf.contrib.framework.is_tensor(self._scale)
              else self._scale.graph_parents +
              [self._shift] if self._shift is not None else []),
          is_constant_jacobian=True,
          dtype=dtype,
          validate_args=validate_args,
          name=name)

  def _create_scale_operator(self, identity_multiplier, diag, tril,
                             perturb_diag, perturb_factor, shift, validate_args,
                             dtype):
    """Construct `scale` from various components.

    Args:
      identity_multiplier: floating point rank 0 `Tensor` representing a scaling
        done to the identity matrix.
      diag: Floating-point `Tensor` representing the diagonal matrix.`diag` has
        shape `[N1, N2, ...  k]`, which represents a k x k diagonal matrix.
      tril: Floating-point `Tensor` representing the lower triangular matrix.
       `tril` has shape `[N1, N2, ...  k, k]`, which represents a k x k lower
       triangular matrix.
      perturb_diag: Floating-point `Tensor` representing the diagonal matrix of
        the low rank update.
      perturb_factor: Floating-point `Tensor` representing factor matrix.
      shift: Floating-point `Tensor` representing `shift in `scale @ X + shift`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      dtype: `DType` for arg `Tensor` conversions.

    Returns:
      scale. In the case of scaling by a constant, scale is a
      floating point `Tensor`. Otherwise, scale is a `LinearOperator`.

    Raises:
      ValueError: if all of `tril`, `diag` and `identity_multiplier` are `None`.
    """
    identity_multiplier = _as_tensor(identity_multiplier, "identity_multiplier",
                                     dtype)
    diag = _as_tensor(diag, "diag", dtype)
    tril = _as_tensor(tril, "tril", dtype)
    perturb_diag = _as_tensor(perturb_diag, "perturb_diag", dtype)
    perturb_factor = _as_tensor(perturb_factor, "perturb_factor", dtype)

    # If possible, use the low rank update to infer the shape of
    # the identity matrix, when scale represents a scaled identity matrix
    # with a low rank update.
    shape_hint = None
    if perturb_factor is not None:
      shape_hint = distribution_util.dimension_size(perturb_factor, axis=-2)

    if self._is_only_identity_multiplier:
      if validate_args:
        return control_flow_ops.with_dependencies([
            tf.assert_none_equal(identity_multiplier,
                                 tf.zeros([], identity_multiplier.dtype),
                                 ["identity_multiplier should be non-zero."])
        ], identity_multiplier)
      return identity_multiplier

    scale = distribution_util.make_tril_scale(
        loc=shift,
        scale_tril=tril,
        scale_diag=diag,
        scale_identity_multiplier=identity_multiplier,
        validate_args=validate_args,
        assert_positive=False,
        shape_hint=shape_hint)

    if perturb_factor is not None:
      return tf.linalg.LinearOperatorLowRankUpdate(
          scale,
          u=perturb_factor,
          diag_update=perturb_diag,
          is_diag_update_positive=perturb_diag is None,
          is_non_singular=True,  # Implied by is_positive_definite=True.
          is_self_adjoint=True,
          is_positive_definite=True,
          is_square=True)

    return scale

  @property
  def shift(self):
    """The `shift` `Tensor` in `Y = scale @ X + shift`."""
    return self._shift

  @property
  def scale(self):
    """The `scale` `LinearOperator` in `Y = scale @ X + shift`."""
    return self._scale

  @property
  def adjoint(self):
    """`bool` indicating `scale` should be used as conjugate transpose."""
    return self._adjoint

  def _forward(self, x):
    y = x
    if self._is_only_identity_multiplier:
      s = (tf.conj(self._scale)
           if self.adjoint and self._scale.dtype.is_complex
           else self._scale)
      y *= s
      if self.shift is not None:
        return y + self.shift
      return y
    with tf.control_dependencies(self._maybe_check_scale()
                                 if self.validate_args else []):
      y = self.scale.matvec(y, adjoint=self.adjoint)
    if self.shift is not None:
      y += self.shift
    return y

  def _inverse(self, y):
    x = y
    if self.shift is not None:
      x -= self.shift
    if self._is_only_identity_multiplier:
      s = (tf.conj(self._scale)
           if self.adjoint and self._scale.dtype.is_complex
           else self._scale)
      return x / s
    # Solve fails if the op is singular so we may safely skip this assertion.
    x = self.scale.solvevec(x, adjoint=self.adjoint)
    return x

  def _forward_log_det_jacobian(self, x):
    # is_constant_jacobian = True for this bijector, hence the
    # `log_det_jacobian` need only be specified for a single input, as this will
    # be tiled to match `event_ndims`.
    if self._is_only_identity_multiplier:
      # We don't pad in this case and instead let the fldj be applied
      # via broadcast.
      log_abs_diag = tf.log(tf.abs(self._scale))
      event_size = tf.shape(x)[-1]
      event_size = tf.cast(event_size, dtype=log_abs_diag.dtype)
      return log_abs_diag * event_size
    return self.scale.log_abs_determinant()

  def _maybe_check_scale(self):
    try:
      return [self.scale.assert_non_singular()]
    except NotImplementedError:
      pass
    return []
