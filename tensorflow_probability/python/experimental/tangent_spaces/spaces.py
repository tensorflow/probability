# Copyright 2021 The TensorFlow Probability Authors.
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

"""Class definitions for tangent spaces."""

import tensorflow.compat.v2 as tf

__all__ = [
    'AxisAlignedSpace',
    'FullSpace',
    'GeneralSpace',
    'TangentSpace',
    'UnspecifiedTangentSpaceError',
    'ZeroSpace',
]


class TangentSpace(object):
  """Represents a tangent space to some manifold M at a point x.

  TFP allows one to transform manifolds via `Bijector`s. Keeping track
  of the tangent space to a manifold allows us to calculate the
  correct push-forward density under such transformations.

  In general, the density correction involves computing the basis of
  the tangent space as well as the image of that basis under the
  transformation. But we can avoid this work in special cases that
  arise from the properties of the transformation f (e.g.,
  dimension-preserving, coordinate-wise) and the density p of the
  manifold (e.g., discrete, supported on all of R^n).

  Each subclass of `TangentSpace` represents a specific property of
  densities seen in the uses of TFP. The methods of `TangentSpace`
  represent the special `Bijector` structures provided by TFP. Each
  subclass thus defines how to compute the density correction under
  each kind of transformation.

  """

  def transform_general(self, x, f, **kwargs):
    """Returns the density correction, in log space, corresponding to f at x.

    Also returns a new `TangentSpace` representing the tangent to fM at f(x).

    Args:
      x: `Tensor` (structure). The point at which to calculate the density.
      f: `Bijector` or one of its subclasses. The transformation that requires a
        density correction based on this tangent space.
      **kwargs: Optional keyword arguments as part of the Bijector.

    Returns:
      log_density: A `Tensor` representing the log density correction of f at x
      space: A `TangentSpace` representing the tangent to fM at f(x)

    Raises:
      NotImplementedError: if the `TangentSpace` subclass does not implement
        this method.
    """
    raise NotImplementedError

  def transform_dimension_preserving(self, x, f, **kwargs):
    """Same as `transform_general`, assuming f goes from R^n to R^n.

    Default falls back to `transform_general`, which may be overridden
    in subclasses.

    Args:
      x: same as in `transform_general`.
      f: same as in `transform_general`.
      **kwargs: same as in `transform_general`.

    Returns:
      log_density: A `Tensor` representing the log density correction of f at x
      space: A `TangentSpace` representing the tangent to fM at f(x)

    Raises:
      NotImplementedError: if the `TangentSpace` subclass does not implement
        `transform_general`.

    """
    return self.transform_general(x, f, **kwargs)

  def transform_projection(self, x, f, **kwargs):
    """Same as `transform_general`, with f a projection (or its inverse).

    Default falls back to `transform_general`, which may be overridden
    in subclasses.

    Args:
      x: same as in `transform_general`.
      f: same as in `transform_general`.
      **kwargs: same as in `transform_general`.

    Returns:
      log_density: A `Tensor` representing the log density correction of f at x
      space: A `TangentSpace` representing the tangent to fM at f(x)

    Raises:
      NotImplementedError: if the `TangentSpace` subclass does not implement
        `transform_general`.
    """
    return self.transform_general(x, f, **kwargs)

  def transform_coordinatewise(self, x, f, **kwargs):
    """Same as `transform_dimension_preserving`, for a coordinatewise f.

    Default falls back to `transform_dimension_preserving`, which may
    be overridden in subclasses.

    Args:
      x: same as in `transform_dimension_preserving`.
      f: same as in `transform_dimension_preserving`.
      **kwargs: same as in `transform_dimension_preserving`.

    Returns:
      log_density: A `Tensor` representing the log density correction of f at x
      space: A `TangentSpace` representing the tangent to fM at f(x)

    Raises:
      NotImplementedError: if the `TangentSpace` subclass does not implement
        `transform_dimension_preserving`.

    """
    return self.transform_dimension_preserving(x, f, **kwargs)


def unit_basis():
  raise NotImplementedError


def unit_basis_on(axis_mask):
  raise NotImplementedError


class AxisAlignedSpace(TangentSpace):
  """Tangent space of M for a distribution with axis-aligned subspace support.

  This subclass covers cases where the support of the distribution is
  on an axis-aligned subspace, such as lower-triangular matrices. In
  this special case we can represent the standard basis of the
  subspace with a mask. The subclass is designed to support axis-aligned
  injections like the `FillTriangular` `Bijector`.

  Any Bijector calling the `transform_projection` method is expected
  to define an `experimental_update_live_dimensions` method.
  """

  def __init__(self, axis_mask):
    """Constructs an AxisAlignedSpace with a set of live dimensions.

    Args:
      axis_mask: `Tensor`. A bit-mask of the live dimensions of the space.
    """
    self.axis_mask = axis_mask

  def transform_general(self, x, f, **kwargs):
    as_general_space = GeneralSpace(unit_basis_on(self.axis_mask), 1)
    return as_general_space.transform_general(x, f, **kwargs)

  def transform_projection(self, x, f, **kwargs):
    if not hasattr(f, 'experimental_update_live_dimensions'):
      msg = ('When calling `transform_projection` the Bijector must implement '
             'the `experimental_update_live_dimensions` method.')
      raise NotImplementedError(msg)
    new_live_dimensions = f.experimental_update_live_dimensions(
        self.axis_mask, **kwargs)
    if all(tf.get_static_value(new_live_dimensions)):
      # Special-case a bijector (direction) that knows that the result
      # of the projection will be a full space
      return 0, FullSpace()
    else:
      return 0, AxisAlignedSpace(new_live_dimensions)

  def transform_coordinatewise(self, x, f, **kwargs):
    # TODO(pravnar): compute the derivative of f along x along the
    # live dimensions.
    raise NotImplementedError


def jacobian_determinant(x, f, **kwargs):
  return f.forward_log_det_jacobian(x, **kwargs)


class FullSpace(TangentSpace):
  """Tangent space of M for distributions supported on all of R^n.

  This subclass covers full-rank distributions on n-dimensional
  manifolds. In this common case we can take the basis to be the
  standard basis for R^n, so we need not explicitly represent it
  at all.
  """

  def transform_general(self, x, f, **kwargs):
    """If the bijector is weird, fall back to the general case."""
    as_general_space = GeneralSpace(unit_basis(), 1)
    return as_general_space.transform_general(x, f, **kwargs)

  def transform_dimension_preserving(self, x, f, **kwargs):
    return jacobian_determinant(x, f, **kwargs), FullSpace()

  def transform_projection(self, x, f, **kwargs):
    return AxisAlignedSpace(tf.ones_like(x)).transform_projection(
        x, f, **kwargs)


def volume_coefficient(basis):
  return tf.multiply(
      tf.linalg.logdet(tf.linalg.matmul(basis, basis, transpose_b=True)), 0.5)


class GeneralSpace(TangentSpace):
  """Arbitrary tangent space when no more-efficient special case applies."""

  def __init__(self, basis, computed_volume=None):
    self.basis = basis
    if computed_volume is None:
      computed_volume = volume_coefficient(basis)
    self.volume = computed_volume

  def transform_general(self, x, f, **kwargs):
    raise NotImplementedError


class ZeroSpace(TangentSpace):
  """Tangent space of M for discrete distributions.

  In this special case the tangent space is 0-dimensional, and the
  basis is represented by a 0x0 matrix, which gives 0 as the density
  correction term.

  """

  def transform_general(self, x, f, **kwargs):
    del x, f
    return 0, ZeroSpace()


class UnspecifiedTangentSpaceError(Exception):
  """An exception raised when a tangent space has not been specified."""

  def __init__(self):
    message = ('Please specify one of the tangent spaces defined at '
               'tensorflow_probability.python.experimental.tangent_spaces.')
    super().__init__(message)
