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

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.linalg import linear_operator_row_block as lorb
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'AxisAlignedSpace',
    'FullSpace',
    'GeneralSpace',
    'SphericalSpace',
    'TangentSpace',
    'UnspecifiedTangentSpaceError',
    'ZeroSpace',
]

JAX_MODE = False
NUMPY_MODE = False
TF_MODE = not (JAX_MODE or NUMPY_MODE)


def _jvp(f, x, b):
  """Computes jvp of `f` with respect to `x`."""
  broadcast_shape = ps.broadcast_shape(ps.shape(x), ps.shape(b))
  b = tf.broadcast_to(b, broadcast_shape)
  if JAX_MODE:
    import jax  # pylint:disable=g-import-not-at-top
    return jax.jvp(f.forward, (x,), (b,))[1]
  elif TF_MODE:
    @tf.function
    def jvp(b):
      with tf.autodiff.ForwardAccumulator(primals=x, tangents=b) as acc:
        y = f.forward(x)
      return acc.jvp(y)
    return jvp(b)


def _elementwise_jvp(f, x):
  """Returns the diagonal of the jacobian of an elementwise `f`."""
  # If `f` acts elementwise, then multiplying by the Jacobian by
  # the ones vector will retrieve the diagonal.
  return _jvp(f, x, tf.ones_like(x))


def _compute_new_basis(f, x, basis):
  # TODO(b/197680518): Add special handling for different kind of bases
  # and `LinearOperator`s in order to be more efficient. For instance, we may be
  # able to avoid densifying the basis in some circumstances.
  def compute_basis(b):
    return _jvp(f, x, b)
  return tf.vectorized_map(compute_basis, basis.to_dense())


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
    x = tf.convert_to_tensor(x)
    return self._transform_general(x, f, **kwargs)

  def _transform_general(self, x, f, **kwargs):
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
    return self._transform_dimension_preserving(x, f, **kwargs)

  def _transform_dimension_preserving(self, x, f, **kwargs):
    return self._transform_general(x, f, **kwargs)

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
    return self._transform_projection(x, f, **kwargs)

  def _transform_projection(self, x, f, **kwargs):
    return self._transform_general(x, f, **kwargs)

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
    x = nest_util.convert_to_nested_tensor(x, dtype=f.dtype)
    return self._transform_coordinatewise(x, f, **kwargs)

  def _transform_coordinatewise(self, x, f, **kwargs):
    return self._transform_dimension_preserving(x, f, **kwargs)

# TODO(b/197680518): Ensure that these methods are implemented.


class Basis:
  """Represents the basis of a `TangentSpace`."""

  def to_dense(self):
    """Returns densified version of this Basis.

    Returns:
      Tensor of shape `[N, B1, ..., Bn, D1, ... Dk], where
      `N` is the number of bases elements, `Bi` are possible
      batch dimensions, and `Di` are the dimensions of the
      tangent space (i.e. `R^{D1 x ... x Dk}`).
    """
    return self._to_dense()

  def to_dense_matrix(self, event_ndims):
    """Returns densified version of this Basis as a matrix.

    Args:
      event_ndims: Python `int` representing the number of right most dimensions
        that are the dimensions of the tangent space.
    Returns:
      Tensor of shape `[B1, ..., Bn, N, F], where
      `N` is the number of bases elements, `Bi` are possible
      batch dimensions, and `F` are the last `event_ndims` dimensions
      reshaped as one dimension.
    """
    return self._to_dense_matrix(event_ndims)

  def _to_dense(self):
    raise NotImplementedError

  def _to_dense_matrix(self, event_ndims):
    dense_tensor = self._to_dense()
    return _reshape_to_matrix(dense_tensor, event_ndims)


class LinearOperatorBasis(Basis):
  """Basis parameterized by a `LinearOperator`."""

  def __init__(self, basis_linop):
    self._basis_linop = basis_linop

  def _to_dense(self):
    dense = self._basis_linop.to_dense()
    return distribution_util.move_dimension(dense, -2, 0)

  def _to_dense_matrix(self, event_ndims):
    del event_ndims
    return self._basis_linop.to_dense()

  @property
  def basis_linop(self):
    return self._basis_linop


class DenseBasis(Basis):
  """Basis parameterized by a dense tensor."""

  def __init__(self, basis_tensor):
    self.basis_tensor = tensor_util.convert_nonref_to_tensor(
        basis_tensor, name='basis_tensor')

  def _to_dense(self):
    return tf.convert_to_tensor(self.basis_tensor)


class FullUnitBasis(Basis):
  """Represents a full axis aligned unit basis."""

  def __init__(self, event_shape, dtype):
    """Initialize a basis of axis aligned unit vectors.

    Args:
      event_shape: object representing the shape of the ambient space;
        convertible to `tf.TensorShape`.
      dtype: `Dtype` of this basis.
    """
    self._event_shape = event_shape
    self._dtype = dtype

  def _to_dense(self):
    size = tensorshape_util.num_elements(self._event_shape)
    result = tf.eye(size, dtype=self._dtype)
    return tf.reshape(
        result,
        ps.concat([[size], self._event_shape], axis=0))


class FullUnitBasisOn(Basis):
  """Represents an axis aligned unit basis on a masked portion of the space."""

  def __init__(self, axis_mask):
    self.axis_mask = axis_mask


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
    self._axis_mask = axis_mask

  def _transform_general(self, x, f, **kwargs):
    as_general_space = GeneralSpace(FullUnitBasisOn(self._axis_mask), 1)
    return as_general_space.transform_general(x, f, **kwargs)

  def _transform_projection(self, x, f, **kwargs):
    if not hasattr(f, 'experimental_update_live_dimensions'):
      msg = ('When calling `transform_projection` the Bijector must implement '
             'the `experimental_update_live_dimensions` method.')
      raise NotImplementedError(msg)
    new_live_dimensions = f.experimental_update_live_dimensions(
        self._axis_mask, **kwargs)
    if all(tf.get_static_value(new_live_dimensions)):
      # Special-case a bijector (direction) that knows that the result
      # of the projection will be a full space
      return 0, FullSpace()
    else:
      return 0, AxisAlignedSpace(new_live_dimensions)

  def _transform_coordinatewise(self, x, f, **kwargs):
    # TODO(b/197680518): compute the derivative of f along x along the
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

  def _transform_general(self, x, f, **kwargs):
    """If the bijector is weird, fall back to the general case."""
    as_general_space = GeneralSpace(
        FullUnitBasis([x.shape[-1]], dtype=f.dtype), 1)
    return as_general_space.transform_general(x, f, **kwargs)

  def _transform_dimension_preserving(self, x, f, **kwargs):
    return jacobian_determinant(x, f, **kwargs), FullSpace()

  def _transform_projection(self, x, f, **kwargs):
    return AxisAlignedSpace(tf.ones_like(x)).transform_projection(
        x, f, **kwargs)


def volume_coefficient(basis):
  result = 0.5 * tf.linalg.logdet(
      tf.linalg.matmul(basis, basis, transpose_b=True))
  return result


class GeneralSpace(TangentSpace):
  """Arbitrary tangent space when no more-efficient special case applies."""

  def __init__(self, basis, computed_log_volume=None):
    """Initialize a GeneralSpace with a basis.

    Args:
      basis: `Basis` object representing a basis of shape
        `[N, B1, ..., Bk, D1, ... Dl]`, where `N` is the
        number of bases vectors, `Bi` are batch dimensions and `Di` refer to
        the dimensions that this basis lives in.
      computed_log_volume: Optional `Tensor` of shape `[B1, ..., Bk]`. Computed
        log-volume coefficient.
        Default `None`.
    """
    self.basis = basis
    self.computed_log_volume = computed_log_volume

  def _transform_from_basis(self, new_basis, event_ndims):
    result = self.computed_log_volume
    if self.computed_log_volume is None:
      basis_matrix = self.basis.to_dense_matrix(event_ndims)
      result = volume_coefficient(basis_matrix)
    new_log_volume = volume_coefficient(new_basis.to_dense_matrix(event_ndims))
    result = new_log_volume - result
    return result, GeneralSpace(
        new_basis, computed_log_volume=new_log_volume)

  def _transform_general(self, x, f, event_ndims=None, **kwargs):
    # TODO(b/197680518): Clean up and extend the following code:
    # 1) Add Multipart Bijector Support.
    if NUMPY_MODE:
      raise ValueError('`transform_general` not available in Numpy')

    new_basis_tensor = _compute_new_basis(f, x, self.basis)
    new_basis = DenseBasis(new_basis_tensor)
    if event_ndims is None:
      event_ndims = f.forward_min_event_ndims
    return self._transform_from_basis(new_basis, event_ndims)

  def _transform_coordinatewise(self, x, f, **kwargs):
    diag_jacobian = self._elementwise_jvp(f, x)
    diag_linop = tf.linalg.LinearOperatorDiag(diag_jacobian)
    if isinstance(self.basis, LinearOperatorBasis):
      new_basis = LinearOperatorBasis(self.basis.basis_linop @ diag_linop)
    else:
      new_basis = LinearOperatorBasis(
          tf.linalg.LinearOperatorFullMatrix(
              self.basis.to_dense_matrix() * diag_jacobian[..., tf.newaxis, :]))
    if event_ndims is None:
      event_ndims = f.forward_min_event_ndims
    return self._transform_from_basis(new_basis, event_ndims)


class ZeroSpace(TangentSpace):
  """Tangent space of M for discrete distributions.

  In this special case the tangent space is 0-dimensional, and the
  basis is represented by a 0x0 matrix, which gives 0 as the density
  correction term.

  """

  def _transform_general(self, x, f, **kwargs):
    del x, f
    return 0, ZeroSpace()


class SphericalSpace(TangentSpace):
  """Tangent space of M for Spherical distributions in R^n."""

  def compute_spherical_basis(self, x):
    """Returns a `Basis` representing the tangent space of the n-sphere."""
    # TODO(b/197680518): Is there a cheaper way to represent this basis /
    # compute JVPs with it (perhaps we don't need to represent this dense
    # basis when doing transformations)?

    # Given the Hairy Ball Theorem, we can't find a non-zero smooth vector field
    # for an n-sphere, for even n.
    # Thus we exclude the north pole of the sphere, and then add a basis for
    # this at the end.

    # Choose the zero vector when `x` is the north pole. This is not the north
    # pole, but it is safe for all further calculations.
    is_north_pole = tf.math.equal(x[..., -1], 1.)
    safe_x = tf.where(is_north_pole[..., tf.newaxis], tf.zeros_like(x), x)

    # The stereographic projection is a diffeomorphism from the n-sphere without
    # the north pole to R^{n}. We can find a basis `B` at `stereographic_x`
    # (the unit basis), and then compute B* = f_*^-1(B), where f_*^-1 is the
    # pushforward of f^-1, and f^-1 is the inverse stereographic projection.
    # The pushforward is
    # just the differential, or in otherwords the the Jacobian matrix of f_*^-1.

    # Using the derivation in `tfb.UnitVector`, we have that
    # J_{i, i} = 2 * (s^2 - 2z_i^2 + 1) / (s^2 + 1)^2
    # J_{i, j} = 4z_iz_j/(s^2 + 1)^2 for 0 <= i, j <= n - 1
    # J_{n, i} = 4z_i/(s^2 + 1)^2
    # where z_i are the stereographic coordinates, and s^2 is the sum of those
    # coordinates squared.
    # Factoring out 1 / (s^2 + 1)^2, we have that the last row is just 4z, and
    # that the rest of the matrix is 2(s^2 + 1)I - 4zz^T.

    # Finally rewriting all this in the original coordinate system,
    # we have (1 - x_n) I - yy^T for the first n - 1 rows, and (1 - x_n) * y for
    # the last row, where y omits the last coordinate of x. We then transpose
    # this to get the number of bases vectors as the first dimension, and
    # rescale by 1 / (1 - x_n) in order to get bases with unit volume.
    # Note: When we are at the north pole, due to computing safe_x, we have
    # x_n = 0 and y = 0, which results in the identity matrix, which is the
    # correct basis.
    y = safe_x[..., :-1]

    base_operator = tf.linalg.LinearOperatorIdentity(
        ps.shape(y)[-1], batch_shape=ps.shape(x)[:-1], dtype=x.dtype)
    block1 = tf.linalg.LinearOperatorLowRankUpdate(
        base_operator=base_operator,
        u=y[..., tf.newaxis],
        diag_update=-tf.math.reciprocal(1. - safe_x[..., -1:]))
    block2 = tf.linalg.LinearOperatorFullMatrix(y[..., tf.newaxis])
    sphere_basis_linop = lorb.LinearOperatorRowBlock([block1, block2])
    return LinearOperatorBasis(sphere_basis_linop)

  def _transform_general(self, x, f, event_ndims=None, **kwargs):
    basis = self.compute_spherical_basis(x)
    new_basis = _compute_new_basis(f, x, basis)
    if event_ndims is None:
      event_ndims = f.forward_min_event_ndims
    new_log_volume = volume_coefficient(
        distribution_util.move_dimension(new_basis, 0, -2))
    # The original basis has 0 log_volume.
    return new_log_volume, GeneralSpace(
        DenseBasis(new_basis), computed_log_volume=new_log_volume)


class ProbabilitySimplexSpace(TangentSpace):
  """Tangent space of M for Simplex distributions in R^n."""

  def compute_basis(self, x):
    """Returns a `TangentSpace` of a n-simplex."""
    # The tangent space of the simplex satisfies `{x | <1, x> = 0}`, where `1`
    # is the vector of all `1`s. This can be seen by the fact that `1` is
    # orthogonal to the unit simplex.
    # We can do this by using the basis:  e_i - e_n, 1 <= i <= n - 1. For n = 4,
    # this looks like:
    # [[1, 0., 0., -1],
    #  [0, 1., 0., -1],
    #  [0, 0., 1., -1]]
    dim = ps.shape(x)[-1]
    block1 = tf.linalg.LinearOperatorIdentity(num_rows=dim - 1, dtype=x.dtype)
    block2 = tf.linalg.LinearOperatorFullMatrix(
        -tf.ones([dim - 1, 1], dtype=x.dtype))
    simplex_basis_linop = lorb.LinearOperatorRowBlock([block1, block2])
    return LinearOperatorBasis(simplex_basis_linop)

  def _transform_general(self, x, f, event_ndims=None, **kwargs):
    basis = self.compute_basis(x)
    # Note that B @ B.T results in the matrix I + 11^T, where 1 is the vector of
    # all ones. By the matrix determinant lemma we have det(I + 11^T) = n + 1,
    # or the dimension of the ambient space.
    dim = ps.shape(x)[-1]
    result = dtype_util.as_numpy_dtype(x.dtype)(0.5 * np.log(dim))
    new_basis_tensor = _compute_new_basis(f, x, basis)
    if event_ndims is None:
      event_ndims = f.forward_min_event_ndims
    new_log_volume = volume_coefficient(
        distribution_util.move_dimension(new_basis_tensor, 0, -2))
    result = new_log_volume - result
    return new_log_volume, GeneralSpace(
        DenseBasis(new_basis_tensor), computed_log_volume=new_log_volume)

  def _transform_coordinatewise(self, x, f, **kwargs):
    # Compute the diagonal. New matrix is Linop that we can easily write.
    dim = ps.shape(x)[-1]
    diag_jacobian = _elementwise_jvp(f, x)
    # Multiplying the basis written in block form as [I, 1] by the diagonal
    # results in this operator:
    block1 = tf.linalg.LinearOperatorDiag(diag_jacobian[..., :-1])
    block2 = tf.linalg.LinearOperatorFullMatrix(
        diag_jacobian[..., -1] * tf.ones([dim - 1, 1], dtype=x.dtype))
    linop = lorb.LinearOperatorRowBlock([block1, block2])
    # The volume can be calculated again by the matrix determinant lemma:
    # det(D**2 + d_n**2 11^T) = (1 + d_n**2 1(D^-1)**21^T) * det(D**2)
    # = (\sum d_i**-2) * \prod d_i**2
    log_diag_jacobian = tf.math.log(tf.math.abs(diag_jacobian))
    log_volume = tf.math.reduce_sum(log_diag_jacobian, axis=-1)
    log_volume = log_volume + 0.5 * tf.math.reduce_logsumexp(
        -2. * log_diag_jacobian, axis=-1)
    return log_volume, GeneralSpace(
        LinearOperatorBasis(linop), computed_log_volume=log_volume)


class UnspecifiedTangentSpaceError(Exception):
  """An exception raised when a tangent space has not been specified."""

  def __init__(self):
    message = ('Please specify one of the tangent spaces defined at '
               'tensorflow_probability.python.experimental.tangent_spaces.')
    super().__init__(message)


def _reshape_to_matrix(basis_tensor, event_ndims):
  # Reshape basis so that there is only one ambient dimension.
  basis_tensor = ps.reshape(
      basis_tensor, ps.concat(
          [ps.shape(basis_tensor)[
              :ps.rank(basis_tensor) - event_ndims], [-1]], axis=0))
  if event_ndims == 0:
    basis_tensor = basis_tensor[..., tf.newaxis]
  # Finally move the basis vector dimension to the end so we have shape [B1,
  # ..., Bk, N, D].
  return distribution_util.move_dimension(basis_tensor, 0, -2)
