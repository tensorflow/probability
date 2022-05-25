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
"""The SchurComplement kernel."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'SchurComplement',
]


def _validate_arg_if_not_none(arg, assertion, validate_args):
  if arg is None:
    return arg
  with tf.control_dependencies([assertion(arg)] if validate_args else []):
    result = tf.identity(arg)
  return result


def _add_diagonal_shift(matrix, shift):
  return tf.linalg.set_diag(
      matrix, tf.linalg.diag_part(matrix) + shift, name='add_diagonal_shift')


def _compute_divisor_matrix(
    base_kernel,
    diag_shift,
    fixed_inputs):
  """Compute the the modified kernel with respect to the fixed inputs."""
  divisor_matrix = base_kernel.matrix(fixed_inputs, fixed_inputs)
  if diag_shift is not None:
    diag_shift = tf.convert_to_tensor(diag_shift)
    broadcast_shape = distribution_util.get_broadcast_shape(
        divisor_matrix, diag_shift[..., tf.newaxis, tf.newaxis])
    divisor_matrix = tf.broadcast_to(divisor_matrix, broadcast_shape)
    divisor_matrix = _add_diagonal_shift(
        divisor_matrix, diag_shift[..., tf.newaxis])
  return divisor_matrix


class SchurComplement(psd_kernel.AutoCompositeTensorPsdKernel):
  """The SchurComplement kernel.

  Given a block matrix `M = [[A, B], [C, D]]`, the Schur complement of D in M is
  written `M / D = A - B @ Inverse(D) @ C`.

  This class represents a PositiveSemidefiniteKernel whose behavior is as
  follows. We compute a matrix, analogous to `D` in the above definition, by
  calling `base_kernel.matrix(fixed_inputs, fixed_inputs)`. Then given new input
  locations `x` and `y`, we can construct the remaining pieces of `M` above, and
  compute the Schur complement of `D` in `M` (see Mathematical Details, below).

  Notably, this kernel uses a bijector (Invert(CholeskyOuterProduct)), as an
  intermediary for the requisite matrix solve, which means we get a caching
  benefit after the first use.

  ### Mathematical Details

  Suppose we have a kernel `k` and some fixed collection of inputs
  `Z = [z0, z1, ..., zN]`. Given new inputs `x` and `y`, we can form a block
  matrix

   ```none
     M = [
       [k(x, y), k(x, z0), ..., k(x, zN)],
       [k(z0, y), k(z0, z0), ..., k(z0, zN)],
       ...,
       [k(zN, y), k(z0, zN), ..., k(zN, zN)],
     ]
   ```

  We might write this, so as to emphasize the block structure,

   ```none
     M = [
       [xy, xZ],
       [yZ^T, ZZ],
     ],

     xy = [k(x, y)]
     xZ = [k(x, z0), ..., k(x, zN)]
     yZ = [k(y, z0), ..., k(y, zN)]
     ZZ = "the matrix of k(zi, zj)'s"
   ```

  Then we have the definition of this kernel's apply method:

  `schur_comp.apply(x, y) = xy - xZ @ ZZ^{-1} @ yZ^T`

  and similarly, if x and y are collections of inputs.

  As with other PSDKernels, the `apply` method acts as a (possibly
  vectorized) scalar function of 2 inputs. Given a single `x` and `y`,
  `apply` will yield a scalar output. Given two (equal size!) collections `X`
  and `Y`, it will yield another (equal size!) collection of scalar outputs.

  ### Examples

  Here's a simple example usage, with no particular motivation.

  ```python
  from tensorflow_probability.math import psd_kernels

  base_kernel = psd_kernels.ExponentiatedQuadratic(amplitude=np.float64(1.))
  # 3 points in 1-dimensional space (shape [3, 1]).
  z = [[0.], [3.], [4.]]

  schur_kernel = psd_kernels.SchurComplement(
      base_kernel=base_kernel,
      fixed_inputs=z)

  # Two individual 1-d points
  x = [1.]
  y = [2.]
  print(schur_kernel.apply(x, y))
  # ==> k(x, y) - k(x, z) @ Inverse(k(z, z)) @ k(z, y)
  ```

  A more motivating application of this kernel is in constructing a Gaussian
  process that is conditioned on some observed data.

  ```python
  from tensorflow_probability import distributions as tfd
  from tensorflow_probability.math import psd_kernels

  base_kernel = psd_kernels.ExponentiatedQuadratic(amplitude=np.float64(1.))
  observation_index_points = np.random.uniform(-1., 1., [50, 1])
  observations = np.sin(2 * np.pi * observation_index_points[..., 0])

  posterior_kernel = psd_kernels.SchurComplement(
      base_kernel=base_kernel,
      fixed_inputs=observation_index_points)

  # Assume we use a zero prior mean, and compute the posterior mean.
  def posterior_mean_fn(x):
    k_x_obs_linop = tf.linalg.LinearOperatorFullMatrix(
        base_kernel.matrix(x, observation_index_points))
    chol_linop = tf.linalg.LinearOperatorLowerTriangular(
        posterior_kernel.divisor_matrix_cholesky())

    return k_x_obs_linop.matvec(
        chol_linop.solvevec(
            chol_linop.solvevec(observations),
            adjoint=True))

  # Construct the GP posterior distribution at some new points.
  gp_posterior = tfp.distributions.GaussianProcess(
      index_points=np.linspace(-1., 1., 100)[..., np.newaxis],
      kernel=posterior_kernel,
      mean_fn=posterior_mean_fn)

  # Draw 5 samples on the above 100-point grid
  samples = gp_posterior.sample(5)
  ```

  """
  # pylint:disable=invalid-name

  @deprecation.deprecated_args(
      '2022-06-23',
      ('The `fixed_inputs_mask` flag is deprecated; instead use '
       '`fixed_inputs_is_missing` (with the opposite sense).'),
      'fixed_inputs_mask')
  def __init__(self,
               base_kernel,
               fixed_inputs,
               fixed_inputs_mask=None,
               fixed_inputs_is_missing=None,
               diag_shift=None,
               cholesky_fn=None,
               validate_args=False,
               name='SchurComplement',
               _precomputed_divisor_matrix_cholesky=None):
    """Construct a SchurComplement kernel instance.

    Args:
      base_kernel: A `PositiveSemidefiniteKernel` instance, the kernel used to
        build the block matrices of which this kernel computes the Schur
        complement.
      fixed_inputs: A Tensor, representing a collection of inputs. The Schur
        complement that this kernel computes comes from a block matrix, whose
        bottom-right corner is derived from `base_kernel.matrix(fixed_inputs,
        fixed_inputs)`, and whose top-right and bottom-left pieces are
        constructed by computing the base_kernel at pairs of input locations
        together with these `fixed_inputs`. `fixed_inputs` is allowed to be an
        empty collection (either `None` or having a zero shape entry), in which
        case the kernel falls back to the trivial application of `base_kernel`
        to inputs. See class-level docstring for more details on the exact
        computation this does; `fixed_inputs` correspond to the `Z` structure
        discussed there. `fixed_inputs` is assumed to have shape `[b1, ..., bB,
        N, f1, ..., fF]` where the `b`'s are batch shape entries, the `f`'s are
        feature_shape entries, and `N` is the number of fixed inputs. Use of
        this kernel entails a 1-time O(N^3) cost of computing the Cholesky
        decomposition of the k(Z, Z) matrix. The batch shape elements of
        `fixed_inputs` must be broadcast compatible with
        `base_kernel.batch_shape`.
      fixed_inputs_mask: Deprecated. A boolean Tensor of shape `[..., N]`. When
        `mask` is not None and an element of `mask` is `False`, this kernel
        will return values computed as if the divisor matrix did not contain the
        corresponding row or column.
      fixed_inputs_is_missing: A boolean Tensor of shape `[..., N]`.
        When `is_missing` is not None and an element of `mask` is `True`,
        this kernel will return values computed as if the divisor matrix did
        not contain the corresponding row or column.
      diag_shift: A floating point scalar to be added to the diagonal of the
        divisor_matrix before computing its Cholesky.
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
        in which case `make_cholesky_with_jitter_fn` is used with the `jitter`
        parameter.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance.
        Default value: `False`
      name: Python `str` name prefixed to Ops created by this class.
        Default value: `"SchurComplement"`
      _precomputed_divisor_matrix_cholesky: Internal parameter -- do not use.
    """
    parameters = dict(locals())

    # Delayed import to avoid circular dependency between `tfp.bijectors` and
    # `tfp.math`
    # pylint: disable=g-import-not-at-top
    from tensorflow_probability.python.bijectors import cholesky_outer_product
    from tensorflow_probability.python.bijectors import invert
    # pylint: enable=g-import-not-at-top
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [base_kernel,
           fixed_inputs,
           diag_shift,
           _precomputed_divisor_matrix_cholesky], tf.float32)
      self._base_kernel = base_kernel
      self._diag_shift = tensor_util.convert_nonref_to_tensor(
          diag_shift, dtype=dtype, name='diag_shift')
      self._fixed_inputs = tensor_util.convert_nonref_to_tensor(
          fixed_inputs, dtype=dtype, name='fixed_inputs')
      if ((fixed_inputs_mask is not None) and
          (fixed_inputs_is_missing is not None)):
        raise ValueError('Expected at most one of `fixed_inputs_mask` or '
                         '`fixed_inputs_is_missing`')
      self._fixed_inputs_mask = tensor_util.convert_nonref_to_tensor(
          fixed_inputs_mask, dtype=tf.bool, name='fixed_inputs_mask')
      self._fixed_inputs_is_missing = tensor_util.convert_nonref_to_tensor(
          fixed_inputs_is_missing,
          dtype=tf.bool, name='fixed_inputs_is_missing')
      self._cholesky_bijector = invert.Invert(
          cholesky_outer_product.CholeskyOuterProduct())
      self._precomputed_divisor_matrix_cholesky = _precomputed_divisor_matrix_cholesky
      if self._precomputed_divisor_matrix_cholesky is not None:
        self._precomputed_divisor_matrix_cholesky = tf.convert_to_tensor(
            self._precomputed_divisor_matrix_cholesky, dtype)
      if cholesky_fn is None:
        from tensorflow_probability.python.distributions import cholesky_util  # pylint:disable=g-import-not-at-top
        cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn()
      self._cholesky_fn = cholesky_fn
      self._cholesky_bijector = invert.Invert(
          cholesky_outer_product.CholeskyOuterProduct(cholesky_fn=cholesky_fn))

      super(SchurComplement, self).__init__(
          base_kernel.feature_ndims,
          dtype=dtype,
          name=name,
          parameters=parameters)

  @staticmethod
  @deprecation.deprecated_args(
      '2022-06-23',
      ('The `fixed_inputs_mask` flag is deprecated; instead use '
       '`fixed_inputs_is_missing` (with the opposite sense).'),
      'fixed_inputs_mask')
  def with_precomputed_divisor(
      base_kernel,
      fixed_inputs,
      fixed_inputs_mask=None,
      fixed_inputs_is_missing=None,
      diag_shift=None,
      cholesky_fn=None,
      validate_args=False,
      name='PrecomputedSchurComplement'):
    """Returns a `SchurComplement` with a precomputed divisor matrix.

    This method is the same as creating a `SchurComplement` kernel, but assumes
    that `fixed_inputs`, `diag_shift` and `base_kernel` are unchanging /
    not parameterized by any mutable state. We explicitly read / concretize
    these values when this method is called, since we can precompute some
    factorizations in order to speed up subsequent invocations of the kernel.

    WARNING: This method assumes passed in arguments are not parameterized
    by mutable state (`fixed_inputs`, `diag_shift` and `base_kernel`), and hence
    is not tape-safe.

    Args:
      base_kernel: A `PositiveSemidefiniteKernel` instance, the kernel used to
        build the block matrices of which this kernel computes the Schur
        complement.
      fixed_inputs: A Tensor, representing a collection of inputs. The Schur
        complement that this kernel computes comes from a block matrix, whose
        bottom-right corner is derived from `base_kernel.matrix(fixed_inputs,
        fixed_inputs)`, and whose top-right and bottom-left pieces are
        constructed by computing the base_kernel at pairs of input locations
        together with these `fixed_inputs`. `fixed_inputs` is allowed to be an
        empty collection (either `None` or having a zero shape entry), in which
        case the kernel falls back to the trivial application of `base_kernel`
        to inputs. See class-level docstring for more details on the exact
        computation this does; `fixed_inputs` correspond to the `Z` structure
        discussed there. `fixed_inputs` is assumed to have shape `[b1, ..., bB,
        N, f1, ..., fF]` where the `b`'s are batch shape entries, the `f`'s are
        feature_shape entries, and `N` is the number of fixed inputs. Use of
        this kernel entails a 1-time O(N^3) cost of computing the Cholesky
        decomposition of the k(Z, Z) matrix. The batch shape elements of
        `fixed_inputs` must be broadcast compatible with
        `base_kernel.batch_shape`.
      fixed_inputs_mask: Deprecated. A boolean Tensor of shape `[..., N]`. When
        `mask` is not None and an element of `mask` is `False`, the returned
        kernel will return values computed as if the divisor matrix did not
        contain the corresponding row or column.
      fixed_inputs_is_missing: A boolean Tensor of shape `[..., N]`.  When
        `is_missing` is not None and an element of `is_missing` is `True`, the
        returned kernel will return values computed as if the divisor matrix
        did not contain the corresponding row or column.
      diag_shift: A floating point scalar to be added to the diagonal of the
        divisor_matrix before computing its Cholesky.
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
        in which case `make_cholesky_with_jitter_fn` is used with the `jitter`
        parameter.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance.
        Default value: `False`
      name: Python `str` name prefixed to Ops created by this class.
        Default value: `"PrecomputedSchurComplement"`
    """
    dtype = dtype_util.common_dtype(
        [base_kernel, fixed_inputs, diag_shift], tf.float32)
    fixed_inputs = tf.convert_to_tensor(fixed_inputs, dtype)
    if ((fixed_inputs_mask is not None) and
        (fixed_inputs_is_missing is not None)):
      raise ValueError('Expected at most one of `fixed_inputs_mask` or '
                       '`fixed_inputs_is_missing`')
    if fixed_inputs_mask is not None:
      fixed_inputs_is_missing = ~tf.convert_to_tensor(
          fixed_inputs_mask, tf.bool)
    if fixed_inputs_is_missing is not None:
      fixed_inputs_is_missing = tf.convert_to_tensor(
          fixed_inputs_is_missing, tf.bool)
    if diag_shift is not None:
      diag_shift = tf.convert_to_tensor(diag_shift, dtype)

    if cholesky_fn is None:
      from tensorflow_probability.python.distributions import cholesky_util  # pylint:disable=g-import-not-at-top
      cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn()

    # TODO(b/196219597): Add a check to ensure that we have a `base_kernel`
    # that is explicitly concretized.
    divisor_matrix_cholesky = cholesky_fn(util.mask_matrix(
        _compute_divisor_matrix(base_kernel,
                                diag_shift=diag_shift,
                                fixed_inputs=fixed_inputs),
        is_missing=fixed_inputs_is_missing))

    schur_complement = SchurComplement(
        base_kernel=base_kernel,
        fixed_inputs=fixed_inputs,
        fixed_inputs_is_missing=fixed_inputs_is_missing,
        diag_shift=diag_shift,
        cholesky_fn=cholesky_fn,
        validate_args=validate_args,
        _precomputed_divisor_matrix_cholesky=divisor_matrix_cholesky,
        name=name)

    return schur_complement

  def _is_fixed_inputs_empty(self):
    # If fixed_inputs are `None` or have size 0, we consider this empty and fall
    # back to (cheaper) trivial behavior.
    if self._fixed_inputs is None:
      return True
    num_fixed_inputs = tf.compat.dimension_value(
        self._fixed_inputs.shape[-(self._base_kernel.feature_ndims + 1)])
    if num_fixed_inputs is not None and num_fixed_inputs == 0:
      return True
    return False

  def _get_fixed_inputs_is_missing(self):
    fixed_inputs_is_missing = self._fixed_inputs_is_missing
    if fixed_inputs_is_missing is not None:
      fixed_inputs_is_missing = tf.convert_to_tensor(fixed_inputs_is_missing)
    if self._fixed_inputs_mask is not None:
      fixed_inputs_is_missing = ~tf.convert_to_tensor(self._fixed_inputs_mask)
    return fixed_inputs_is_missing

  def _apply(self, x1, x2, example_ndims):
    # In the shape annotations below,
    #
    #  - x1 has shape B1 + E1 + F  (batch, example, feature),
    #  - x2 has shape B2 + E2 + F,
    #  - z refers to self.fixed_inputs, and has shape Bz + [ez] + F, ie its
    #    example ndims is exactly 1,
    #  - self.base_kernel has batch shape Bk,
    #  - bc(A, B, C) means "the result of broadcasting shapes A, B, and C".

    # Shape: bc(Bk, B1, B2) + bc(E1, E2)
    k12 = self.base_kernel.apply(x1, x2, example_ndims)
    if self._is_fixed_inputs_empty():
      return k12

    fixed_inputs = tf.convert_to_tensor(self._fixed_inputs)
    fixed_inputs_is_missing = self._get_fixed_inputs_is_missing()
    if fixed_inputs_is_missing is not None:
      fixed_inputs_is_missing = util.pad_shape_with_ones(
          fixed_inputs_is_missing, example_ndims, -2)

    # Shape: bc(Bk, B1, Bz) + E1 + [ez]
    k1z = self.base_kernel.tensor(x1, fixed_inputs,
                                  x1_example_ndims=example_ndims,
                                  x2_example_ndims=1)
    if fixed_inputs_is_missing is not None:
      k1z = tf.where(fixed_inputs_is_missing, tf.zeros([], k1z.dtype), k1z)

    # Shape: bc(Bk, B2, Bz) + E2 + [ez]
    k2z = self.base_kernel.tensor(x2, fixed_inputs,
                                  x1_example_ndims=example_ndims,
                                  x2_example_ndims=1)
    if fixed_inputs_is_missing is not None:
      k2z = tf.where(fixed_inputs_is_missing, tf.zeros([], k2z.dtype), k2z)

    # Shape: bc(Bz, Bk) + [ez, ez]
    div_mat_chol = self._divisor_matrix_cholesky(
        fixed_inputs=fixed_inputs)

    # Shape: bc(Bz, Bk) + [1, ..., 1] + [ez, ez]
    #                      `--------'
    #                          `-- (example_ndims - 1) ones
    # This reshape ensures that the batch shapes here align correctly with the
    # batch shape of k2z, below: `example_ndims` because E2 has rank
    # `example_ndims`, and "- 1" because one of the ez's here already "pushed"
    # the batch dims over by one.
    div_mat_chol = util.pad_shape_with_ones(div_mat_chol, example_ndims - 1, -3)
    div_mat_chol_linop = tf.linalg.LinearOperatorLowerTriangular(div_mat_chol)

    # Shape: bc(Bz, Bk, B2) + E1 + [ez]
    cholinv_kz1 = tf.linalg.matrix_transpose(
        div_mat_chol_linop.solve(k1z, adjoint_arg=True))
    # Shape: bc(Bz, Bk, B2) + E2 + [ez]
    cholinv_kz2 = tf.linalg.matrix_transpose(
        div_mat_chol_linop.solve(k2z, adjoint_arg=True))
    k1z_kzzinv_kz2 = tf.reduce_sum(cholinv_kz1 * cholinv_kz2, axis=-1)

    # Shape: bc(Bz, Bk, B1, B2) + bc(E1, E2)
    return k12 - k1z_kzzinv_kz2

  def _matrix(self, x1, x2):
    k12 = self.base_kernel.matrix(x1, x2)
    if self._is_fixed_inputs_empty():
      return k12

    fixed_inputs = tf.convert_to_tensor(self._fixed_inputs)
    fixed_inputs_is_missing = self._get_fixed_inputs_is_missing()
    if fixed_inputs_is_missing is not None:
      fixed_inputs_is_missing = fixed_inputs_is_missing[..., tf.newaxis, :]

    # Shape: bc(Bk, B1, Bz) + [e1] + [ez]
    k1z = self.base_kernel.matrix(x1, fixed_inputs)
    if fixed_inputs_is_missing is not None:
      k1z = tf.where(fixed_inputs_is_missing, tf.zeros([], k1z.dtype), k1z)

    # Shape: bc(Bk, B2, Bz) + [e2] + [ez]
    k2z = self.base_kernel.matrix(x2, fixed_inputs)
    if fixed_inputs_is_missing is not None:
      k2z = tf.where(fixed_inputs_is_missing, tf.zeros([], k2z.dtype), k2z)

    # Shape: bc(Bz, Bk) + [ez, ez]
    div_mat_chol = self._divisor_matrix_cholesky(
        fixed_inputs=fixed_inputs)

    div_mat_chol_linop = tf.linalg.LinearOperatorLowerTriangular(div_mat_chol)

    # Shape: bc(Bz, Bk, B2) + [ez] + [e1]
    cholinv_kz1 = div_mat_chol_linop.solve(k1z, adjoint_arg=True)
    # Shape: bc(Bz, Bk, B2) + [ez] + [e2]
    cholinv_kz2 = div_mat_chol_linop.solve(k2z, adjoint_arg=True)
    k1z_kzzinv_kz2 = tf.linalg.matmul(
        cholinv_kz1, cholinv_kz2, transpose_a=True)
    # Shape: bc(Bz, Bk, B1, B2) + [e1, e2]
    return k12 - k1z_kzzinv_kz2

  @property
  def fixed_inputs(self):
    return self._fixed_inputs

  @property
  def base_kernel(self):
    return self._base_kernel

  @property
  def diag_shift(self):
    return self._diag_shift

  @property
  def cholesky_fn(self):
    return self._cholesky_fn

  @property
  def cholesky_bijector(self):
    return self._cholesky_bijector

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        base_kernel=parameter_properties.BatchedComponentProperties(),
        fixed_inputs=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.base_kernel.feature_ndims + 1),
        fixed_inputs_mask=parameter_properties.ParameterProperties(
            event_ndims=1),
        fixed_inputs_is_missing=parameter_properties.ParameterProperties(
            event_ndims=1),
        diag_shift=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        _precomputed_divisor_matrix_cholesky=(
            parameter_properties.ParameterProperties(event_ndims=2)))

  def _divisor_matrix(self, fixed_inputs=None, fixed_inputs_is_missing=None):
    fixed_inputs = tf.convert_to_tensor(
        self._fixed_inputs if fixed_inputs is None else fixed_inputs)
    if fixed_inputs_is_missing is None:
      fixed_inputs_is_missing = self._get_fixed_inputs_is_missing()
    # NOTE: Replacing masked-out rows/columns of the divisor matrix with
    # rows/columns from the identity matrix is equivalent to using a divisor
    # matrix in which those rows and columns have been dropped.
    return util.mask_matrix(
        _compute_divisor_matrix(self._base_kernel,
                                diag_shift=self._diag_shift,
                                fixed_inputs=fixed_inputs),
        is_missing=fixed_inputs_is_missing)

  def divisor_matrix(self):
    return self._divisor_matrix()

  def _divisor_matrix_cholesky(
      self,
      fixed_inputs=None,
      fixed_inputs_is_missing=None):
    if self._precomputed_divisor_matrix_cholesky is not None:
      return self._precomputed_divisor_matrix_cholesky
    return self.cholesky_bijector.forward(
        self._divisor_matrix(fixed_inputs, fixed_inputs_is_missing))

  def divisor_matrix_cholesky(
      self,
      fixed_inputs=None,
      fixed_inputs_mask=None,
      fixed_inputs_is_missing=None):
    if self._precomputed_divisor_matrix_cholesky is not None:
      return self._precomputed_divisor_matrix_cholesky
    if ((fixed_inputs_mask is not None) and
        (fixed_inputs_is_missing is not None)):
      raise ValueError('Expected only one of `fixed_inputs_mask` or '
                       '`fixed_inputs_is_missing` to be set.')
    if fixed_inputs_mask is not None:
      fixed_inputs_is_missing = ~tf.convert_to_tensor(fixed_inputs_mask)
    return self._divisor_matrix_cholesky(fixed_inputs, fixed_inputs_is_missing)
