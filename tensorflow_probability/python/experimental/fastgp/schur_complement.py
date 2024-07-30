# Copyright 2024 The TensorFlow Probability Authors.
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

import jax
import jax.numpy as jnp
from tensorflow_probability.python.experimental.fastgp import mbcg
from tensorflow_probability.substrates.jax.bijectors import softplus
from tensorflow_probability.substrates.jax.internal import distribution_util
from tensorflow_probability.substrates.jax.internal import dtype_util
from tensorflow_probability.substrates.jax.internal import nest_util
from tensorflow_probability.substrates.jax.internal import parameter_properties
from tensorflow_probability.substrates.jax.math.psd_kernels import positive_semidefinite_kernel
from tensorflow_probability.substrates.jax.math.psd_kernels.internal import util


__all__ = [
    'SchurComplement',
]


def _add_diagonal_shift(matrix, shift):
  return matrix + shift[..., jnp.newaxis] * jnp.eye(
      matrix.shape[-1], dtype=matrix.dtype)


def _compute_divisor_matrix(
    base_kernel, diag_shift, fixed_inputs):
  """Compute the modified kernel with respect to the fixed inputs."""
  divisor_matrix = base_kernel.matrix(fixed_inputs, fixed_inputs)
  if diag_shift is not None:
    broadcast_shape = distribution_util.get_broadcast_shape(
        divisor_matrix, diag_shift[..., jnp.newaxis, jnp.newaxis]
    )
    divisor_matrix = jnp.broadcast_to(divisor_matrix, broadcast_shape)
    divisor_matrix = _add_diagonal_shift(
        divisor_matrix, diag_shift[..., jnp.newaxis])
  return divisor_matrix


class SchurComplement(
    positive_semidefinite_kernel.AutoCompositeTensorPsdKernel
):
  """The fast SchurComplement kernel.

  See tfp.math.psd_kernels.SchurComplement for more details.

  """

  def __init__(self,
               base_kernel,
               fixed_inputs,
               preconditioner_fn,
               diag_shift=None):
    """Construct a SchurComplement kernel instance.

    Args:
      base_kernel: A `PositiveSemidefiniteKernel` instance, the kernel used to
        build the block matrices of which this kernel computes the Schur
        complement.
      fixed_inputs: A (nested) Tensor, representing a collection of inputs. The
        Schur complement that this kernel computes comes from a block matrix,
        whose bottom-right corner is derived from
        `base_kernel.matrix(fixed_inputs, fixed_inputs)`, and whose top-right
        and bottom-left pieces are constructed by computing the base_kernel
        at pairs of input locations together with these `fixed_inputs`.
        `fixed_inputs` is allowed to be an empty collection (either `None` or
        having a zero shape entry), in which case the kernel falls back to
        the trivial application of `base_kernel` to inputs. See class-level
        docstring for more details on the exact computation this does;
        `fixed_inputs` correspond to the `Z` structure discussed there.
        `fixed_inputs` (or each of its nested components) is assumed to have
        shape `[b1, ..., bB, N, f1, ..., fF]` where the `b`'s are batch shape
        entries, the `f`'s are feature_shape entries, and `N` is the number
        of fixed inputs.
      preconditioner_fn: A function that applies an invertible linear
        transformation to its input, designed to increase the rate of
        convergence by decreasing the condition number. The preconditioner_fn
        should act like left application of an n by n linear operator, i.e.
        preconditioner_fn(n x m) should have shape n x m.
      diag_shift: A floating point scalar to be added to the diagonal of the
        divisor_matrix.
    """
    # TODO(srvasude): Support masking.
    parameters = dict(locals())

    if jax.tree_util.treedef_is_leaf(
        jax.tree_util.tree_structure(base_kernel.feature_ndims)):
      dtype = dtype_util.common_dtype(
          [base_kernel, fixed_inputs],
          dtype_hint=nest_util.broadcast_structure(
              base_kernel.feature_ndims, jnp.float32
          ),
      )
    else:
      # If the fixed inputs are not nested, we assume they are of the same
      # float dtype as the remaining parameters.
      dtype = dtype_util.common_dtype(
          [base_kernel, fixed_inputs, diag_shift], jnp.float32
      )

    self._base_kernel = base_kernel
    self._diag_shift = diag_shift
    self._fixed_inputs = fixed_inputs
    self._preconditioner_fn = preconditioner_fn

    super(SchurComplement, self).__init__(
        base_kernel.feature_ndims,
        dtype=dtype,
        name='SchurComplement',
        parameters=parameters)

  def _is_fixed_inputs_empty(self):
    # If fixed_inputs are `None` or have size 0, we consider this empty and fall
    # back to (cheaper) trivial behavior.
    if self._fixed_inputs is None:
      return True
    num_fixed_inputs = jax.tree_util.tree_map(
        lambda t, nd: t.shape[-(nd + 1)],
        self._fixed_inputs, self._base_kernel.feature_ndims)
    if all(n is not None and n == 0 for n in jax.tree_util.tree_leaves(
        num_fixed_inputs)):
      return True
    return False

  def _apply(self, x1, x2, example_ndims):
    # In the shape annotations below,
    #
    #  - x1 has shape B1 + E1 + F  (batch, example, feature),
    #  - x2 has shape B2 + E2 + F,
    #  - z refers to self.fixed_inputs, and has shape Bz + [ez] + F, ie its
    #    example ndims is exactly 1,
    #  - self.base_kernel has batch shape Bk,
    #  - bc(A, B, C) means "the result of broadcasting shapes A, B, and C".
    fixed_inputs = self._fixed_inputs

    # Shape: bc(Bk, B1, B2) + bc(E1, E2)
    k12 = self.base_kernel.apply(x1, x2, example_ndims)
    if self._is_fixed_inputs_empty():
      return k12

    # Shape: bc(Bk, B1, Bz) + E1 + [ez]
    k1z = self.base_kernel.tensor(x1, fixed_inputs,
                                  x1_example_ndims=example_ndims,
                                  x2_example_ndims=1)
    # Shape: bc(Bk, B2, Bz) + E2 + [ez]
    k2z = self.base_kernel.tensor(x2, fixed_inputs,
                                  x1_example_ndims=example_ndims,
                                  x2_example_ndims=1)
    k2z = jnp.reshape(k2z, [-1, k2z.shape[-1]])
    # Shape: bc(Bz, Bk) + [ez, ez]
    div_mat = self._divisor_matrix(fixed_inputs=fixed_inputs)

    div_mat = util.pad_shape_with_ones(div_mat, example_ndims - 1, -3)

    kzzinv_kz2, _ = mbcg.modified_batched_conjugate_gradients(
        lambda x: div_mat @ x,
        jnp.transpose(k2z),
        preconditioner_fn=self.preconditioner_fn,
        max_iters=20)
    kzzinv_kz2 = jnp.transpose(kzzinv_kz2)
    k1z_kzzinv_kz2 = jnp.sum(k1z * kzzinv_kz2, axis=-1)

    return k12 - k1z_kzzinv_kz2

  def _matrix(self, x1, x2):
    k12 = self.base_kernel.matrix(x1, x2)
    if self._is_fixed_inputs_empty():
      return k12
    fixed_inputs = self._fixed_inputs

    # Shape: bc(Bk, B1, Bz) + [e1] + [ez]
    k1z = self.base_kernel.matrix(x1, fixed_inputs)

    # Shape: bc(Bk, B2, Bz) + [e2] + [ez]
    k2z = self.base_kernel.matrix(x2, fixed_inputs)

    # Shape: bc(Bz, Bk) + [ez, ez]
    div_mat = self._divisor_matrix(fixed_inputs=fixed_inputs)

    kzzinv_kz2, _ = mbcg.modified_batched_conjugate_gradients(
        lambda x: div_mat @ x,
        jnp.transpose(k2z),
        preconditioner_fn=self.preconditioner_fn,
        max_iters=20)

    k1z_kzzinv_kz2 = k1z @ kzzinv_kz2
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
  def preconditioner_fn(self):
    return self._preconditioner_fn

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        base_kernel=parameter_properties.BatchedComponentProperties(),
        fixed_inputs=parameter_properties.ParameterProperties(
            event_ndims=lambda self: jax.tree_util.tree_map(  # pylint: disable=g-long-lambda
                lambda nd: nd + 1, self.base_kernel.feature_ndims
            )
        ),
        diag_shift=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(  # pylint: disable=g-long-lambda
                    low=dtype_util.eps(dtype)
                )
            )
        ),
    )

  def _divisor_matrix(self, fixed_inputs=None):
    fixed_inputs = self._fixed_inputs if fixed_inputs is None else fixed_inputs
    return _compute_divisor_matrix(
        self._base_kernel,
        diag_shift=self._diag_shift,
        fixed_inputs=fixed_inputs)

  def divisor_matrix(self):
    return self._divisor_matrix()
