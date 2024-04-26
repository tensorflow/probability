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
"""Functions for quickly computing approximate log det of a big PSD matrix.

It's recommended to use `fast_log_det` in `float64` mode only.
"""

import enum
import functools

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.experimental.fastgp import mbcg
from tensorflow_probability.python.experimental.fastgp import partial_lanczos
from tensorflow_probability.python.experimental.fastgp import preconditioners
from tensorflow_probability.python.internal.backend import jax as tf2jax
from tensorflow_probability.substrates.jax.mcmc import sample_halton_sequence_lib

Array = jnp.ndarray

# pylint: disable=invalid-name


class ProbeVectorType(enum.IntEnum):
  RADEMACHER = 0
  NORMAL = 1
  NORMAL_ORTHOGONAL = 2
  NORMAL_QMC = 3


@jax.named_call
def make_probe_vectors(
    n: int,
    num_probe_vectors: int,
    key: jax.Array,
    probe_vector_type: ProbeVectorType,
    dtype: jnp.dtype,
) -> Array:
  """Return num_probe_vectors n-dim random vectors with mean zero."""
  if probe_vector_type == ProbeVectorType.RADEMACHER:
    return jax.random.choice(
        key, jnp.array([-1.0, 1.0], dtype=dtype), shape=(n, num_probe_vectors)
    )

  if probe_vector_type == ProbeVectorType.NORMAL:
    return jax.random.normal(key, shape=(n, num_probe_vectors), dtype=dtype)

  if probe_vector_type == ProbeVectorType.NORMAL_ORTHOGONAL:
    if num_probe_vectors > n:
      print(f'Warning, make_probe_vectors(normal_orthogonal) called with '
            f'{num_probe_vectors=} > {n=}  Falling back on normal.')
      return jax.random.normal(key, shape=(n, num_probe_vectors), dtype=dtype)
    # Sample a random orthogonal matrix.
    key1, key2 = jax.random.split(key)
    samples = jax.random.normal(key1, shape=(n, num_probe_vectors), dtype=dtype)
    q, _ = jnp.linalg.qr(samples, mode='reduced')
    # Rescale up by a chi random variable.
    norm = jnp.sqrt(
        jax.random.chisquare(
            key2, df=n, shape=(num_probe_vectors,), dtype=dtype))
    return q * norm

  if probe_vector_type == ProbeVectorType.NORMAL_QMC:
    uniforms = sample_halton_sequence_lib.sample_halton_sequence(
        dim=n, num_results=num_probe_vectors, dtype=dtype, seed=key
    )
    return jnp.transpose(jax.scipy.special.ndtri(uniforms))

  raise ValueError(
      f'Unknown probe vector type {probe_vector_type}.'
      '  Try NORMAL, NORMAL_QMC or RADEMACHER.'
  )


@jax.named_call
def log_det_jvp(primals, tangents, f, num_iters):
  """Jacobian-Vector product for log_det.

  This can be used to provide a jax.custom_jvp for a function with the
  signature (matrix, preconditioner, probe_vectors, key,
             *optional_args, num_iters) -> Float.

  [num_iters needs to be last in the signature because it needs to be
  a non-diff argument of the custom.jvp, which means it isn't passed in
  the primals and we need to put it back in here.]

  Args:
    primals: The arguments f was called with.
    tangents: The differentials of the primals.
    f: The log det function log_det_jvp is the custom.jvp of.
    num_iters: The number of iterations to run conjugate gradients for.

  Returns:
    The pair (f(primals), df).
  """
  primal_out = f(*primals, num_iters=num_iters)

  M = primals[0]
  M_dot = tangents[0]
  preconditioner = primals[1]
  probe_vectors = primals[2]
  num_probe_vectors = probe_vectors.shape[-1]

  # d(log det M) = tr(M^(-1) dM)
  # Traditionally, this trace is approximated using Hutchinson's trick.
  # However, we find that that estimate has a large variance when the
  # operator A = M^(-1) dM is badly conditioned.  So we use our preconditioner
  # P to make B = - P^(-1) dM, and get tr(A) = tr(A+B) - tr(B), with the
  # first term coming from Hutchinson's trick with hopefully low variance and
  # the second tr(B) term being computed directly.
  trace_B = -preconditioner.trace_of_inverse_product(M_dot)

  # tr(A+B) = tr( M^(-1) dM - P^(-1) dM).
  # = E[ v^t (M^(-1) dM - P^(-1) dM) v ]
  # = E[ (v^t M^(-1) - v^t P^(-1)) (dM v) ]

  # left_factor1 = v^t M^(-1)
  left_factor1, _ = mbcg.modified_batched_conjugate_gradients(
      lambda x: M @ x, probe_vectors,
      preconditioner.full_preconditioner().solve,
      max_iters=num_iters
  )
  # left_factor2 = v^t P^(-1)
  left_factor2 = preconditioner.full_preconditioner().solvevec(probe_vectors.T)
  left_factor = left_factor1 - left_factor2.T
  # right_factor = dM probe_vectors
  right_factor = M_dot @ probe_vectors

  unnormalized_trace_of_A_plus_B = jnp.einsum(
      'ij,ij->', left_factor, right_factor
  )
  trace_of_A_plus_B = unnormalized_trace_of_A_plus_B / num_probe_vectors

  tangent_out = trace_of_A_plus_B - trace_B

  return primal_out, tangent_out


@jax.named_call
def _log_det_rational_approx_with_hutchinson(
    shifts,
    coefficients,
    bias,
    preconditioner: preconditioners.Preconditioner,
    probe_vectors: Array,
    key: jax.Array,
    num_iters: int,
):
  """Approximate log det using a rational function.

  We calculate log det M as the trace of log M, and we approximate the
  trace of log M using Hutchinson's trick.  We then approximate
  (log M) @ probe_vector using the partial fraction decomposition
  log M ~ bias + sum_i coefficients[i] / (M - shifts[i])
  and finally we get the (1 / (M - shifts[i]) ) @ probe_vector parts
  using a multishift solver.

  Args:
    shifts:  An array of length r.  When approximating log z with p(x)/q(x),
      shifts will contain the roots of q(x).
    coefficients:  An array of length r.
    bias:  A scalar Float.
    preconditioner:  A preconditioner of M.  It is used both in speeding the
      convergence of the multishift solve, and in reducing the variance of the
      approximation used to compute the derivative of this function.
    probe_vectors:  An array of shape (n, num_probe_vectors).  Each probe vector
      should be composed of i.i.d. random variables with mean 0 and variance 1.
    key:  The RNG key.
    num_iters: The number of iterations to run the partial Lanczos algorithm
      for.

  Returns:
    An approximation to the log det of M.
  """
  num_probe_vectors = probe_vectors.shape[-1]

  solutions = partial_lanczos.psd_solve_multishift(
      preconditioner.preconditioned_operator().matmul,
      probe_vectors,
      shifts,
      key,
      num_iters,
  )
  # solutions will be (num_shifts, num_probes, n)
  weighted_solutions = jnp.einsum('i,ijk->kj', coefficients, solutions)
  # logM_pv is our approximation to (log M) @ probe_vectors
  logM_pv = bias * probe_vectors + weighted_solutions

  return (
      preconditioner.log_det()
      + jnp.einsum('ij,ij->', probe_vectors, logM_pv) / num_probe_vectors
  )


R1_SHIFTS = np.array([-1.0], dtype=np.float64)
R1_COEFFICIENTS = np.array([-4.0], dtype=np.float64)

R2_SHIFTS = np.array(
    [-5.828427124746191, -0.1715728752538099], dtype=np.float64
)
R2_COEFFICIENTS = np.array(
    [-23.313708498984763, -0.6862915010152396], dtype=np.float64
)

R3_SHIFTS = np.array(
    [-13.92820323027551, -1.0, -0.0717967697244908], dtype=np.float64
)
R3_COEFFICIENTS = np.array(
    [-49.52250037431294, -2.2222222222222214, -0.2552774034648563],
    dtype=np.float64,
)

R4_SHIFTS = np.array(
    [
        -25.27414236908818,
        -2.2398288088435496,
        -0.4464626921716892,
        -0.03956612989657948,
    ],
    dtype=np.float64,
)
R4_COEFFICIENTS = np.array(
    [
        -91.22640292804368,
        -3.861145971009117,
        -0.7696381162159669,
        -0.1428129847311194,
    ],
    dtype=np.float64,
)

R5_SHIFTS = np.array(
    [
        -3.9863458189061411e01,
        -3.8518399963191827,
        -1.0,
        -2.5961618368249978e-01,
        -2.5085630936916615e-02,
    ],
    dtype=np.float64,
)
R5_COEFFICIENTS = np.array(
    [
        -1.4008241129102026e02,
        -6.1858406006156228e00,
        -1.2266666666666659e00,
        -4.1692913805732562e-01,
        -8.8152303639431204e-02,
    ],
    dtype=np.float64,
)

R6_SHIFTS = np.array(
    [
        -5.7695480540981052e01,
        -5.8284271247461907e00,
        -1.6983963724170996e00,
        -5.8879070648086351e-01,
        -1.7157287525380993e-01,
        -1.7332380120999309e-02,
    ],
    dtype=np.float64,
)
R6_COEFFICIENTS = np.array(
    [
        -2.0440306874472464e02,
        -8.8074009885053552e00,
        -1.8333009080451452e00,
        -6.3555866838298825e-01,
        -2.5926567816131274e-01,
        -6.1405012180561776e-02,
    ],
    dtype=np.float64,
)


@functools.partial(jax.custom_jvp, nondiff_argnums=(4,))
def _r1(
    unused_M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    probe_vectors: Array,
    key: jax.Array,
    num_iters: int,
):
  """Approximate log det using a 1st order rational function."""
  return _log_det_rational_approx_with_hutchinson(
      jnp.asarray(R1_SHIFTS, dtype=probe_vectors.dtype),
      jnp.asarray(R1_COEFFICIENTS, dtype=probe_vectors.dtype),
      2.0,
      preconditioner,
      probe_vectors,
      key,
      num_iters,
  )


@functools.partial(jax.custom_jvp, nondiff_argnums=(4,))
def _r2(
    unused_M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    probe_vectors: Array,
    key: jax.Array,
    num_iters: int,
):
  """Approximate log det using a 2nd order rational function."""
  return _log_det_rational_approx_with_hutchinson(
      jnp.asarray(R2_SHIFTS, dtype=probe_vectors.dtype),
      jnp.asarray(R2_COEFFICIENTS, dtype=probe_vectors.dtype),
      4.0,
      preconditioner,
      probe_vectors,
      key,
      num_iters,
  )


@functools.partial(jax.custom_jvp, nondiff_argnums=(4,))
def _r3(
    unused_M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    probe_vectors: Array,
    key: jax.Array,
    num_iters: int,
):
  """Approximate log det using a 4th order rational function."""
  return _log_det_rational_approx_with_hutchinson(
      jnp.asarray(R3_SHIFTS, dtype=probe_vectors.dtype),
      jnp.asarray(R3_COEFFICIENTS, dtype=probe_vectors.dtype),
      14.0 / 3.0,
      preconditioner,
      probe_vectors,
      key,
      num_iters,
  )


@functools.partial(jax.custom_jvp, nondiff_argnums=(4,))
def _r4(
    unused_M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    probe_vectors: Array,
    key: jax.Array,
    num_iters: int,
):
  """Approximate log det using a 4th order rational function."""
  return _log_det_rational_approx_with_hutchinson(
      jnp.asarray(R4_SHIFTS, dtype=probe_vectors.dtype),
      jnp.asarray(R4_COEFFICIENTS, dtype=probe_vectors.dtype),
      16.0 / 3.0,
      preconditioner,
      probe_vectors,
      key,
      num_iters,
  )


@functools.partial(jax.custom_jvp, nondiff_argnums=(4,))
def _r5(
    unused_M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    probe_vectors: Array,
    key: jax.Array,
    num_iters: int,
):
  """Approximate log det using a 4th order rational function."""
  return _log_det_rational_approx_with_hutchinson(
      jnp.asarray(R5_SHIFTS, dtype=probe_vectors.dtype),
      jnp.asarray(R5_COEFFICIENTS, dtype=probe_vectors.dtype),
      86.0 / 15.0,
      preconditioner,
      probe_vectors,
      key,
      num_iters,
  )


@functools.partial(jax.custom_jvp, nondiff_argnums=(4,))
def _r6(
    unused_M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    probe_vectors: Array,
    key: jax.Array,
    num_iters: int,
):
  """Approximate log det using a 4th order rational function."""
  return _log_det_rational_approx_with_hutchinson(
      jnp.asarray(R6_SHIFTS, dtype=probe_vectors.dtype),
      jnp.asarray(R6_COEFFICIENTS, dtype=probe_vectors.dtype),
      92.0 / 15.0,
      preconditioner,
      probe_vectors,
      key,
      num_iters,
  )


@_r1.defjvp
def _r1_jvp(num_iters, primals, tangents):
  """Jacobian-Vector product for r1."""
  return log_det_jvp(primals, tangents, _r1, num_iters)


@_r2.defjvp
def _r2_jvp(num_iters, primals, tangents):
  """Jacobian-Vector product for r2."""
  return log_det_jvp(primals, tangents, _r2, num_iters)


@_r3.defjvp
def _r3_jvp(num_iters, primals, tangents):
  """Jacobian-Vector product for r3."""
  return log_det_jvp(primals, tangents, _r3, num_iters)


@_r4.defjvp
def _r4_jvp(num_iters, primals, tangents):
  """Jacobian-Vector product for r4."""
  return log_det_jvp(primals, tangents, _r4, num_iters)


@_r5.defjvp
def _r5_jvp(num_iters, primals, tangents):
  """Jacobian-Vector product for r5."""
  return log_det_jvp(primals, tangents, _r5, num_iters)


@_r6.defjvp
def _r6_jvp(num_iters, primals, tangents):
  """Jacobian-Vector product for r6."""
  return log_det_jvp(primals, tangents, _r6, num_iters)


@jax.named_call
def r1(
    M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    key: jax.Array,
    num_probe_vectors: int = 25,
    probe_vector_type: ProbeVectorType = ProbeVectorType.RADEMACHER,
    num_iters: int = 20,
    **unused_kwargs,
):
  """Approximate log det using a 2nd order rational function."""
  n = M.shape[-1]
  key1, key2 = jax.random.split(key)
  probe_vectors = make_probe_vectors(
      n, num_probe_vectors, key1, probe_vector_type, dtype=M.dtype
  )

  return _r1(M, preconditioner, probe_vectors, key2, num_iters)


@jax.named_call
def r2(
    M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    key: jax.Array,
    num_probe_vectors: int = 25,
    probe_vector_type: ProbeVectorType = ProbeVectorType.RADEMACHER,
    num_iters: int = 20,
    **unused_kwargs,
):
  """Approximate log det using a 2nd order rational function."""
  n = M.shape[-1]
  key1, key2 = jax.random.split(key)
  probe_vectors = make_probe_vectors(
      n, num_probe_vectors, key1, probe_vector_type, dtype=M.dtype
  )

  return _r2(M, preconditioner, probe_vectors, key2, num_iters)


@jax.named_call
def r3(
    M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    key: jax.Array,
    num_probe_vectors: int = 25,
    probe_vector_type: ProbeVectorType = ProbeVectorType.RADEMACHER,
    num_iters: int = 20,
    **unused_kwargs,
):
  """Approximate log det using a 3rd order rational function."""
  n = M.shape[-1]
  key1, key2 = jax.random.split(key)
  probe_vectors = make_probe_vectors(
      n, num_probe_vectors, key1, probe_vector_type, dtype=M.dtype
  )

  return _r3(M, preconditioner, probe_vectors, key2, num_iters)


@jax.named_call
def r4(
    M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    key: jax.Array,
    num_probe_vectors: int = 25,
    probe_vector_type: ProbeVectorType = ProbeVectorType.RADEMACHER,
    num_iters: int = 20,
    **unused_kwargs,
):
  """Approximate log det using a 4th order rational function."""
  n = M.shape[-1]
  key1, key2 = jax.random.split(key)
  probe_vectors = make_probe_vectors(
      n, num_probe_vectors, key1, probe_vector_type, dtype=M.dtype
  )

  return _r4(M, preconditioner, probe_vectors, key2, num_iters)


@jax.named_call
def r5(
    M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    key: jax.Array,
    num_probe_vectors: int = 25,
    probe_vector_type: ProbeVectorType = ProbeVectorType.RADEMACHER,
    num_iters: int = 20,
    **unused_kwargs,
):
  """Approximate log det using a 5th order rational function."""
  n = M.shape[-1]
  key1, key2 = jax.random.split(key)
  probe_vectors = make_probe_vectors(
      n, num_probe_vectors, key1, probe_vector_type, dtype=M.dtype
  )

  return _r5(M, preconditioner, probe_vectors, key2, num_iters)


@jax.named_call
def r6(
    M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    key: jax.Array,
    num_probe_vectors: int = 25,
    probe_vector_type: ProbeVectorType = ProbeVectorType.RADEMACHER,
    num_iters: int = 20,
    **unused_kwargs,
):
  """Approximate log det using a 6th order rational function."""
  n = M.shape[-1]
  key1, key2 = jax.random.split(key)
  probe_vectors = make_probe_vectors(
      n, num_probe_vectors, key1, probe_vector_type, dtype=M.dtype
  )

  return _r6(M, preconditioner, probe_vectors, key2, num_iters)


def log00(diag: Array, off_diag: Array) -> Array:
  """Return the (0, 0)-th entry of the log of the tridiagonal matrix."""
  n = diag.shape[-1]
  if n == 1:
    return jnp.log(diag[0])
  m = jnp.diag(diag) + jnp.diag(off_diag, -1) + jnp.diag(off_diag, 1)
  # Use jax.numpy.linalg.eigh instead of scipy.linalg.eigh_tridiagonal
  # because the later isn't yet jax-able.  We would use
  # jax.scipy.linalg.eigh_tridiagonal, but that doesn't yet return
  # eigenvectors.  TODO(thomaswc): Switch when it does.
  evalues, evectors = jax.numpy.linalg.eigh(m)
  log_evalues = jnp.log(evalues)
  first_components = evectors[0, :]
  return jnp.einsum('i,i,i->', first_components, log_evalues, first_components)


@jax.named_call
def batch_log00(ts: mbcg.SymmetricTridiagonalMatrix) -> Array:
  """Return the (0, 0)-th entries of the log of the tridiagonal matrices."""
  return jax.vmap(log00)(ts.diag, ts.off_diag)


@functools.partial(jax.custom_jvp, nondiff_argnums=(4, 5))
@functools.partial(
    jax.jit, static_argnames=['probe_vectors_are_rademacher', 'num_iters']
)
def _stochastic_lanczos_quadrature_log_det(
    M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    probe_vectors: Array,
    unused_key,
    probe_vectors_are_rademacher: bool,
    num_iters: int,
):
  """Fast log det using the alg. from https://arxiv.org/pdf/1809.11165.pdf ."""
  n = M.shape[-1]

  _, tridiagonals = mbcg.modified_batched_conjugate_gradients(
      lambda x: M @ x,
      probe_vectors,
      preconditioner.full_preconditioner().solve,
      num_iters
      # TODO(thomaswc): Pass tolerance=-1 here to make sure mbcg never
      # stops early.  Currently that is broken.
  )

  # The modified_batched_conjugate_gradients applies a rotation so that the
  # tridiagonal matrices are written in a basis where e_0 is parallel to the
  # probe vector.  Therefore,
  # probe_vector^T A probe_vector = (a e_0)^T A (a e_0)
  #                               = a^2 A_00
  # for a = len(probe_vector).
  sum_squared_probe_vector_lengths = None
  if probe_vectors_are_rademacher:
    sum_squared_probe_vector_lengths = n
  else:
    sum_squared_probe_vector_lengths = jnp.einsum(
        'ij,ij->j', probe_vectors, probe_vectors
    )
  trace_log_estimates = sum_squared_probe_vector_lengths * batch_log00(
      tridiagonals
  )

  return preconditioner.log_det() + jnp.average(trace_log_estimates)


@jax.named_call
def stochastic_lanczos_quadrature_log_det(
    M: tf2jax.linalg.LinearOperator,
    preconditioner: preconditioners.Preconditioner,
    key: jax.Array,
    num_probe_vectors: int = 25,
    probe_vector_type: ProbeVectorType = ProbeVectorType.RADEMACHER,
    num_iters: int = 25,
    **unused_kwargs,
):
  """Fast log det using the alg. from https://arxiv.org/pdf/1809.11165.pdf ."""
  n = M.shape[-1]
  num_iters = min(n, num_iters)

  probe_vectors = make_probe_vectors(
      n, num_probe_vectors, key, probe_vector_type, dtype=M.dtype
  )

  return _stochastic_lanczos_quadrature_log_det(
      M,
      preconditioner,
      probe_vectors,
      None,
      probe_vector_type == ProbeVectorType.RADEMACHER,
      num_iters,
  )


@_stochastic_lanczos_quadrature_log_det.defjvp
def _stochastic_lanczos_quadrature_log_det_jvp(
    probe_vectors_are_rademacher, num_iters, primals, tangents
):
  """Jacobian-Vector product for @_stochastic_lanczos_quadrature_log_det."""
  def slq_f(M, preconditioner, probe_vectors, unused_key, num_iters):
    return _stochastic_lanczos_quadrature_log_det(
        M, preconditioner, probe_vectors, unused_key,
        probe_vectors_are_rademacher, num_iters)

  return log_det_jvp(primals, tangents, slq_f, num_iters)


LOG_DET_REGISTRY = {
    'r1': r1,
    'r2': r2,
    'r3': r3,
    'r4': r4,
    'r5': r5,
    'r6': r6,
    'slq': stochastic_lanczos_quadrature_log_det,
}


@jax.named_call
def get_log_det_algorithm(alg_name: str):
  try:
    return LOG_DET_REGISTRY[alg_name]
  except KeyError as key_error:
    raise ValueError(
        'Unknown algorithm name {}, known log det algorithms are {}'.format(
            alg_name, LOG_DET_REGISTRY.keys()
        )
    ) from key_error


# The below log det algorithms are not yet useful enough to make it into the
# LOG_DET_REGISTRY.


def log_det_taylor_series_with_hutchinson(
    M: tf2jax.linalg.LinearOperator,
    num_probe_vectors: int,
    key: jax.Array,
    num_taylor_series_iterations: int = 10,
):
  """Return an approximation of log det M."""
  # TODO(thomaswc): Consider having this support a batch of LinearOperators.
  n = M.shape[0]
  A = M - jnp.identity(n)
  probe_vectors = jax.random.choice(
      key, jnp.array([-1.0, 1.0], dtype=M.dtype), shape=(n, num_probe_vectors))
  estimate = 0.0
  Apv = probe_vectors
  sign = 1
  for i in range(1, num_taylor_series_iterations + 1):
    Apv = A @ Apv
    trace_estimate = 0
    if i == 1:
      trace_estimate = jnp.trace(A)
    elif i == 2:
      # tr A^2 = sum_i (A^2)_ii = sum_i sum_j A_ij A_ji
      # = sum_i sum_j A_ij A_ij since A is symmetric.
      trace_estimate = jnp.einsum('ij,ij->', A, A)
    else:
      trace_estimate = (
          jnp.einsum('ij,ij->', Apv, probe_vectors) / num_probe_vectors
      )
    estimate = estimate + sign * trace_estimate / i
    sign = -sign

  return estimate
