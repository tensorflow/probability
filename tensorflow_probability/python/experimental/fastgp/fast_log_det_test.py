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
"""Tests for fast_log_det.py."""

import math

from absl.testing import parameterized
import jax
from jax import config
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.experimental.fastgp import fast_log_det
from tensorflow_probability.python.experimental.fastgp import preconditioners
from tensorflow_probability.substrates import jax as tfp

from absl.testing import absltest

# pylint: disable=invalid-name


def rational_at_one(shifts, coefficients):
  """Return sum_i coefficients[i] / (1.0 - shifts[i])."""
  n = shifts.shape[-1]
  s = 0.0
  for i in range(n):
    s += coefficients[i] / (1.0 - shifts[i])
  return s


class _FastLogDetTest(parameterized.TestCase):
  def test_make_probe_vectors_rademacher(self):
    pvs = fast_log_det.make_probe_vectors(
        10,
        5,
        jax.random.PRNGKey(0),
        fast_log_det.ProbeVectorType.RADEMACHER,
        dtype=self.dtype)
    for i in range(10):
      for j in range(5):
        self.assertIn(float(pvs[i, j]), {-1.0, 1.0})

  @parameterized.parameters(
      fast_log_det.ProbeVectorType.NORMAL,
      fast_log_det.ProbeVectorType.NORMAL_ORTHOGONAL,
      fast_log_det.ProbeVectorType.NORMAL_QMC)
  def test_make_probe_vectors(self, probe_vector_type):
    pvs = fast_log_det.make_probe_vectors(
        10,
        5,
        jax.random.PRNGKey(0),
        probe_vector_type,
        dtype=self.dtype)
    for i in range(10):
      for j in range(5):
        self.assertNotIn(float(pvs[i, j]), {-1.0, 1.0})

  def test_rational_parameters(self):
    self.assertAlmostEqual(
        0.0,
        4.0
        + rational_at_one(fast_log_det.R2_SHIFTS, fast_log_det.R2_COEFFICIENTS),
    )
    self.assertAlmostEqual(
        0.0,
        14.0 / 3.0
        + rational_at_one(fast_log_det.R3_SHIFTS, fast_log_det.R3_COEFFICIENTS),
    )
    self.assertAlmostEqual(
        0.0,
        16.0 / 3.0
        + rational_at_one(fast_log_det.R4_SHIFTS, fast_log_det.R4_COEFFICIENTS),
    )
    self.assertAlmostEqual(
        0.0,
        86.0 / 15.0
        + rational_at_one(fast_log_det.R5_SHIFTS, fast_log_det.R5_COEFFICIENTS),
    )
    self.assertAlmostEqual(
        0.0,
        92.0 / 15.0
        + rational_at_one(fast_log_det.R6_SHIFTS, fast_log_det.R6_COEFFICIENTS),
    )

  def test_r2_same_as_rational(self):
    num_probe_vectors = 5
    M = jnp.array([[1.0, -0.5, 0.0], [-0.5, 1.0, -0.5], [0.0, -0.5, 1.0]],
                  dtype=self.dtype)
    I = jnp.eye(3, dtype=self.dtype)
    pvs = fast_log_det.make_probe_vectors(
        3,
        num_probe_vectors,
        jax.random.PRNGKey(1),
        fast_log_det.ProbeVectorType.RADEMACHER,
        dtype=self.dtype
    )
    pc = preconditioners.IdentityPreconditioner(M)
    _r2_answer = fast_log_det._r2(M, pc, pvs, jax.random.PRNGKey(2), 20)
    # r2(z) = 4(z-1)(1+z)/(1 + 6z + z^2)
    _rat_answer = (
        4.0
        * (M - I)
        @ (M + I)
        @ jnp.linalg.inv(I + 6.0 * M + M @ M)
        @ pvs
    )
    _rat_answer = jnp.einsum('ij,ij->', pvs, _rat_answer) / float(
        num_probe_vectors
    )
    np.testing.assert_allclose(_r2_answer, _rat_answer, atol=0.1)

  @parameterized.parameters(
      ('r1', 1),
      ('r2', 1),
      ('r3', 2),
      ('r4', 2),
      ('r5', 2),
      ('r6', 2),
      ('slq', 2),
  )
  def test_log_det_algorithm_in_low_dim(self, log_det_alg, num_places):
    lda = fast_log_det.get_log_det_algorithm(log_det_alg)

    # For 1x1 arrays, for Rademacher probe vectors, v^t A v = A = tr A, so
    # only one probe vector is necessary.
    m = jnp.array([[1.5]], dtype=self.dtype)
    pc = preconditioners.IdentityPreconditioner(m)
    log_det = lda(
        m,
        pc,
        jax.random.PRNGKey(0),
        1,
    )
    self.assertAlmostEqual(log_det, math.log(1.5), places=num_places)

    m = jnp.array([[0.5]], dtype=self.dtype)
    pc = preconditioners.IdentityPreconditioner(m)
    log_det = lda(
        m,
        pc,
        jax.random.PRNGKey(1),
        1,
    )
    self.assertAlmostEqual(log_det, math.log(0.5), places=num_places)

    m = jnp.array([[1.0, 0.1], [0.1, 1.0]], dtype=self.dtype)
    pc = preconditioners.IdentityPreconditioner(m)
    log_det = lda(
        m,
        pc,
        jax.random.PRNGKey(2),
        num_probe_vectors=200,
    )
    self.assertAlmostEqual(log_det, math.log(0.99), places=num_places)

  @parameterized.parameters(
      ('r1', 2.0),
      ('r2', 1.0),
      ('r3', 0.2),
      ('r4', 0.1),
      ('r5', 0.1),
      ('r6', 0.01),
      ('slq', 0.00001),
  )
  def test_log_det_algorithm_diagonal_matrices(self, log_det_alg, delta):
    lda = fast_log_det.get_log_det_algorithm(log_det_alg)
    m = jnp.identity(5)
    pc = preconditioners.IdentityPreconditioner(m)
    self.assertAlmostEqual(lda(m, pc, jax.random.PRNGKey(0)), 0.0, delta=delta)

    m = jnp.diag(
        jnp.arange(
            1.0,
            9.0,
        )
    )
    pc = preconditioners.IdentityPreconditioner(m)
    self.assertAlmostEqual(
        lda(m, pc, jax.random.PRNGKey(1)),
        # log 8! =
        10.6046029027,
        delta=delta,
    )

  @parameterized.parameters(
      (fast_log_det.ProbeVectorType.NORMAL, 0.5),
      (fast_log_det.ProbeVectorType.NORMAL_ORTHOGONAL, 0.4),
      (fast_log_det.ProbeVectorType.NORMAL_QMC, 0.9),
      (fast_log_det.ProbeVectorType.RADEMACHER, 0.01)
  )
  def test_r4_jits_different_probe_vector_types(self, probe_vector_type, delta):
    my_log_det = jax.jit(
        fast_log_det.r4, static_argnames=[
            'num_probe_vectors', 'probe_vector_type'])
    m = jnp.array([[1.5]], dtype=self.dtype)
    pc = preconditioners.IdentityPreconditioner(m)
    ld = my_log_det(
        m,
        pc,
        jax.random.PRNGKey(0),
        num_probe_vectors=1,
        probe_vector_type=probe_vector_type
    )
    self.assertAlmostEqual(ld, math.log(1.5), delta=delta)

    m = jnp.array([[0.5]])
    pc = preconditioners.IdentityPreconditioner(m)
    self.assertAlmostEqual(
        my_log_det(
            m,
            pc,
            jax.random.PRNGKey(1),
            num_probe_vectors=1,
        ),
        math.log(0.5),
        delta=delta
    )

  @parameterized.parameters(
      ('r1', 1),
      ('r2', 1),
      ('r3', 3),
      ('r4', 3),
      ('r5', 3),
      ('r6', 3),
      ('slq', 3),
  )
  def test_log_det_algorithm_jits(self, log_det_alg, num_places):
    lda = fast_log_det.get_log_det_algorithm(log_det_alg)

    my_log_det = jax.jit(lda, static_argnames=['num_probe_vectors'])
    m = jnp.array([[1.5]], dtype=self.dtype)
    pc = preconditioners.IdentityPreconditioner(m)
    ld = my_log_det(
        m,
        pc,
        jax.random.PRNGKey(0),
        num_probe_vectors=1,
    )
    self.assertAlmostEqual(ld, math.log(1.5), places=num_places)

    m = jnp.array([[0.5]])
    pc = preconditioners.IdentityPreconditioner(m)
    self.assertAlmostEqual(
        my_log_det(
            m,
            pc,
            jax.random.PRNGKey(1),
            num_probe_vectors=1,
        ),
        math.log(0.5),
        places=num_places,
    )

  @parameterized.parameters(
      ('r1', 5),
      ('r2', 5),
      ('r3', 5),
      ('r4', 5),
      ('r5', 5),
      ('r6', 5),
      ('slq', 5),
  )
  def test_log_det_algorithm_derivative(self, log_det_alg, num_places):
    lda = fast_log_det.get_log_det_algorithm(log_det_alg)
    n = 10
    def log_det_of_scaled_identity(scale):
      m = scale * jnp.identity(n).astype(self.dtype)
      pc = preconditioners.IdentityPreconditioner(m)
      return lda(m, pc, jax.random.PRNGKey(0))

    d_scale = jax.grad(log_det_of_scaled_identity)(self.dtype(2.0))
    # det(scale I) = scale^n
    # log det(scale I) = n log scale
    # d log det(scale I) = n (d scale) / scale
    # d log det(scale I) / d scale = n / scale
    self.assertAlmostEqual(d_scale, n / 2.0, places=num_places)

  # TODO(thomaswc,srvasude): Investigate why these numbers got so much worse
  # after cl/564448268.  Before that, the deltas were:
  # ('r2', 0.3), ('r3', 0.3), ('r4', 0.2), ('r5', 0.2), ('r6', 0.3),
  # ('slq', 0.5).
  @parameterized.parameters(
      ('r1', 0.8),
      ('r2', 0.8),
      ('r3', 0.7),
      ('r4', 0.6),
      ('r5', 0.6),
      ('r6', 0.6),
      ('slq', 1.4),
  )
  def test_log_det_grad_of_random(self, log_det_alg, delta):
    if self.dtype == np.float32:
      self.skipTest('Numerically unstable in float32.')
    lda = fast_log_det.get_log_det_algorithm(log_det_alg)
    # Generate two random PSD matrices.
    A = jax.random.uniform(jax.random.PRNGKey(0), shape=(10, 10),
                           minval=-1.0, maxval=1.0, dtype=self.dtype)
    A = A @ jnp.transpose(A)
    B = jax.random.uniform(jax.random.PRNGKey(1), shape=(10, 10),
                           minval=-1.0, maxval=1.0, dtype=self.dtype)
    B = B @ jnp.transpose(B)

    def my_log_det(alpha):
      m = A + alpha * B
      pc = preconditioners.DiagonalSplitPreconditioner(m)
      return lda(m, pc, jax.random.PRNGKey(2))

    def std_log_det(alpha):
      _, logdet = jnp.linalg.slogdet(A + alpha * B)
      return logdet

    my_ld, my_grad = jax.value_and_grad(my_log_det)(self.dtype(1.0))
    std_ld, std_grad = jax.value_and_grad(std_log_det)(self.dtype(1.0))
    self.assertAlmostEqual(my_ld, std_ld, delta=delta)
    self.assertAlmostEqual(my_grad, std_grad, delta=delta)

  def test_log00(self):
    self.assertAlmostEqual(
        fast_log_det.log00(
            jnp.array([2.0, 3.0], dtype=self.dtype),
            jnp.array([1.0], dtype=self.dtype)),
        0.5895146,
        places=6,
    )
    self.assertAlmostEqual(
        fast_log_det.log00(
            jnp.array([5.0, 6.0], dtype=self.dtype),
            jnp.array([4.0], dtype=self.dtype)),
        1.2035519,
        places=6,
    )

  @parameterized.parameters(
      (fast_log_det.ProbeVectorType.NORMAL, 1.1, 0.4),
      (fast_log_det.ProbeVectorType.NORMAL_ORTHOGONAL, 1.7, 0.6),
      (fast_log_det.ProbeVectorType.NORMAL_QMC, 0.6, 0.3))
  def test_stochastic_lanczos_quadrature_normal_log_det(
      self, probe_vector_type, error_float32, error_float64):
    error = error_float32 if self.dtype == np.float32 else error_float64
    m = jnp.identity(5).astype(self.dtype)
    pc = preconditioners.IdentityPreconditioner(m)
    num_probe_vectors = 25
    if probe_vector_type == fast_log_det.ProbeVectorType.NORMAL_ORTHOGONAL:
      num_probe_vectors = 5
    self.assertAlmostEqual(
        fast_log_det.stochastic_lanczos_quadrature_log_det(
            m,
            pc,
            jax.random.PRNGKey(0),
            num_probe_vectors=num_probe_vectors,
            probe_vector_type=probe_vector_type,
        ),
        0.0,
    )
    m = jnp.diag(jnp.arange(1., 9.,).astype(self.dtype))
    if probe_vector_type == fast_log_det.ProbeVectorType.NORMAL_ORTHOGONAL:
      num_probe_vectors = 8
    pc = preconditioners.IdentityPreconditioner(m)
    self.assertAlmostEqual(
        fast_log_det.stochastic_lanczos_quadrature_log_det(
            m,
            pc,
            jax.random.PRNGKey(1),
            num_probe_vectors=num_probe_vectors,
            probe_vector_type=probe_vector_type,
        ),
        # log 8! =
        10.6046029027,
        # At least for this example, normal probe vectors really degrade
        # accuracy.
        delta=error,
    )

  def test_log_det_gradient_leaks(self):
    n = 10

    def scan_body(scale, _):
      m = scale * jnp.identity(n).astype(self.dtype)
      pc = preconditioners.IdentityPreconditioner(m)
      return fast_log_det.log_det_order_four_rational_with_hutchinson(
          m, pc, jax.random.PRNGKey(0))

    def log_det_of_scaled_identity(scale):
      _, y = jax.lax.scan(scan_body, scale, jnp.arange(1))
      return y[0]

    # Because of b/266429021, using nondiff_argnums over inputs that are or
    # contain JAX arrays will cause a memory leak when used in a loop like a
    # scan.
    with self.assertRaises(Exception):
      with jax.checking_leaks():
        unused_d_scale = jax.grad(log_det_of_scaled_identity)(2.0)

  @parameterized.parameters('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'slq')
  def test_log_det_gradient_hard(self, algname):
    log_det_fn = fast_log_det.get_log_det_algorithm(algname)
    b = (
        jnp.diag(jnp.full(10, 2.0))
        + jnp.diag(jnp.full(9, 1.0), 1)
        + jnp.diag(jnp.full(9, 1.0), -1)
    ).astype(self.dtype)

    def fast_logdet(jitter):
      m = b + jitter * jnp.identity(10).astype(self.dtype)
      pc = preconditioners.PartialCholeskySplitPreconditioner(m)
      return log_det_fn(m, pc, jax.random.PRNGKey(1))

    def slow_logdet(jitter):
      m = b + jitter * jnp.identity(10).astype(self.dtype)
      return tfp.math.hpsd_logdet(m)

    d_fast = jax.grad(fast_logdet)
    d_slow = jax.grad(slow_logdet)

    self.assertAlmostEqual(d_fast(0.1), d_slow(0.1), places=5)
    self.assertAlmostEqual(d_fast(1.0), d_slow(1.0), places=6)

  def test_log_det_jvp(self):
    if self.dtype == np.float32:
      self.skipTest('Numerically unstable in float32.')
    M = (
        jnp.diag(jnp.full(10, 2.0))
        + jnp.diag(jnp.full(9, 1.0), 1)
        + jnp.diag(jnp.full(9, 1.0), -1)
    ).astype(self.dtype)
    M_dot = jnp.diag(jnp.full(10, 0.1).astype(self.dtype))
    pc = preconditioners.PartialCholeskySplitPreconditioner(M)
    num_probe_vectors = 25
    probe_vectors = fast_log_det.make_probe_vectors(
        10,
        num_probe_vectors,
        jax.random.PRNGKey(0),
        fast_log_det.ProbeVectorType.RADEMACHER,
        dtype=self.dtype,
    )
    _, tangent_out = fast_log_det.log_det_jvp(
        (M, pc, probe_vectors, num_probe_vectors),
        (M_dot, None, None, None),
        lambda a, b, c, d, **kwargs: 0.0,
        20,
    )
    truth = jnp.trace(jnp.linalg.inv(M) @ M_dot)
    self.assertAlmostEqual(tangent_out, truth, places=6)

  @parameterized.parameters(
      (fast_log_det.ProbeVectorType.NORMAL, 6),
      (fast_log_det.ProbeVectorType.NORMAL_ORTHOGONAL, 6),
      (fast_log_det.ProbeVectorType.NORMAL_QMC, 5))
  def test_log_det_jvp_normal_probe_vectors(
      self, probe_vector_type, places):
    M = (
        jnp.diag(jnp.full(10, 2.0))
        + jnp.diag(jnp.full(9, 1.0), 1)
        + jnp.diag(jnp.full(9, 1.0), -1)
    ).astype(self.dtype)
    M_dot = jnp.diag(jnp.full(10, 0.1).astype(self.dtype))
    pc = preconditioners.PartialCholeskySplitPreconditioner(M)
    num_probe_vectors = 25
    if probe_vector_type == fast_log_det.ProbeVectorType.NORMAL_ORTHOGONAL:
      num_probe_vectors = 10
    probe_vectors = fast_log_det.make_probe_vectors(
        10,
        num_probe_vectors,
        jax.random.PRNGKey(0),
        probe_vector_type,
        dtype=self.dtype)
    _, tangent_out = fast_log_det.log_det_jvp(
        (M, pc, probe_vectors, num_probe_vectors),
        (M_dot, None, None, None),
        lambda a, b, c, d, **kwargs: 0.0,
        20,
    )
    truth = jnp.trace(jnp.linalg.inv(M) @ M_dot)
    self.assertAlmostEqual(tangent_out, truth, places=places)

  def test_log_det_jvp_hard(self):
    if self.dtype == np.float32:
      self.skipTest('Numerically unstable in float32.')
    # Example from fast_gp_test.py:test_gaussian_process_log_prob_gradient
    M = jnp.array([
        [
            1.0020001,
            0.88311934,
            0.9894911,
            0.9695768,
            0.9987461,
            0.98577714,
            0.97863793,
            0.9880289,
            0.7110599,
            0.7718459,
        ],
        [
            0.88311934,
            1.0020001,
            0.9395206,
            0.7564426,
            0.86025584,
            0.94721663,
            0.7791884,
            0.8075757,
            0.9478641,
            0.9758552,
        ],
        [
            0.9894911,
            0.9395206,
            1.0020001,
            0.92534095,
            0.98108065,
            0.9997143,
            0.93953925,
            0.95583755,
            0.79332554,
            0.84795874,
        ],
        [
            0.9695768,
            0.7564426,
            0.92534095,
            1.0020001,
            0.98049456,
            0.91640615,
            0.9991695,
            0.99564964,
            0.5614807,
            0.6257758,
        ],
        [
            0.9987461,
            0.86025584,
            0.98108065,
            0.98049456,
            1.0020001,
            0.97622854,
            0.98763895,
            0.99449164,
            0.6813891,
            0.74358207,
        ],
        [
            0.98577714,
            0.94721663,
            0.9997143,
            0.91640615,
            0.97622854,
            1.0020001,
            0.9313745,
            0.9487237,
            0.80610526,
            0.859435,
        ],
        [
            0.97863793,
            0.7791884,
            0.93953925,
            0.9991695,
            0.98763895,
            0.9313745,
            1.0020001,
            0.99861676,
            0.5861309,
            0.65042824,
        ],
        [
            0.9880289,
            0.8075757,
            0.95583755,
            0.99564964,
            0.99449164,
            0.9487237,
            0.99861676,
            1.0020001,
            0.61803514,
            0.68201244,
        ],
        [
            0.7110599,
            0.9478641,
            0.79332554,
            0.5614807,
            0.6813891,
            0.80610526,
            0.5861309,
            0.61803514,
            1.0020001,
            0.9943819,
        ],
        [
            0.7718459,
            0.9758552,
            0.84795874,
            0.6257758,
            0.74358207,
            0.859435,
            0.65042824,
            0.68201244,
            0.9943819,
            1.0020001,
        ],
    ], dtype=self.dtype)
    M_dot = jnp.diag(jnp.full(10, 1.0).astype(self.dtype))
    pc = preconditioners.PartialCholeskySplitPreconditioner(M)
    num_probe_vectors = 25
    probe_vectors = fast_log_det.make_probe_vectors(
        10,
        num_probe_vectors,
        jax.random.PRNGKey(1),
        fast_log_det.ProbeVectorType.RADEMACHER,
        dtype=self.dtype)
    _, tangent_out = fast_log_det.log_det_jvp(
        (M, pc, probe_vectors, num_probe_vectors),
        (M_dot, None, None, None),
        lambda a, b, c, d, **kwargs: 0.0,
        20,
    )
    truth = jnp.trace(jnp.linalg.inv(M) @ M_dot)
    self.assertAlmostEqual(tangent_out, truth, delta=0.02)

  def test_log_det_taylor_series_with_hutchinson(self):
    # For 1x1 arrays, for Rademacher probe vectors, v^t A v = A = tr A, so
    # only one probe vector is necessary.
    log_det = fast_log_det.log_det_taylor_series_with_hutchinson(
        jnp.array([1.5], dtype=self.dtype), 1, jax.random.PRNGKey(0)
    )
    self.assertAlmostEqual(log_det, math.log(1.5), places=3)
    log_det = fast_log_det.log_det_taylor_series_with_hutchinson(
        jnp.array([0.5], dtype=self.dtype), 1, jax.random.PRNGKey(1)
    )
    self.assertAlmostEqual(log_det, math.log(0.5), places=3)
    log_det = fast_log_det.log_det_taylor_series_with_hutchinson(
        jnp.array([[1.0, 0.1], [0.1, 1.0]], dtype=self.dtype),
        200,
        jax.random.PRNGKey(2),
    )
    self.assertAlmostEqual(log_det, math.log(0.99), places=3)

  def test_log_det_taylor_series_with_hutchinson_order_two_isnt_random(self):
    log_det = fast_log_det.log_det_taylor_series_with_hutchinson(
        jnp.array([[1.0, 0.1], [0.1, 1.0]], dtype=self.dtype),
        0,
        jax.random.PRNGKey(0),
        2,
    )
    self.assertAlmostEqual(log_det, math.log(0.99), places=3)
    log_det2 = fast_log_det.log_det_taylor_series_with_hutchinson(
        jnp.array([[1.0, 0.1], [0.1, 1.0]], dtype=self.dtype),
        0,
        jax.random.PRNGKey(1),
        2,
    )
    self.assertAlmostEqual(log_det, log_det2, places=8)


class FastLogDetTestFloat32(_FastLogDetTest):
  dtype = np.float32


class FastLogDetTestFloat64(_FastLogDetTest):
  dtype = np.float64


del _FastLogDetTest


if __name__ == '__main__':
  config.update('jax_enable_x64', True)
  absltest.main()
