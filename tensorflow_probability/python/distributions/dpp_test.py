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
"""Tests for the DeterminantalPointProcess distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import bernoulli as bernoulli_lib
from tensorflow_probability.python.distributions import dpp as dpp_lib
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfpk = tfp.math.psd_kernels


@contextlib.contextmanager
def _capture_bernoulli_samples():
  """Use monkey-patching to capture the output of an Bernoulli sample."""
  observations = []
  true_sample = bernoulli_lib.Bernoulli.sample

  def _capturing_sample(
      self, sample_shape=(), seed=None, name='sample', **kwargs):
    samples = true_sample(self, sample_shape, seed, name, **kwargs)
    observations.append(samples)
    return samples

  bernoulli_lib.Bernoulli.sample = _capturing_sample
  try:
    yield observations
  finally:
    bernoulli_lib.Bernoulli.sample = true_sample


def kernel_over_unit_square(n_points, batch_shape=(), dtype=tf.float32):
  kernel = tfpk.ExponentiatedQuadratic(amplitude=tf.ones([], dtype),
                                       length_scale=0.1)
  pts = tfd.Uniform(tf.zeros([2], dtype), tf.ones([2], dtype)).sample(
      tuple(batch_shape) + (n_points,), seed=test_util.test_seed())
  kernel_mat = kernel.matrix(pts, pts) + 1e-3 * tf.eye(n_points, dtype=dtype)
  eigvals, eigvecs = tf.linalg.eigh(kernel_mat)
  return kernel_mat, eigvals, eigvecs


@test_util.test_all_tf_execution_regimes
class _DppTest(test_util.TestCase):

  fast_path_enabled = True
  param_dtype = tf.float32

  def setUp(self):
    dpp_lib.FAST_PATH_ENABLED = self.fast_path_enabled
    super().setUp()

  @parameterized.parameters(
      (
          [10],
          [5, 10, 10],
          [5],
          10,
      ),
      (
          [5, 10],
          [10, 10],
          [5],
          10,
      ),
      (
          [10],
          [10, 10],
          [],
          10,
      ),
      (
          [1, 10],
          [10, 10],
          [1],
          10,
      ),
      (
          [5, 3, 4],
          [4, 4],
          [5, 3],
          4,
      ),
  )
  def testShapes(self, eigvals_shape, eigvecs_shape, expected_batch_shape,
                 n_points):
    eigvals = tf.ones(eigvals_shape, dtype=self.param_dtype)
    eigvecs = tf.zeros(eigvecs_shape, dtype=self.param_dtype)
    dpp = tfd.DeterminantalPointProcess(eigvals, eigvecs)

    self.assertAllEqual(expected_batch_shape, dpp.batch_shape)
    self.assertAllEqual(expected_batch_shape, dpp.batch_shape_tensor())

    self.assertAllEqual([n_points], dpp.event_shape_tensor())
    self.assertAllEqual([n_points], dpp.event_shape)

  @parameterized.named_parameters(
      ('empty', (), 10, []),
      ('size5', (), 10, [0, 1, 4, 7, 9]),
      ('full', (), 10, list(range(10))),
      ('noindices', (), 10, None),
      ('batch_5', (5,), 3, [0, 2]),
      ('batch_6x3', (6, 3), 10, [0, 1, 5, 6]),
      ('batch_1x2x3', (1, 2, 3), 4, [1]),
      ('batch_noindices', (1, 2, 3), 4, None),
  )
  def testReconstructMatrix(self, batch_shape, n_points, indices):
    matrices, eigvals, eigvecs = self.evaluate(kernel_over_unit_square(
        n_points, batch_shape=batch_shape, dtype=self.param_dtype))
    if indices is not None:
      indices = np.array(indices, dtype=np.int32)
      one_hot_indices = tf.constant(
          [(1 if i in indices else 0) for i in range(n_points)], dtype=tf.int32)

      expected = matrices[..., indices[:, np.newaxis], indices]
    else:
      one_hot_indices = None
      expected = matrices

    reconstructed = dpp_lib._reconstruct_matrix(
        tf.constant(eigvals), tf.constant(eigvecs), one_hot_indices)
    if indices is None:
      # When no indices are specified, we can also check the full matrix.
      self.assertAllClose(expected, reconstructed, rtol=1e-5)
    # logdet must always agree, even if shape does not.
    self.assertAllClose(np.linalg.slogdet(expected)[1],
                        tf.linalg.logdet(reconstructed),
                        rtol=1e-5, atol=1e-5)

  @parameterized.parameters(
      ([5], [5, 5], 10, [10, 5]),
      ([2, 3, 5], [5, 5], 10, [10, 2, 3, 5]),
      ([5], [2, 3, 5, 5], 10, [10, 2, 3, 5]),
  )
  def testSampleElementaryDppShape(self, eigvals_shape, eigvecs_shape,
                                   n_samples, expected):
    eigvals = tf.ones(eigvals_shape, dtype=self.param_dtype)
    eigvecs = tf.eye(eigvecs_shape[-1], batch_shape=eigvecs_shape[:-2],
                     dtype=self.param_dtype)
    dpp = tfd.DeterminantalPointProcess(eigvals, eigvecs)
    with _capture_bernoulli_samples() as sampled_edpp_indices:
      dpp.sample(n_samples, seed=test_util.test_seed())
    self.assertLen(sampled_edpp_indices, 1)
    self.assertAllEqual(sampled_edpp_indices[0].shape, expected)

  def testSampleElementaryDppPoints(self):
    """Checks we don't sample points with corresponding eigenvalue = 0."""
    n_points, batch_size = 5, 10
    _, eigvals, eigvecs = kernel_over_unit_square(n_points,
                                                  dtype=self.param_dtype)
    eigvals = tf.one_hot(1, n_points, dtype=self.param_dtype)
    dpp = tfd.DeterminantalPointProcess(eigvals, eigvecs)
    with _capture_bernoulli_samples() as sampled_edpp_indices:
      dpp.sample(batch_size, seed=test_util.test_seed())
    self.assertLen(sampled_edpp_indices, 1)
    sample = sampled_edpp_indices[0]
    self.assertAllEqual(sample[:, 0], tf.zeros([batch_size]))
    self.assertAllEqual(sample[:, 1] * (1 - sample[:, 1]),
                        tf.zeros([batch_size]))
    self.assertAllEqual(sample[:, 2:], tf.zeros([batch_size, n_points - 2]))

  def testSampleFromEDppSize(self):
    """Tests that the selected E-DPP size is equal to the sampled size."""
    n_points = 5
    edpp_indices = tf.constant(
        [
            [0] * n_points,  # Empty set.
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [1] * n_points
        ],
        dtype=tf.int32)
    eigenvectors = tf.random.uniform(
        [edpp_indices.shape[0], n_points, n_points], seed=test_util.test_seed(),
        dtype=self.param_dtype)
    samples = self.evaluate(
        dpp_lib._sample_from_edpp(eigenvectors, edpp_indices,
                                  seed=test_util.test_seed()))
    actual_sizes = np.sum(samples, axis=-1)
    expected_sizes = tf.reduce_sum(edpp_indices, axis=-1)
    self.assertAllEqual(actual_sizes, expected_sizes)

  def testSampleFromEDppDeterministic(self):
    """Tests that for diagonal kernels, we select points w/ eigenvalues > 0."""
    edpp_indices = tf.constant([0, 0, 0, 1, 1, 1, 0, 1, 0, 0], dtype=tf.int32)
    eigvecs = tf.eye(10, dtype=self.param_dtype)
    samples = dpp_lib._sample_from_edpp(
        eigvecs, edpp_indices, seed=test_util.test_seed())
    self.assertAllEqual(edpp_indices, samples)

  def testOrthogonalComplementEi(self):
    """Checks that row i=0 after orthogonalization."""
    dim, n_vectors, i = 10, 5, 3
    vectors = tf.random.normal([dim, n_vectors], seed=test_util.test_seed(),
                               dtype=self.param_dtype)

    ortho = self.evaluate(dpp_lib._orthogonal_complement_e_i(vectors, i,
                                                             n_vectors))

    self.assertAllEqual(ortho.shape, [dim, n_vectors])
    self.assertAllClose(ortho[:, -1], np.zeros(dim))
    self.assertAllClose(ortho[i, :], np.zeros(n_vectors))

  def testDppLEnsembleMatrix(self):
    n_points = 20
    true_kernel, eigvals, eigvecs = self.evaluate(
        kernel_over_unit_square(n_points, dtype=self.param_dtype))
    dpp = tfd.DeterminantalPointProcess(eigvals, eigvecs)
    self.assertAllClose(
        true_kernel,
        dpp.l_ensemble_matrix(),
        rtol=1e-5, atol=1e-5)

  def testDppMarginalKernel(self):
    n_points = 20
    true_kernel, eigvals, eigvecs = kernel_over_unit_square(
        n_points, dtype=self.param_dtype)
    dpp = tfd.DeterminantalPointProcess(eigvals, eigvecs)
    marginal_kernel = tf.matmul(true_kernel,
                                tf.linalg.inv(true_kernel + np.eye(n_points)))
    self.assertAllClose(
        marginal_kernel - dpp.marginal_kernel(),
        tf.zeros([n_points, n_points]),
        atol=1e-5)

  @parameterized.named_parameters(dict(testcase_name='_3', n_points=3),
                                  dict(testcase_name='_4', n_points=4),
                                  dict(testcase_name='_5', n_points=5))
  def testDppLogPDF(self, n_points):
    true_kernel, eigvals, eigvecs = self.evaluate(
        kernel_over_unit_square(n_points, dtype=self.param_dtype))
    dpp = tfd.DeterminantalPointProcess(eigvals, eigvecs)
    log_probs = []
    for i in range(2**n_points):  # n_points is small so we can enumerate sets.
      binary = bin(i)[2:]
      subset = [0] * n_points
      subset[-len(binary):] = [int(c) for c in binary]
      mask = np.array(subset, np.bool)
      submatrix = true_kernel[mask][:, mask]
      expected = (tf.linalg.logdet(submatrix) -
                  tf.linalg.logdet(true_kernel +
                                   tf.eye(n_points, dtype=self.param_dtype)))
      log_probs.append(dpp.log_prob(tf.constant(subset)))
      self.assertAllClose(expected, log_probs[-1],
                          atol=1e-4,
                          msg=str(subset))
    self.assertAllClose(1., tf.reduce_sum(tf.math.exp(log_probs)))

  def testDppSample(self):
    n_points = 50
    _, eigvals, eigvecs = kernel_over_unit_square(n_points,
                                                  dtype=self.param_dtype)
    dpp = tfd.DeterminantalPointProcess(eigvals, eigvecs)
    n = 10
    samples = dpp.sample(n, seed=test_util.test_seed())
    self.assertEqual(samples.shape, (n, n_points))

  def testDppSampleStats(self):
    n_points = 5
    _, eigvals, eigvecs = kernel_over_unit_square(n_points,
                                                  dtype=self.param_dtype)
    dpp = tfd.DeterminantalPointProcess(eigvals, eigvecs)

    n = 500
    samples, expected_marginals = self.evaluate(
        [dpp.sample(n, seed=test_util.test_seed()),
         tf.linalg.diag_part(dpp.marginal_kernel())])

    counts = np.zeros(n_points)
    for s in samples:
      counts[np.nonzero(s)[0]] += 1.

    self.assertAllClose(counts / n, expected_marginals, atol=.1)

  def testEigvalsAsserts(self):
    with self.assertRaisesOpError(r'must be positive'):
      dpp = tfd.DeterminantalPointProcess(
          tf.constant([1, 2, 3, 0.], dtype=self.param_dtype),
          tf.eye(4, dtype=self.param_dtype),
          validate_args=True)
      self.evaluate(dpp.sample(seed=test_util.test_seed()))

    v = tf.Variable(tf.constant([1, 2, -3, 4.], dtype=self.param_dtype))
    self.evaluate(v.initializer)
    dpp = tfd.DeterminantalPointProcess(v, tf.eye(4, dtype=self.param_dtype),
                                        validate_args=True)
    with self.assertRaisesOpError(r'must be positive'):
      self.evaluate(dpp.sample(seed=test_util.test_seed()))

  def testEigvecsAsserts(self):
    with self.assertRaisesOpError(r'must be orthonormal'):
      dpp = tfd.DeterminantalPointProcess(
          tf.ones([4], dtype=self.param_dtype),
          tf.ones([4, 4], dtype=self.param_dtype) / 2,
          validate_args=True)
      self.evaluate(dpp.sample(seed=test_util.test_seed()))

    v = tf.Variable(tf.ones([4, 4], dtype=self.param_dtype) / 2)
    self.evaluate(v.initializer)
    dpp = tfd.DeterminantalPointProcess(tf.ones([4], dtype=self.param_dtype), v,
                                        validate_args=True)
    with self.assertRaisesOpError(r'must be orthonormal'):
      self.evaluate(dpp.sample(seed=test_util.test_seed()))

    with self.assertRaisesOpError(r'must be orthonormal'):
      dpp = dpp.copy(eigenvectors=tf.eye(4, dtype=self.param_dtype) * .1)
      self.evaluate(dpp.sample(seed=test_util.test_seed()))

    self.evaluate(v.assign(tf.eye(4, dtype=self.param_dtype) * .1))
    dpp = dpp.copy(eigenvectors=v)
    with self.assertRaisesOpError(r'must be orthonormal'):
      self.evaluate(dpp.sample(seed=test_util.test_seed()))

  def testXLASample(self):
    self.skip_if_no_xla()

    _, eigvals, eigvecs = kernel_over_unit_square(20, dtype=self.param_dtype)
    for n in 1, 5:
      @tf.function(jit_compile=True)
      def f(eigvals):
        return tfd.DeterminantalPointProcess(eigvals, eigvecs).sample(
            n, seed=test_util.test_seed())  # pylint: disable=cell-var-from-loop
      self.evaluate(f(eigvals))


class DppTestFast32(_DppTest):
  param_dtype = tf.float32
  fast_path_enabled = True


class DppTestFast64(_DppTest):
  param_dtype = tf.float64
  fast_path_enabled = True


class DppTest32(_DppTest):
  param_dtype = tf.float32
  fast_path_enabled = False


class DppTest64(_DppTest):
  param_dtype = tf.float64
  fast_path_enabled = False


del _DppTest


if __name__ == '__main__':
  tf.test.main()
