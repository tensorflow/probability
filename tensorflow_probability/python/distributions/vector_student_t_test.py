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
"""Tests for MultivariateStudentsT Distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import linalg
from scipy import special
import tensorflow as tf

from tensorflow_probability.python.distributions.vector_student_t import _VectorStudentT

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


class _FakeVectorStudentT(object):
  """Fake scipy implementation for Multivariate Student's t-distribution.

  Technically we don't need to test the `Vector Student's t-distribution` since
  its composed of only unit-tested parts.  However this _FakeVectorStudentT
  serves as something like an end-to-end test of the
  `TransformedDistribution + Affine` API.

  Other `Vector*` implementations need only test new code. That we don't need
  to test every Vector* distribution is good because there aren't SciPy
  analogs and reimplementing everything in NumPy sort of defeats the point of
  having the `TransformedDistribution + Affine` API.
  """

  def __init__(self, df, loc, scale_tril):
    self._df = np.asarray(df)
    self._loc = np.asarray(loc)
    self._scale_tril = np.asarray(scale_tril)

  def log_prob(self, x):
    def _compute(df, loc, scale_tril, x):
      k = scale_tril.shape[-1]
      ildj = np.sum(np.log(np.abs(np.diag(scale_tril))), axis=-1)
      logz = ildj + k * (0.5 * np.log(df) +
                         0.5 * np.log(np.pi) +
                         special.gammaln(0.5 * df) -
                         special.gammaln(0.5 * (df + 1.)))
      y = linalg.solve_triangular(scale_tril, np.matrix(x - loc).T,
                                  lower=True, overwrite_b=True)
      logs = -0.5 * (df + 1.) * np.sum(np.log1p(y**2. / df), axis=-2)
      return logs - logz
    if not self._df.shape:
      return _compute(self._df, self._loc, self._scale_tril, x)
    return np.concatenate([
        [_compute(self._df[i], self._loc[i], self._scale_tril[i], x[:, i, :])]
        for i in range(len(self._df))]).T

  def prob(self, x):
    return np.exp(self.log_prob(x))


@test_util.run_all_in_graph_and_eager_modes
class VectorStudentTTest(tf.test.TestCase):

  def setUp(self):
    super(VectorStudentTTest, self).setUp()
    self._rng = np.random.RandomState(42)

  def testProbStaticScalar(self):
    # Scalar batch_shape.
    df = np.asarray(3., dtype=np.float32)
    # Scalar batch_shape.
    loc = np.asarray([1], dtype=np.float32)
    scale_diag = np.asarray([2.], dtype=np.float32)
    scale_tril = np.diag(scale_diag)

    expected_mst = _FakeVectorStudentT(df=df, loc=loc, scale_tril=scale_tril)

    actual_mst = _VectorStudentT(
        df=df, loc=loc, scale_diag=scale_diag, validate_args=True)
    x = 2. * self._rng.rand(4, 1).astype(np.float32) - 1.

    self.assertAllClose(
        expected_mst.log_prob(x),
        self.evaluate(actual_mst.log_prob(x)),
        rtol=0.,
        atol=1e-5)
    self.assertAllClose(
        expected_mst.prob(x),
        self.evaluate(actual_mst.prob(x)),
        rtol=0.,
        atol=1e-5)

  def testProbStatic(self):
    # Non-scalar batch_shape.
    df = np.asarray([1., 2, 3], dtype=np.float32)
    # Non-scalar batch_shape.
    loc = np.asarray([[0., 0, 0],
                      [1, 2, 3],
                      [1, 0, 1]],
                     dtype=np.float32)
    scale_diag = np.asarray([[1., 2, 3],
                             [2, 3, 4],
                             [4, 5, 6]],
                            dtype=np.float32)
    scale_tril = np.concatenate([[np.diag(scale_diag[i])]
                                 for i in range(len(scale_diag))])
    x = 2. * self._rng.rand(4, 3, 3).astype(np.float32) - 1.

    expected_mst = _FakeVectorStudentT(
        df=df, loc=loc, scale_tril=scale_tril)

    actual_mst = _VectorStudentT(
        df=df, loc=loc, scale_diag=scale_diag, validate_args=True)
    self.assertAllClose(
        expected_mst.log_prob(x),
        self.evaluate(actual_mst.log_prob(x)),
        rtol=0.,
        atol=1e-5)
    self.assertAllClose(
        expected_mst.prob(x),
        self.evaluate(actual_mst.prob(x)),
        rtol=0.,
        atol=1e-5)

  def testProbDynamic(self):
    # Non-scalar batch_shape.
    df = np.asarray([1., 2, 3], dtype=np.float32)
    # Non-scalar batch_shape.
    loc = np.asarray([[0., 0, 0],
                      [1, 2, 3],
                      [1, 0, 1]],
                     dtype=np.float32)
    scale_diag = np.asarray([[1., 2, 3],
                             [2, 3, 4],
                             [4, 5, 6]],
                            dtype=np.float32)
    scale_tril = np.concatenate([[np.diag(scale_diag[i])]
                                 for i in range(len(scale_diag))])
    x = 2. * self._rng.rand(4, 3, 3).astype(np.float32) - 1.

    expected_mst = _FakeVectorStudentT(
        df=df, loc=loc, scale_tril=scale_tril)

    df_pl = tf.compat.v1.placeholder_with_default(
        input=df, shape=df.shape, name="df")
    loc_pl = tf.compat.v1.placeholder_with_default(
        input=loc, shape=loc.shape, name="loc")
    scale_diag_pl = tf.compat.v1.placeholder_with_default(
        input=scale_diag, shape=scale_diag.shape, name="scale_diag")
    actual_mst = _VectorStudentT(
        df=df_pl, loc=loc_pl, scale_diag=scale_diag_pl, validate_args=True)
    self.assertAllClose(
        expected_mst.log_prob(x),
        self.evaluate(actual_mst.log_prob(x)),
        rtol=0.,
        atol=1e-5)
    self.assertAllClose(
        expected_mst.prob(x),
        self.evaluate(actual_mst.prob(x)),
        rtol=0.,
        atol=1e-5)

  def testProbScalarBaseDistributionNonScalarTransform(self):
    # Scalar batch_shape.
    df = np.asarray(2., dtype=np.float32)
    # Non-scalar batch_shape.
    loc = np.asarray([[0., 0, 0],
                      [1, 2, 3],
                      [1, 0, 1]],
                     dtype=np.float32)
    scale_diag = np.asarray([[1., 2, 3],
                             [2, 3, 4],
                             [4, 5, 6]],
                            dtype=np.float32)
    scale_tril = np.concatenate([[np.diag(scale_diag[i])]
                                 for i in range(len(scale_diag))])
    x = 2. * self._rng.rand(4, 3, 3).astype(np.float32) - 1.

    expected_mst = _FakeVectorStudentT(
        df=np.tile(df, reps=len(scale_diag)),
        loc=loc,
        scale_tril=scale_tril)

    actual_mst = _VectorStudentT(
        df=df, loc=loc, scale_diag=scale_diag, validate_args=True)
    self.assertAllClose(
        expected_mst.log_prob(x),
        self.evaluate(actual_mst.log_prob(x)),
        rtol=0.,
        atol=1e-5)
    self.assertAllClose(
        expected_mst.prob(x),
        self.evaluate(actual_mst.prob(x)),
        rtol=0.,
        atol=1e-5)

  def testProbScalarBaseDistributionNonScalarTransformDynamic(self):
    # Scalar batch_shape.
    df = np.asarray(2., dtype=np.float32)
    # Non-scalar batch_shape.
    loc = np.asarray([[0., 0, 0],
                      [1, 2, 3],
                      [1, 0, 1]],
                     dtype=np.float32)
    scale_diag = np.asarray([[1., 2, 3],
                             [2, 3, 4],
                             [4, 5, 6]],
                            dtype=np.float32)
    scale_tril = np.concatenate([[np.diag(scale_diag[i])]
                                 for i in range(len(scale_diag))])
    x = 2. * self._rng.rand(4, 3, 3).astype(np.float32) - 1.

    expected_mst = _FakeVectorStudentT(
        df=np.tile(df, reps=len(scale_diag)),
        loc=loc,
        scale_tril=scale_tril)

    df_pl = tf.compat.v1.placeholder_with_default(
        input=df, shape=df.shape, name="df")
    loc_pl = tf.compat.v1.placeholder_with_default(
        input=loc, shape=loc.shape, name="loc")
    scale_diag_pl = tf.compat.v1.placeholder_with_default(
        input=scale_diag, shape=scale_diag.shape, name="scale_diag")
    actual_mst = _VectorStudentT(
        df=df_pl, loc=loc_pl, scale_diag=scale_diag_pl, validate_args=True)
    self.assertAllClose(
        expected_mst.log_prob(x),
        self.evaluate(actual_mst.log_prob(x)),
        rtol=0.,
        atol=1e-5)
    self.assertAllClose(
        expected_mst.prob(x),
        self.evaluate(actual_mst.prob(x)),
        rtol=0.,
        atol=1e-5)

  def testProbNonScalarBaseDistributionScalarTransform(self):
    # Non-scalar batch_shape.
    df = np.asarray([1., 2., 3.], dtype=np.float32)
    # Scalar batch_shape.
    loc = np.asarray([1, 2, 3], dtype=np.float32)
    scale_diag = np.asarray([2, 3, 4], dtype=np.float32)
    scale_tril = np.diag(scale_diag)
    x = 2. * self._rng.rand(4, 3, 3).astype(np.float32) - 1.

    expected_mst = _FakeVectorStudentT(
        df=df,
        loc=np.tile(loc[tf.newaxis, :], reps=[len(df), 1]),
        scale_tril=np.tile(scale_tril[tf.newaxis, :, :], reps=[len(df), 1, 1]))

    actual_mst = _VectorStudentT(
        df=df, loc=loc, scale_diag=scale_diag, validate_args=True)
    self.assertAllClose(
        expected_mst.log_prob(x),
        self.evaluate(actual_mst.log_prob(x)),
        rtol=0.,
        atol=1e-5)
    self.assertAllClose(
        expected_mst.prob(x),
        self.evaluate(actual_mst.prob(x)),
        rtol=0.,
        atol=1e-5)

  def testProbNonScalarBaseDistributionScalarTransformDynamic(self):
    # Non-scalar batch_shape.
    df = np.asarray([1., 2., 3.], dtype=np.float32)
    # Scalar batch_shape.
    loc = np.asarray([1, 2, 3], dtype=np.float32)
    scale_diag = np.asarray([2, 3, 4], dtype=np.float32)
    scale_tril = np.diag(scale_diag)

    x = 2. * self._rng.rand(4, 3, 3).astype(np.float32) - 1.

    expected_mst = _FakeVectorStudentT(
        df=df,
        loc=np.tile(loc[tf.newaxis, :], reps=[len(df), 1]),
        scale_tril=np.tile(scale_tril[tf.newaxis, :, :], reps=[len(df), 1, 1]))

    df_pl = tf.compat.v1.placeholder_with_default(
        input=df, shape=df.shape, name="df")
    loc_pl = tf.compat.v1.placeholder_with_default(
        input=loc, shape=loc.shape, name="loc")
    scale_diag_pl = tf.compat.v1.placeholder_with_default(
        input=scale_diag, shape=scale_diag.shape, name="scale_diag")
    actual_mst = _VectorStudentT(
        df=df_pl, loc=loc_pl, scale_diag=scale_diag_pl, validate_args=True)
    self.assertAllClose(
        expected_mst.log_prob(x),
        self.evaluate(actual_mst.log_prob(x)),
        rtol=0.,
        atol=1e-5)
    self.assertAllClose(
        expected_mst.prob(x),
        self.evaluate(actual_mst.prob(x)),
        rtol=0.,
        atol=1e-5)


if __name__ == "__main__":
  tf.test.main()
