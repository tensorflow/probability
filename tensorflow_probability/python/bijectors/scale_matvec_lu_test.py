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
"""Tests for ScaleMatvecLU Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


def trainable_lu_factorization(
    event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
  with tf.name_scope(name or 'trainable_lu_factorization'):
    event_size = tf.convert_to_tensor(
        value=event_size, dtype_hint=tf.int32, name='event_size')
    batch_shape = tf.convert_to_tensor(
        value=batch_shape, dtype_hint=event_size.dtype, name='batch_shape')
    random_matrix = tf.random.uniform(
        shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
        dtype=dtype,
        seed=seed)
    random_orthonormal = tf.linalg.qr(random_matrix)[0]
    lower_upper, permutation = tf.linalg.lu(random_orthonormal)
    lower_upper = tf.Variable(
        initial_value=lower_upper, trainable=True, name='lower_upper')
    # Initialize a non-trainable variable for the permutation indices so
    # that its value isn't re-sampled from run-to-run.
    permutation = tf.Variable(
        initial_value=permutation, trainable=False, name='permutation')
    return lower_upper, permutation


@test_util.test_all_tf_execution_regimes
class ScaleMatvecLUTest(test_util.TestCase):

  def test_invertible_from_trainable_lu_factorization(self):
    channels = 3
    lower_upper, permutation = trainable_lu_factorization(channels, seed=42)
    conv1x1 = tfb.ScaleMatvecLU(lower_upper, permutation, validate_args=True)

    self.assertIs(lower_upper, conv1x1.lower_upper)
    self.evaluate([v.initializer for v in conv1x1.variables])

    x = tf.random.uniform(shape=[2, 28, 28, channels])

    fwd = conv1x1.forward(x)
    rev_fwd = conv1x1.inverse(fwd)
    fldj = conv1x1.forward_log_det_jacobian(x, event_ndims=3)

    rev = conv1x1.inverse(x)
    fwd_rev = conv1x1.forward(rev)
    ildj = conv1x1.inverse_log_det_jacobian(x, event_ndims=3)

    [x_, fwd_, rev_, fwd_rev_, rev_fwd_, fldj_, ildj_] = self.evaluate([
        x, fwd, rev, fwd_rev, rev_fwd, fldj, ildj])

    self.assertAllClose(x_, fwd_rev_, atol=1e-3, rtol=1e-6)
    self.assertAllClose(x_, rev_fwd_, atol=1e-3, rtol=1e-6)

    self.assertEqual(fldj_, -ildj_)
    self.assertNear(0., fldj_, err=1e-3)

    # We now check that the bijector isn't simply the identity function. We do
    # this by checking that at least 50% of pixels differ by at least 10%.
    self.assertTrue(np.mean(np.abs(x_ - fwd_) > 0.1 * x_) > 0.5)
    self.assertTrue(np.mean(np.abs(x_ - rev_) > 0.1 * x_) > 0.5)

  def test_trainable_lu_factorization_init(self):
    """Initial LU factorization parameters do not change per execution."""
    channels = 8
    lower_upper, permutation = trainable_lu_factorization(channels, seed=42)
    conv1x1 = tfb.ScaleMatvecLU(lower_upper, permutation, validate_args=True)

    self.evaluate([v.initializer for v in conv1x1.variables])

    lower_upper_1, permutation_1 = self.evaluate([lower_upper, permutation])
    lower_upper_2, permutation_2 = self.evaluate([lower_upper, permutation])

    self.assertAllEqual(lower_upper_1, lower_upper_2)
    self.assertAllEqual(permutation_1, permutation_2)

  def test_invertible_from_lu(self):
    lower_upper, permutation = tf.linalg.lu(
        [[1., 2, 3],
         [4, 5, 6],
         [0.5, 0., 0.25]])

    conv1x1 = tfb.ScaleMatvecLU(lower_upper=lower_upper,
                                permutation=permutation,
                                validate_args=True)

    channels = tf.compat.dimension_value(lower_upper.shape[-1])
    x = tf.random.uniform(shape=[2, 28, 28, channels])

    fwd = conv1x1.forward(x)
    rev_fwd = conv1x1.inverse(fwd)
    fldj = conv1x1.forward_log_det_jacobian(x, event_ndims=3)

    rev = conv1x1.inverse(x)
    fwd_rev = conv1x1.forward(rev)
    ildj = conv1x1.inverse_log_det_jacobian(x, event_ndims=3)

    [x_, fwd_, rev_, fwd_rev_, rev_fwd_, fldj_, ildj_] = self.evaluate([
        x, fwd, rev, fwd_rev, rev_fwd, fldj, ildj])

    self.assertAllClose(x_, fwd_rev_, atol=1e-3, rtol=1e-6)
    self.assertAllClose(x_, rev_fwd_, atol=1e-3, rtol=1e-6)

    self.assertEqual(fldj_, -ildj_)
    self.assertTrue(fldj_ > 1.)  # Notably, bounded away from zero.

    # We now check that the bijector isn't simply the identity function. We do
    # this by checking that at least 50% of pixels differ by at least 10%.
    self.assertTrue(np.mean(np.abs(x_ - fwd_) > 0.1 * x_) > 0.5)
    self.assertTrue(np.mean(np.abs(x_ - rev_) > 0.1 * x_) > 0.5)

  def testTheoreticalFldj(self):
    raw_mat = tf.constant([[1., 2, 3],
                           [4, 5, 6],
                           [0.5, 0., 0.25]])
    nbatch = 5
    batch_mats = raw_mat * tf.range(1., nbatch + 1.)[:, tf.newaxis, tf.newaxis]
    lower_upper, permutation = tf.linalg.lu(tf.cast(batch_mats, tf.float64))

    bijector = tfb.ScaleMatvecLU(
        lower_upper=lower_upper, permutation=permutation, validate_args=True)
    self.assertEqual(tf.float64, bijector.dtype)

    channels = tf.compat.dimension_value(lower_upper.shape[-1])
    x = np.random.uniform(size=[2, 7, nbatch, channels]).astype(np.float64)
    y = self.evaluate(bijector.forward(x))
    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x,
        y,
        eval_func=self.evaluate,
        event_ndims=1,
        inverse_event_ndims=1,
        rtol=1e-5)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    # The jacobian is not yet broadcast, since it is constant.
    fldj = fldj + tf.zeros(tf.shape(x)[:-1], dtype=x.dtype)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector, x, event_ndims=1)
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)

  def testNonInvertibleLUAssert(self):
    lower_upper, permutation = self.evaluate(
        tf.linalg.lu([[1., 2, 3], [4, 5, 6], [0.5, 0., 0.25]]))
    lower_upper = tf.Variable(lower_upper)
    self.evaluate(lower_upper.initializer)
    bijector = tfb.ScaleMatvecLU(
        lower_upper=lower_upper, permutation=permutation, validate_args=True)

    self.evaluate(bijector.forward([1., 2, 3]))

    with tf.control_dependencies([
        lower_upper[1, 1].assign(-lower_upper[1, 1])]):
      self.evaluate(bijector.forward([1., 2, 3]))

    with self.assertRaisesOpError('`lower_upper` must have nonzero diagonal'):
      with tf.control_dependencies([lower_upper[1, 1].assign(0)]):
        self.evaluate(bijector.forward([1., 2, 3]))


if __name__ == '__main__':
  tf.test.main()
