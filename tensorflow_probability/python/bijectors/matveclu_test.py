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
"""Tests for MatvecLU Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


def trainable_lu_factorization(
    event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
  with tf.compat.v1.name_scope(name, 'trainable_lu_factorization',
                               [event_size, batch_shape]):
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
    lower_upper = tf.compat.v2.Variable(
        initial_value=lower_upper,
        trainable=True,
        name='lower_upper')
    return lower_upper, permutation


@test_util.run_all_in_graph_and_eager_modes
class MatvecLUTest(tf.test.TestCase):

  def test_invertible_from_trainable_lu_factorization(self):
    channels = 3
    conv1x1 = tfb.MatvecLU(*trainable_lu_factorization(channels, seed=42),
                           validate_args=True)

    self.evaluate(tf.compat.v1.global_variables_initializer())

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

  def test_invertible_from_lu(self):
    lower_upper, permutation = tf.linalg.lu(
        [[1., 2, 3],
         [4, 5, 6],
         [0.5, 0., 0.25]])

    conv1x1 = tfb.MatvecLU(lower_upper=lower_upper,
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


if __name__ == '__main__':
  tf.test.main()
