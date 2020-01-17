# Copyright 2019 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for `internal.backend.numpy.TensorArray`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np  # Rewritten by script to import jax.numpy
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util as test_util
from tensorflow_probability.python.internal.backend import numpy as nptf

JAX_MODE = False


class TensorArrayTest(test_util.TestCase):

  def test_jax_jit(self):
    if not JAX_MODE:
      self.skipTest('JAX only')

    import jax  # pylint: disable=g-import-not-at-top
    @jax.jit
    def f(ta, index, value):
      return ta.write(index, value)

    ta = nptf.compat.v2.TensorArray(
        dtype=tf.float32, size=4, element_shape=(3,))
    ta = f(ta, 1, [2, 3, 4.])
    ta = f(ta, 3, [4, 5, 6.])
    self.assertAllEqual([[0, 0, 0], [2, 3, 4], [0, 0, 0], [4, 5, 6.]],
                        ta.stack())

  def test_write_read(self):
    ta = nptf.compat.v2.TensorArray(
        dtype=tf.float32,
        tensor_array_name='foo',
        infer_shape=False,
        dynamic_size=True)
    w0 = ta.write(0, [[4., 5.]])
    w1 = w0.write(1, [[1.]])
    w2 = w1.write(2, -3.)
    r0 = w2.read(0)
    r1 = w2.read(1)
    r2 = w2.read(2)
    self.assertAllEqual([[4.0, 5.0]], r0)
    self.assertAllEqual([[1.0]], r1)
    self.assertAllEqual(-3.0, r2)

  def test_gather(self):
    ta = nptf.compat.v2.TensorArray(
        dtype=tf.float32, tensor_array_name='foo', size=3)
    w0 = ta.write(0, -3.)
    w1 = w0.write(1, -2.)
    w2 = w1.write(2, -1.)
    self.assertAllEqual([-3., -3., -1., -2.], w2.gather([0, 0, 2, 1]))

  def test_gather_nonscalar_elements(self):
    ta = nptf.compat.v2.TensorArray(
        dtype=tf.int32, tensor_array_name='foo', size=3).unstack(
            np.int32([[0, 0], [1, 1], [2, 2]]))
    self.assertAllEqual([1, 1], ta.gather([1])[0])

  def test_stack(self):
    ta = nptf.compat.v2.TensorArray(
        dtype=tf.float32, tensor_array_name='foo', size=3)
    w0 = ta.write(0, -3.)
    w1 = w0.write(1, -2.)
    w2 = w1.write(2, -1.)
    self.assertAllEqual(np.array([-3., -2., -1.]), w2.stack())

  def test_unstack(self):
    ta = nptf.compat.v2.TensorArray(
        dtype=tf.float32, size=3, tensor_array_name='foo')
    value = np.array([[1., -1.], [10., -10.]])
    w = ta.unstack(value)
    r0 = w.read(0)
    r1 = w.read(1)
    self.assertAllEqual([1., -1.], r0)
    self.assertAllEqual([10., -10.], r1)


if __name__ == '__main__':
  tf.test.main()
