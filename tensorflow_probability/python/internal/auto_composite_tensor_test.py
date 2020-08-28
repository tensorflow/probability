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
"""Tests for auto_composite_tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import auto_composite_tensor as auto_ct
from tensorflow_probability.python.internal import test_util


AutoIdentity = auto_ct.auto_composite_tensor(tf.linalg.LinearOperatorIdentity)
AutoDiag = auto_ct.auto_composite_tensor(tf.linalg.LinearOperatorDiag)
AutoBlockDiag = auto_ct.auto_composite_tensor(tf.linalg.LinearOperatorBlockDiag)


class AutoCompositeTensorTest(test_util.TestCase):

  def test_example(self):
    @auto_ct.auto_composite_tensor
    class Adder(object):

      def __init__(self, x, y):
        self._x = tf.convert_to_tensor(x)
        self._y = tf.convert_to_tensor(y)

      def xpy(self):
        return self._x + self._y

    def body(obj):
      return Adder(obj.xpy(), 1.),

    result, = tf.while_loop(
        cond=lambda _: True,
        body=body,
        loop_vars=(Adder(1., 1.),),
        maximum_iterations=3)
    self.assertAllClose(5., result.xpy())

  def test_function(self):
    lop = AutoDiag(2. * tf.ones([3]))
    self.assertAllClose(
        6. * tf.ones([3]),
        tf.function(lambda lop: lop.matvec(3. * tf.ones([3])))(lop))

  def test_loop(self):
    def body(lop):
      return AutoDiag(lop.matvec(tf.ones([3]) * 2.)),
    init_lop = AutoDiag(tf.ones([3]))
    lop, = tf.while_loop(
        cond=lambda _: True,
        body=body,
        loop_vars=(init_lop,),
        maximum_iterations=3)
    self.assertAllClose(2.**3 * tf.ones([3]), lop.matvec(tf.ones([3])))

  def test_nested(self):
    lop = AutoBlockDiag([AutoDiag(tf.ones([2]) * 2), AutoIdentity(1)])
    self.assertAllClose(
        tf.constant([6., 6, 3]),
        tf.function(lambda lop: lop.matvec(3. * tf.ones([3])))(lop))


if __name__ == '__main__':
  tf.test.main()
