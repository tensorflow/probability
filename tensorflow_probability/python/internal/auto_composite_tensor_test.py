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

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


AutoIdentity = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorIdentity, omit_kwargs=('name',))
AutoDiag = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorDiag, omit_kwargs=('name',))
AutoBlockDiag = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorBlockDiag, omit_kwargs=('name',))
AutoTriL = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorLowerTriangular, omit_kwargs=('name',))


@test_util.test_all_tf_execution_regimes
class AutoCompositeTensorTest(test_util.TestCase):

  def test_example(self):
    @tfp.experimental.auto_composite_tensor(omit_kwargs=('name',))
    class Adder(object):

      def __init__(self, x, y, name=None):
        with tf.name_scope(name or 'Adder') as name:
          self._x = tf.convert_to_tensor(x)
          self._y = tf.convert_to_tensor(y)
          self._name = name

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

  def test_preconditioner(self):
    xs = self.evaluate(tf.random.uniform([30, 30], seed=test_util.test_seed()))
    cov_linop = tf.linalg.LinearOperatorFullMatrix(
        tf.matmul(xs, xs, transpose_b=True) + tf.linalg.eye(30) * 1e-3,
        is_self_adjoint=True,
        is_positive_definite=True)

    tfd = tfp.experimental.distributions
    auto_ct_mvn_prec_linop = tfp.experimental.auto_composite_tensor(
        tfd.MultivariateNormalPrecisionFactorLinearOperator,
        omit_kwargs=('name',))
    tril = AutoTriL(**cov_linop.cholesky().parameters)
    momentum_distribution = auto_ct_mvn_prec_linop(precision_factor=tril)
    def body(d):
      return d.copy(precision_factor=AutoTriL(
          **dict(d.precision_factor.parameters,
                 tril=d.precision_factor.to_dense() + tf.linalg.eye(30),))),
    after_loop = tf.while_loop(lambda d: True, body, (momentum_distribution,),
                               maximum_iterations=1)
    tf.nest.map_structure(self.evaluate,
                          after_loop,
                          expand_composites=True)

  def test_already_ct_subclass(self):

    @tfp.experimental.auto_composite_tensor
    class MyCT(tfp.experimental.AutoCompositeTensor):

      def __init__(self, tensor_param, non_tensor_param, maybe_tensor_param):
        self._tensor_param = tf.convert_to_tensor(tensor_param)
        self._non_tensor_param = non_tensor_param
        self._maybe_tensor_param = maybe_tensor_param

    def body(obj):
      return MyCT(obj._tensor_param + 1,
                  obj._non_tensor_param,
                  obj._maybe_tensor_param),

    init = MyCT(0., 0, 0)
    result, = tf.while_loop(
        cond=lambda *_: True,
        body=body,
        loop_vars=(init,),
        maximum_iterations=3)
    self.assertAllClose(3., result._tensor_param)

    init = MyCT(0., 0, tf.constant(0))
    result, = tf.while_loop(
        cond=lambda *_: True,
        body=body,
        loop_vars=(init,),
        maximum_iterations=3)
    self.assertAllClose(3., result._tensor_param)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
