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

import functools

import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions


AutoIdentity = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorIdentity, omit_kwargs=('name',))
AutoDiag = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorDiag, omit_kwargs=('name',))
AutoBlockDiag = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorBlockDiag, omit_kwargs=('name',))
AutoTriL = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorLowerTriangular, omit_kwargs=('name',))

AutoNormal = tfp.experimental.auto_composite_tensor(
    tfd.Normal, omit_kwargs=('name',))
AutoIndependent = tfp.experimental.auto_composite_tensor(
    tfd.Independent, omit_kwargs=('name',))
AutoReshape = tfp.experimental.auto_composite_tensor(
    tfb.Reshape, omit_kwargs=('name',))


@test_util.test_all_tf_execution_regimes
class AutoCompositeTensorTest(test_util.TestCase):

  def test_example(self):
    @tfp.experimental.auto_composite_tensor(omit_kwargs=('name',))
    class Adder(object):

      def __init__(self, x, y, name=None):
        with tf.name_scope(name or 'Adder') as name:
          self._x = tensor_util.convert_nonref_to_tensor(x)
          self._y = tensor_util.convert_nonref_to_tensor(y)
          self._name = name

      def xpy(self):
        return self._x + self._y

    x = 1.
    y = tf.Variable(1.)
    self.evaluate(y.initializer)

    def body(obj):
      return Adder(obj.xpy(), y),

    result, = tf.while_loop(
        cond=lambda _: True,
        body=body,
        loop_vars=(Adder(x, y),),
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

  def test_shape_parameters(self):
    dist = AutoIndependent(AutoNormal(0, tf.ones([1])),
                           reinterpreted_batch_ndims=1)
    stream = test_util.test_seed_stream()
    lp = dist.log_prob(dist.sample(seed=stream()))
    lp, _ = tf.while_loop(
        lambda *_: True,
        lambda lp, d: (d.log_prob(d.sample(seed=stream())), d),
        (lp, dist),
        maximum_iterations=2)
    self.evaluate(lp)

  def test_prefer_static_shape_params(self):
    @tf.function
    def f(b):
      return b
    b = AutoReshape(
        event_shape_out=[2, 3],
        event_shape_in=[tf.reduce_prod([2, 3])])  # Tensor in a list.
    f(b)

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

    tfed = tfp.experimental.distributions
    auto_ct_mvn_prec_linop = tfp.experimental.auto_composite_tensor(
        tfed.MultivariateNormalPrecisionFactorLinearOperator,
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

  def test_parameters_lookup(self):

    @tfp.experimental.auto_composite_tensor
    class ThingWithParametersButNoAttrs(tfp.experimental.AutoCompositeTensor):

      def __init__(self, a, b):
        self.a = tf.convert_to_tensor(a, dtype_hint=tf.float32, name='a')
        self.b = tf.convert_to_tensor(b, dtype_hint=tf.float32, name='a')
        self.parameters = dict(a=self.a, b=self.b)

    t = ThingWithParametersButNoAttrs(1., 2.)
    self.assertIsInstance(t, tf.__internal__.CompositeTensor)

    ts = t._type_spec
    components = ts._to_components(t)
    self.assertAllEqualNested(components, dict(a=1., b=2.))

    t2 = ts._from_components(components)
    self.assertIsInstance(t2, ThingWithParametersButNoAttrs)

  def test_wrapped_constructor(self):
    def add_tag(f):
      @functools.wraps(f)
      def wrapper(*args, **kwargs):
        args[0]._tag = 'tagged'
        return f(*args, **kwargs)
      return wrapper

    @tfp.experimental.auto_composite_tensor
    class ThingWithWrappedInit(tfp.experimental.AutoCompositeTensor):

      @add_tag
      def __init__(self, value):
        self.value = tf.convert_to_tensor(value)

    init = ThingWithWrappedInit(3)
    def body(obj):
      return ThingWithWrappedInit(value=obj.value + 1),

    out, = tf.while_loop(
        cond=lambda *_: True,
        body=body,
        loop_vars=(init,),
        maximum_iterations=3)
    self.assertEqual(self.evaluate(out.value), 6)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
