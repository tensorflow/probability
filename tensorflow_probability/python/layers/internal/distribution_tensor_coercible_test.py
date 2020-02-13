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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import operator

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.layers.internal import distribution_tensor_coercible

from tensorflow.python.framework import test_util as tf_test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

dtc = distribution_tensor_coercible


class FakeBoolDistribution(tfd.Distribution):
  """Fake distribution for testing logical operators."""

  def __init__(self):
    super(FakeBoolDistribution, self).__init__(
        dtype=tf.bool,
        reparameterization_type=tfd.FULLY_REPARAMETERIZED,
        validate_args=False,
        allow_nan_stats=True)

  def _mean(self):
    return tf.convert_to_tensor([True, False], dtype=tf.bool)


class Normal(tfd.Normal):
  """Vanilla `Normal` except has enhanced `__add__`."""

  def __add__(self, x):
    """Distributional arithmetic unless both have value semantics."""
    if (isinstance(x, (Normal, tfd.Normal)) and
        not isinstance(x, dtc._TensorCoercible) and
        not isinstance(self, dtc._TensorCoercible)):
      return Normal(loc=self.loc + x.loc,
                    scale=tf.sqrt(self.scale**2. + x.scale**2.))
    # testUserCustomOperators shows that this is never called.
    return super(Normal, self).__add__(x)


@test_util.test_all_tf_execution_regimes
class DistributionTensorConversionTest(test_util.TestCase):

  def testErrorsByDefault(self):
    x = tfd.Normal(loc=0., scale=1.)
    if tf.executing_eagerly():
      with self.assertRaises(ValueError):
        tf.convert_to_tensor(x)
    else:
      with self.assertRaises(TypeError):
        tf.convert_to_tensor(x)

  def testConvertToTensor(self):
    x = dtc._TensorCoercible(tfd.Normal(loc=1.5, scale=1),
                             tfd.Distribution.mean)
    y = tf.convert_to_tensor(x)
    y1 = x._value()
    self.assertIs(y, y1)
    self.assertEqual([1.5, 1.5], self.evaluate([y, y1]))

  def testConvertFromExplicit(self):
    x = dtc._TensorCoercible(tfd.Normal(loc=1.25, scale=1),
                             lambda self: 42.)
    y = tf.convert_to_tensor(x)
    y1 = x._value()
    self.assertIs(y, y1)
    self.assertEqual([42., 42.], self.evaluate([y, y1]))

  def testReproducible(self):
    u = dtc._TensorCoercible(tfd.Uniform(low=-100., high=100),
                             tfd.Distribution.sample)
    # Small scale means only the mean really matters.
    x = tfd.Normal(loc=u, scale=0.0001)
    [u_, x1_, x2_] = self.evaluate([
        tf.convert_to_tensor(x.loc), x.sample(), x.sample()])
    self.assertNear(u_, x1_, err=0.01)
    self.assertNear(u_, x2_, err=0.01)

  def testArrayPriority(self):
    x = dtc._TensorCoercible(tfd.Normal(loc=1.5, scale=1),
                             tfd.Distribution.mean)
    y = np.array(3., dtype=np.float32)
    self.assertEqual(2., self.evaluate(y / x))

  @parameterized.parameters(
      operator.add,
      operator.sub,
      operator.mul,
      operator.floordiv,
      operator.truediv,
      operator.pow,
      operator.mod,
      operator.gt,
      operator.ge,
      operator.lt,
      operator.le,
  )
  def testOperatorBinary(self, op):
    loc = np.array([-0.25, 0.5], dtype=np.float32)
    x = dtc._TensorCoercible(tfd.Normal(loc=loc, scale=1),
                             tfd.Distribution.mean)
    # Left operand does not support corresponding op and the operands are of
    # different types. Eg: `__radd__`.
    y1 = op(1., x)
    # Left operand supports op since right operand is implicitly converted by
    # usual `convert_to_tensor` semantics. Eg: `__add__`.
    y2 = op(x, 2.)
    self.assertAllEqual([op(1., loc), op(loc, 2.)],
                        self.evaluate([y1, y2]))

  @parameterized.parameters(
      operator.abs,
      operator.neg,
  )
  def testOperatorUnary(self, op):
    loc = np.array([-0.25, 0.5], dtype=np.float32)
    x = dtc._TensorCoercible(tfd.Normal(loc=loc, scale=1),
                             tfd.Distribution.mean)
    self.assertAllEqual(op(loc), self.evaluate(op(x)))

  @parameterized.parameters(
      operator.and_,
      operator.or_,
      operator.xor,
  )
  def testOperatorBinaryLogical(self, op):
    loc = np.array([True, False])
    x = dtc._TensorCoercible(FakeBoolDistribution(), tfd.Distribution.mean)
    y1 = op(True, x)
    y2 = op(x, False)
    self.assertAllEqual([op(True, loc), op(loc, False)],
                        self.evaluate([y1, y2]))

  # `~` is the only supported unary logical operator.
  # Note: 'boolean operator' is distinct from 'logical operator'. (The former
  # generally being not overrideable.)
  def testOperatorUnaryLogical(self):
    loc = np.array([True, False])
    x = dtc._TensorCoercible(FakeBoolDistribution(), tfd.Distribution.mean)
    self.assertAllEqual(*self.evaluate([~tf.convert_to_tensor(loc), ~x]))

  def testOperatorBoolNonzero(self):
    loc = np.array([-0.25, 0.5], dtype=np.float32)
    x = dtc._TensorCoercible(tfd.Normal(loc=loc, scale=1),
                             tfd.Distribution.mean)
    with self.assertRaises(TypeError):
      _ = not x

  def testOperatorGetitem(self):
    loc = np.linspace(-1., 1., 12).reshape(3, 4)
    x = dtc._TensorCoercible(tfd.Normal(loc=loc, scale=1),
                             tfd.Distribution.mean)
    self.assertAllEqual(loc[:2, 1:], self.evaluate(x[:2, 1:]))

  def testOperatorIter(self):
    loc = np.array([-0.25, 0.5], dtype=np.float32)
    x = dtc._TensorCoercible(tfd.Normal(loc=loc, scale=1),
                             tfd.Distribution.mean)
    if tf.executing_eagerly():
      for expected_, actual_ in zip(loc, iter(x)):
        self.assertEqual(expected_, actual_.numpy())
    else:
      with self.assertRaises(TypeError):
        for _ in iter(x):
          pass

  def testUserCustomOperators(self):
    # First we show that adding constants has arithmetic semantics.
    x = Normal(0, 1)
    y = x + Normal(1, 1)
    self.assertIsInstance(y, Normal)

    x_coerce = dtc._TensorCoercible(x, tfd.Distribution.variance)
    var_plus_2 = x_coerce + tf.constant(2.)

    # Now we show that adding two distributions has random variable semantics.
    y_coerce = dtc._TensorCoercible(y, tfd.Distribution.variance)
    self.assertIsInstance(y_coerce, Normal)
    var = tf.convert_to_tensor(y_coerce)

    # Note: it is not possible to have user-defined operators and also use the
    # _TensorCoercible class.
    self.assertIsInstance(x_coerce + y_coerce, tf.Tensor)

    # When adding `2` as a constant (to the variance) we get `3`.
    # When adding as a distribution, the variances add.
    self.assertAllClose([3., 2.], self.evaluate([var_plus_2, var]),
                        atol=0, rtol=1e-3)

  def _testWhileLoop(self):
    """Shows misuse of `dtc._TensorCoercible(distribution)` in `tf.while_loop`.

    Since `dtc._TensorCoercible(distribution)` only creates the `Tensor` on an
    as-needed basis, care must be taken that the Tensor is created outside the
    body of a `tf.while_loop`, if the result is going to be used outside of the
    `tf.while_loop`. Although this is the case for any use of `tf.while_loop` we
    write this unit-test as a reminder of the behavior.
    """
    mean_ = 0.5
    stddev_ = 2.
    num_iter_ = 4

    x = dtc._TensorCoercible(tfd.Normal(mean_, 0.75), tfd.Distribution.mean)

    # Note: in graph mode we can make the assertion not raise
    # if we make sure to create the Tensor outside the loop. Ie,
    # tf.convert_to_tensor(x)

    def _body(iter_, d):
      y = dtc._TensorCoercible(tfd.Normal(0, stddev_), tfd.Distribution.stddev)
      return [iter_ + 1, d + y + x]

    _, mean_plus_numiter_times_stddev = tf.while_loop(
        cond=lambda iter_, *args: iter_ < num_iter_,
        body=_body,
        loop_vars=[0, mean_])

    if not tf.executing_eagerly():
      # In graph mode we cannot access the cached value of x outside the
      # tf.while_loop. To make this exception not occur, simply call
      # `tf.convert_to_tensor(x)` prior to the `tf.while_loop`.  Doing so will
      # cause the value of `x` to exist both outside and inside the
      # `tf.while_loop`.
      if tf1.control_flow_v2_enabled():
        error_regex = r'Tensor.*must be from the same graph as Tensor.*'
      else:
        error_regex = 'Cannot use.*in a while loop'
      with self.assertRaisesRegexp(ValueError, error_regex):
        _ = x + tf.constant(3.)
      return

    # Things work in Eager mode because it has regular Python semantics,
    # ie, lexical scope rules.
    self.assertAllEqual([
        mean_ + 3.,
        mean_ + num_iter_ * (mean_ + stddev_),
    ], self.evaluate([
        x + tf.constant(3.),
        mean_plus_numiter_times_stddev,
    ]))

  def testWhileLoop(self):
    self._testWhileLoop()

  def testWhileLoopWithControlFlowV2(self):
    tf_test_util.enable_control_flow_v2(self._testWhileLoop)()


@test_util.test_all_tf_execution_regimes
class MemoryLeakTest(test_util.TestCase):

  def testTypeObjectLeakage(self):
    if not tf.executing_eagerly():
      self.skipTest('only relevant to eager')

    layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    x = tf.constant([-.23, 1.23, 1.42])
    dist = layer(x)
    gc.collect()
    before_objs = len(gc.get_objects())
    for _ in range(int(1e2)):
      dist = layer(x)
    gc.collect()
    after_objs = len(gc.get_objects())
    del dist

    # This was 43150(py2)/43750(py3) before PR#532.
    self.assertLess(after_objs - before_objs, 1)


if __name__ == '__main__':
  tf.test.main()
