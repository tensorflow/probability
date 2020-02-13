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
"""Tests for random variable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


class FakeDistribution(tfd.Distribution):
  """Fake distribution class for testing."""

  def __init__(self):
    super(FakeDistribution, self).__init__(
        dtype=None,
        reparameterization_type=tfd.FULLY_REPARAMETERIZED,
        validate_args=False,
        allow_nan_stats=True)


@test_util.test_all_tf_execution_regimes
class RandomVariableTest(test_util.TestCase):

  def testConstructor(self):
    x = ed.RandomVariable(tfd.Poisson(rate=tf.ones([2, 5])),
                          value=tf.ones([2, 5]))
    x_sample, x_value = self.evaluate([tf.convert_to_tensor(value=x), x.value])
    self.assertAllEqual(x_sample, x_value)
    with self.assertRaises(ValueError):
      _ = ed.RandomVariable(tfd.Bernoulli(probs=0.5),
                            value=tf.zeros([2, 5], dtype=tf.int32))
    x = ed.RandomVariable(FakeDistribution())
    with self.assertRaises(NotImplementedError):
      _ = x.value

  def testGradientsFirstOrder(self):
    f = lambda x: 2. * x
    x = ed.RandomVariable(tfd.Normal(0., 1.))
    _, dydx = tfp.math.value_and_gradient(f, x)
    self.assertEqual(self.evaluate(dydx), 2.)

  def testGradientsSecondOrder(self):
    f = lambda x: 2. * x**2.
    df = lambda x: tfp.math.value_and_gradient(f, x)[1]
    x = ed.RandomVariable(tfd.Normal(0., 1.))
    _, d2ydx2 = tfp.math.value_and_gradient(df, x)
    self.assertEqual(self.evaluate(d2ydx2), 4.)

  def testStr(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0), value=1.234)
    if tf.executing_eagerly():
      pattern = "RandomVariable(\"1.234\", shape=(), dtype=float32"
    else:
      pattern = "RandomVariable(\"Normal\", shape=(), dtype=float32"
    regexp = re.escape(pattern)
    self.assertRegexpMatches(str(x), regexp)

  def testRepr(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0), value=1.234)
    if tf.executing_eagerly():
      string = ("<ed.RandomVariable 'Normal' shape=() "
                "dtype=float32 numpy=1.234>")
    else:
      string = "<ed.RandomVariable 'Normal' shape=() dtype=float32>"
    self.assertEqual(repr(x), string)

  def testNumpy(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0), value=1.23)
    if tf.executing_eagerly():
      self.assertEqual(x.numpy(), tf.constant(1.23).numpy())
    else:
      with self.assertRaises(NotImplementedError):
        _ = x.numpy()

  def testOperatorsAdd(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x + y
    z_value = x.value + y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsRadd(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y + x
    z_value = y + x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsSub(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x - y
    z_value = x.value - y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsRsub(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y - x
    z_value = y - x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsMul(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x * y
    z_value = x.value * y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsRmul(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y * x
    z_value = y * x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsDiv(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x / y
    z_value = x.value / y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsRdiv(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y / x
    z_value = y / x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsFloordiv(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x // y
    z_value = x.value // y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsRfloordiv(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y // x
    z_value = y // x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsMod(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x % y
    z_value = x.value % y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsRmod(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y % x
    z_value = y % x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsLt(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x < y
    z_value = x.value < y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsLe(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x <= y
    z_value = x.value <= y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsGt(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x > y
    z_value = x.value > y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsGe(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x >= y
    z_value = x.value >= y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsGetitem(self):
    x = ed.RandomVariable(tfd.Normal(tf.zeros([3, 4]), tf.ones([3, 4])))
    z = x[0:2, 2:3]
    z_value = x.value[0:2, 2:3]
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsPow(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x ** y
    z_value = x.value ** y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsRpow(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y ** x
    z_value = y ** x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsNeg(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    z = -x
    z_value = -x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsAbs(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    z = abs(x)
    z_value = abs(x.value)
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testOperatorsHash(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    self.assertNotEqual(hash(x), hash(y))
    self.assertEqual(hash(x), id(x))

  def testOperatorsEq(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    self.assertEqual(x, x)

  def testOperatorsNe(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    self.assertNotEqual(x, y)

  def testOperatorsBoolNonzero(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    with self.assertRaises(TypeError):
      _ = not x

  def testArrayPriority(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = np.array(5.0, dtype=np.float32)
    z = y / x
    z_value = y / x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  def testConvertToTensor(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 0.1))
    with self.assertRaises(ValueError):
      _ = tf.convert_to_tensor(value=x, dtype=tf.int32)

  def testSessionEval(self):
    if tf.executing_eagerly(): return
    with self.cached_session() as sess:
      x = ed.RandomVariable(tfd.Normal(0.0, 0.1))
      x_ph = tf1.placeholder(tf.float32, [])
      y = ed.RandomVariable(tfd.Normal(x_ph, 0.1))
      self.assertLess(x.eval(), 5.0)
      self.assertLess(x.eval(sess), 5.0)
      self.assertLess(x.eval(feed_dict={x_ph: 100.0}), 5.0)
      self.assertGreater(y.eval(feed_dict={x_ph: 100.0}), 5.0)
      self.assertGreater(y.eval(sess, feed_dict={x_ph: 100.0}), 5.0)
      self.assertRaises(tf.errors.InvalidArgumentError, y.eval)
      self.assertRaises(tf.errors.InvalidArgumentError, y.eval, sess)

  def testSessionRun(self):
    if tf.executing_eagerly(): return
    with self.cached_session() as sess:
      x = ed.RandomVariable(tfd.Normal(0.0, 0.1))
      x_ph = tf1.placeholder(tf.float32, [])
      y = ed.RandomVariable(tfd.Normal(x_ph, 0.1))
      self.assertLess(sess.run(x), 5.0)
      self.assertLess(sess.run(x, feed_dict={x_ph: 100.0}), 5.0)
      self.assertGreater(sess.run(y, feed_dict={x_ph: 100.0}), 5.0)
      self.assertRaises(tf.errors.InvalidArgumentError, sess.run, y)

  # Note: we must defer creation of any tensors until after tf.test.main().
  # pylint: disable=g-long-lambda
  @parameterized.parameters(
      {"rv": lambda: ed.RandomVariable(tfd.Bernoulli(probs=0.5)),
       "sample_shape": [],
       "batch_shape": [],
       "event_shape": []},
      {"rv": lambda: ed.RandomVariable(tfd.Bernoulli(tf.zeros([2, 3]))),
       "sample_shape": [],
       "batch_shape": [2, 3],
       "event_shape": []},
      {"rv": lambda: ed.RandomVariable(tfd.Bernoulli(probs=0.5),
                                       sample_shape=2),
       "sample_shape": [2],
       "batch_shape": [],
       "event_shape": []},
      {"rv": lambda: ed.RandomVariable(tfd.Bernoulli(probs=0.5),
                                       sample_shape=[2, 1]),
       "sample_shape": [2, 1],
       "batch_shape": [],
       "event_shape": []},
      {"rv": lambda: ed.RandomVariable(tfd.Bernoulli(probs=0.5),
                                       sample_shape=tf.constant([2])),
       "sample_shape": [2],
       "batch_shape": [],
       "event_shape": []},
      {"rv": lambda: ed.RandomVariable(tfd.Bernoulli(probs=0.5),
                                       sample_shape=tf.constant([2, 4])),
       "sample_shape": [2, 4],
       "batch_shape": [],
       "event_shape": []},
  )
  # pylint: enable=g-long-lambda
  def testShape(self, rv, sample_shape, batch_shape, event_shape):
    rv = rv()
    self.assertEqual(rv.shape, sample_shape + batch_shape + event_shape)
    self.assertEqual(rv.shape, rv.shape)
    self.assertEqual(rv.sample_shape, sample_shape)
    self.assertEqual(rv.distribution.batch_shape, batch_shape)
    self.assertEqual(rv.distribution.event_shape, event_shape)

  def testRandomTensorSample(self):
    num_samples = tf.cast(tfd.Poisson(rate=5.).sample(), tf.int32)
    _ = ed.RandomVariable(tfd.Normal(loc=0.0, scale=1.0),
                          sample_shape=num_samples)


if __name__ == "__main__":
  tf.test.main()
