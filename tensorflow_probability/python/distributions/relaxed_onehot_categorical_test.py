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
"""Tests for Relaxed One-Hot Categorical distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy.special import gamma
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_case
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions


def make_relaxed_categorical(batch_shape, num_classes, dtype=tf.float32):
  logits = tf.random.uniform(
      list(batch_shape) + [num_classes], -10, 10, dtype=dtype) - 50.
  temperatures = tf.random.uniform(list(batch_shape), 0.1, 10, dtype=tf.float32)
  return tfd.RelaxedOneHotCategorical(temperatures, logits, validate_args=True)


@test_util.run_all_in_graph_and_eager_modes
class ExpRelaxedOneHotCategoricalTest(test_case.TestCase):

  def testProbs(self):
    temperature = 1.0
    logits = [2.0, 3.0, -4.0]
    dist = tfd.ExpRelaxedOneHotCategorical(
        temperature, logits, validate_args=True)
    expected_p = np.exp(logits)/np.sum(np.exp(logits))
    self.assertAllClose(expected_p, self.evaluate(dist.probs_parameter()))
    self.assertAllEqual([3], dist.probs_parameter().shape)

  def testPdf(self):
    temperature = .4
    logits = [.3, .1, .4]
    k = len(logits)
    p = np.exp(logits)/np.sum(np.exp(logits))
    dist = tfd.ExpRelaxedOneHotCategorical(
        temperature, logits, validate_args=True)
    x = self.evaluate(dist.sample())
    # analytical ExpConcrete density presented in Maddison et al. 2016
    prod_term = p * np.exp(-temperature * x)
    expected_pdf = (
        gamma(k) * np.power(temperature, k - 1) * np.prod(
            prod_term / np.sum(prod_term)))
    pdf = self.evaluate(dist.prob(x))
    self.assertAllClose(expected_pdf, pdf)


def analytical_pdf(x, temperature, logits):
  # analytical density of RelaxedOneHotCategorical
  temperature = np.reshape(temperature, (-1, 1))
  if len(x.shape) == 1:
    x = np.expand_dims(x, 0)
  k = logits.shape[-1]
  p = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
  term1 = gamma(k) * np.power(temperature, k-1)
  term2 = np.sum(p / (np.power(x, temperature)), axis=-1, keepdims=True)
  term3 = np.prod(p / (np.power(x, temperature+1)), axis=-1, keepdims=True)
  expected_pdf = term1 * np.power(term2, -k) * term3
  return expected_pdf


@test_util.run_all_in_graph_and_eager_modes
class RelaxedOneHotCategoricalTest(tf.test.TestCase):

  def assertRaises(self, error_class, msg):
    if tf.executing_eagerly():
      return self.assertRaisesRegexp(error_class, msg)
    return self.assertRaisesOpError(msg)

  def testProbs(self):
    temperature = 1.0
    probs = [0.1, 0.5, 0.4]
    dist = tfd.RelaxedOneHotCategorical(temperature, probs=probs)
    self.assertAllClose(probs, self.evaluate(dist.probs))
    self.assertAllEqual([3], dist.probs.shape)

  def testLogits(self):
    temperature = 1.0
    logits = [2.0, 3.0, -4.0]
    dist = tfd.RelaxedOneHotCategorical(temperature, logits)
    # check p for ExpRelaxed base distribution
    self.assertAllClose(logits, self.evaluate(dist.logits))
    self.assertAllEqual([3], dist.logits.shape)

  def testParamBroadcasting(self):
    temperature = [1.0, 1.4]
    logits = [2.0, 3.0, -4.0]
    dist = tfd.RelaxedOneHotCategorical(temperature, logits)
    self.assertAllEqual([2], dist.batch_shape)
    self.assertAllEqual([3], dist.event_shape)

  def testSample(self):
    temperature = 1.4
    # single logit
    logits = [.3, .1, .4]
    dist = tfd.RelaxedOneHotCategorical(temperature, logits)
    self.assertAllEqual([3], self.evaluate(dist.sample()).shape)
    self.assertAllEqual([5, 3], self.evaluate(dist.sample(5)).shape)
    # multiple distributions
    logits = [[2.0, 3.0, -4.0], [.3, .1, .4]]
    dist = tfd.RelaxedOneHotCategorical(temperature, logits)
    self.assertAllEqual([2, 3], self.evaluate(dist.sample()).shape)
    self.assertAllEqual([5, 2, 3], self.evaluate(dist.sample(5)).shape)
    # multiple distributions
    logits = np.random.uniform(size=(4, 1, 3)).astype(np.float32)
    dist = tfd.RelaxedOneHotCategorical(temperature, logits)
    self.assertAllEqual([4, 1, 3], self.evaluate(dist.sample()).shape)
    self.assertAllEqual([5, 4, 1, 3], self.evaluate(dist.sample(5)).shape)

  def testPdf(self):
    temperature = .4
    logits = np.array([[.3, .1, .4]]).astype(np.float32)
    dist = tfd.RelaxedOneHotCategorical(temperature, logits)
    x = self.evaluate(dist.sample())
    pdf = self.evaluate(dist.prob(x))
    expected_pdf = analytical_pdf(x, temperature, logits)
    self.assertAllClose(expected_pdf.flatten(), pdf, rtol=1e-4)

    # variable batch size
    logits = np.array([[.3, .1, .4], [.6, -.1, 2.]]).astype(np.float32)
    temperatures = np.array([0.4, 2.3]).astype(np.float32)
    dist = tfd.RelaxedOneHotCategorical(temperatures, logits)
    x = self.evaluate(dist.sample())
    pdf = self.evaluate(dist.prob(x))
    expected_pdf = analytical_pdf(x, temperatures, logits)
    self.assertAllClose(expected_pdf.flatten(), pdf, rtol=1e-4)

    # broadcast logits over temparatures
    logits = np.array([.3, .1, .4]).astype(np.float32)
    temperatures = np.array([0.4, 2.3]).astype(np.float32)
    dist = tfd.RelaxedOneHotCategorical(temperatures, logits)
    x = self.evaluate(dist.sample())
    pdf = self.evaluate(dist.prob(x))
    expected_pdf = analytical_pdf(x, temperatures, logits)
    self.assertAllClose(expected_pdf.flatten(), pdf, rtol=1e-4)

  def testShapes(self):
    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_relaxed_categorical(batch_shape, 10)
      self.assertAllEqual(batch_shape,
                          tensorshape_util.as_list(dist.batch_shape))
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([10], self.evaluate(dist.event_shape_tensor()))
      self.assertAllEqual([10], self.evaluate(dist.event_shape_tensor()))

    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_relaxed_categorical(batch_shape,
                                      tf.constant(10, dtype=tf.int32))
      self.assertAllEqual(
          len(batch_shape), tensorshape_util.rank(dist.batch_shape))
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([10], self.evaluate(dist.event_shape_tensor()))
      self.assertAllEqual([10], self.evaluate(dist.event_shape_tensor()))

  def testUnknownShape(self):
    logits_pl = tf1.placeholder_with_default(input=[.3, .1, .4], shape=None)
    temperature = 1.0
    dist = tfd.ExpRelaxedOneHotCategorical(
        temperature, logits_pl, validate_args=True)
    self.assertAllEqual([3], self.evaluate(dist.sample()).shape)
    self.assertAllEqual([5, 3], self.evaluate(dist.sample(5)).shape)

  def testUnknownAndInvalidShape(self):
    logits = tf1.placeholder_with_default(19.84, shape=None)
    with self.assertRaises(
        ValueError, 'Argument `logits` must have rank at least 1.'):
      dist = tfd.ExpRelaxedOneHotCategorical(
          0.75, logits=logits, validate_args=True)
      self.evaluate(dist.sample())

    logits = tf1.placeholder_with_default([19.84], shape=None)
    with self.assertRaises(
        ValueError, 'Argument `logits` must have final dimension >= 2.'):
      dist = tfd.ExpRelaxedOneHotCategorical(
          12.0, logits=logits, validate_args=True)
      self.evaluate(dist.sample())

  def testDTypes(self):
    # check that sampling and log_prob work for a range of dtypes
    for dtype in (tf.float16, tf.float32, tf.float64):
      logits = tf.random.uniform(shape=[3, 3], dtype=dtype)
      dist = tfd.RelaxedOneHotCategorical(temperature=0.5, logits=logits)
      dist.log_prob(dist.sample())

  def testParamTensorFromLogits(self):
    x = tf.constant([-1., 0.5, 1.])
    d = tfd.ExpRelaxedOneHotCategorical(
        temperature=1., logits=x, validate_args=True)
    self.assertAllClose(
        *self.evaluate([x, d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([tf.math.softmax(x),
                        d.probs_parameter()]),
        atol=0,
        rtol=1e-4)

  def testParamTensorFromProbs(self):
    x = tf.constant([0.1, 0.5, 0.4])
    d = tfd.ExpRelaxedOneHotCategorical(
        temperature=1., probs=x, validate_args=True)
    self.assertAllClose(
        *self.evaluate([tf.math.log(x), d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([x, d.probs_parameter()]),
        atol=0, rtol=1e-4)


@test_util.run_all_in_graph_and_eager_modes
class ExpRelaxedOneHotCategoricalFromVariableTest(test_case.TestCase):

  def testGradientLogits(self):
    t = tf.Variable([0.01, 1.])
    logits = tf.Variable([[-1., 0., 1], [3., 3., 3.]])
    d = tfd.ExpRelaxedOneHotCategorical(t, logits=logits, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob(tf.math.log_softmax([[-1., 0., 0.], [0., 0., 1.]]))
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 2)
    self.assertAllNotNone(g)

  def testGradientProbs(self):
    t = tf.Variable(0.4)
    probs = tf.Variable([0.1, 0.7, 0.2])
    d = tfd.ExpRelaxedOneHotCategorical(t, probs=probs, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob(tf.math.log_softmax([[1., 0., 0.], [0., 0., 1.]]))
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 2)
    self.assertAllNotNone(g)

  def testAssertionsProbs(self):
    probs = tf.Variable([0.1, 0.7, 0.0])
    with self.assertRaisesOpError('Argument `probs` must sum to 1.'):
      d = tfd.ExpRelaxedOneHotCategorical(0.3, probs=probs, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.sample())

  def testAssertionsProbsAfterMutation(self):
    probs = tf.Variable([0.25, 0.25, 0.5])
    d = tfd.ExpRelaxedOneHotCategorical(0.1337, probs=probs, validate_args=True)
    with self.assertRaisesOpError('Condition x >= 0 did not hold element-wise'):
      self.evaluate([v.initializer for v in d.variables])
      with tf.control_dependencies([probs.assign([-0.25, 0.75, 0.5])]):
        self.evaluate(d.logits_parameter())

  def testAssertionsLogits(self):
    logits = tfp.util.DeferredTensor(tf.identity, tf.Variable(0.), shape=None)
    with self.assertRaisesRegexp(
        ValueError, 'Argument `logits` must have rank at least 1.'):
      d = tfd.ExpRelaxedOneHotCategorical(
          0.7, logits=logits, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.sample())

  def testAssertionsTemperatureAfterMutation(self):
    t = tf.Variable(7.7)
    d = tfd.ExpRelaxedOneHotCategorical(t, probs=[0.5, 0.5], validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError('Condition x > 0 did not hold element-wise'):
      with tf.control_dependencies([t.assign(-0.07)]):
        self.evaluate(d.logits_parameter())


if __name__ == '__main__':
  tf.test.main()
