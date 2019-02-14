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
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


def make_relaxed_categorical(batch_shape, num_classes, dtype=tf.float32):
  logits = tf.random.uniform(
      list(batch_shape) + [num_classes], -10, 10, dtype=dtype) - 50.
  temperatures = tf.random.uniform(list(batch_shape), 0.1, 10, dtype=tf.float32)
  return tfd.RelaxedOneHotCategorical(temperatures, logits)


@test_util.run_all_in_graph_and_eager_modes
class ExpRelaxedOneHotCategoricalTest(tf.test.TestCase):

  def testP(self):
    temperature = 1.0
    logits = [2.0, 3.0, -4.0]
    dist = tfd.ExpRelaxedOneHotCategorical(temperature, logits)
    expected_p = np.exp(logits)/np.sum(np.exp(logits))
    self.assertAllClose(expected_p, self.evaluate(dist.probs))
    self.assertAllEqual([3], dist.probs.shape)

  def testPdf(self):
    temperature = .4
    logits = [.3, .1, .4]
    k = len(logits)
    p = np.exp(logits)/np.sum(np.exp(logits))
    dist = tfd.ExpRelaxedOneHotCategorical(temperature, logits)
    x = self.evaluate(dist.sample())
    # analytical ExpConcrete density presented in Maddison et al. 2016
    prod_term = p * np.exp(-temperature * x)
    expected_pdf = (
        gamma(k) * np.power(temperature, k - 1) * np.prod(
            prod_term / np.sum(prod_term)))
    pdf = self.evaluate(dist.prob(x))
    self.assertAllClose(expected_pdf, pdf)


@test_util.run_all_in_graph_and_eager_modes
class RelaxedOneHotCategoricalTest(tf.test.TestCase):

  def testLogits(self):
    temperature = 1.0
    logits = [2.0, 3.0, -4.0]
    dist = tfd.RelaxedOneHotCategorical(temperature, logits)
    # check p for ExpRelaxed base distribution
    self.assertAllClose(logits, self.evaluate(dist._distribution.logits))
    self.assertAllEqual([3], dist._distribution.logits.shape)

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
    def analytical_pdf(x, temperature, logits):
      # analytical density of RelaxedOneHotCategorical
      temperature = np.reshape(temperature, (-1, 1))
      if len(x.shape) == 1:
        x = np.expand_dims(x, 0)
      k = logits.shape[1]
      p = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
      term1 = gamma(k)*np.power(temperature, k-1)
      term2 = np.sum(p/(np.power(x, temperature)), axis=1, keepdims=True)
      term3 = np.prod(p/(np.power(x, temperature+1)), axis=1, keepdims=True)
      expected_pdf = term1*np.power(term2, -k)*term3
      return expected_pdf

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

  def testShapes(self):
    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_relaxed_categorical(batch_shape, 10)
      self.assertAllEqual(batch_shape, dist.batch_shape.as_list())
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([10], self.evaluate(dist.event_shape_tensor()))
      self.assertAllEqual([10], self.evaluate(dist.event_shape_tensor()))

    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_relaxed_categorical(batch_shape,
                                      tf.constant(10, dtype=tf.int32))
      self.assertAllEqual(len(batch_shape), dist.batch_shape.ndims)
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([10], self.evaluate(dist.event_shape_tensor()))
      self.assertAllEqual([10], self.evaluate(dist.event_shape_tensor()))

  def testUnknownShape(self):
    logits_pl = tf.compat.v1.placeholder_with_default(
        input=[.3, .1, .4], shape=None)
    temperature = 1.0
    dist = tfd.ExpRelaxedOneHotCategorical(temperature, logits_pl)
    self.assertAllEqual([3], self.evaluate(dist.sample()).shape)
    self.assertAllEqual([5, 3], self.evaluate(dist.sample(5)).shape)

  def testDTypes(self):
    # check that sampling and log_prob work for a range of dtypes
    for dtype in (tf.float16, tf.float32, tf.float64):
      logits = tf.random.uniform(shape=[3, 3], dtype=dtype)
      dist = tfd.RelaxedOneHotCategorical(temperature=0.5, logits=logits)
      dist.log_prob(dist.sample())


if __name__ == "__main__":
  tf.test.main()
