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
"""Tests for FiniteDiscrete distribution classs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import finite_discrete
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class FiniteDiscreteTest(object):

  def _build_tensor(self, ndarray):
    # Enforce parameterized dtype and static/dynamic testing.
    ndarray = np.asarray(ndarray)
    return tf.compat.v1.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)

  def _get_shape(self, tensor):
    return tensor.shape if self.use_static_shape else tf.shape(input=tensor)


class FiniteDiscreteValidateArgsTest(FiniteDiscreteTest):

  def testInequalLastDimRaises(self):
    outcomes = self._build_tensor([1.0, 2.0])
    probs = self._build_tensor([0.25, 0.25, 0.5])
    with self.assertRaisesWithPredicateMatch(
        Exception, 'Last dimension of outcomes and probs must be equal size'):
      dist = finite_discrete.FiniteDiscrete(
          outcomes, probs=probs, validate_args=True)
      self.evaluate(dist.outcomes)

  def testRankOfOutcomesLargerThanOneRaises(self):
    outcomes = self._build_tensor([[1.0, 2.0], [3.0, 4.0]])
    probs = self._build_tensor([0.5, 0.5])
    with self.assertRaisesWithPredicateMatch(Exception,
                                             'Rank of outcomes must be 1.'):
      dist = finite_discrete.FiniteDiscrete(
          outcomes, probs=probs, validate_args=True)
      self.evaluate(dist.outcomes)

  def testSizeOfOutcomesIsZeroRaises(self):
    outcomes = self._build_tensor([])
    probs = self._build_tensor([])
    with self.assertRaisesWithPredicateMatch(
        Exception, 'Size of outcomes must be greater than 0.'):
      dist = finite_discrete.FiniteDiscrete(
          outcomes, probs=probs, validate_args=True)
      self.evaluate(dist.outcomes)

  def testOutcomesNotStrictlyIncreasingRaises(self):
    outcomes = self._build_tensor([1.0, 1.0, 2.0, 2.0])
    probs = self._build_tensor([0.25, 0.25, 0.25, 0.25])
    with self.assertRaisesWithPredicateMatch(
        Exception, 'outcomes is not strictly increasing.'):
      dist = finite_discrete.FiniteDiscrete(
          outcomes, probs=probs, validate_args=True)
      self.evaluate(dist.outcomes)


class FiniteDiscreteScalarTest(FiniteDiscreteTest):
  """Tests FiniteDiscrete when `logits` or `probs` is a 1-D tensor."""

  def testShape(self):
    outcomes = self._build_tensor([0.0, 0.2, 0.3, 0.5])
    logits = self._build_tensor([-0.1, 0.0, 0.1, 0.2])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, logits=logits, validate_args=True)
    if self.use_static_shape:
      self.assertAllEqual([], dist.batch_shape)
    self.assertAllEqual([], dist.batch_shape_tensor())
    self.assertAllEqual([], dist.event_shape)
    self.assertAllEqual([], dist.event_shape_tensor())

  def testMean(self):
    outcomes = self._build_tensor([1.0, 2.0])
    probs = self._build_tensor([0.5, 0.5])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    mean = dist.mean()
    self.assertAllEqual((), self._get_shape(mean))
    self.assertAllClose(1.5, mean)

  def testStddevAndVariance(self):
    outcomes = self._build_tensor([1.0, 2.0])
    probs = self._build_tensor([0.5, 0.5])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    stddev = dist.stddev()
    self.assertAllEqual((), self._get_shape(stddev))
    self.assertAllClose(0.5, stddev)
    variance = dist.variance()
    self.assertAllEqual((), self._get_shape(variance))
    self.assertAllClose(0.25, variance)

  def testEntropy(self):
    outcomes = self._build_tensor([1, 2, 3, 4])
    probs = np.array([0.125, 0.125, 0.25, 0.5])
    probs_tensor = self._build_tensor(probs)
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs_tensor, validate_args=True)
    entropy = dist.entropy()
    self.assertAllEqual((), self._get_shape(entropy))
    self.assertAllClose(np.sum(-probs * np.log(probs)), entropy)

  def testMode(self):
    outcomes = self._build_tensor([1.0, 2.0, 3.0])
    probs = self._build_tensor([0.3, 0.1, 0.6])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    mode = dist.mode()
    self.assertAllEqual((), self._get_shape(mode))
    self.assertAllClose(3.0, mode)

  def testModeWithIntegerOutcomes(self):
    outcomes = self._build_tensor([1, 2, 3])
    probs = self._build_tensor([0.3, 0.1, 0.6])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    mode = dist.mode()
    self.assertAllEqual((), self._get_shape(mode))
    self.assertAllEqual(3, mode)

  def testSample(self):
    outcomes = self._build_tensor([1.0, 2.0])
    probs = self._build_tensor([0.2, 0.8])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    samples = self.evaluate(dist.sample(5000, seed=1234))
    self.assertAllEqual((5000,), self._get_shape(samples))
    self.assertAllClose(np.mean(samples), dist.mean(), atol=0.1)
    self.assertAllClose(np.std(samples), dist.stddev(), atol=0.1)

  def testSampleWithIntegerOutcomes(self):
    outcomes = self._build_tensor([1, 2])
    probs = self._build_tensor([0.2, 0.8])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    samples = self.evaluate(dist.sample(5000, seed=1234))
    self.assertAllClose(np.mean(samples), dist.mean(), atol=0.1)
    self.assertAllClose(np.std(samples), dist.stddev(), atol=0.1)

  def testPMF(self):
    outcomes = self._build_tensor([1.0, 2.0, 4.0, 8.0])
    probs = self._build_tensor([0.0, 0.1, 0.2, 0.7])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    prob = dist.prob(4.0)
    self.assertAllEqual((), self._get_shape(prob))
    self.assertAllClose(0.2, prob)
    # Outcome with zero probability.
    prob = dist.prob(1.0)
    self.assertAllEqual((), self._get_shape(prob))
    self.assertAllClose(0.0, prob)
    # Input that is not in the list of possible outcomes.
    prob = dist.prob(3.0)
    self.assertAllEqual((), self._get_shape(prob))
    self.assertAllClose(0.0, prob)

  def testPMFWithBatchSampleShape(self):
    outcomes = self._build_tensor([1.0, 2.0, 4.0, 8.0])
    probs = self._build_tensor([0.0, 0.1, 0.2, 0.7])
    x = self._build_tensor([[1.0], [2.0], [3.0], [4.0], [8.0]])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    prob = dist.prob(x)
    self.assertAllEqual((5, 1), self._get_shape(prob))
    self.assertAllClose([[0.0], [0.1], [0.0], [0.2], [0.7]], prob)

  def testPMFWithIntegerOutcomes(self):
    outcomes = self._build_tensor([1, 2, 4, 8])
    probs = self._build_tensor([0.0, 0.1, 0.2, 0.7])
    x = self._build_tensor([[1], [2], [3], [4], [8]])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    prob = dist.prob(x)
    self.assertAllEqual((5, 1), self._get_shape(prob))
    self.assertAllClose([[0.0], [0.1], [0.0], [0.2], [0.7]], prob)

  def testCDF(self):
    outcomes = self._build_tensor([0.1, 0.2, 0.4, 0.8])
    probs = self._build_tensor([0.0, 0.1, 0.2, 0.7])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    cdf = dist.cdf(0.4)
    self.assertAllEqual((), self._get_shape(cdf))
    self.assertAllClose(0.3, cdf)

  def testCDFWithBatchSampleShape(self):
    outcomes = self._build_tensor([0.1, 0.2, 0.4, 0.8])
    probs = self._build_tensor([0.0, 0.1, 0.2, 0.7])
    x = self._build_tensor([[0.0999, 0.1], [0.2, 0.4], [0.8, 0.8001]])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    cdf = dist.cdf(x)
    self.assertAllEqual((3, 2), self._get_shape(cdf))
    self.assertAllClose([[0.0, 0.0], [0.1, 0.3], [1.0, 1.0]], cdf)

  def testCDFWithIntegerOutcomes(self):
    outcomes = self._build_tensor([1, 2, 4, 8])
    probs = self._build_tensor([0.0, 0.1, 0.2, 0.7])
    x = self._build_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    cdf = dist.cdf(x)
    self.assertAllEqual((10,), self._get_shape(cdf))
    self.assertAllClose([0.0, 0.0, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 1.0, 1.0], cdf)

  def testCDFWithDifferentAtol(self):
    outcomes = self._build_tensor([0.1, 0.2, 0.4, 0.8])
    probs = self._build_tensor([0.0, 0.1, 0.2, 0.7])
    x = self._build_tensor([[0.095, 0.095], [0.395, 0.395]])
    dist1 = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, atol=0.001, validate_args=True)
    cdf = dist1.cdf(x)
    self.assertAllEqual((2, 2), self._get_shape(cdf))
    self.assertAllClose([[0.0, 0.0], [0.1, 0.1]], cdf)
    dist2 = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, atol=0.01, validate_args=True)
    cdf = dist2.cdf(x)
    self.assertAllEqual((2, 2), self._get_shape(cdf))
    self.assertAllClose([[0.0, 0.0], [0.3, 0.3]], cdf)


class FiniteDiscreteVectorTest(FiniteDiscreteTest):
  """Tests FiniteDiscrete when `logits` or `probs` is a tensor with rank >= 2."""

  def testShapes(self):
    outcomes = [0.0, 0.2, 0.3, 0.5]
    outcomes_tensor = self._build_tensor(outcomes)
    for batch_shape in ([1], [2], [3, 4, 5]):
      logits = self._build_tensor(
          np.random.uniform(-1, 1, size=list(batch_shape) + [len(outcomes)]))
      dist = finite_discrete.FiniteDiscrete(
          outcomes_tensor, logits=logits, validate_args=True)
      if self.use_static_shape:
        self.assertAllEqual(batch_shape, dist.batch_shape)
      self.assertAllEqual(batch_shape, dist.batch_shape_tensor())
      self.assertAllEqual([], dist.event_shape)
      self.assertAllEqual([], dist.event_shape_tensor())

  def testMean(self):
    outcomes = self._build_tensor([1.0, 2.0])
    probs = self._build_tensor([[0.5, 0.5], [0.2, 0.8]])
    expected_means = [1.5, 1.8]
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    mean = dist.mean()
    self.assertAllEqual((2,), self._get_shape(mean))
    self.assertAllClose(expected_means, mean)

  def testStddevAndVariance(self):
    outcomes = self._build_tensor([1.0, 2.0])
    probs = self._build_tensor([[0.5, 0.5], [0.2, 0.8]])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    stddev = dist.stddev()
    self.assertAllEqual((2,), self._get_shape(stddev))
    self.assertAllClose([0.5, 0.4], stddev)
    variance = dist.variance()
    self.assertAllEqual((2,), self._get_shape(variance))
    self.assertAllClose([0.25, 0.16], variance)

  def testMode(self):
    outcomes = self._build_tensor([1.0, 2.0, 3.0])
    probs = self._build_tensor([[0.3, 0.1, 0.6], [0.5, 0.4, 0.1],
                                [0.3, 0.5, 0.2]])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    mode = dist.mode()
    self.assertAllEqual((3,), self._get_shape(mode))
    self.assertAllClose([3.0, 1.0, 2.0], mode)

  def testEntropy(self):
    outcomes = self._build_tensor([1, 2, 3, 4])
    probs = np.array([[0.125, 0.125, 0.25, 0.5], [0.25, 0.25, 0.25, 0.25]])
    probs_tensor = self._build_tensor(probs)
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs_tensor, validate_args=True)
    entropy = dist.entropy()
    self.assertAllEqual((2,), self._get_shape(entropy))
    self.assertAllClose(np.sum(-probs * np.log(probs), axis=1), entropy)

  def testSample(self):
    outcomes = self._build_tensor([1.0, 2.0])
    probs = self._build_tensor([[0.2, 0.8], [0.8, 0.2]])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    samples = self.evaluate(dist.sample(5000, seed=1234))
    self.assertAllEqual((5000, 2), self._get_shape(samples))
    self.assertAllClose(np.mean(samples, axis=0), dist.mean(), atol=0.1)
    self.assertAllClose(np.std(samples, axis=0), dist.stddev(), atol=0.1)

  def testPMF(self):
    outcomes = self._build_tensor([1.0, 2.0, 4.0, 8.0])
    probs = self._build_tensor([[0.0, 0.1, 0.2, 0.7], [0.5, 0.3, 0.2, 0.0]])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    prob = dist.prob(8.0)
    self.assertAllEqual((2,), self._get_shape(prob))
    self.assertAllClose([0.7, 0.0], prob)

  def testPMFWithBatchSampleShape(self):
    outcomes = self._build_tensor([1.0, 2.0, 4.0, 8.0])
    probs = self._build_tensor([[0.0, 0.1, 0.2, 0.7], [0.5, 0.3, 0.2, 0.0]])
    x = self._build_tensor([[1.0], [2.0], [3.0], [4.0], [8.0]])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    prob = dist.prob(x)
    self.assertAllEqual((5, 2), self._get_shape(prob))
    self.assertAllClose(
        [[0.0, 0.5], [0.1, 0.3], [0.0, 0.0], [0.2, 0.2], [0.7, 0.0]], prob)

  def testCDF(self):
    outcomes = self._build_tensor([0.1, 0.2, 0.4, 0.8])
    probs = self._build_tensor([[0.0, 0.1, 0.2, 0.7], [0.5, 0.3, 0.2, 0.0]])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    cdf = dist.cdf(0.4)
    self.assertAllEqual((2,), self._get_shape(cdf))
    self.assertAllClose([0.3, 1.0], cdf)

  def testCDFWithBatchSampleShape(self):
    outcomes = self._build_tensor([0.1, 0.2, 0.4, 0.8])
    probs = self._build_tensor([[0.0, 0.1, 0.2, 0.7], [0.5, 0.3, 0.2, 0.0]])
    x = self._build_tensor([[0.0999, 0.0999], [0.1, 0.1], [0.2, 0.2],
                            [0.4, 0.4], [0.8, 0.8], [0.8001, 0.8001]])
    dist = finite_discrete.FiniteDiscrete(
        outcomes, probs=probs, validate_args=True)
    cdf = dist.cdf(x)
    self.assertAllEqual((6, 2), self._get_shape(cdf))
    self.assertAllClose([[0.0, 0.0], [0.0, 0.5], [0.1, 0.8], [0.3, 1.0],
                         [1.0, 1.0], [1.0, 1.0]], cdf)


class FiniteDiscreteValidateArgsStaticShapeTest(FiniteDiscreteValidateArgsTest,
                                                tf.test.TestCase):
  use_static_shape = True


class FiniteDiscreteValidateArgsDynamicShapeTest(FiniteDiscreteValidateArgsTest,
                                                 tf.test.TestCase):
  use_static_shape = False


class FiniteDiscreteScalarStaticShapeTest(FiniteDiscreteScalarTest,
                                          tf.test.TestCase):
  use_static_shape = True


class FiniteDiscreteScalarDynamicShapeTest(FiniteDiscreteScalarTest,
                                           tf.test.TestCase):
  use_static_shape = False


class FiniteDiscreteVectorStaticShapeTest(FiniteDiscreteVectorTest,
                                          tf.test.TestCase):
  use_static_shape = True


class FiniteDiscreteVectorDynamicShapeTest(FiniteDiscreteVectorTest,
                                           tf.test.TestCase):
  use_static_shape = False


if __name__ == '__main__':
  tf.test.main()
