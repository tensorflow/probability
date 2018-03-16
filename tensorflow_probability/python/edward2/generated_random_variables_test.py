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
"""Tests for generated random variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import numpy as np
import tensorflow as tf

from tensorflow_probability import edward2 as ed

tfd = tf.contrib.distributions
tfe = tf.contrib.eager


class GeneratedRandomVariablesTest(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes()
  def testBernoulliDoc(self):
    self.assertGreater(len(ed.Bernoulli.__doc__), 0)
    self.assertTrue(inspect.cleandoc(tfd.Bernoulli.__init__.__doc__) in
                    ed.Bernoulli.__doc__)
    self.assertEqual(ed.Bernoulli.__name__, "Bernoulli")

  def _testBernoulliLogProb(self, probs, n):
    rv = ed.Bernoulli(probs)
    dist = tfd.Bernoulli(probs)
    x = rv.distribution.sample(n)
    rv_log_prob, dist_log_prob = self.evaluate(
        [rv.distribution.log_prob(x), dist.log_prob(x)])
    self.assertAllEqual(rv_log_prob, dist_log_prob)

  @tfe.run_test_in_graph_and_eager_modes()
  def testBernoulliLogProb1D(self):
    self._testBernoulliLogProb(tf.zeros([1]) + 0.5, [1])
    self._testBernoulliLogProb(tf.zeros([1]) + 0.5, [5])
    self._testBernoulliLogProb(tf.zeros([5]) + 0.5, [1])
    self._testBernoulliLogProb(tf.zeros([5]) + 0.5, [5])

  def _testBernoulliSample(self, probs, n):
    rv = ed.Bernoulli(probs)
    dist = tfd.Bernoulli(probs)
    self.assertEqual(rv.distribution.sample(n).shape, dist.sample(n).shape)

  @tfe.run_test_in_graph_and_eager_modes()
  def testBernoulliSample0D(self):
    self._testBernoulliSample(0.5, [1])
    self._testBernoulliSample(np.array(0.5), [1])
    self._testBernoulliSample(tf.constant(0.5), [1])

  @tfe.run_test_in_graph_and_eager_modes()
  def testBernoulliSample1D(self):
    self._testBernoulliSample(np.array([0.5]), [1])
    self._testBernoulliSample(np.array([0.5]), [5])
    self._testBernoulliSample(np.array([0.2, 0.8]), [1])
    self._testBernoulliSample(np.array([0.2, 0.8]), [10])
    self._testBernoulliSample(tf.constant([0.5]), [1])
    self._testBernoulliSample(tf.constant([0.5]), [5])
    self._testBernoulliSample(tf.constant([0.2, 0.8]), [1])
    self._testBernoulliSample(tf.constant([0.2, 0.8]), [10])

  def _testShape(self, rv, sample_shape, batch_shape, event_shape):
    self.assertEqual(rv.shape, sample_shape + batch_shape + event_shape)
    self.assertEqual(rv.sample_shape, sample_shape)
    self.assertEqual(rv.distribution.batch_shape, batch_shape)
    self.assertEqual(rv.distribution.event_shape, event_shape)

  @tfe.run_test_in_graph_and_eager_modes()
  def testShapeBernoulli(self):
    self._testShape(ed.Bernoulli(probs=0.5), [], [], [])
    self._testShape(ed.Bernoulli(tf.zeros([2, 3])), [], [2, 3], [])
    self._testShape(ed.Bernoulli(probs=0.5, sample_shape=2), [2], [], [])
    self._testShape(
        ed.Bernoulli(probs=0.5, sample_shape=[2, 1]), [2, 1], [], [])

  @tfe.run_test_in_graph_and_eager_modes()
  def testShapeDirichlet(self):
    self._testShape(ed.Dirichlet(tf.zeros(3)), [], [], [3])
    self._testShape(ed.Dirichlet(tf.zeros([2, 3])), [], [2], [3])
    self._testShape(ed.Dirichlet(tf.zeros(3), sample_shape=1), [1], [], [3])
    self._testShape(
        ed.Dirichlet(tf.zeros(3), sample_shape=[2, 1]), [2, 1], [], [3])

  def _testValueShapeAndDtype(self, cls, value, *args, **kwargs):
    rv = cls(*args, value=value, **kwargs)
    value_shape = rv.value.shape
    expected_shape = rv.sample_shape.concatenate(
        rv.distribution.batch_shape).concatenate(rv.distribution.event_shape)
    self.assertEqual(value_shape, expected_shape)
    self.assertEqual(rv.distribution.dtype, rv.value.dtype)

  @tfe.run_test_in_graph_and_eager_modes()
  def testValueShapeAndDtype(self):
    self._testValueShapeAndDtype(ed.Normal, 2, loc=0.5, scale=1.0)
    self._testValueShapeAndDtype(ed.Normal, [2], loc=[0.5], scale=[1.0])
    self._testValueShapeAndDtype(ed.Poisson, 2, rate=0.5)

  def testValueUnknownShape(self):
    # should not raise error
    ed.Bernoulli(probs=0.5, value=tf.placeholder(tf.int32))

  @tfe.run_test_in_graph_and_eager_modes()
  def testValueMismatchRaises(self):
    with self.assertRaises(ValueError):
      self._testValueShapeAndDtype(ed.Normal, 2, loc=[0.5, 0.5], scale=1.0)
    with self.assertRaises(ValueError):
      self._testValueShapeAndDtype(ed.Normal, 2, loc=[0.5], scale=[1.0])
    with self.assertRaises(ValueError):
      self._testValueShapeAndDtype(
          ed.Normal, np.zeros([10, 3]), loc=[0.5, 0.5], scale=[1.0, 1.0])


if __name__ == "__main__":
  tf.test.main()
