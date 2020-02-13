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
from absl.testing import parameterized
import numpy as np
import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class GeneratedRandomVariablesTest(test_util.TestCase):

  def testBernoulliDoc(self):
    self.assertGreater(len(ed.Bernoulli.__doc__), 0)
    self.assertIn(inspect.cleandoc(tfd.Bernoulli.__init__.__doc__),
                  ed.Bernoulli.__doc__)
    self.assertEqual(ed.Bernoulli.__name__, "Bernoulli")

  @parameterized.named_parameters(
      {"testcase_name": "1d_rv_1d_event", "logits": np.zeros(1), "n": [1]},
      {"testcase_name": "1d_rv_5d_event", "logits": np.zeros(1), "n": [5]},
      {"testcase_name": "5d_rv_1d_event", "logits": np.zeros(5), "n": [1]},
      {"testcase_name": "5d_rv_5d_event", "logits": np.zeros(5), "n": [5]},
  )
  def testBernoulliLogProb(self, logits, n):
    rv = ed.Bernoulli(logits)
    dist = tfd.Bernoulli(logits)
    x = rv.distribution.sample(n)
    rv_log_prob, dist_log_prob = self.evaluate(
        [rv.distribution.log_prob(x), dist.log_prob(x)])
    self.assertAllEqual(rv_log_prob, dist_log_prob)

  @parameterized.named_parameters(
      {"testcase_name": "0d_rv_0d_sample",
       "logits": 0.,
       "n": 1},
      {"testcase_name": "0d_rv_1d_sample",
       "logits": 0.,
       "n": [1]},
      {"testcase_name": "1d_rv_1d_sample",
       "logits": np.array([0.]),
       "n": [1]},
      {"testcase_name": "1d_rv_5d_sample",
       "logits": np.array([0.]),
       "n": [5]},
      {"testcase_name": "2d_rv_1d_sample",
       "logits": np.array([-0.2, 0.8]),
       "n": [1]},
      {"testcase_name": "2d_rv_5d_sample",
       "logits": np.array([-0.2, 0.8]),
       "n": [5]},
  )
  def testBernoulliSample(self, logits, n):
    rv = ed.Bernoulli(logits)
    dist = tfd.Bernoulli(logits)
    self.assertEqual(rv.distribution.sample(n).shape, dist.sample(n).shape)

  # Note: we must defer creation of any tensors until after tf.test.main().
  @parameterized.named_parameters(
      {"testcase_name": "0d_bernoulli",
       "rv": lambda: ed.Bernoulli(probs=0.5),
       "sample_shape": [],
       "batch_shape": [],
       "event_shape": []},
      {"testcase_name": "2d_bernoulli",
       "rv": lambda: ed.Bernoulli(tf.zeros([2, 3])),
       "sample_shape": [],
       "batch_shape": [2, 3],
       "event_shape": []},
      {"testcase_name": "2x0d_bernoulli",
       "rv": lambda: ed.Bernoulli(probs=0.5, sample_shape=2),
       "sample_shape": [2],
       "batch_shape": [],
       "event_shape": []},
      {"testcase_name": "2x1d_bernoulli",
       "rv": lambda: ed.Bernoulli(probs=0.5, sample_shape=[2, 1]),
       "sample_shape": [2, 1],
       "batch_shape": [],
       "event_shape": []},
      {"testcase_name": "3d_dirichlet",
       "rv": lambda: ed.Dirichlet(tf.zeros(3)),
       "sample_shape": [],
       "batch_shape": [],
       "event_shape": [3]},
      {"testcase_name": "2x3d_dirichlet",
       "rv": lambda: ed.Dirichlet(tf.zeros([2, 3])),
       "sample_shape": [],
       "batch_shape": [2],
       "event_shape": [3]},
      {"testcase_name": "1x3d_dirichlet",
       "rv": lambda: ed.Dirichlet(tf.zeros(3), sample_shape=1),
       "sample_shape": [1],
       "batch_shape": [],
       "event_shape": [3]},
      {"testcase_name": "2x1x3d_dirichlet",
       "rv": lambda: ed.Dirichlet(tf.zeros(3), sample_shape=[2, 1]),
       "sample_shape": [2, 1],
       "batch_shape": [],
       "event_shape": [3]},
  )
  def testShape(self, rv, sample_shape, batch_shape, event_shape):
    rv = rv()
    self.assertEqual(rv.shape, sample_shape + batch_shape + event_shape)
    self.assertEqual(rv.sample_shape, sample_shape)
    self.assertEqual(rv.distribution.batch_shape, batch_shape)
    self.assertEqual(rv.distribution.event_shape, event_shape)

  def _testValueShapeAndDtype(self, cls, value, **kwargs):
    rv = cls(value=value, **kwargs)
    value_shape = rv.value.shape
    expected_shape = rv.sample_shape.concatenate(
        rv.distribution.batch_shape).concatenate(rv.distribution.event_shape)
    self.assertEqual(value_shape, expected_shape)
    self.assertEqual(rv.distribution.dtype, rv.value.dtype)

  @parameterized.parameters(
      {"cls": ed.Normal, "value": 2, "kwargs": {"loc": 0.5, "scale": 1.0}},
      {"cls": ed.Normal, "value": [2],
       "kwargs": {"loc": [0.5], "scale": [1.0]}},
      {"cls": ed.Poisson, "value": 2, "kwargs": {"rate": 0.5}},
  )
  def testValueShapeAndDtype(self, cls, value, kwargs):
    self._testValueShapeAndDtype(cls, value, **kwargs)

  def testValueMismatchRaises(self):
    with self.assertRaises(ValueError):
      self._testValueShapeAndDtype(ed.Normal, 2, loc=[0.5, 0.5], scale=1.0)
    with self.assertRaises(ValueError):
      self._testValueShapeAndDtype(ed.Normal, 2, loc=[0.5], scale=[1.0])
    with self.assertRaises(ValueError):
      self._testValueShapeAndDtype(
          ed.Normal, np.zeros([10, 3]), loc=[0.5, 0.5], scale=[1.0, 1.0])

  def testValueUnknownShape(self):
    if tf.executing_eagerly(): return
    # should not raise error
    ed.Bernoulli(probs=0.5, value=tf1.placeholder(tf.int32))

  def testAsRandomVariable(self):
    # A wrapped Normal distribution should behave identically to
    # the builtin Normal RV.
    def model_builtin():
      return ed.Normal(1., 0.1, name="x")

    def model_wrapped():
      return ed.as_random_variable(tfd.Normal(1., 0.1, name="x"))

    # Check that both models are interceptable and yield
    # identical log probs.
    log_joint_builtin = ed.make_log_joint_fn(model_builtin)
    log_joint_wrapped = ed.make_log_joint_fn(model_wrapped)
    self.assertEqual(self.evaluate(log_joint_builtin(x=7.)),
                     self.evaluate(log_joint_wrapped(x=7.)))

    # Check that our attempt to back out the variable name from the
    # Distribution name is robust to name scoping.
    with tf1.name_scope("nested_scope"):
      dist = tfd.Normal(1., 0.1, name="x")
      def model_scoped():
        return ed.as_random_variable(dist)
      log_joint_scoped = ed.make_log_joint_fn(model_scoped)
      self.assertEqual(self.evaluate(log_joint_builtin(x=7.)),
                       self.evaluate(log_joint_scoped(x=7.)))

  def testAllDistributionsAreRVs(self):
    for (dist_name, dist_class)  in six.iteritems(tfd.__dict__):
      if inspect.isclass(dist_class) and issubclass(
          dist_class, tfd.Distribution):
        self.assertIn(dist_name, ed.__dict__)

if __name__ == "__main__":
  tf.test.main()
