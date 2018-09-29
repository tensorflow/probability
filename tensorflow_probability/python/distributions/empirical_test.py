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
"""Tests for the Empirical distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import empirical
from tensorflow_probability.python.internal import test_util
from tensorflow.python.framework import test_util as tf_test_util

tfd = tfp.distributions

def entropy(probs):
  probs = np.array(probs)
  return np.sum(- probs * np.log(probs))

def random_samples(shape):
  return np.random.uniform(size=list(shape))


class EmpiricalScalarTest(test_util.VectorDistributionTestHelpers):

  def testSamples(self):
    for samples_shape in ([2], [2, 4], [4, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=samples_shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllClose(input_ph, self.evaluate(dist.samples))

    invalid_sample = 0
    with self.assertRaises(ValueError):
      dist = empirical.Empirical(samples=invalid_sample)

  def testShapes(self):
    for samples_shape in ([2], [2, 4], [4, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=samples_shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllEqual(samples_shape[:-1],
                          self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([],
                          self.evaluate(dist.event_shape_tensor()))

  def testCdfNoBatch(self):
    sample = [0, 1, 1, 2]
    events = [
        [1],
        [0, 1],
        [[0, 1], [1, 2]]
    ]
    expected_cdfs = [
        [0.75],
        [0.25, 0.75],
        [[0.25, 0.75], [0.75, 1]]
    ]

    for event, expected_cdf in zip(events, expected_cdfs):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllClose(self.evaluate(dist.cdf(event)),
                          expected_cdf)
      self.assertAllClose(self.evaluate(dist.log_cdf(event)),
                          np.log(expected_cdf))

  def testCdfWithBatch(self):
    sample = [[0, 0, 1, 2], [0, 1, 1, 2]]
    events = [
        [0],
        [0, 1],
        [[0, 1], [1, 2]]
    ]
    expected_cdfs = [
        [0.5, 0.25],
        [0.5, 0.75],
        [[0.5, 0.75], [0.75, 1]]
    ]

    for event, expected_cdf in zip(events, expected_cdfs):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllClose(self.evaluate(dist.cdf(event)),
                          expected_cdf)
      self.assertAllClose(self.evaluate(dist.log_cdf(event)),
                          np.log(expected_cdf))

  def testPmfNoBatch(self):
    sample = [0, 1, 1, 2]
    events = [
        [0],
        [0, 1],
        [[0, 1], [1, 2]]
    ]
    expected_pmfs = [
        [0.25],
        [0.25, 0.5],
        [[0.25, 0.5], [0.5, 0.25]]
    ]

    for event, expected_pmf in zip(events, expected_pmfs):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllClose(self.evaluate(dist.prob(event)),
                          expected_pmf)

  def testPmfWithBatch(self):
    sample = [[0, 0, 1, 2], [0, 1, 1, 2]]
    events = [
        [0],
        [0, 1],
        [[0, 1], [1, 2]]
    ]
    expected_pmfs = [
        [0.5, 0.25],
        [0.5, 0.5],
        [[0.5, 0.5], [0.25, 0.25]]
    ]

    for event, expected_pmf in zip(events, expected_pmfs):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllClose(self.evaluate(dist.prob(event)),
                          expected_pmf)

  def testPmfInvalid(self):
    dist = empirical.Empirical(samples=random_samples([2, 2]),
                               validate_args=True)
    invalid_events = [
        [],
        [0, 0, 1]
    ]
    for event in invalid_events:
      with self.assertRaises(ValueError):
        self.evaluate(dist.prob(event))

  def testEntropy(self):
    samples = [
        [0],
        [0, 1, 0, 2, 2],
        [[0, 0], [0, 1], [1, 2]],
        [[[0, 0, 0, 1], [0, 0, 1, 1]],
         [[1, 1, 1, 2], [0, 1, 2, 2]]]
    ]
    expected_entropys = [
        entropy([1]),
        entropy([0.4, 0.2, 0.4]),
        [entropy([1]), entropy([0.5, 0.5]), entropy([0.5, 0.5])],
        [[entropy([0.75, 0.25]), entropy([0.5, 0.5])],
         [entropy([0.75, 0.25]), entropy([0.25, 0.25, 0.5])]]
    ]

    for sample, expected_entropy in zip(samples, expected_entropys):
      input_ = tf.convert_to_tensor(sample, dtype=np.float64)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllClose(self.evaluate(dist.entropy()), expected_entropy)

  def testSampleN(self):
    for samples_shape in ([2], [2, 4], [4, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=samples_shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllEqual(self.evaluate(tf.shape(dist.sample())),
                          dist.batch_shape_tensor())

      n = 1000
      seed = 42
      self.assertAllEqual(self.evaluate(dist.sample(n, seed)),
                          self.evaluate(dist.sample(n, seed)))

  def testMean(self):
    samples = [
        [0, 1, 1, 2],
        [[0, 0, 0, 1], [0, 1, 1, 2]],
        [[[0, 0], [1, 1]],
         [[0, 1], [0, 2]]]
    ]
    expected_means = [
        1.,
        [0.25, 1],
        [[0, 1], [0.5, 1]]
    ]

    for sample, expected_mean in zip(samples, expected_means):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllClose(self.evaluate(dist.mean()), expected_mean)

  def testMode(self):
    samples = [
        [0, 1, 1, 2],
        [[0, 0, 1], [0, 1, 1], [2, 2, 2]],
        [[[0, 0, 0, 0], [0, 0, 1, 1]],
         [[1, 1, 1, 2], [0, 1, 2, 2]]]
    ]
    expected_modes = [
        1,
        [0, 1, 2],
        [[0, 0], [1, 2]]
    ]

    for sample, expected_mode in zip(samples, expected_modes):
      input_ = tf.convert_to_tensor(sample, dtype=np.int32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllClose(self.evaluate(dist.mode()), expected_mode)

  def testVarianceAndStd(self):
    samples = [
        [0, 1, 1, 2],
        [[1, 3], [2, 6]]
    ]
    expected_variances = [
        0.5,
        [1., 4.],
    ]

    for sample, expected_variance in zip(samples, expected_variances):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      self.assertAllClose(self.evaluate(dist.variance()),
                          expected_variance)
      self.assertAllClose(self.evaluate(dist.stddev()),
                          np.sqrt(expected_variance))


class EmpiricalVectorTest(test_util.VectorDistributionTestHelpers):

  def testSamples(self):
    for samples_shape in ([2, 4], [4, 2, 4], [2, 2, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=samples_shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=1)
      self.assertAllClose(input_ph, self.evaluate(dist.samples))

    invalid_samples = [
        0,
        [0, 1],
    ]
    for samples in invalid_samples:
      with self.assertRaises(ValueError):
        dist = empirical.Empirical(samples=samples, event_ndims=1)

  def testShapes(self):
    for samples_shape in ([2, 4], [4, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=samples_shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=1)
      self.assertAllEqual(samples_shape[:-2],
                          self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual(samples_shape[-1:],
                          self.evaluate(dist.event_shape_tensor()))

  def testCdfNoBatch(self):
    sample = [[0, 0], [0, 1], [0, 1], [1, 2]]
    events = [
        [0, 1],
        [[0, 1], [1, 2]],
        [[[0, 0], [0, 1]],
         [[1, 1], [1, 2]]]
    ]
    expected_cdfs = [
        [0.75, 0.75],
        [[0.75, 0.75],
         [1, 1]],
        [[[0.75, 0.25], [0.75, 0.75]],
         [[1., 0.75], [1., 1.]]]
    ]

    for event, expected_cdf in zip(events, expected_cdfs):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=1)
      self.assertAllClose(self.evaluate(dist.cdf(event)),
                          expected_cdf)
      self.assertAllClose(self.evaluate(dist.log_cdf(event)),
                          np.log(expected_cdf))

  def testCdfWithBatch(self):
    sample = [[[0, 0], [0, 1], [0, 1], [1, 1]],
              [[0, 1], [2, 2], [2, 2], [2, 2]]]
    events = [
        [0, 1],
        [[0, 0], [2, 2]],
        [[[0, 0], [0, 1]],
         [[0, 1], [2, 2]]]
    ]
    expected_cdfs = [
        [[0.75, 1.],
         [0.25, 0.25]],
        [[0.75, 0.25],
         [1., 1.]],
        [[[0.75, 0.25], [0.25, 0.25]],
         [[0.75, 1.], [1., 1.]]]
    ]

    for event, expected_cdf in zip(events, expected_cdfs):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=1)
      self.assertAllClose(self.evaluate(dist.cdf(event)),
                          expected_cdf)

  def testPmfNoBatch(self):
    sample = [[0, 0], [0, 1], [0, 1], [1, 2]]
    events = [
        [0, 1],
        [[0, 1], [1, 2]],
        [[[0, 0], [0, 1]],
         [[1, 1], [1, 2]]]
    ]
    expected_pmfs = [
        0.5,
        [0.5, 0.25],
        [[0.25, 0.5], [0, 0.25]]
    ]

    for event, expected_pmf in zip(events, expected_pmfs):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=1)
      self.assertAllClose(self.evaluate(dist.prob(event)),
                          expected_pmf)

  def testPmfWithBatch(self):
    sample = [[[0, 0], [0, 1], [0, 1], [1, 1]],
              [[0, 1], [2, 2], [2, 2], [2, 2]]]
    events = [
        [0, 1],
        [[0, 0], [2, 2]],
        [[[0, 0], [0, 1]],
         [[0, 1], [2, 2]]]
    ]
    expected_pmfs = [
        [0.5, 0.25],
        [0.25, 0.75],
        [[0.25, 0.25], [0.5, 0.75]]
    ]

    for event, expected_pmf in zip(events, expected_pmfs):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=1)
      self.assertAllClose(self.evaluate(dist.prob(event)),
                          expected_pmf)
      self.assertAllClose(self.evaluate(dist.log_prob(event)),
                          np.log(expected_pmf))

  def testSampleN(self):
    for samples_shape in ([2, 4], [4, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=samples_shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      expected_shape = tf.concat(
          [dist.batch_shape_tensor(), dist.event_shape_tensor()],
          axis=0)
      self.assertAllEqual(self.evaluate(tf.shape(dist.sample())),
                          expected_shape)

      n = 1000
      seed = 42
      self.assertAllEqual(self.evaluate(dist.sample(n, seed)),
                          self.evaluate(dist.sample(n, seed)))

  def testMean(self):
    samples = [
        [[0, 0, 1, 2], [0, 1, 1, 2]],
        [[[0, 0], [1, 2]],
         [[0, 1], [2, 4]]]
    ]
    expected_means = [
        [0, 0.5, 1, 2],
        [[0.5, 1], [1, 2.5]]
    ]

    for sample, expected_mean in zip(samples, expected_means):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=1)
      self.assertAllClose(self.evaluate(dist.mean()), expected_mean)

  def testMode(self):
    samples = [
        [[0, 0], [0, 1], [0, 1]],
        [[[0, 0], [0, 1], [0, 1], [1, 1]],
         [[1, 1], [2, 2], [2, 2], [2, 2]]]
    ]
    expected_modes = [
        [0, 1],
        [[0, 1], [2, 2]]
    ]

    for sample, expected_mode in zip(samples, expected_modes):
      input_ = tf.convert_to_tensor(sample, dtype=np.int32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=1)
      self.assertAllClose(self.evaluate(dist.mode()), expected_mode)

  def testEntropy(self):
    samples = [
        [[0, 0], [0, 1]],
        [[[0, 0], [0, 1], [0, 1], [1, 1]],
         [[1, 1], [2, 2], [2, 2], [2, 2]]]
    ]
    expected_entropys = [
        entropy([0.5, 0.5]),
        [entropy([0.25, 0.5, 0.25]), entropy([0.25, 0.75])]
    ]

    for sample, expected_entropy in zip(samples, expected_entropys):
      input_ = tf.convert_to_tensor(sample, dtype=np.float64)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=1)
      self.assertAllClose(self.evaluate(dist.entropy()), expected_entropy)

  def testVarianceAndStd(self):
    samples = [
        [[0, 1], [2, 3]],
        [[[1, 2], [1, 4]],
         [[1, 2], [3, 6]]]
    ]
    expected_variances = [
        [1, 1],
        [[0, 1], [1, 4]]
    ]

    for sample, expected_variance in zip(samples, expected_variances):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=1)
      self.assertAllClose(self.evaluate(dist.variance()),
                          expected_variance)
      self.assertAllClose(self.evaluate(dist.stddev()),
                          np.sqrt(expected_variance))


class EmpiricalNdTest(test_util.VectorDistributionTestHelpers):

  def testSamples(self):
    for samples_shape in ([4, 2, 4], [4, 2, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=samples_shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=2)
      self.assertAllClose(input_ph, self.evaluate(dist.samples))

    invalid_samples = [
        0,
        [0, 1],
        [[0, 1], [1, 2]]
    ]
    for samples in invalid_samples:
      with self.assertRaises(ValueError):
        dist = empirical.Empirical(samples=samples, event_ndims=2)

  def testShapes(self):
    for samples_shape in ([2, 2, 4], [4, 2, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=samples_shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=2)
      self.assertAllEqual(samples_shape[:-3],
                          self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual(samples_shape[-2:],
                          self.evaluate(dist.event_shape_tensor()))

  def testCdfNoBatch(self):
    sample = [[[0, 0], [0, 1]],
              [[0, 0], [1, 2]],
              [[0, 1], [1, 2]],
              [[1, 2], [2, 4]]]
    events = [
        [[0, 0], [1, 1]],
        [[[0, 1], [1, 2]],
         [[0, 2], [2, 4]]]
    ]
    expected_cdfs = [
        [[0.75, 0.5], [0.75, 0.25]],
        [[[0.75, 0.75], [0.75, 0.75]],
         [[0.75, 1], [1, 1]]]
    ]

    for event, expected_cdf in zip(events, expected_cdfs):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=2)
      self.assertAllClose(self.evaluate(dist.cdf(event)),
                          expected_cdf)
      self.assertAllClose(self.evaluate(dist.log_cdf(event)),
                          np.log(expected_cdf))

  def testPmfNoBatch(self):
    sample = [[[0, 0], [0, 1]],
              [[0, 1], [1, 2]],
              [[0, 1], [1, 2]],
              [[0, 2], [2, 4]]]
    events = [
        [[0, 0], [0, 1]],
        [[[0, 1], [1, 2]],
         [[0, 2], [2, 4]]]
    ]
    expected_pmfs = [
        0.25,
        [0.5, 0.25]
    ]

    for event, expected_pmf in zip(events, expected_pmfs):
      input_ = tf.convert_to_tensor(sample, dtype=np.float32)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=input_.shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph, event_ndims=2)
      self.assertAllClose(self.evaluate(dist.prob(event)),
                          expected_pmf)

  def testSampleN(self):
    for samples_shape in ([2, 2, 4], [4, 2, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf.placeholder_with_default(
          input=input_,
          shape=samples_shape if self.static_shape else None)
      dist = empirical.Empirical(samples=input_ph)
      expected_shape = tf.concat(
          [dist.batch_shape_tensor(), dist.event_shape_tensor()],
          axis=0)
      self.assertAllEqual(self.evaluate(tf.shape(dist.sample())),
                          expected_shape)

      n = 1000
      seed = 42
      self.assertAllEqual(self.evaluate(dist.sample(n, seed)),
                          self.evaluate(dist.sample(n, seed)))

  def testMean(self):
    sample = [[[0, 0], [0, 1]],
              [[0, 1], [1, 2]],
              [[0, 1], [1, 2]],
              [[0, 2], [2, 4]]]

    expected_mean = [[0, 1], [1, 2.25]]

    input_ = tf.convert_to_tensor(sample, dtype=np.float32)
    input_ph = tf.placeholder_with_default(
        input=input_,
        shape=input_.shape if self.static_shape else None)
    dist = empirical.Empirical(samples=input_ph, event_ndims=2)
    self.assertAllClose(self.evaluate(dist.mean()), expected_mean)

  def testMode(self):
    sample = [[[0, 0], [0, 1]],
              [[0, 1], [1, 2]],
              [[0, 1], [1, 2]],
              [[0, 2], [2, 4]]]

    expected_mode = [[0, 1], [1, 2]]

    input_ = tf.convert_to_tensor(sample, dtype=np.int32)
    input_ph = tf.placeholder_with_default(
        input=input_,
        shape=input_.shape if self.static_shape else None)
    dist = empirical.Empirical(samples=input_ph, event_ndims=2)
    self.assertAllClose(self.evaluate(dist.mode()), expected_mode)

  def testEntropy(self):
    sample = [[[0, 0], [0, 1]],
              [[0, 1], [1, 2]],
              [[0, 1], [1, 2]],
              [[0, 2], [2, 4]]]

    expected_entropy = entropy([0.25, 0.5, 0.25])

    input_ = tf.convert_to_tensor(sample, dtype=np.float64)
    input_ph = tf.placeholder_with_default(
        input=input_,
        shape=input_.shape if self.static_shape else None)
    dist = empirical.Empirical(samples=input_ph, event_ndims=2)
    self.assertAllClose(self.evaluate(dist.entropy()), expected_entropy)

  def testVarianceAndStd(self):
    sample = [[[0, 1], [1, 2]],
              [[0, 3], [3, 6]]]

    expected_variance = [[0, 1], [1, 4]]

    input_ = tf.convert_to_tensor(sample, dtype=np.float64)
    input_ph = tf.placeholder_with_default(
        input=input_,
        shape=input_.shape if self.static_shape else None)
    dist = empirical.Empirical(samples=input_ph, event_ndims=2)
    self.assertAllClose(self.evaluate(dist.variance()),
                        expected_variance)
    self.assertAllClose(self.evaluate(dist.stddev()),
                        np.sqrt(expected_variance))


@tf_test_util.run_all_in_graph_and_eager_modes
class EmpiricalScalarStaticShapeTest(EmpiricalScalarTest, tf.test.TestCase):
  static_shape = True

@tf_test_util.run_all_in_graph_and_eager_modes
class EmpiricalScalarDynamicShapeTest(EmpiricalScalarTest, tf.test.TestCase):
  static_shape = False

@tf_test_util.run_all_in_graph_and_eager_modes
class EmpiricalVectorStaticShapeTest(EmpiricalVectorTest, tf.test.TestCase):
  static_shape = True

@tf_test_util.run_all_in_graph_and_eager_modes
class EmpiricalVectorDynamicShapeTest(EmpiricalVectorTest, tf.test.TestCase):
  static_shape = False

@tf_test_util.run_all_in_graph_and_eager_modes
class EmpiricalNdStaticShapeTest(EmpiricalNdTest, tf.test.TestCase):
  static_shape = True
#
@tf_test_util.run_all_in_graph_and_eager_modes
class EmpiricalNdDynamicShapeTest(EmpiricalNdTest, tf.test.TestCase):
  static_shape = False


if __name__ == "__main__":
  tf.test.main()
