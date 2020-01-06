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
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


def entropy(probs):
  probs = np.array(probs)
  return np.sum(-probs * np.log(probs))


def random_samples(shape):
  return np.random.uniform(size=list(shape))


@test_util.test_all_tf_execution_regimes
class EmpiricalScalarTest(test_util.VectorDistributionTestHelpers):

  def testSamples(self):
    for samples_shape in ([2], [2, 4], [4, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf1.placeholder_with_default(
          input_, shape=samples_shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
      self.assertAllClose(input_ph, self.evaluate(dist.samples))

    invalid_sample = 0
    with self.assertRaises(ValueError):
      dist = tfd.Empirical(samples=invalid_sample, validate_args=True)

  def testShapes(self):
    for samples_shape in ([2], [2, 4], [4, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf1.placeholder_with_default(
          input_, shape=samples_shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
      self.assertAllClose(self.evaluate(dist.cdf(event)),
                          expected_cdf)
      self.assertAllClose(self.evaluate(dist.log_cdf(event)),
                          np.log(expected_cdf))

  def testCdfWithBatch(self):
    sample = [[0, 0, 1, 2], [0, 10, 20, 40]]
    events = [
        [0],
        [0, 10],
        [[0, 1], [10, 20]]
    ]
    expected_cdfs = [
        [0.5, 0.25],
        [0.5, 0.5],
        [[0.5, 0.25], [1, 0.75]]
    ]

    for event, expected_cdf in zip(events, expected_cdfs):
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
      self.assertAllClose(self.evaluate(dist.prob(event)),
                          expected_pmf)
      self.assertAllClose(self.evaluate(dist.log_prob(event)),
                          np.log(expected_pmf))

  def testPmfWithBatch(self):
    sample = [[0, 0, 1, 2], [0, 10, 20, 40]]
    events = [
        [0],
        [0, 10],
        [[0, 1], [10, 20]]
    ]
    expected_pmfs = [
        [0.5, 0.25],
        [0.5, 0.25],
        [[0.5, 0], [0, 0.25]]
    ]

    for event, expected_pmf in zip(events, expected_pmfs):
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
      self.assertAllClose(self.evaluate(dist.prob(event)),
                          expected_pmf)
      self.assertAllClose(self.evaluate(dist.log_prob(event)),
                          np.log(expected_pmf))

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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float64)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
      self.assertAllClose(self.evaluate(dist.entropy()), expected_entropy)

  def testSampleN(self):
    for samples_shape in ([2], [2, 4], [4, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf1.placeholder_with_default(
          input_, shape=samples_shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
      self.assertAllEqual(
          self.evaluate(
              tf.shape(dist.sample(seed=test_util.test_seed()))),
          dist.batch_shape_tensor())

      n = 1000
      seed = tf.random.set_seed(42) if tf.executing_eagerly() else 42
      samples1 = dist.sample(n, seed)
      seed = tf.random.set_seed(42) if tf.executing_eagerly() else 42
      samples2 = dist.sample(n, seed)
      self.assertAllEqual(
          self.evaluate(samples1), self.evaluate(samples2))

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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.int32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, validate_args=True)
      self.assertAllClose(self.evaluate(dist.variance()),
                          expected_variance)
      self.assertAllClose(self.evaluate(dist.stddev()),
                          np.sqrt(expected_variance))


@test_util.test_all_tf_execution_regimes
class EmpiricalVectorTest(test_util.VectorDistributionTestHelpers):

  def testSamples(self):
    for samples_shape in ([2, 4], [4, 2, 4], [2, 2, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf1.placeholder_with_default(
          input_, shape=samples_shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
      self.assertAllClose(input_ph, self.evaluate(dist.samples))

    invalid_samples = [
        0,
        [0, 1],
    ]
    for samples in invalid_samples:
      with self.assertRaises(ValueError):
        dist = tfd.Empirical(samples=samples, event_ndims=1, validate_args=True)

  def testShapes(self):
    for samples_shape in ([2, 4], [4, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf1.placeholder_with_default(
          input_, shape=samples_shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
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
        0.75,
        [0.75, 1],
        [[0.25, 0.75], [0.75, 1]]
    ]

    for event, expected_cdf in zip(events, expected_cdfs):
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
      self.assertAllClose(self.evaluate(dist.cdf(event)),
                          expected_cdf)
      self.assertAllClose(self.evaluate(dist.log_cdf(event)),
                          np.log(expected_cdf))

  def testCdfWithBatch(self):
    sample = [[[0, 0], [0, 1], [0, 1], [1, 1]],
              [[0, 10], [10, 10], [10, 20], [20, 20]]]
    events = [
        [0, 1],
        [[0, 1], [10, 10]],
        [[[0, 0], [0, 10]],
         [[0, 1], [10, 20]]],
        [[0], [10]],
        [[[0, 1]], [[10, 20]]]
    ]
    expected_cdfs = [
        [0.75, 0],
        [0.75, 0.5],
        [[0.25, 0.25], [0.75, 0.75]],
        [0.25, 0.5],
        [[0.75, 0], [1, 0.75]]
    ]

    for event, expected_cdf in zip(events, expected_cdfs):
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
      self.assertAllClose(self.evaluate(dist.cdf(event)),
                          expected_cdf)
      self.assertAllClose(self.evaluate(dist.log_cdf(event)),
                          np.log(expected_cdf))

  def testPmfNoBatch(self):
    samples = [[0, 0], [0, 1], [0, 1], [1, 2]]
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
      input_ = tf.convert_to_tensor(value=samples, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
      self.assertAllClose(self.evaluate(dist.prob(event)),
                          expected_pmf)
      self.assertAllClose(self.evaluate(dist.log_prob(event)),
                          np.log(expected_pmf))

  def testPmfWithBatch(self):
    sample = [[[0, 0], [0, 1], [0, 1], [1, 1]],
              [[0, 10], [10, 10], [10, 20], [20, 20]]]
    events = [
        [0, 1],
        [[0, 1], [10, 10]],
        [[[0, 0], [0, 10]],
         [[0, 1], [10, 20]]],
        [[0], [10]],
        [[[0, 1]], [[10, 20]]]

    ]
    expected_pmfs = [
        [0.5, 0],
        [0.5, 0.25],
        [[0.25, 0.25], [0.5, 0.25]],
        [0.25, 0.25],
        [[0.5, 0], [0, 0.25]]
    ]

    for event, expected_pmf in zip(events, expected_pmfs):
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
      self.assertAllClose(self.evaluate(dist.prob(event)),
                          expected_pmf)
      self.assertAllClose(self.evaluate(dist.log_prob(event)),
                          np.log(expected_pmf))

  def testSampleN(self):
    for samples_shape in ([2, 4], [4, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf1.placeholder_with_default(
          input_, shape=samples_shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
      expected_shape = tf.concat(
          [dist.batch_shape_tensor(), dist.event_shape_tensor()],
          axis=0)
      self.assertAllEqual(
          self.evaluate(
              tf.shape(dist.sample(seed=test_util.test_seed()))),
          expected_shape)

      n = 1000
      seed = tf.random.set_seed(42) if tf.executing_eagerly() else 42
      samples1 = dist.sample(n, seed)
      seed = tf.random.set_seed(42) if tf.executing_eagerly() else 42
      samples2 = dist.sample(n, seed)
      self.assertAllEqual(
          self.evaluate(samples1), self.evaluate(samples2))

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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.int32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float64)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=1, validate_args=True)
      self.assertAllClose(self.evaluate(dist.variance()),
                          expected_variance)
      self.assertAllClose(self.evaluate(dist.stddev()),
                          np.sqrt(expected_variance))


@test_util.test_all_tf_execution_regimes
class EmpiricalNdTest(test_util.VectorDistributionTestHelpers,
                      test_util.TestCase):

  def testSamples(self):
    for samples_shape in ([4, 2, 4], [4, 2, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf1.placeholder_with_default(
          input_, shape=samples_shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=2, validate_args=True)
      self.assertAllClose(input_ph, self.evaluate(dist.samples))

  @parameterized.named_parameters(
      {'testcase_name': 'ZeroRank', 'samples': 0},
      {'testcase_name': 'TooFewDims1', 'samples': [0, 1]},
      {'testcase_name': 'TooFewDims2', 'samples': [[0, 1], [1, 2]]},
  )
  def testInvalidSamples(self, samples):
    input_ph = tf1.placeholder_with_default(
        samples,
        shape=np.array(samples).shape if self.static_shape else None)
    with self.assertRaises(Exception):
      dist = tfd.Empirical(samples=input_ph, event_ndims=2, validate_args=True)
      self.evaluate(dist.mean())

  def testShapes(self):
    for samples_shape in ([2, 2, 4], [4, 2, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf1.placeholder_with_default(
          input_, shape=samples_shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=2, validate_args=True)
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
        [[[0, 0], [1, 2]],
         [[0, 2], [2, 4]]]
    ]
    expected_cdfs = [
        0.25,
        [0.5, 0.75]
    ]

    for event, expected_cdf in zip(events, expected_cdfs):
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=2, validate_args=True)
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
      input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
      input_ph = tf1.placeholder_with_default(
          input_, shape=input_.shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=2, validate_args=True)
      self.assertAllClose(self.evaluate(dist.prob(event)),
                          expected_pmf)
      self.assertAllClose(self.evaluate(dist.log_prob(event)),
                          np.log(expected_pmf))

  def testSampleN(self):
    for samples_shape in ([2, 2, 4], [4, 2, 2, 4]):
      input_ = random_samples(samples_shape)
      input_ph = tf1.placeholder_with_default(
          input_, shape=samples_shape if self.static_shape else None)
      dist = tfd.Empirical(samples=input_ph, event_ndims=2, validate_args=True)
      expected_shape = tf.concat(
          [dist.batch_shape_tensor(), dist.event_shape_tensor()],
          axis=0)
      self.assertAllEqual(
          self.evaluate(tf.shape(
              dist.sample(seed=test_util.test_seed()))), expected_shape)

      n = 1000
      seed = tf.random.set_seed(42) if tf.executing_eagerly() else 42
      samples1 = dist.sample(n, seed)
      seed = tf.random.set_seed(42) if tf.executing_eagerly() else 42
      samples2 = dist.sample(n, seed)
      self.assertAllEqual(
          self.evaluate(samples1), self.evaluate(samples2))

  def testMean(self):
    sample = [[[0, 0], [0, 1]],
              [[0, 1], [1, 2]],
              [[0, 1], [1, 2]],
              [[0, 2], [2, 4]]]

    expected_mean = [[0, 1], [1, 2.25]]

    input_ = tf.convert_to_tensor(value=sample, dtype=np.float32)
    input_ph = tf1.placeholder_with_default(
        input_, shape=input_.shape if self.static_shape else None)
    dist = tfd.Empirical(samples=input_ph, event_ndims=2, validate_args=True)
    self.assertAllClose(self.evaluate(dist.mean()), expected_mean)

  def testMode(self):
    sample = [[[0, 0], [0, 1]],
              [[0, 1], [1, 2]],
              [[0, 1], [1, 2]],
              [[0, 2], [2, 4]]]

    expected_mode = [[0, 1], [1, 2]]

    input_ = tf.convert_to_tensor(value=sample, dtype=np.int32)
    input_ph = tf1.placeholder_with_default(
        input_, shape=input_.shape if self.static_shape else None)
    dist = tfd.Empirical(samples=input_ph, event_ndims=2, validate_args=True)
    self.assertAllClose(self.evaluate(dist.mode()), expected_mode)

  def testEntropy(self):
    sample = [[[0, 0], [0, 1]],
              [[0, 1], [1, 2]],
              [[0, 1], [1, 2]],
              [[0, 2], [2, 4]]]

    expected_entropy = entropy([0.25, 0.5, 0.25])

    input_ = tf.convert_to_tensor(value=sample, dtype=np.float64)
    input_ph = tf1.placeholder_with_default(
        input_, shape=input_.shape if self.static_shape else None)
    dist = tfd.Empirical(samples=input_ph, event_ndims=2, validate_args=True)
    self.assertAllClose(self.evaluate(dist.entropy()), expected_entropy)

  def testVarianceAndStd(self):
    sample = [[[0, 1], [1, 2]],
              [[0, 3], [3, 6]]]

    expected_variance = [[0, 1], [1, 4]]

    input_ = tf.convert_to_tensor(value=sample, dtype=np.float64)
    input_ph = tf1.placeholder_with_default(
        input_, shape=input_.shape if self.static_shape else None)
    dist = tfd.Empirical(samples=input_ph, event_ndims=2, validate_args=True)
    self.assertAllClose(self.evaluate(dist.variance()),
                        expected_variance)
    self.assertAllClose(self.evaluate(dist.stddev()),
                        np.sqrt(expected_variance))

  def testLogProbAfterSlice(self):
    samples = np.random.randn(6, 5, 4)
    dist = tfd.Empirical(samples=samples, event_ndims=1, validate_args=True)
    self.assertAllEqual((6,), dist.batch_shape)
    self.assertAllEqual((4,), dist.event_shape)
    sliced_dist = dist[:, tf.newaxis]
    samples = self.evaluate(dist.sample(seed=test_util.test_seed()))
    self.assertAllEqual((6, 4), samples.shape)
    lp, sliced_lp = self.evaluate([
        dist.log_prob(samples), sliced_dist.log_prob(samples[:, tf.newaxis])])
    self.assertAllEqual(lp[:, tf.newaxis], sliced_lp)


class EmpiricalScalarStaticShapeTest(
    EmpiricalScalarTest, test_util.TestCase):
  static_shape = True


class EmpiricalScalarDynamicShapeTest(
    EmpiricalScalarTest, test_util.TestCase):
  static_shape = False


class EmpiricalVectorStaticShapeTest(
    EmpiricalVectorTest, test_util.TestCase):
  static_shape = True


class EmpiricalVectorDynamicShapeTest(
    EmpiricalVectorTest, test_util.TestCase):
  static_shape = False


class EmpiricalNdStaticShapeTest(
    EmpiricalNdTest):
  static_shape = True


class EmpiricalNdDynamicShapeTest(
    EmpiricalNdTest):
  static_shape = False
del EmpiricalNdTest


if __name__ == '__main__':
  tf.test.main()
