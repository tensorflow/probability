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
"""Tests for MCMC drivers (e.g., `sample_chain`)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util

TestTransitionKernelResults = collections.namedtuple(
    'TestTransitionKernelResults', 'counter_1, counter_2')


@test_util.test_all_tf_execution_regimes
class TestTransitionKernel(tfp.mcmc.TransitionKernel):

  def __init__(self, is_calibrated=True):
    self._is_calibrated = is_calibrated

  def one_step(self, current_state, previous_kernel_results):
    return current_state + 1, TestTransitionKernelResults(
        counter_1=previous_kernel_results.counter_1 + 1,
        counter_2=previous_kernel_results.counter_2 + 2)

  def bootstrap_results(self, current_state):
    return TestTransitionKernelResults(counter_1=0, counter_2=0)

  @property
  def is_calibrated(self):
    return self._is_calibrated


class SampleChainTest(test_util.TestCase):

  def setUp(self):
    self._shape_param = 5.
    self._rate_param = 10.

    super(SampleChainTest, self).setUp()
    tf.random.set_seed(10003)
    np.random.seed(10003)

  def testChainWorksCorrelatedMultivariate(self):
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5],
                      [0.5, 1]])
    num_results = 3000
    counter = collections.Counter()
    def target_log_prob(x, y):
      counter['target_calls'] += 1
      # Corresponds to unnormalized MVN.
      # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
      z = tf.stack([x, y], axis=-1) - true_mean
      z = tf.squeeze(
          tf.linalg.triangular_solve(
              np.linalg.cholesky(true_cov),
              z[..., tf.newaxis]),
          axis=-1)
      return -0.5 * tf.reduce_sum(z**2., axis=-1)

    if tf.executing_eagerly():
      tf.random.set_seed(54)
    states, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=[dtype(-2), dtype(2)],
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            step_size=[0.5, 0.5],
            num_leapfrog_steps=2,
            seed=None if tf.executing_eagerly() else 54),
        num_burnin_steps=200,
        num_steps_between_results=1,
        parallel_iterations=1)
    if not tf.executing_eagerly():
      self.assertAllEqual(dict(target_calls=4), counter)
    states = tf.stack(states, axis=-1)
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    x = states - sample_mean
    sample_cov = tf.matmul(x, x, transpose_a=True) / dtype(num_results)
    sample_mean_, sample_cov_ = self.evaluate([sample_mean, sample_cov])
    self.assertAllClose(true_mean, sample_mean_,
                        atol=0.05, rtol=0.)
    self.assertAllClose(true_cov, sample_cov_,
                        atol=0., rtol=0.1)

  def testBasicOperation(self):
    kernel = TestTransitionKernel()
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=2, current_state=0, kernel=kernel)

    self.assertAllClose(
        [2], tensorshape_util.as_list(samples.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_2.shape))

    samples, kernel_results = self.evaluate([samples, kernel_results])
    self.assertAllClose([1, 2], samples)
    self.assertAllClose([1, 2], kernel_results.counter_1)
    self.assertAllClose([2, 4], kernel_results.counter_2)

  def testBurnin(self):
    kernel = TestTransitionKernel()
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=2, current_state=0, kernel=kernel, num_burnin_steps=1)

    self.assertAllClose([2], tensorshape_util.as_list(samples.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_2.shape))

    samples, kernel_results = self.evaluate([samples, kernel_results])
    self.assertAllClose([2, 3], samples)
    self.assertAllClose([2, 3], kernel_results.counter_1)
    self.assertAllClose([4, 6], kernel_results.counter_2)

  def testThinning(self):
    kernel = TestTransitionKernel()
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=2,
        current_state=0,
        kernel=kernel,
        num_steps_between_results=2)

    self.assertAllClose([2], tensorshape_util.as_list(samples.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_2.shape))

    samples, kernel_results = self.evaluate([samples, kernel_results])
    self.assertAllClose([1, 4], samples)
    self.assertAllClose([1, 4], kernel_results.counter_1)
    self.assertAllClose([2, 8], kernel_results.counter_2)

  def testDefaultTraceNamedTuple(self):
    kernel = TestTransitionKernel()
    res = tfp.mcmc.sample_chain(num_results=2, current_state=0, kernel=kernel)

    self.assertAllClose([2], tensorshape_util.as_list(res.all_states.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(res.trace.counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(res.trace.counter_2.shape))

    res = self.evaluate(res)
    self.assertAllClose([1, 2], res.all_states)
    self.assertAllClose([1, 2], res.trace.counter_1)
    self.assertAllClose([2, 4], res.trace.counter_2)

  def testNoTraceFn(self):
    kernel = TestTransitionKernel()
    samples = tfp.mcmc.sample_chain(
        num_results=2, current_state=0, kernel=kernel, trace_fn=None)

    self.assertAllClose([2], tensorshape_util.as_list(samples.shape))

    samples = self.evaluate(samples)
    self.assertAllClose([1, 2], samples)

  def testCustomTrace(self):
    kernel = TestTransitionKernel()
    res = tfp.mcmc.sample_chain(
        num_results=2,
        current_state=0,
        kernel=kernel,
        trace_fn=lambda *args: args)

    self.assertAllClose([2], tensorshape_util.as_list(res.all_states.shape))
    self.assertAllClose([2], tensorshape_util.as_list(res.trace[0].shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(res.trace[1].counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(res.trace[1].counter_2.shape))

    res = self.evaluate(res)
    self.assertAllClose([1, 2], res.all_states)
    self.assertAllClose([1, 2], res.trace[0])
    self.assertAllClose([1, 2], res.trace[1].counter_1)
    self.assertAllClose([2, 4], res.trace[1].counter_2)

  def testCheckpointing(self):
    kernel = TestTransitionKernel()
    res = tfp.mcmc.sample_chain(
        num_results=2,
        current_state=0,
        kernel=kernel,
        trace_fn=None,
        return_final_kernel_results=True)

    self.assertAllClose([2], tensorshape_util.as_list(res.all_states.shape))
    self.assertEqual((), res.trace)
    self.assertAllClose(
        [], tensorshape_util.as_list(res.final_kernel_results.counter_1.shape))
    self.assertAllClose(
        [], tensorshape_util.as_list(res.final_kernel_results.counter_2.shape))

    res = self.evaluate(res)
    self.assertAllClose([1, 2], res.all_states)
    self.assertAllClose(2, res.final_kernel_results.counter_1)
    self.assertAllClose(4, res.final_kernel_results.counter_2)

  def testWarningsDefault(self):
    with warnings.catch_warnings(record=True) as triggered:
      kernel = TestTransitionKernel()
      tfp.mcmc.sample_chain(num_results=2, current_state=0, kernel=kernel)
    self.assertTrue(
        any('Tracing all kernel results by default is deprecated' in str(
            warning.message) for warning in triggered))

  def testNoWarningsExplicit(self):
    with warnings.catch_warnings(record=True) as triggered:
      kernel = TestTransitionKernel()
      tfp.mcmc.sample_chain(
          num_results=2,
          current_state=0,
          kernel=kernel,
          trace_fn=lambda current_state, kernel_results: kernel_results)
    self.assertFalse(
        any('Tracing all kernel results by default is deprecated' in str(
            warning.message) for warning in triggered))

  def testIsCalibrated(self):
    with warnings.catch_warnings(record=True) as triggered:
      kernel = TestTransitionKernel(False)
      tfp.mcmc.sample_chain(
          num_results=2,
          current_state=0,
          kernel=kernel,
          trace_fn=lambda current_state, kernel_results: kernel_results)
    self.assertTrue(
        any('supplied `TransitionKernel` is not calibrated.' in str(
            warning.message) for warning in triggered))


if __name__ == '__main__':
  tf.test.main()
