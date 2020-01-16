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
"""Shared library for `text_messages_hmc_{graph,eager}_test.py`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
# Dependency imports
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


__all__ = [
    'TextMessagesHmcBenchmarkTestHarness',
    'benchmark_text_messages_hmc',
    'text_messages_joint_log_prob',
]


def mvn(*args, **kwargs):
  """Convenience function to efficiently construct a MultivariateNormalDiag."""
  # Faster than using `tfd.MultivariateNormalDiag`.
  return tfd.Independent(tfd.Normal(*args, **kwargs),
                         reinterpreted_batch_ndims=1)


def text_messages_joint_log_prob(count_data, lambda_1, lambda_2, tau):
  """Joint log probability function."""
  alpha = (1. / tf.reduce_mean(count_data))
  rv_lambda = tfd.Exponential(rate=alpha)

  rv_tau = tfd.Uniform()

  lambda_ = tf.gather(
      [lambda_1, lambda_2],
      indices=tf.cast(
          tau * tf.cast(tf.size(count_data), dtype=tf.float32) <= tf.cast(
              tf.range(tf.size(count_data)), dtype=tf.float32),
          dtype=tf.int32))
  rv_observation = tfd.Poisson(rate=lambda_)

  return (rv_lambda.log_prob(lambda_1) + rv_lambda.log_prob(lambda_2) +
          rv_tau.log_prob(tau) +
          tf.reduce_sum(rv_observation.log_prob(count_data)))


def benchmark_text_messages_hmc(
    num_results=int(3e3),
    num_burnin_steps=int(3e3),
    num_leapfrog_steps=3):
  """Runs HMC on the text-messages unnormalized posterior."""

  if not tf.executing_eagerly():
    tf1.reset_default_graph()

  # Build a static, pretend dataset.
  count_data = tf.cast(
      tf.concat(
          [tfd.Poisson(rate=15.).sample(43),
           tfd.Poisson(rate=25.).sample(31)],
          axis=0),
      dtype=tf.float32)
  if tf.executing_eagerly():
    count_data = count_data.numpy()
  else:
    with tf1.Session():
      count_data = count_data.eval()

  # Define a closure over our joint_log_prob.
  def unnormalized_log_posterior(lambda1, lambda2, tau):
    return text_messages_joint_log_prob(count_data, lambda1, lambda2, tau)

  if tf.executing_eagerly():
    sample_chain = tf.function(tfp.mcmc.sample_chain)
  else:
    sample_chain = tfp.mcmc.sample_chain

  # Initialize the step_size. (It will be automatically adapted.)
  step_size = tf.Variable(
      name='step_size',
      initial_value=tf.constant(0.05, dtype=tf.float32),
      trainable=False)

  def computation():
    """The benchmark computation."""

    initial_chain_state = [
        tf.constant(count_data.mean(), name='init_lambda1'),
        tf.constant(count_data.mean(), name='init_lambda2'),
        tf.constant(0.5, name='init_tau'),
    ]

    unconstraining_bijectors = [
        tfp.bijectors.Exp(),       # Maps a positive real to R.
        tfp.bijectors.Exp(),       # Maps a positive real to R.
        tfp.bijectors.Sigmoid(),   # Maps [0,1] to R.
    ]

    _, kernel_results = sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_chain_state,
        kernel=tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_posterior,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size,
                step_size_update_fn=
                tfp.mcmc.make_simple_step_size_update_policy(num_burnin_steps),
                state_gradients_are_stopped=True),
            bijector=unconstraining_bijectors))

    return kernel_results.inner_results.is_accepted

  # Let's force evaluation of graph to ensure build time is not part of our time
  # trial.
  is_accepted_tensor = computation()
  if not tf.executing_eagerly():
    session = tf1.Session()
    session.run(tf1.global_variables_initializer())
    session.run(is_accepted_tensor)

  start_time = time.time()
  if tf.executing_eagerly():
    is_accepted = computation()
  else:
    is_accepted = session.run(is_accepted_tensor)
  wall_time = time.time() - start_time

  num_accepted = np.sum(is_accepted)
  acceptance_rate = np.float32(num_accepted) / np.float32(num_results)

  return dict(
      iters=(num_results + num_burnin_steps) * num_leapfrog_steps,
      extras={'acceptance_rate': acceptance_rate},
      wall_time=wall_time)


class TextMessagesHmcBenchmarkTestHarness(object):
  """Test harness for running HMC benchmark tests in graph/eager modes."""

  def __init__(self):
    self._mode = 'eager' if tf.executing_eagerly() else 'graph'

  def benchmark_text_messages_hmc_num_leapfrog_1(self):
    self.report_benchmark(
        name=self._mode + '_text_messages_hmc_num_leapfrog_1',
        **benchmark_text_messages_hmc(num_leapfrog_steps=1))

  def benchmark_text_messages_hmc_num_leapfrog_2(self):
    self.report_benchmark(
        name=self._mode + '_text_messages_hmc_num_leapfrog_2',
        **benchmark_text_messages_hmc(num_leapfrog_steps=2))

  def benchmark_text_messages_hmc_num_leapfrog_3(self):
    self.report_benchmark(
        name=self._mode + '_text_messages_hmc_num_leapfrog_3',
        **benchmark_text_messages_hmc(num_leapfrog_steps=3))

  def benchmark_text_messages_hmc_num_leapfrog_10(self):
    self.report_benchmark(
        name=self._mode + '_text_messages_hmc_num_leapfrog_10',
        **benchmark_text_messages_hmc(num_leapfrog_steps=10))

  def benchmark_text_messages_hmc_num_leapfrog_20(self):
    self.report_benchmark(
        name=self._mode + '_text_messages_hmc_num_leapfrog_20',
        **benchmark_text_messages_hmc(num_leapfrog_steps=20))
