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
"""Shared library for `eight_schools_hmc_{graph,eager}_test.py`."""

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
    'EightSchoolsHmcBenchmarkTestHarness',
    'benchmark_eight_schools_hmc',
    'eight_schools_joint_log_prob',
]


def mvn(*args, **kwargs):
  """Convenience function to efficiently construct a MultivariateNormalDiag."""
  # Faster than using `tfd.MultivariateNormalDiag`.
  return tfd.Independent(tfd.Normal(*args, **kwargs),
                         reinterpreted_batch_ndims=1)


def eight_schools_joint_log_prob(
    treatment_effects, treatment_stddevs,
    avg_effect, avg_stddev, school_effects_standard):
  """Eight-schools joint log-prob."""
  rv_avg_effect = tfd.Normal(loc=0., scale=10.)
  rv_avg_stddev = tfd.Normal(loc=5., scale=1.)
  rv_school_effects_standard = mvn(
      loc=tf.zeros_like(school_effects_standard),
      scale=tf.ones_like(school_effects_standard))
  rv_treatment_effects = mvn(
      loc=(avg_effect + tf.math.exp(avg_stddev) * school_effects_standard),
      scale=treatment_stddevs)
  return (
      rv_avg_effect.log_prob(avg_effect) +
      rv_avg_stddev.log_prob(avg_stddev) +
      rv_school_effects_standard.log_prob(school_effects_standard) +
      rv_treatment_effects.log_prob(treatment_effects))


def benchmark_eight_schools_hmc(
    num_results=int(5e3),
    num_burnin_steps=int(3e3),
    num_leapfrog_steps=3,
    step_size=0.4):
  """Runs HMC on the eight-schools unnormalized posterior."""

  num_schools = 8
  treatment_effects = tf.constant(
      [28, 8, -3, 7, -1, 1, 18, 12],
      dtype=np.float32,
      name='treatment_effects')
  treatment_stddevs = tf.constant(
      [15, 10, 16, 11, 9, 11, 10, 18],
      dtype=np.float32,
      name='treatment_stddevs')

  def unnormalized_posterior_log_prob(
      avg_effect, avg_stddev, school_effects_standard):
    """Eight-schools unnormalized log posterior."""
    return eight_schools_joint_log_prob(
        treatment_effects, treatment_stddevs,
        avg_effect, avg_stddev, school_effects_standard)

  if tf.executing_eagerly():
    sample_chain = tf.function(tfp.mcmc.sample_chain)
  else:
    sample_chain = tfp.mcmc.sample_chain

  def computation():
    """The benchmark computation."""
    _, kernel_results = sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=(
            tf.zeros([], name='init_avg_effect'),
            tf.zeros([], name='init_avg_stddev'),
            tf.ones([num_schools], name='init_school_effects_standard'),
        ),
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps))

    return kernel_results.is_accepted

  # Let's force evaluation of graph to ensure build time is not part of our time
  # trial.
  is_accepted_tensor = computation()
  if not tf.executing_eagerly():
    session = tf1.Session()
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


class EightSchoolsHmcBenchmarkTestHarness(object):
  """Test harness for running HMC benchmark tests in graph/eager modes."""

  def __init__(self):
    self._mode = 'eager' if tf.executing_eagerly() else 'graph'

  def benchmark_eight_schools_hmc_num_leapfrog_1(self):
    self.report_benchmark(
        name=self._mode + '_eight_schools_hmc_num_leapfrog_1',
        **benchmark_eight_schools_hmc(num_leapfrog_steps=1))

  def benchmark_eight_schools_hmc_num_leapfrog_2(self):
    self.report_benchmark(
        name=self._mode + '_eight_schools_hmc_num_leapfrog_2',
        **benchmark_eight_schools_hmc(num_leapfrog_steps=2))

  def benchmark_eight_schools_hmc_num_leapfrog_3(self):
    self.report_benchmark(
        name=self._mode + '_eight_schools_hmc_num_leapfrog_3',
        **benchmark_eight_schools_hmc(num_leapfrog_steps=3))

  def benchmark_eight_schools_hmc_num_leapfrog_10(self):
    self.report_benchmark(
        name=self._mode + '_eight_schools_hmc_num_leapfrog_10',
        **benchmark_eight_schools_hmc(num_leapfrog_steps=10))

  def benchmark_eight_schools_hmc_num_leapfrog_20(self):
    self.report_benchmark(
        name=self._mode + '_eight_schools_hmc_num_leapfrog_20',
        **benchmark_eight_schools_hmc(num_leapfrog_steps=20))
