# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for benchmarking Gibbs sampler performance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.experimental.sts_gibbs import gibbs_sampler
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


tfd = tfp.distributions
tfl = tf.linalg


@test_util.run_all_in_graph_and_eager_modes
class XLABenchmarkTests(tfp_test_util.TestCase):

  def _build_test_model(self,
                        num_timesteps=5,
                        num_features=2,
                        batch_shape=(),
                        missing_prob=0,
                        true_noise_scale=0.1,
                        true_level_scale=0.04,
                        dtype=tf.float32):
    strm = tfp_test_util.test_seed_stream()
    design_matrix = tf.random.normal([num_timesteps, num_features],
                                     dtype=dtype, seed=strm())
    weights = tf.random.normal(list(batch_shape) + [num_features],
                               dtype=dtype, seed=strm())
    regression = tf.linalg.matvec(design_matrix, weights)
    noise = tf.random.normal(list(batch_shape) + [num_timesteps],
                             dtype=dtype, seed=strm()) * true_noise_scale
    level = tf.cumsum(
        tf.random.normal(list(batch_shape) + [num_timesteps],
                         dtype=dtype, seed=strm()) * true_level_scale, axis=-1)
    time_series = (regression + noise + level)
    is_missing = tf.random.uniform(list(batch_shape) + [num_timesteps],
                                   dtype=dtype, seed=strm()) < missing_prob

    model = gibbs_sampler.build_model_for_gibbs_fitting(
        observed_time_series=tfp.sts.MaskedTimeSeries(
            time_series[..., tf.newaxis], is_missing),
        design_matrix=design_matrix,
        weights_prior=tfd.Normal(loc=tf.cast(0., dtype),
                                 scale=tf.cast(10.0, dtype)),
        level_variance_prior=tfd.InverseGamma(
            concentration=tf.cast(0.01, dtype),
            scale=tf.cast(0.01 * 0.01, dtype)),
        observation_noise_variance_prior=tfd.InverseGamma(
            concentration=tf.cast(0.01, dtype),
            scale=tf.cast(0.01 * 0.01, dtype)))
    return model, time_series, is_missing

  def test_benchmark_sampling_with_xla(self):
    if not tf.executing_eagerly():
      return
    seed = tfp_test_util.test_seed()
    model, observed_time_series, is_missing = self._build_test_model(
        num_timesteps=336, batch_shape=[])

    t0 = time.time()
    samples = gibbs_sampler.fit_with_gibbs_sampling(
        model, tfp.sts.MaskedTimeSeries(
            observed_time_series[..., tf.newaxis], is_missing),
        num_results=500, num_warmup_steps=100, seed=seed,
        compile_steps_with_xla=True)
    t1 = time.time()
    print('Drew (100+500) samples in time', t1-t0)
    print('Results:', samples)


if __name__ == '__main__':
  tf.test.main()
