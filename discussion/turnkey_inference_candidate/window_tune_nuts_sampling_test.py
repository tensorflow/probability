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
"""Tests for window_tune_nuts_sampling."""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from discussion import turnkey_inference_candidate
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


class WindowTuneNutsSamplingTest(test_util.TestCase):

  def test_docstring_example(self):
    strm = test_util.test_seed_stream()
    dtype = np.float32

    nd = 50
    concentration = 1.

    prior_dist = tfd.Sample(tfd.Normal(tf.constant(0., dtype), 100.), nd)

    mu = tf.cast(np.linspace(-100, 100, nd), dtype=dtype)
    sigma = tf.cast(np.exp(np.linspace(-1, 1.5, nd)), dtype=dtype)
    corr_tril = tfd.CholeskyLKJ(
        dimension=nd, concentration=concentration).sample(seed=strm())
    scale_tril = tf.linalg.matmul(tf.linalg.diag(sigma), corr_tril)
    target_dist = tfd.MultivariateStudentTLinearOperator(
        df=5.,
        loc=mu,
        scale=tf.linalg.LinearOperatorLowerTriangular(scale_tril))

    target_log_prob = lambda *x: (  # pylint: disable=g-long-lambda
        prior_dist.log_prob(*x) + target_dist.log_prob(*x))

    (
        [mcmc_samples],
        diagnostic, conditioner  # pylint: disable=unused-variable
    ) = turnkey_inference_candidate.window_tune_nuts_sampling(
        target_log_prob,
        [prior_dist.sample(2000, seed=strm())],
        seed=strm(),
        parallel_iterations=1)

    self.assertAllClose(
        tf.reduce_mean(mcmc_samples, [0, 1]), mu, atol=.3, rtol=.05)

    self.assertAllClose(tf.math.reduce_std(mcmc_samples, [0, 1]),
                        target_dist._stddev(), atol=.5, rtol=.1)


if __name__ == '__main__':
  tf.test.main()
