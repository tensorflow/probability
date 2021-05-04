# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for MultivariateNormalDiagPlusLowRankCovariance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import stats as tfps
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class MultivariateNormalLowRankUpdateLinearOperatorCovarianceTest(
    test_util.TestCase):
  """Short test of this derived class -- most tests are for base class."""

  def _construct_mvn(self,
                     loc_shape,
                     diag_shape,
                     update_shape,
                     dtype=np.float32):
    """Construct LinearOperatorLowRankUpdate covariance with random params."""
    if loc_shape is not None:
      loc = np.random.normal(size=loc_shape).astype(dtype)
    else:
      loc = None
    cov_diag_factor = np.random.uniform(
        low=1., high=2., size=diag_shape).astype(dtype)
    cov_perturb_factor = np.random.normal(
        size=update_shape).astype(dtype)
    return tfd.MultivariateNormalDiagPlusLowRankCovariance(
        loc=loc,
        cov_diag_factor=cov_diag_factor,
        cov_perturb_factor=cov_perturb_factor)

  def testSampleStatsMatchDistributionStats(self):
    mvn = self._construct_mvn(
        loc_shape=(2, 3),
        diag_shape=(3, 2, 3),
        update_shape=(3, 1),
    )
    n = 1000
    samples = mvn.sample(n, seed=test_util.test_seed())

    s_mean, mean, s_cov, cov = self.evaluate([
        tf.reduce_mean(samples, axis=0),
        mvn.mean(),
        tfps.covariance(samples, sample_axis=0),
        mvn.covariance(),
    ])

    maxstddev = np.sqrt(np.max(cov))

    self.assertAllClose(s_mean, mean, atol=5 * maxstddev / np.sqrt(n))
    self.assertAllClose(s_cov, cov, atol=5 * maxstddev**2 / np.sqrt(n))


if __name__ == '__main__':
  tf.test.main()
