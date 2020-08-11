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
# Lint as: python3
"""Tests for tensorflow_probability.experimental.lazybones.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
lb = tfp.experimental.lazybones


@test_util.test_all_tf_execution_regimes
class DeferredProbabilityUtilsTest(test_util.TestCase):

  def test_log_prob(self):
    tfw = lb.DeferredInput(tf)
    tfdw = lb.DeferredInput(tfd)

    a = tfdw.Normal(0, 1)
    b = a.sample(seed=test_util.test_seed())
    c = tfw.exp(b)
    d = tfdw.Gamma(c, 2.)
    e = d.sample(seed=test_util.test_seed())
    actual1_lp = lb.utils.log_prob([b, e], [3., 4.])
    actual2_lp = lb.utils.log_prob([e, b], [4., 3.])
    actual1_p = lb.utils.prob([b, e], [3., 4.])
    actual2_p = lb.utils.prob([e, b], [4., 3.])

    jd = tfd.JointDistributionSequential([
        tfd.Normal(0, 1),
        lambda b: tfd.Gamma(tf.math.exp(b), 2.)
    ])
    expected_lp = jd.log_prob([3., 4.])
    expected_p = jd.prob([3., 4.])

    [
        expected_lp_, actual1_lp_, actual2_lp_,
        expected_p_, actual1_p_, actual2_p_,
    ] = self.evaluate(
        lb.DeferredInput([
            expected_lp, actual1_lp, actual2_lp,
            expected_p, actual1_p, actual2_p,
        ]).eval())

    self.assertAlmostEqual(expected_lp_, actual1_lp_, places=4)
    self.assertAlmostEqual(expected_lp_, actual2_lp_, places=4)
    self.assertAlmostEqual(expected_p_, actual1_p_, places=4)
    self.assertAlmostEqual(expected_p_, actual2_p_, places=4)


if __name__ == '__main__':
  tf.test.main()
