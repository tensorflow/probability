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

import numpy as np
import scipy as osp
from scipy import optimize
from scipy import stats
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

lb = tfp.experimental.lazybones
tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class DeferredProbabilityUtilsTest(test_util.TestCase):

  def test_log_prob(self):
    tfw = lb.DeferredInput(tf)
    tfbw = lb.DeferredInput(tfb)
    tfdw = lb.DeferredInput(tfd)

    a = tfdw.Normal(0, 1)
    b = a.sample(seed=test_util.test_seed())
    c = tfw.exp(b)
    d = tfbw.Exp()(tfdw.Normal(c, 2.))
    e = d.sample(seed=test_util.test_seed())
    actual1_lp = lb.utils.log_prob([b, e], [3., 4.])
    actual2_lp = lb.utils.log_prob([e, b], [4., 3.])
    actual1_p = lb.utils.prob([b, e], [3., 4.])
    actual2_p = lb.utils.prob([e, b], [4., 3.])

    jd = tfd.JointDistributionSequential([
        tfd.Normal(0, 1),
        lambda b: tfb.Exp()(tfd.Normal(tf.math.exp(b), 2.)),
    ])
    expected_lp = jd.log_prob([3., 4.])
    expected_p = jd.prob([3., 4.])

    [
        expected_lp_,
        actual1_lp_,
        actual2_lp_,
        expected_p_,
        actual1_p_,
        actual2_p_,
    ] = self.evaluate(
        lb.DeferredInput([
            expected_lp,
            actual1_lp,
            actual2_lp,
            expected_p,
            actual1_p,
            actual2_p,
        ]).eval())

    self.assertAlmostEqual(expected_lp_, actual1_lp_, places=4)
    self.assertAlmostEqual(expected_lp_, actual2_lp_, places=4)
    self.assertAlmostEqual(expected_p_, actual1_p_, places=4)
    self.assertAlmostEqual(expected_p_, actual2_p_, places=4)

  def test_end_to_end_scipy_ppl(self):
    sp = lb.DeferredInput(osp)
    st = lb.DeferredInput(stats)

    # Data simulation
    n_feature = 15
    n_obs = 1000
    hyper_mu_ = np.array(10.)
    hyper_sigma_ = np.array(2.)
    sigma_ = np.array(1.5)

    beta_ = hyper_mu_ + hyper_sigma_ * np.random.randn(n_feature)
    design_matrix = np.random.rand(n_obs, n_feature)
    y_ = design_matrix @ beta_ + np.random.randn(n_obs) * sigma_

    # Lazybones model
    hyper_mu = st.norm(0., 100.).rvs()
    hyper_sigma = st.halfnorm(0., 5.).rvs()
    beta = st.norm(hyper_mu, hyper_sigma).rvs(n_feature)
    y_hat = sp.matmul(design_matrix, beta)
    sigma = st.halfnorm(0., 5.).rvs()
    y = st.norm(y_hat, sigma).mean()

    # Inference with MAP
    def target_log_prob_fn(*values):
      return lb.utils.distribution_measure(
          vertexes=[hyper_mu, hyper_sigma, beta, sigma, y],
          values=[*values, y_],
          get_attr_fn=lambda dist: dist.logpdf,
          combine=sum,
          reduce_op=np.sum)

    def loss_fn(x):
      return -target_log_prob_fn(x[0], np.exp(x[1]), x[2:-1], np.exp(x[-1]))

    output = optimize.minimize(
        loss_fn, np.random.randn(n_feature + 3), method='L-BFGS-B')

    x = np.concatenate([
        hyper_mu_[None],
        np.log(hyper_sigma_[None]), beta_,
        np.log(sigma_[None])
    ])
    self.assertAllClose(x, output.x, rtol=.1)


if __name__ == '__main__':
  tf.test.main()
