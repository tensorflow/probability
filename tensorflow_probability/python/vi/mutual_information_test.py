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
"""Tests for mutual information estimators and helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util as tfp_test_util

mi = tfp.vi.mutual_information
tfd = tfp.distributions


LOWER_BOUND_MIN_GAP = 0.3
LOWER_BOUND_MAX_GAP = 0.1


class MutualInformationTest(tfp_test_util.TestCase):

  def setUp(self):
    super(MutualInformationTest, self).setUp()
    self.seed = tfp_test_util.test_seed()
    np.random.seed(self.seed)

    self.scores = np.random.normal(
        loc=1.0,
        scale=2.0,
        size=[13, 17])

    batch_size = 1000
    rho = 0.8
    dim = 2
    x, eps = tf.split(value=tf.random.normal(shape=(2*batch_size, dim),
                                             seed=self.seed),
                      num_or_size_splits=2, axis=0)
    mean = rho * x
    stddev = tf.sqrt(1. - tf.square(rho))
    y = mean + stddev * eps
    conditional_dist = tfd.MultivariateNormalDiag(
        mean, scale_identity_multiplier=stddev)
    marginal_dist = tfd.MultivariateNormalDiag(tf.zeros(dim), tf.ones(dim))

    # The conditional_scores has its shape [y_batch_dim, distibution_batch_dim]
    # as the `lower_bound_info_nce` requires `scores[i, j] = f(x[i], y[j])
    # = log p(x[i] | y[j])`.
    self.conditional_scores = conditional_dist.log_prob(y[:, tf.newaxis, :])
    self.marginal_scores = marginal_dist.log_prob(y)[:, tf.newaxis]
    self.optimal_critic = 1 + self.conditional_scores - self.marginal_scores
    self.theoretical_mi = np.float32(-0.5 * np.log(1. - rho**2) * dim)
    # Y is N-D standard normal distributed.
    self.differential_entropy_y = 0.5 * np.log(2 * np.pi * np.e) * dim

  def test_check_and_get_mask(self):
    test_scores = tf.ones([2, 3])
    positive_mask = np.eye(N=2, M=3, dtype=bool)

    # create default masks
    r_scores, r_pos_mask = mi._check_and_get_mask(test_scores)
    self.assertEqual(r_scores.shape, [2, 3])
    self.assertAllEqual(self.evaluate(r_pos_mask), positive_mask)

  def test_get_masked_scores(self):
    scores = np.array([[2., 5., -1e-3],
                       [-1073., 4.2, -4.]]).astype(np.float32)
    mask = scores < 3.
    target_res = np.array([[2., -np.inf, -1e-3],
                           [-1073., -np.inf, -4.]]).astype(np.float32)
    func_res = mi._get_masked_scores(scores, mask)
    self.assertAllEqual(self.evaluate(func_res), target_res)

  def test_masked_logmeanexp(self):
    # test1: compare against numpy/scipy implementation.
    masked_scores = self.scores
    num_masked_ele = np.sum(masked_scores > 0.)
    masked_scores[masked_scores <= 0.] = -np.inf
    numpy_impl = np.float32(
        scipy.special.logsumexp(masked_scores) - np.log(num_masked_ele))
    result_0d = mi._masked_logmeanexp(self.scores, self.scores > 0, axis=None)
    self.assertAllClose(self.evaluate(result_0d), numpy_impl)

    # test2: test against results from composition of numpy functions.
    scores_2 = np.array([[2., 5., -1e-3],
                         [-1073., 4.2, -4.]], dtype=np.float32)
    result_empty_sum = mi._masked_logmeanexp(
        scores_2, scores_2 < 0., axis=None)
    numpy_result = np.log(np.mean(np.exp(scores_2[scores_2 < 0.])))
    self.assertAllClose(self.evaluate(result_empty_sum),
                        numpy_result.astype(np.float32))

    # test3: whether `axis` arg works as expected.
    result_1d = mi._masked_logmeanexp(self.scores, self.scores > 0, axis=[1,])
    self.assertEqual(result_1d.shape, [13,])

  def test_lower_bound_barber_agakov(self):
    # Test1: against numpy reimplementation
    test_scores = tf.random.normal(shape=[100,], stddev=5.)
    test_entropy = tf.random.normal(shape=[], stddev=10.)
    impl_estimation, test_scores, test_entropy = self.evaluate(
        [mi.lower_bound_barber_agakov(logu=test_scores, entropy=test_entropy),
         test_scores, test_entropy])
    numpy_estimation = np.mean(test_scores) + test_entropy
    self.assertAllClose(impl_estimation, numpy_estimation)

    # Test2: batched input
    test_scores_2 = tf.random.normal(shape=[13, 5], stddev=5.)
    test_entropy_2 = tf.random.normal(shape=[13,], stddev=10.)
    impl_estimation_2, test_scores_2, test_entropy_2 = self.evaluate(
        [mi.lower_bound_barber_agakov(
            logu=test_scores_2, entropy=test_entropy_2),
         test_scores_2, test_entropy_2])
    numpy_estimation_2 = np.mean(test_scores_2, axis=-1) + test_entropy_2
    self.assertAllClose(impl_estimation_2, numpy_estimation_2)

    # Test3: test example, since the estimation is a lower bound, we test
    # by range.
    impl_estimation_3 = self.evaluate(
        mi.lower_bound_barber_agakov(
            logu=tf.linalg.diag_part(self.conditional_scores),
            entropy=self.differential_entropy_y))
    self.assertAllInRange(
        impl_estimation_3,
        self.theoretical_mi-LOWER_BOUND_MIN_GAP,
        self.theoretical_mi+LOWER_BOUND_MAX_GAP)

  def test_lower_bound_info_nce(self):
    # Numerical test with correlated gaussian as random variables.
    info_nce_bound = self.evaluate(
        mi.lower_bound_info_nce(self.conditional_scores))
    self.assertAllInRange(
        info_nce_bound,
        lower_bound=self.theoretical_mi-LOWER_BOUND_MIN_GAP,
        upper_bound=self.theoretical_mi+LOWER_BOUND_MAX_GAP)

    # Check the masked against none masked version
    info_nce_bound_1 = self.evaluate(
        mi.lower_bound_info_nce(self.scores))
    positive_mask = np.eye(self.scores.shape[0], self.scores.shape[1])
    info_nce_bound_2 = self.evaluate(
        mi.lower_bound_info_nce(self.scores, positive_mask, validate_args=True))
    self.assertAllClose(info_nce_bound_1, info_nce_bound_2)

    # Check batched against none batched version
    info_nce_bound_3 = self.evaluate(
        mi.lower_bound_info_nce(tf.tile(self.scores[None, :, :], [3, 1, 1])))
    self.assertAllClose(
        info_nce_bound_3,
        self.evaluate(tf.tile(info_nce_bound_1[tf.newaxis,], [3])))

  def test_lower_bound_jensen_shannon(self):
    # Check against numpy implementation.
    log_f = self.optimal_critic
    js_bound, log_f = self.evaluate([mi.lower_bound_jensen_shannon(log_f),
                                     log_f])
    # The following numpy softplus is numerically stable when x is large
    # log(1+exp(x)) = log(1+exp(x)) - log(exp(x)) + x = log(1+exp(-x)) + x
    numpy_softplus = lambda x: np.log(1+np.exp(-np.abs(x))) + np.maximum(x, 0)

    log_f_diag = np.diag(log_f)
    n = np.float32(log_f.shape[0])
    first_term = np.mean(-numpy_softplus(-log_f_diag))
    second_term = (np.sum(numpy_softplus(log_f)) -
                   np.sum(numpy_softplus(log_f_diag))) / (n * (n - 1.))
    numpy_implementation = first_term - second_term
    self.assertAllClose(js_bound, numpy_implementation, rtol=1e-5)

    # Check the masked against none masked version
    js_bound_1 = mi.lower_bound_jensen_shannon(self.scores)
    positive_mask = np.eye(self.scores.shape[0], self.scores.shape[1])
    js_bound_2 = self.evaluate(
        mi.lower_bound_jensen_shannon(self.scores, positive_mask,
                                      validate_args=True))
    self.assertAllClose(js_bound_1, js_bound_2)

    # Check batched against none batched version
    js_bound_3 = self.evaluate(
        mi.lower_bound_jensen_shannon(
            tf.tile(self.scores[tf.newaxis, :, :], [3, 1, 1])))
    self.assertAllClose(
        js_bound_3, self.evaluate(tf.tile(js_bound_1[tf.newaxis,], [3])))

  def test_lower_bound_nguyen_wainwright_jordan(self):
    # Numerical test against theoretical values
    nwj_bound = self.evaluate(
        mi.lower_bound_nguyen_wainwright_jordan(self.optimal_critic))
    self.assertAllInRange(
        nwj_bound,
        lower_bound=self.theoretical_mi-LOWER_BOUND_MIN_GAP,
        upper_bound=self.theoretical_mi+LOWER_BOUND_MAX_GAP)

    # Check the masked against none masked version
    nwj_bound_1 = mi.lower_bound_nguyen_wainwright_jordan(self.scores)
    positive_mask = np.eye(self.scores.shape[0], self.scores.shape[1])
    nwj_bound_2 = self.evaluate(
        mi.lower_bound_nguyen_wainwright_jordan(
            self.scores, positive_mask, validate_args=True))
    self.assertAllClose(nwj_bound_1, nwj_bound_2)

    # Check batched against none batched version
    nwj_bound_3 = self.evaluate(
        mi.lower_bound_nguyen_wainwright_jordan(
            tf.tile(self.scores[tf.newaxis, :, :], [3, 1, 1])))
    self.assertAllClose(
        nwj_bound_3, self.evaluate(tf.tile(nwj_bound_1[None,], [3])))


if __name__ == '__main__':
  tf.test.main()
