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
"""Tests for GLM Fisher Scoring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class FitTestFast(test_util.TestCase):

  dtype = np.float32
  fast = True

  def make_dataset(self, n, d, link, offset=None, scale=1.):
    seed = tfp.util.SeedStream(
        seed=213356351, salt='tfp.glm.fisher_scoring_test')
    model_coefficients = tfd.Uniform(
        low=np.array(-0.5, self.dtype),
        high=np.array(0.5, self.dtype)).sample(d, seed=seed())
    radius = np.sqrt(2.)
    model_coefficients *= radius / tf.linalg.norm(tensor=model_coefficients)
    model_matrix = tfd.Normal(
        loc=np.array(0, self.dtype),
        scale=np.array(1, self.dtype)).sample([n, d], seed=seed())
    scale = tf.convert_to_tensor(value=scale, dtype=self.dtype)
    linear_response = tf.tensordot(
        model_matrix, model_coefficients, axes=[[1], [0]])
    if offset is not None:
      linear_response += offset
    if link == 'linear':
      response = tfd.Normal(loc=linear_response, scale=scale).sample(
          seed=seed())
    elif link == 'probit':
      response = tf.cast(
          tfd.Normal(loc=linear_response, scale=scale).sample(seed=seed()) > 0,
          self.dtype)
    elif link == 'logit':
      response = tfd.Bernoulli(logits=linear_response).sample(seed=seed())
    else:
      raise ValueError('unrecognized true link: {}'.format(link))
    return model_matrix, response, model_coefficients, linear_response

  def testProbitWorksCorrectly(self):
    [
        model_matrix,
        response,
        model_coefficients_true,
        linear_response_true,
    ] = self.make_dataset(n=int(1e4), d=3, link='probit')
    model_coefficients, linear_response, is_converged, num_iter = tfp.glm.fit(
        model_matrix,
        response,
        tfp.glm.BernoulliNormalCDF(),
        fast_unsafe_numerics=self.fast,
        maximum_iterations=10)
    [
        model_coefficients_,
        linear_response_,
        is_converged_,
        num_iter_,
        model_coefficients_true_,
        linear_response_true_,
        response_,
    ] = self.evaluate([
        model_coefficients,
        linear_response,
        is_converged,
        num_iter,
        model_coefficients_true,
        linear_response_true,
        response,
    ])
    prediction = linear_response_ > 0.
    accuracy = np.mean(response_ == prediction)
    # Since both the true data generating process and model are the same, the
    # diff between true and predicted linear responses should be zero, on
    # average.
    avg_response_diff = np.mean(linear_response_ - linear_response_true_)

    self.assertTrue(num_iter_ < 10)
    self.assertNear(0., avg_response_diff, err=4e-3)
    self.assertAllClose(0.8, accuracy, atol=0., rtol=0.03)
    self.assertAllClose(model_coefficients_true_, model_coefficients_,
                        atol=0.03, rtol=0.15)
    self.assertTrue(is_converged_)

  def testLinearWorksCorrectly(self):
    [
        model_matrix,
        response,
        model_coefficients_true,
        linear_response_true,
    ] = self.make_dataset(n=int(1e4), d=3, link='linear')
    model_coefficients, linear_response, is_converged, num_iter = tfp.glm.fit(
        model_matrix,
        response,
        tfp.glm.Normal(),
        fast_unsafe_numerics=self.fast,
        maximum_iterations=10)
    [
        model_coefficients_,
        linear_response_,
        is_converged_,
        num_iter_,
        model_coefficients_true_,
        linear_response_true_,
    ] = self.evaluate([
        model_coefficients,
        linear_response,
        is_converged,
        num_iter,
        model_coefficients_true,
        linear_response_true,
    ])
    # Since both the true data generating process and model are the same, the
    # diff between true and predicted linear responses should be zero, on
    # average.
    avg_response_diff = np.mean(linear_response_ - linear_response_true_)
    self.assertNear(0., avg_response_diff, err=3e-3)
    self.assertAllClose(model_coefficients_true_, model_coefficients_,
                        atol=0.03, rtol=0.15)
    self.assertTrue(is_converged_)
    # Since linear regression is a quadratic objective and because
    # we're using a Newton-Raphson solver, we actually expect to obtain the
    # solution in one step. It takes two because the way we structure the while
    # loop means that the procedure can only terminate on the second iteration.
    self.assertTrue(num_iter_ < 3)

  def testBatchedOperationConverges(self):
    model_1 = self.make_dataset(n=10, d=3, link='linear')
    model_2 = self.make_dataset(n=10, d=3, link='probit')
    model_matrices = [model_1[0], model_2[0]]
    responses = [model_1[1], model_2[1]]

    _, _, is_converged, _ = self.evaluate(
        tfp.glm.fit(
            model_matrices,
            responses,
            tfp.glm.Normal(),
            fast_unsafe_numerics=self.fast,
            maximum_iterations=10))
    self.assertTrue(is_converged)

  def testOffsetWorksCorrectly(self):
    n = int(1e5)
    offset = tf.fill([n], 1.0)
    [
        model_matrix,
        response,
        model_coefficients_true,
        linear_response_true,
    ] = self.make_dataset(
        n=n, d=3, link='probit', offset=offset)
    model_coefficients, linear_response, is_converged, _ = tfp.glm.fit(
        model_matrix,
        response,
        tfp.glm.BernoulliNormalCDF(),
        offset=offset,
        fast_unsafe_numerics=self.fast,
        maximum_iterations=20)
    [
        model_coefficients_,
        linear_response_,
        is_converged_,
        model_coefficients_true_,
        linear_response_true_,
    ] = self.evaluate([
        model_coefficients,
        linear_response,
        is_converged,
        model_coefficients_true,
        linear_response_true,
    ])

    self.assertTrue(is_converged_)
    avg_response_diff = np.mean(linear_response_ - linear_response_true_)
    self.assertNear(0., avg_response_diff, err=3e-3)
    self.assertAllClose(
        model_coefficients_true_, model_coefficients_, atol=0.03, rtol=0.15)

  def testRegularizationWithPenaltyFactor(self):
    n = int(1e4)
    [
        model_matrix,
        response,
        _,  # model_coefficients_true
        _,  # linear_response_true
    ] = self.make_dataset(
        n=n, d=2, link='linear')
    # Really strong regularization should bring all regularized coefficients to
    # approximately 0.
    l2_regularizer = np.array(1e9 * n, model_matrix.dtype.as_numpy_dtype)

    model_coefficients_uniform_penalty, _, is_converged_uniform_penalty, _ = (
        tfp.glm.fit(
            model_matrix,
            response,
            tfp.glm.Normal(),
            l2_regularizer=l2_regularizer,
            fast_unsafe_numerics=self.fast,
            maximum_iterations=10))

    model_coefficients_penalty_factor, _, is_converged_penalty_factor, _ = (
        tfp.glm.fit(
            model_matrix,
            response,
            tfp.glm.Normal(),
            l2_regularizer=l2_regularizer,
            # only penalize (apply regularization to) second coefficient
            l2_regularization_penalty_factor=[0.0, 1.0],
            fast_unsafe_numerics=self.fast,
            maximum_iterations=10))

    [
        model_coefficients_uniform_penalty_, is_converged_uniform_penalty_,
        model_coefficients_penalty_factor_, is_converged_penalty_factor_
    ] = self.evaluate([
        model_coefficients_uniform_penalty, is_converged_uniform_penalty,
        model_coefficients_penalty_factor, is_converged_penalty_factor
    ])

    self.assertTrue(is_converged_uniform_penalty_)
    self.assertTrue(is_converged_penalty_factor_)
    # When regularization is applied to all coefficients, they should all be
    # close to 0.
    self.assertAllClose([0., 0.],
                        model_coefficients_uniform_penalty_,
                        rtol=1e-6,
                        atol=1e-6)
    # When regularization is applied only to second coefficient, only it should
    # be close to 0.
    self.assertGreater(np.abs(model_coefficients_penalty_factor_[0]), 1e-6)
    self.assertNear(0., model_coefficients_penalty_factor_[1], err=1e-6)


class FitTestSlow(FitTestFast):

  fast = False

  # Only need to run this test once since it compares fast to slow.
  # We use `fast` as a baseline since core TF implements the L2 regularization
  # in this case.
  def _testL2RegularizationWorksCorrectly(self, static_l2):
    n = int(1e3)
    [
        model_matrix,
        response,
        _,  # model_coefficients_true
        _,  # linear_response_true
    ] = self.make_dataset(n=n, d=3, link='probit')
    l2_regularizer = np.array(0.07 * n, model_matrix.dtype.as_numpy_dtype)
    if not static_l2:
      l2_regularizer = tf1.placeholder_with_default(
          l2_regularizer, shape=[])
    [
        expected_model_coefficients,
        expected_linear_response,
        expected_is_converged,
        expected_num_iter,
    ] = tfp.glm.fit(
        model_matrix,
        response,
        tfp.glm.BernoulliNormalCDF(),
        l2_regularizer=l2_regularizer,
        fast_unsafe_numerics=True,
        maximum_iterations=10)
    [
        actual_model_coefficients,
        actual_linear_response,
        actual_is_converged,
        actual_num_iter,
    ] = tfp.glm.fit(
        model_matrix,
        response,
        tfp.glm.BernoulliNormalCDF(),
        l2_regularizer=l2_regularizer,
        fast_unsafe_numerics=False,
        maximum_iterations=10)

    [
        expected_model_coefficients_,
        expected_linear_response_,
        expected_is_converged_,
        expected_num_iter_,
        actual_model_coefficients_,
        actual_linear_response_,
        actual_is_converged_,
        actual_num_iter_,
    ] = self.evaluate([
        expected_model_coefficients,
        expected_linear_response,
        expected_is_converged,
        expected_num_iter,
        actual_model_coefficients,
        actual_linear_response,
        actual_is_converged,
        actual_num_iter,
    ])

    self.assertAllClose(
        expected_model_coefficients_, actual_model_coefficients_,
        atol=1e-6, rtol=1e-6)
    self.assertAllClose(
        expected_linear_response_, actual_linear_response_,
        atol=1e-5, rtol=1e-5)
    self.assertEqual(expected_is_converged_, actual_is_converged_)
    self.assertEqual(expected_num_iter_, actual_num_iter_)

  def testStaticL2RegularizationWorksCorrectly(self):
    self._testL2RegularizationWorksCorrectly(static_l2=True)

# TODO(jvdillon): Re-enable once matrix_solve_ls correctly casts
# l2_regularization.
# def testDynamicL2RegularizationWorksCorrectly(self):
#   self._testL2RegularizationWorksCorrectly(static_l2=False)


# TODO(b/79377499): Add additional unit-tests, esp, those to cover cases when
# grad_mean=variance=0 or either isn't finite.


if __name__ == '__main__':
  tf.test.main()
