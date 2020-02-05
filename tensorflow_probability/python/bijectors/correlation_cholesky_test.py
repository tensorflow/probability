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
"""Tests for CorrelationCholesky bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.distributions import beta
from tensorflow_probability.python.distributions import cholesky_lkj
from tensorflow_probability.python.distributions import lkj
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import sample
from tensorflow_probability.python.mcmc import transformed_kernel


# Bijector for converting the image of CorrelationCholesky bijector to
# unconstrained space; by only considering the strictly lower triangular entries
# of the output matrices.
class OutputToUnconstrained(tfb.Bijector):

  def __init__(self, name="output_to_unconstrained"):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(OutputToUnconstrained, self).__init__(
          validate_args=True,
          forward_min_event_ndims=2,
          inverse_min_event_ndims=1,
          parameters=parameters,
          name=name)

  def _forward(self, x):
    x = tf.convert_to_tensor(x)
    # Remove the first row and last column so we can extract strictly
    # lower triangular entries.
    n = x.shape[-1]
    t = tf.linalg.band_part(x[..., 1:, :-1], num_lower=n - 1, num_upper=0)
    return tfb.FillTriangular().inverse(t)

  def _inverse_log_det_jacobian(self, y):
    return tf.zeros_like(y[..., 0])


@test_util.test_all_tf_execution_regimes
class CorrelationCholeskyBijectorTest(test_util.TestCase):
  """Tests the correctness of the CorrelationCholesky bijector."""

  def testBijector(self):
    x = np.float32(np.array([7., -5., 5., 1., 2., -2.]))
    y = np.float32(
        np.array([[1., 0., 0., 0.], [0.707107, 0.707107, 0., 0.],
                  [-0.666667, 0.666667, 0.333333, 0.], [0.5, -0.5, 0.7, 0.1]]))

    b = tfb.CorrelationCholesky()

    y_ = self.evaluate(b.forward(x))
    self.assertAllClose(y, y_, atol=1e-5, rtol=1e-5)

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_, atol=1e-5, rtol=1e-5)

    expected_fldj = -0.5 * np.sum([3, 4, 5] * np.log([2, 9, 100]))

    fldj = self.evaluate(b.forward_log_det_jacobian(x, event_ndims=1))
    self.assertAllClose(expected_fldj, fldj)

    ildj = self.evaluate(b.inverse_log_det_jacobian(y, event_ndims=2))
    self.assertAllClose(-expected_fldj, ildj)

  def testBijectorBatch(self):
    x = np.float32([[7., -5., 5., 1., 2., -2.], [1., 3., -5., 1., -4., 8.]])
    y = np.float32([
        [[1., 0., 0., 0.], [0.707107, 0.707107, 0., 0.],
         [-0.666667, 0.666667, 0.333333, 0.], [0.5, -0.5, 0.7, 0.1]],
        [[1., 0., 0., 0.], [0.707107, 0.707107, 0., 0.],
         [0.888889, -0.444444, 0.111111, 0.],
         [-0.833333, 0.5, 0.166667, 0.166667]],
    ])

    b = tfb.CorrelationCholesky()

    y_ = self.evaluate(b.forward(x))
    self.assertAllClose(y, y_, atol=1e-5, rtol=1e-5)

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_, atol=1e-5, rtol=1e-5)

    expected_fldj = -0.5 * np.sum(
        [3, 4, 5] * np.log([[2, 9, 100], [2, 81, 36]]), axis=-1)

    fldj = self.evaluate(b.forward_log_det_jacobian(x, event_ndims=1))
    self.assertAllClose(expected_fldj, fldj)

    ildj = self.evaluate(b.inverse_log_det_jacobian(y, event_ndims=2))
    self.assertAllClose(-expected_fldj, ildj)

  def testShape(self):
    x_shape = tf.TensorShape([5, 4, 6])
    y_shape = tf.TensorShape([5, 4, 4, 4])

    b = tfb.CorrelationCholesky(validate_args=True)

    x = tf.ones(shape=x_shape, dtype=tf.float32)
    y_ = b.forward(x)
    self.assertAllEqual(
        tensorshape_util.as_list(y_.shape), tensorshape_util.as_list(y_shape))
    x_ = b.inverse(y_)
    self.assertAllEqual(
        tensorshape_util.as_list(x_.shape), tensorshape_util.as_list(x_shape))

    y_shape_ = b.forward_event_shape(x_shape)
    self.assertAllEqual(
        tensorshape_util.as_list(y_shape_), tensorshape_util.as_list(y_shape))
    x_shape_ = b.inverse_event_shape(y_shape)
    self.assertAllEqual(
        tensorshape_util.as_list(x_shape_), tensorshape_util.as_list(x_shape))

    y_shape_tensor = self.evaluate(
        b.forward_event_shape_tensor(tensorshape_util.as_list(x_shape)))
    self.assertAllEqual(y_shape_tensor, tensorshape_util.as_list(y_shape))
    x_shape_tensor = self.evaluate(
        b.inverse_event_shape_tensor(tensorshape_util.as_list(y_shape)))
    self.assertAllEqual(x_shape_tensor, tensorshape_util.as_list(x_shape))

  def testShapeError(self):

    b = tfb.FillTriangular(validate_args=True)

    x_shape_bad = tf.TensorShape([5, 4, 7])
    with self.assertRaisesRegexp(ValueError, "is not a triangular number"):
      b.forward_event_shape(x_shape_bad)
    with self.assertRaisesOpError("is not a triangular number"):
      self.evaluate(
          b.forward_event_shape_tensor(tensorshape_util.as_list(x_shape_bad)))

    y_shape_bad = tf.TensorShape([5, 4, 4, 3])
    with self.assertRaisesRegexp(ValueError, "Matrix must be square"):
      b.inverse_event_shape(y_shape_bad)
    with self.assertRaisesOpError("Matrix must be square"):
      self.evaluate(
          b.inverse_event_shape_tensor(tensorshape_util.as_list(y_shape_bad)))

  @test_util.test_graph_mode_only
  def testSampleMarginals(self):
    # Verify that the marginals of the LKJ distribution are distributed
    # according to a (scaled) Beta distribution. The LKJ distributed samples are
    # obtained by sampling a CholeskyLKJ distribution using HMC and the
    # CorrelationCholesky bijector.
    dim = 4
    concentration = np.array(2.5, dtype=np.float64)
    beta_concentration = np.array(.5 * dim + concentration - 1, np.float64)
    beta_dist = beta.Beta(
        concentration0=beta_concentration, concentration1=beta_concentration)

    inner_kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=cholesky_lkj.CholeskyLKJ(
            dimension=dim, concentration=concentration).log_prob,
        num_leapfrog_steps=3,
        step_size=0.3,
        seed=test_util.test_seed())

    kernel = transformed_kernel.TransformedTransitionKernel(
        inner_kernel=inner_kernel, bijector=tfb.CorrelationCholesky())

    num_chains = 10
    num_total_samples = 30000

    # Make sure that we have enough samples to catch a wrong sampler to within
    # a small enough discrepancy.
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_total_samples)

    @tf.function  # Ensure that MCMC sampling is done efficiently.
    def sample_mcmc_chain():
      return sample.sample_chain(
          num_results=num_total_samples // num_chains,
          num_burnin_steps=1000,
          current_state=tf.eye(dim, batch_shape=[num_chains], dtype=tf.float64),
          trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
          kernel=kernel,
          parallel_iterations=1)

    # Draw samples from the HMC chains.
    chol_lkj_samples, is_accepted = self.evaluate(sample_mcmc_chain())

    # Ensure that the per-chain acceptance rate is high enough.
    self.assertAllGreater(np.mean(is_accepted, axis=0), 0.8)

    # Transform from Cholesky LKJ samples to LKJ samples.
    lkj_samples = tf.matmul(chol_lkj_samples, chol_lkj_samples, adjoint_b=True)
    lkj_samples = tf.reshape(lkj_samples, shape=[num_total_samples, dim, dim])

    # Only look at the entries strictly below the diagonal which is achieved by
    # the OutputToUnconstrained bijector. Also scale the marginals from the
    # range [-1,1] to [0,1].
    scaled_lkj_samples = .5 * (OutputToUnconstrained().forward(lkj_samples) + 1)

    # Each of the off-diagonal marginals should be distributed according to a
    # Beta distribution.
    for i in range(dim * (dim - 1) // 2):
      self.evaluate(
          st.assert_true_cdf_equal_by_dkwm(
              scaled_lkj_samples[..., i],
              cdf=beta_dist.cdf,
              false_fail_rate=1e-9))

  def testTheoreticalFldj(self):
    bijector = tfb.CorrelationCholesky()
    x = np.linspace(-50, 50, num=30).reshape(5, 6).astype(np.float64)
    y = self.evaluate(bijector.forward(x))
    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x,
        y,
        eval_func=self.evaluate,
        event_ndims=1,
        inverse_event_ndims=2,
        rtol=1e-5)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector,
        x,
        event_ndims=1,
        inverse_event_ndims=2,
        output_to_unconstrained=OutputToUnconstrained())
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)

  def testBijectorWithVariables(self):
    x_ = np.array([1.], dtype=np.float32)
    y_ = np.array([[1., 0.], [0.707107, 0.707107]], dtype=np.float32)

    x = tf.Variable(x_, dtype=tf.float32)
    y = tf.Variable(y_, dtype=tf.float32)
    forward_event_ndims = tf.Variable(1, dtype=tf.int32)
    inverse_event_ndims = tf.Variable(2, dtype=tf.int32)
    self.evaluate([
        v.initializer for v in (x, y, forward_event_ndims, inverse_event_ndims)
    ])

    bijector = tfb.CorrelationCholesky()
    self.assertAllClose(
        y_, self.evaluate(bijector.forward(x)), atol=1e-5, rtol=1e-5)
    self.assertAllClose(
        x_, self.evaluate(bijector.inverse(y)), atol=1e-5, rtol=1e-5)

    fldj = bijector.forward_log_det_jacobian(x, event_ndims=forward_event_ndims)
    self.assertAllClose(-3 * 0.5 * np.log(2), self.evaluate(fldj))

    ildj = bijector.inverse_log_det_jacobian(y, event_ndims=inverse_event_ndims)
    self.assertAllClose(3 * 0.5 * np.log(2), ildj)

  @parameterized.parameters(itertools.product([2, 3, 4, 5, 6, 7], [1., 2., 3.]))
  def testBijectiveWithLKJSamples(self, dimension, concentration):
    bijector = tfb.CorrelationCholesky()
    lkj_dist = lkj.LKJ(
        dimension=dimension,
        concentration=np.float64(concentration),
        input_output_cholesky=True)
    batch_size = 10
    y = self.evaluate(
        lkj_dist.sample([batch_size], seed=test_util.test_seed()))
    x = self.evaluate(bijector.inverse(y))

    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x,
        y,
        eval_func=self.evaluate,
        event_ndims=1,
        inverse_event_ndims=2,
        rtol=1e-5)

  @parameterized.parameters(itertools.product([2, 3, 4, 5, 6, 7], [1., 2., 3.]))
  @test_util.numpy_disable_gradient_test
  def testJacobianWithLKJSamples(self, dimension, concentration):
    bijector = tfb.CorrelationCholesky()
    lkj_dist = lkj.LKJ(
        dimension=dimension,
        concentration=np.float64(concentration),
        input_output_cholesky=True)
    batch_size = 10
    y = self.evaluate(lkj_dist.sample([batch_size], seed=test_util.test_seed()))
    x = self.evaluate(bijector.inverse(y))

    fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector,
        x,
        event_ndims=1,
        inverse_event_ndims=2,
        output_to_unconstrained=OutputToUnconstrained())
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)


if __name__ == "__main__":
  tf.test.main()
