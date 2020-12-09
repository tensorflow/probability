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
"""Tests for ScalarFunctionWithInferredInverse bijector."""


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions
tfbe = tfp.experimental.bijectors


@test_util.test_all_tf_execution_regimes
class ScalarFunctionWithInferredInverseTests(test_util.TestCase):

  @test_util.numpy_disable_gradient_test
  def test_student_t_cdf(self):
    dist = tfd.StudentT(df=7, loc=3., scale=2.)
    xs = self.evaluate(dist.sample([100], seed=test_util.test_seed()))

    bij = tfbe.ScalarFunctionWithInferredInverse(dist.cdf)
    ys = bij.forward(xs)
    xxs = bij.inverse(ys)
    self.assertAllClose(xs, xxs)

  @test_util.numpy_disable_gradient_test
  def test_normal_cdf_gradients(self):
    dist = tfd.Normal(loc=3., scale=2.)
    bij = tfbe.ScalarFunctionWithInferredInverse(dist.cdf)

    ys = self.evaluate(samplers.uniform([100], seed=test_util.test_seed()))
    xs_true, grad_true = tfp.math.value_and_gradient(dist.quantile, ys)
    xs_numeric, grad_numeric = tfp.math.value_and_gradient(bij.inverse, ys)
    self.assertAllClose(xs_true, xs_numeric, atol=1e-4)
    self.assertAllClose(grad_true, grad_numeric, rtol=1e-4)

  @test_util.numpy_disable_gradient_test
  def test_domain_constraint_fn(self):
    dist = tfd.Beta(concentration0=5., concentration1=3.)
    xs = self.evaluate(dist.sample([100], seed=test_util.test_seed()))

    bij = tfbe.ScalarFunctionWithInferredInverse(
        dist.cdf,
        domain_constraint_fn=dist.experimental_default_event_space_bijector())
    self.assertAllClose(xs, bij.inverse(bij.forward(xs)))

  @test_util.numpy_disable_gradient_test
  def test_transformed_distribution_log_prob_and_grads(self):
    normal = tfd.Normal(loc=0., scale=1.)
    xs = self.evaluate(normal.sample(100, seed=test_util.test_seed()))
    lp_true, lp_grad_true = tfp.math.value_and_gradient(normal.log_prob, xs)

    # Define a normal distribution using inverse-CDF sampling. Computing
    # log probs under this definition requires inverting the quantile function,
    # i.e., numerically approximating `normal.cdf`.
    uniform = tfd.Uniform(low=0, high=1.)
    inverse_transform_normal = tfbe.ScalarFunctionWithInferredInverse(
        fn=normal.quantile,
        domain_constraint_fn=uniform.experimental_default_event_space_bijector()
        )(uniform)
    lp, lp_grad = tfp.math.value_and_gradient(inverse_transform_normal.log_prob,
                                              xs)
    self.assertAllClose(lp_true, lp, atol=1e-4)
    self.assertAllClose(lp_grad_true, lp_grad, atol=1e-4)

  @test_util.numpy_disable_gradient_test
  def test_ildj_gradients(self):
    bij = tfbe.ScalarFunctionWithInferredInverse(lambda x: x**2)
    ys = tf.convert_to_tensor([0.25, 1., 4., 9.])
    ildj, ildj_grad = tfp.math.value_and_gradient(
        lambda y: bij.inverse_log_det_jacobian(y, event_ndims=0),
        ys)

    # Compare ildjs from inferred inverses to ildjs from the true inverse.
    def ildj_fn(y):
      _, inverse_grads = tfp.math.value_and_gradient(tf.sqrt, y)
      return tf.math.log(tf.abs(inverse_grads))
    ildj_true, ildj_grad_true = tfp.math.value_and_gradient(ildj_fn, ys)
    self.assertAllClose(ildj, ildj_true, atol=1e-4)
    self.assertAllClose(ildj_grad, ildj_grad_true, rtol=1e-4)

if __name__ == '__main__':
  tf.test.main()
