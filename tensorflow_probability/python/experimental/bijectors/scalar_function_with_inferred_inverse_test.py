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

from absl.testing import parameterized

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.experimental.bijectors import scalar_function_with_inferred_inverse
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

from tensorflow.python import tf2  # pylint: disable=g-direct-tensorflow-import

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

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'ScalarPower',
       'params': (2.5,),
       'bijector_fn': tfb.Power,
       'forward_fn': lambda x, p: x ** p,
       'domain_constraint_fn': tf.nn.softplus,
       'y': [2., 4., 6.]},
      {'testcase_name': 'BatchPower',
       'params': ([2.5, 3., 4.],),
       'bijector_fn': tfb.Power,
       'forward_fn': lambda x, p: x ** p,
       'domain_constraint_fn': tf.nn.softplus,
       'y': [2., 4., 6.]},
      {'testcase_name': 'PowerShiftChainTwoParameters',
       'params': (-3., [2., -2.]),
       'bijector_fn': lambda a, b: tfb.Chain([tfb.Power(b), tfb.Shift(a)]),
       'forward_fn': lambda x, a, b: (x + a) ** b,
       'domain_constraint_fn': lambda ux: tf.nn.softplus(ux) + 3.,
       'y': [2., 3.]},
      {'testcase_name': 'Scale',
       'params': (0.5,),
       'bijector_fn': tfb.Scale,
       'forward_fn': lambda x, c: c * x,  # Second derivative wrt c is `None`.
       'domain_constraint_fn': None,
       'y': 17.})
  def test_gradient_with_additional_parameters(self,
                                               params,
                                               bijector_fn,
                                               forward_fn,
                                               domain_constraint_fn, y):
    y = tf.convert_to_tensor(y)
    params = [tf.convert_to_tensor(x) for x in params]

    # Define a drop-in replacement for 'bijector_fn'.
    implicit_bijector_fn = (
        lambda *params: tfbe.ScalarFunctionWithInferredInverse(  # pylint: disable=g-long-lambda
            forward_fn,
            domain_constraint_fn=domain_constraint_fn,
            additional_scalar_parameters_requiring_gradients=params))

    # Check the inverse and its first-order derivatives.
    x, [
        dx_dy,
        dx_dparams
    ] = tfp.math.value_and_gradient(
        lambda y, params: bijector_fn(*params).inverse(y),
        y, params)
    implicit_x, [
        implicit_dx_dy,
        implicit_dx_dparams
    ] = tfp.math.value_and_gradient(
        lambda y, params: implicit_bijector_fn(*params).inverse(y),
        y, params)
    self.assertAllClose(x, implicit_x)
    self.assertAllClose(dx_dy, implicit_dx_dy)
    self.assertAllCloseNested(dx_dparams, implicit_dx_dparams)

    # Check second-order derivatives, by way of ildj and its gradient.
    y = tf.identity(y)  # Bypass the global bijector cache.
    ildj, [
        dildj_dy,
        dildj_dparams
    ] = tfp.math.value_and_gradient(
        lambda y, params: bijector_fn(*params).inverse_log_det_jacobian(y),
        y, params)
    implicit_ildj, [
        implicit_dildj_dy,
        implicit_dildj_dparams
    ] = tfp.math.value_and_gradient(
        lambda y, params: implicit_bijector_fn(  # pylint: disable=g-long-lambda
            *params).inverse_log_det_jacobian(y),
        y, params)
    self.assertAllClose(ildj, implicit_ildj)
    self.assertAllClose(
        0. if dildj_dy is None else dildj_dy,
        implicit_dildj_dy)
    self.assertAllCloseNested(
        [0 if g is None else g for g in dildj_dparams],
        implicit_dildj_dparams)

    # Directly test second-order derivatives wrt params, since ildj doesn't use
    # them.
    y = tf.identity(y)  # Bypass the global bijector cache.
    _, second_derivatives_wrt_params = tfp.math.value_and_gradient(
        lambda params: tfp.math.value_and_gradient(  # pylint: disable=g-long-lambda
            lambda params: bijector_fn(*params).inverse(y),
            params, auto_unpack_single_arg=False)[1],
        params, auto_unpack_single_arg=False)
    _, implicit_second_derivatives_wrt_params = tfp.math.value_and_gradient(
        lambda params: tfp.math.value_and_gradient(  # pylint: disable=g-long-lambda
            lambda params: implicit_bijector_fn(*params).inverse(y),
            params, auto_unpack_single_arg=False)[1],
        params, auto_unpack_single_arg=False)
    self.assertAllCloseNested(
        [0. if g is None else g for g in second_derivatives_wrt_params],
        implicit_second_derivatives_wrt_params)


class TestMakeGradientFunctionOfY(test_util.TestCase):
  """Direct tests for `_make_gradient_fn_of_y`."""

  @test_util.numpy_disable_gradient_test
  def test_grad_fn_of_y_grad(self):
    y = tf.convert_to_tensor([2., 3., 4.])
    x = tf.sqrt(y)
    dy_dx_fn = scalar_function_with_inferred_inverse._make_gradient_fn_of_y(
        lambda x: x ** 2, x)
    true_dy_dx_fn = lambda y: [2 * tf.math.pow(y, 1./2)]

    dy_dx, grad_dy_dx_wrt_y = tfp.math.value_and_gradient(dy_dx_fn, y)
    true_dy_dx, true_grad_dy_dx_wrt_y = tfp.math.value_and_gradient(
        true_dy_dx_fn, y)
    self.assertAllClose(dy_dx, true_dy_dx)
    self.assertAllClose(grad_dy_dx_wrt_y, true_grad_dy_dx_wrt_y)

  @test_util.numpy_disable_gradient_test
  def test_grad_fn_of_y_grad_with_argument(self):
    if not tf2.enabled():
      self.skipTest('TF1 appears to find incorrect gradients. This is likely '
                    'not worth debugging.')
    power = tf.convert_to_tensor([2., 3., 4.])
    y = tf.convert_to_tensor([2., 4., 6.])
    x = tf.math.pow(y, 1. / power)
    grad_fn_of_y = scalar_function_with_inferred_inverse._make_gradient_fn_of_y(
        lambda x, p: x ** p, x)
    true_dy_dx_fn = lambda y, p: p * tf.math.pow(y, (p - 1) / p)
    true_dy_dpower_fn = lambda y, p: y * tf.math.log(y) / p

    (dy_dx,
     (grad_dy_dx_wrt_y,
      grad_dy_dx_wrt_power)) = tfp.math.value_and_gradient(
          lambda y, p: grad_fn_of_y(y, p)[0], y, power)
    (true_dy_dx,
     (true_grad_dy_dx_wrt_y,
      true_grad_dy_dx_wrt_power)) = tfp.math.value_and_gradient(
          true_dy_dx_fn, y, power)
    self.assertAllClose(dy_dx, true_dy_dx)
    self.assertAllClose(grad_dy_dx_wrt_y, true_grad_dy_dx_wrt_y)
    self.assertAllClose(grad_dy_dx_wrt_power, true_grad_dy_dx_wrt_power)

    (dy_dpower,
     (grad_dy_dpower_wrt_y,
      grad_dy_dpower_wrt_power)) = tfp.math.value_and_gradient(
          lambda y, p: grad_fn_of_y(y, p)[1], y, power)
    (true_dy_dpower,
     (true_grad_dy_dpower_wrt_y,
      true_grad_dy_dpower_wrt_power)) = tfp.math.value_and_gradient(
          true_dy_dpower_fn, y, power)
    self.assertAllClose(dy_dpower, true_dy_dpower)
    self.assertAllClose(grad_dy_dpower_wrt_y, true_grad_dy_dpower_wrt_y)
    self.assertAllClose(grad_dy_dpower_wrt_power, true_grad_dy_dpower_wrt_power)

if __name__ == '__main__':
  tf.test.main()
