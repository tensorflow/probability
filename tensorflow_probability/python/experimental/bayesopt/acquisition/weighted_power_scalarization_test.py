# Copyright 2023 The TensorFlow Probability Authors.
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
"""Tests for weighted_power_scalarization."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import gaussian_process
from tensorflow_probability.python.experimental.bayesopt.acquisition import expected_improvement
from tensorflow_probability.python.experimental.bayesopt.acquisition import upper_confidence_bound
from tensorflow_probability.python.experimental.bayesopt.acquisition import weighted_power_scalarization as wps
from tensorflow_probability.python.experimental.distributions import multitask_gaussian_process as mtgp
from tensorflow_probability.python.experimental.psd_kernels import multitask_kernel
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic


@test_util.test_all_tf_execution_regimes
class _WeightedPowerScalarizationTest(object):

  @test_util.numpy_disable_gradient_test
  def test_weighted_power_scalarization(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude=self.dtype(1.)
    )
    mt_kernel = multitask_kernel.Independent(
        base_kernel=kernel,
        num_tasks=4)
    shape = [5, 1, 20]
    index_points = 2. * np.random.uniform(size=shape).astype(dtype=self.dtype)
    observations = 2. * np.random.uniform(size=[10, 4]).astype(dtype=self.dtype)
    dist = mtgp.MultiTaskGaussianProcess(
        kernel=mt_kernel,
        index_points=index_points,
        observation_noise_variance=1e-4)

    weights = np.array([0.8, 1., 1.1, 0.5]).astype(self.dtype)

    acquisition_function_classes = [
        expected_improvement.GaussianProcessExpectedImprovement,
        upper_confidence_bound.GaussianProcessUpperConfidenceBound,
        expected_improvement.GaussianProcessExpectedImprovement,
        upper_confidence_bound.GaussianProcessUpperConfidenceBound]

    acquisition_kwargs_list = 4 * [None]

    cheb_scalar_fn = wps.WeightedPowerScalarization(
        predictive_distribution=dist,
        acquisition_function_classes=acquisition_function_classes,
        acquisition_kwargs_list=acquisition_kwargs_list,
        observations=observations,
        weights=weights,
        seed=test_util.test_seed())
    actual_cheb_scalar, grad = self.evaluate(
        gradient.value_and_gradient(
            lambda x: cheb_scalar_fn(index_points=x),
            tf.convert_to_tensor(index_points)
        )
    )

    gp_per_task = gaussian_process.GaussianProcess(
        kernel=kernel,
        index_points=index_points,
        observation_noise_variance=1e-4)

    expected_cheb_scalars = []
    for i in range(4):
      acquisition_kwargs = acquisition_kwargs_list[i]
      if acquisition_kwargs is None:
        acquisition_kwargs = {}
      expected_cheb_scalars.append(
          weights[i] * tf.math.abs(
              acquisition_function_classes[i](
                  predictive_distribution=gp_per_task,
                  observations=observations[..., i],
                  seed=test_util.test_seed(),
                  **acquisition_kwargs)()))
    expected_cheb_scalar = tf.reduce_max(
        tf.concat(expected_cheb_scalars, axis=-1), axis=-1)
    self.assertAllNotNan(grad)
    self.assertAllClose(actual_cheb_scalar, expected_cheb_scalar, rtol=6e-4)
    self.assertDTypeEqual(actual_cheb_scalar, self.dtype)


@test_util.test_all_tf_execution_regimes
class WeightedPowerScalarizationFloat32Test(_WeightedPowerScalarizationTest,
                                            test_util.TestCase):

  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class WeightedPowerScalarizationFloat64Test(_WeightedPowerScalarizationTest,
                                            test_util.TestCase):

  dtype = np.float64


del _WeightedPowerScalarizationTest


if __name__ == '__main__':
  test_util.main()
