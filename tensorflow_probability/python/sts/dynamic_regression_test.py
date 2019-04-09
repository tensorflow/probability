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
"""Dynamic Linear Regression State Space Model Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.sts import DynamicLinearRegressionStateSpaceModel

from tensorflow.python.framework import test_util
from tensorflow.python.ops.linalg import linear_operator_util

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class _DynamicLinearRegressionStateSpaceModelTest(object):

  def test_basic_statistics_no_latent_variance(self):
    batch_shape = [1]
    num_timesteps = 7
    num_features = 3
    weights_scale = 0.

    design_matrix = self._build_placeholder(
        np.random.randn(*(batch_shape + [num_timesteps, num_features])))

    initial_state_mean = self._build_placeholder(
        np.random.randn(*(batch_shape + [num_features])))
    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=initial_state_mean, scale_diag=tf.zeros_like(initial_state_mean))

    ssm = DynamicLinearRegressionStateSpaceModel(
        num_timesteps=num_timesteps,
        design_matrix=design_matrix,
        weights_scale=weights_scale,
        initial_state_prior=initial_state_prior,
    )

    predicted_time_series = linear_operator_util.matmul_with_broadcast(
        design_matrix, initial_state_mean[..., tf.newaxis])

    self.assertAllEqual(self.evaluate(ssm.mean()), predicted_time_series)
    self.assertAllEqual(*self.evaluate((ssm.stddev(),
                                        tf.zeros_like(predicted_time_series))))

  def _build_placeholder(self, ndarray):
    ndarray = np.asarray(ndarray).astype(self.dtype)
    return tf.compat.v1.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.run_all_in_graph_and_eager_modes
class DynamicRegressionStateSpaceModelTestStaticShape32(
    tf.test.TestCase, _DynamicLinearRegressionStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class DynamicRegressionStateSpaceModelTestDynamicShape32(
    tf.test.TestCase, _DynamicLinearRegressionStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class DynamicRegressionStateSpaceModelTestStaticShape64(
    tf.test.TestCase, _DynamicLinearRegressionStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == "__main__":
  tf.test.main()
