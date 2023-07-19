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
"""Tests for AcquisitionFunctions."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.bayesopt.acquisition import acquisition_function
from tensorflow_probability.python.internal import test_util


# Returns mean.
class TestFunction(acquisition_function.AcquisitionFunction):
  def __init__(self, predictive_distribution, observations, seed=None):
    super(TestFunction, self).__init__(
        predictive_distribution=predictive_distribution,
        observations=observations,
        seed=seed)

  def __call__(self, **kwargs):
    return self.predictive_distribution.mean(**kwargs)


@test_util.test_all_tf_execution_regimes
class AcquisitionFunctionTest(test_util.TestCase):

  def test_mean_fn(self):
    shape = [6, 2, 15]
    loc = 2. * np.ones(shape, dtype=np.float32)
    stddev = 3.
    observations = [2., 3., 6.]
    model = normal.Normal(loc, stddev, validate_args=True)
    actual_mean = TestFunction(
        predictive_distribution=model,
        observations=observations)()
    expected_mean = model.mean()
    self.assertAllEqual(actual_mean.shape, shape)
    self.assertAllClose(actual_mean, expected_mean)

  @parameterized.parameters(0, 1, 2)
  def test_mcmc_reducer(self, reduce_dims):
    shape = [6, 2, 15]
    loc = 2. * np.ones(shape, dtype=np.float32)
    stddev = 3.
    observations = [2., 3., 6.]
    model = normal.Normal(loc, stddev, validate_args=True)
    actual_mean = acquisition_function.MCMCReducer(
        predictive_distribution=model,
        observations=observations,
        reduce_dims=reduce_dims,
        acquisition_class=TestFunction)()
    expected_mean = tf.reduce_mean(
        model.mean(), axis=list(range(reduce_dims)))
    self.assertAllClose(actual_mean, expected_mean)


if __name__ == '__main__':
  test_util.main()
