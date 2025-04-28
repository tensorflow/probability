# Copyright 2025 The TensorFlow Probability Authors.
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
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import dirichlet


@test_util.multi_backend_test(globals(),
                              'targets.dirichlet_test')
class DirichletTest(test_util.InferenceGymTestCase):

  @parameterized.parameters(np.float32, np.float64)
  def testBasic(self, dtype):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.

    Args:
      dtype: Dtype to use for floating point computations.
    """
    model = dirichlet.Dirichlet(
        concentration_vector=np.ones(10, dtype=dtype),
        dtype=dtype
    )
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(identity=[10],),
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
        dtype=dtype,
    )

  def testMC(self):
    """Checks true samples from the model against the ground truth."""
    model = dirichlet.Dirichlet(
        concentration_vector=np.ones(10, dtype=np.float32),
        dtype=tf.float32,
    )
    self.validate_ground_truth_using_monte_carlo(
        model,
        num_samples=int(1e6),
    )

  @test_util.numpy_disable_gradient_test
  def testHMC(self):
    """Checks true samples from the model against the ground truth."""
    model = dirichlet.Dirichlet(
        concentration_vector=np.ones(10, dtype=np.float32),
        dtype=np.float32,
    )
    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=4000,
        num_leapfrog_steps=10,
        step_size=0.025,
    )


if __name__ == '__main__':
  tfp_test_util.main()
