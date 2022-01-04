# Lint as: python3
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
"""Tests for inference_gym.targets.bayesian_model."""

import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym import targets
from inference_gym.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


class TestModel(targets.BayesianModel):

  def __init__(self):
    self._prior_val = tfd.Exponential(0.)
    self._evidence = 1.

    def log_likelihood_fn(value):
      return tfd.Normal(value, 1.).log_prob(self._evidence)

    self._log_likelihood_fn = log_likelihood_fn

    super(TestModel, self).__init__(
        default_event_space_bijector=tfb.Exp(),
        event_shape=self._prior_val.event_shape,
        dtype=self._prior_val.dtype,
        name='test_model',
        pretty_name='TestModel',
        sample_transformations=dict(
            identity=targets.Model.SampleTransformation(
                fn=lambda x: x,
                pretty_name='Identity',
            ),),
    )

  def _prior_distribution(self):
    return self._prior_val

  def _log_likelihood(self, value):
    return self._log_likelihood_fn(value)


@test_util.multi_backend_test(globals(), 'targets.bayesian_model_test')
@test_util.test_all_tf_execution_regimes
class BayesianModelTest(test_util.InferenceGymTestCase):

  def testUnnormalizedLogProb(self):
    x = 1.

    model = TestModel()
    unnormalized_log_prob = model.unnormalized_log_prob(x)
    manual_unnormalized_log_prob = (
        model.prior_distribution().log_prob(x) + model.log_likelihood(x))

    self.assertAllClose(
        self.evaluate(manual_unnormalized_log_prob),
        self.evaluate(unnormalized_log_prob))


if __name__ == '__main__':
  tfp_test_util.main()
