# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

tfb = tfp.bijectors
tfd = tfp.distributions
gym = tfp.experimental.inference_gym


class TestModel(gym.targets.BayesianModel):

  def __init__(self):
    self._joint_distribution_val = tfd.JointDistributionSequential([
        tfd.Exponential(0.),
        lambda s: tfd.Normal(0., s),
    ])
    self._evidence_val = 1.

    super(TestModel, self).__init__(
        default_event_space_bijector=tfb.Exp(),
        event_shape=self._joint_distribution_val.event_shape[0],
        dtype=self._joint_distribution_val.dtype[0],
        name='test_model',
        pretty_name='TestModel',
        sample_transformations=dict(
            identity=gym.targets.BayesianModel.SampleTransformation(
                fn=lambda x: x,
                pretty_name='Identity',
                ground_truth_mean=1.,  # It's wrong, but we'll use it for
                                       # testing.
            ),),
    )

  def _joint_distribution(self):
    return self._joint_distribution_val

  def _evidence(self):
    return self._evidence_val

  def _unnormalized_log_prob(self, x):
    return self.joint_distribution().log_prob([x, self.evidence()])


@test_util.test_all_tf_execution_regimes
class BayesianModelTest(test_util.TestCase):

  def testExampleDoc1(self):
    seed = test_util.test_seed_stream()
    model = TestModel()
    unconstrained_values = tf.nest.map_structure(
        lambda d, s: tf.random.normal(s, dtype=d, seed=seed()),
        model.dtype,
        model.event_shape,
    )
    constrained_values = nest.map_structure_up_to(
        model.default_event_space_bijector,
        lambda b, v: b(v),
        model.default_event_space_bijector,
        unconstrained_values,
    )
    self.assertGreaterEqual(self.evaluate(constrained_values), 0.)

  def testExampleDoc2(self):
    samples = np.ones(10)
    model = TestModel()
    for _, sample_transformation in model.sample_transformations.items():
      transformed_samples = sample_transformation(samples)
      square_diff = tf.nest.map_structure(
          lambda gtm, sm: (gtm - tf.reduce_mean(sm, axis=0))**2,
          sample_transformation.ground_truth_mean,
          transformed_samples,
      )
      self.assertLess(self.evaluate(square_diff), 1e-3)

  def testNames(self):
    model = TestModel()
    self.assertEqual('test_model', model.name)
    self.assertEqual('TestModel', str(model))
    self.assertEqual('Identity', str(model.sample_transformations['identity']))


if __name__ == '__main__':
  tf.test.main()
