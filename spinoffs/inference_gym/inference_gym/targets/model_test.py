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
"""Tests for inference_gym.targets.model."""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym import targets
from inference_gym.internal import test_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

tfb = tfp.bijectors
tfd = tfp.distributions


class TestModel(targets.Model):

  def __init__(self):
    super(TestModel, self).__init__(
        default_event_space_bijector=tfb.Exp(),
        event_shape=[],
        dtype=tf.float32,
        name='test_model',
        pretty_name='TestModel',
        sample_transformations=dict(
            identity=targets.Model.SampleTransformation(
                fn=lambda x: x,
                pretty_name='Identity',
                ground_truth_mean=1.,  # It's wrong, but we'll use it for
                                       # testing.
            ),),
    )

  def _unnormalized_log_prob(self, value):
    return 1. + value - tf.math.exp(value)


@test_util.multi_backend_test(globals(), 'targets.model_test')
@test_util.test_all_tf_execution_regimes
class ModelTest(test_util.InferenceGymTestCase):

  def testExampleDoc1(self):
    seed = tfp_test_util.test_seed(sampler_type='stateless')
    model = TestModel()

    num_seeds = len(tf.nest.flatten(model.dtype))
    flat_seed = tf.unstack(tfp.random.split_seed(seed, num_seeds), axis=0)
    seed = tf.nest.pack_sequence_as(model.dtype, flat_seed)

    unconstrained_values = tf.nest.map_structure(
        lambda d, s, seed: tf.random.stateless_normal(s, dtype=d, seed=seed),
        model.dtype,
        model.event_shape,
        seed,
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
  tfp_test_util.main()
