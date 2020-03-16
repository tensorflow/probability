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
"""Test utilities for the Inference Gym."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


class InferenceGymTestCase(test_util.TestCase):
  """A TestCase mixin for common tests on inference gym targets."""

  def validate_log_prob_and_transforms(
      self,
      model,
      sample_transformation_shapes,
      seed=None,
  ):
    """Validate that the model's log probability and sample transformations run.

    This checks that unconstrained values passed through the event space
    bijectors into `unnormalized_log_prob` and sample transformations yield
    finite values. This also verifies that the transformed values have the
    expected shape.

    Args:
      model: The model to validate.
      sample_transformation_shapes: Shapes of the transformation outputs.
      seed: Optional seed to use. By default, `test_util.test_seed()` is used.
    """
    batch_size = 16

    if seed is not None:
      seed = tfp.util.SeedStream(seed, 'validate_log_prob_and_transforms')
    else:
      seed = test_util.test_seed_stream()

    def _random_element(shape, dtype, event_space_bijector):
      unconstrained_shape = event_space_bijector.inverse_event_shape(shape)
      unconstrained_shape = tf.TensorShape([batch_size
                                           ]).concatenate(unconstrained_shape)
      return event_space_bijector.forward(
          tf.random.normal(unconstrained_shape, dtype=dtype, seed=seed()))

    test_points = tf.nest.map_structure(_random_element, model.event_shape,
                                        model.dtype,
                                        model.default_event_space_bijector)
    log_prob = self.evaluate(model.unnormalized_log_prob(test_points))

    self.assertAllFinite(log_prob)
    self.assertEqual((batch_size,), log_prob.shape)

    for name, sample_transformation in model.sample_transformations.items():
      transformed_points = self.evaluate(sample_transformation(test_points))

      def _assertions_part(expected_shape, transformed_part):
        self.assertAllFinite(transformed_part)
        self.assertEqual(
            (batch_size,) + tuple(list(tf.TensorShape(expected_shape))),
            transformed_part.shape)

      self.assertAllAssertsNested(
          _assertions_part,
          sample_transformation_shapes[name],
          transformed_points,
          shallow=transformed_points,
          msg='Comparing sample transformation: {}'.format(name))
