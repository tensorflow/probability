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
"""Tests for inference_gym.targets.vector_model."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from inference_gym import targets
from inference_gym.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


class TestStructuredModel(targets.Model):

  def __init__(self):
    self._model = tfd.JointDistributionSequential([
        tfd.Sample(tfd.Normal(0., 1.), 3),
        tfd.CholeskyLKJ(dimension=3, concentration=1.),
    ])

    super(TestStructuredModel, self).__init__(
        default_event_space_bijector=[
            tfb.Identity(), tfb.CorrelationCholesky()
        ],
        event_shape=self._model.event_shape,
        dtype=self._model.dtype,
        name='test_structured_model',
        pretty_name='TestStructuredModel',
        sample_transformations=dict(
            first_moment=targets.Model.SampleTransformation(
                fn=lambda x: x,
                pretty_name='First moment',
                # These are wrong, we'll only be checking them for
                # shapes/dtypes.
                ground_truth_mean=(tf.nest.map_structure(
                    tf.zeros, self._model.event_shape, self._model.dtype)),
                ground_truth_standard_deviation=tf.nest.map_structure(
                    tf.zeros, self._model.event_shape, self._model.dtype),
                ground_truth_mean_standard_error=tf.nest.map_structure(
                    tf.zeros, self._model.event_shape, self._model.dtype),
                ground_truth_standard_deviation_standard_error=(
                    tf.nest.map_structure(tf.zeros, self._model.event_shape,
                                          self._model.dtype)),
            ),
            second_moment=targets.Model.SampleTransformation(
                fn=lambda x: tf.nest.map_structure(tf.square, x),
                pretty_name='Second moment',
                # These are wrong, we'll only be checking them for
                # shapes/dtypes.
                ground_truth_mean=(tf.nest.map_structure(
                    tf.zeros, self._model.event_shape, self._model.dtype)),
                ground_truth_standard_deviation=tf.nest.map_structure(
                    tf.zeros, self._model.event_shape, self._model.dtype),
                ground_truth_mean_standard_error=tf.nest.map_structure(
                    tf.zeros, self._model.event_shape, self._model.dtype),
                ground_truth_standard_deviation_standard_error=(
                    tf.nest.map_structure(tf.zeros, self._model.event_shape,
                                          self._model.dtype)),
            ),
        ),
    )

  def _unnormalized_log_prob(self, value):
    return self._model.log_prob(value)


class TestUnstructuredModel(targets.Model):

  def __init__(self):
    self._model = tfd.CholeskyLKJ(dimension=3, concentration=1.)

    super(TestUnstructuredModel, self).__init__(
        default_event_space_bijector=tfb.CorrelationCholesky(),
        event_shape=self._model.event_shape,
        dtype=self._model.dtype,
        name='test_unstructured_model',
        pretty_name='TestUnstructuredModel',
        sample_transformations=dict(
            first_moment=targets.Model.SampleTransformation(
                fn=lambda x: x,
                pretty_name='First moment',
            ),
            second_moment=targets.Model.SampleTransformation(
                fn=lambda x: x**2,
                pretty_name='Second moment',
            ),
        ),
    )

  def _unnormalized_log_prob(self, value):
    return self._model.log_prob(value)


class TestBadModel(targets.Model):
  """This model mixes dtypes, making it un-flattenable."""

  def __init__(self):
    self._model = tfd.JointDistributionSequential([
        tfd.Sample(tfd.Normal(0., 1.), 3),
        tfd.Categorical([0., 0.]),
    ])

    super(TestBadModel, self).__init__(
        default_event_space_bijector=[tfb.Identity(),
                                      tfb.Identity()],
        event_shape=self._model.event_shape,
        dtype=self._model.dtype,
        name='test_bad_model',
        pretty_name='TestBadModel',
        sample_transformations={},
    )

  def _unnormalized_log_prob(self, value):
    return self._model.log_prob(value)


class TestBadSampleTransformationModel(targets.Model):
  """Has a sample transform that mixes dtypes, making it un-flattenable."""

  def __init__(self):
    self._model = tfd.CholeskyLKJ(dimension=3, concentration=1.)

    super(TestBadSampleTransformationModel, self).__init__(
        default_event_space_bijector=tfb.CorrelationCholesky(),
        event_shape=self._model.event_shape,
        dtype=self._model.dtype,
        name='test_bad_sample_transformation_model',
        pretty_name='TestBadSampleTransformationModel',
        sample_transformations=dict(
            bad=targets.Model.SampleTransformation(
                fn=lambda x: (x, tf.cast(x, tf.float64)),
                pretty_name='Bad',
                dtype=(tf.float32, tf.float64),
            ),
        ),
    )

  def _unnormalized_log_prob(self, value):
    return self._model.log_prob(value)


class TestScalarModel(targets.Model):

  def __init__(self):
    self._model = tfd.Normal(0., 1.)

    super(TestScalarModel, self).__init__(
        default_event_space_bijector=tfb.Identity(),
        event_shape=self._model.event_shape,
        dtype=self._model.dtype,
        name='test_scalar_model',
        pretty_name='TestScalarModel',
        sample_transformations=dict(
            first_moment=targets.Model.SampleTransformation(
                fn=lambda x: x,
                pretty_name='First moment',
            ),
            second_moment=targets.Model.SampleTransformation(
                fn=lambda x: x**2,
                pretty_name='Second moment',
            ),
        ),
    )

  def _unnormalized_log_prob(self, value):
    return self._model.log_prob(value)


@test_util.multi_backend_test(globals(), 'targets.vector_model_test')
@test_util.test_all_tf_execution_regimes
class VectorModelTest(test_util.InferenceGymTestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('Structured', TestStructuredModel, 3 + 9, []),
      ('Unstructured', TestUnstructuredModel, 9, []),
      ('Scalar', TestScalarModel, 1, []),
      ('BatchedStructured', TestStructuredModel, 3 + 9, [2]),
      ('BatchedUnstructured', TestUnstructuredModel, 9, [2]),
      ('BatchedScalar', TestScalarModel, 1, [2]),
  )
  def testBasic(self, model_class, vec_event_size, batch_shape):
    base_model = model_class()
    vec_model = targets.VectorModel(base_model)

    # We can randomize only one element, as otherwise we'd need to know the
    # details of the reshaping/flattening which is outside the scope of this
    # test.
    rand_elem = tf.constant(np.random.randn(), tf.float32)

    self.assertEqual('vector_' + base_model.name, vec_model.name)
    self.assertEqual(str(base_model), str(vec_model))
    self.assertEqual(tf.float32, vec_model.dtype)
    self.assertEqual([vec_event_size], list(vec_model.event_shape))

    # z - unconstrained space
    structured_z = tf.nest.map_structure(
        lambda s, d, b: tf.fill(  # pylint: disable=g-long-lambda
            batch_shape + list(b.inverse_event_shape(s)), tf.cast(rand_elem, d)
        ),
        base_model.event_shape,
        base_model.dtype,
        base_model.default_event_space_bijector)

    # x - constrained space
    structured_x = tf.nest.map_structure(
        lambda b, z: b(z), base_model.default_event_space_bijector,
        structured_z)

    vec_z = tf.fill(
        batch_shape + list(
            vec_model.default_event_space_bijector.inverse_event_shape(
                vec_model.event_shape)),
        tf.cast(rand_elem, vec_model.dtype),
    )
    vec_x = vec_model.default_event_space_bijector(vec_z)

    # Utility transforms.
    self.assertAllEqualNested(structured_x,
                              vec_model.vector_event_to_structured(vec_x))
    self.assertAllEqual(vec_x,
                        vec_model.structured_event_to_vector(structured_x))

    self.assertAllEqual(
        base_model.unnormalized_log_prob(structured_x),
        vec_model.unnormalized_log_prob(vec_x))
    self.assertAllEqualNested(
        base_model.sample_transformations['first_moment'](structured_x),
        vec_model.sample_transformations['first_moment'](vec_x))
    self.assertAllEqualNested(
        base_model.sample_transformations['second_moment'](structured_x),
        vec_model.sample_transformations['second_moment'](vec_x))

    # Verify that generating values directly from the vector event space works
    # as well.
    vec_x = tf.zeros(batch_shape + vec_model.event_shape, vec_model.dtype)
    self.assertAllEqual(batch_shape,
                        vec_model.unnormalized_log_prob(vec_x).shape)

  @parameterized.named_parameters(
      ('Unbatched', []),
      ('Batched', [2]),
  )
  def testFlattenSampleTransformations(self, batch_shape):
    model = targets.VectorModel(
        TestStructuredModel(), flatten_sample_transformations=True)

    x_init = tf.zeros(batch_shape + model.event_shape, model.dtype)
    first_moment_transform = model.sample_transformations['first_moment']
    first_moment = first_moment_transform(x_init)
    self.assertAllEqual(batch_shape + model.event_shape, first_moment.shape)
    self.assertAllEqual(
        model.event_shape,
        first_moment_transform.ground_truth_mean.shape,
    )
    self.assertAllEqual(
        model.event_shape,
        first_moment_transform.ground_truth_standard_deviation.shape,
    )
    self.assertAllEqual(
        model.event_shape,
        first_moment_transform.ground_truth_mean_standard_error.shape,
    )
    self.assertAllEqual(
        model.event_shape,
        first_moment_transform.ground_truth_standard_deviation_standard_error
        .shape,
    )

  def testBadModel(self):
    with self.assertRaisesRegex(TypeError,
                                'Model must have only one Tensor dtype'):
      targets.VectorModel(TestBadModel())

  def testBadSampleTransformation(self):
    # If we don't flatten the transformations, we don't expect an error
    targets.VectorModel(TestBadSampleTransformationModel())
    with self.assertRaisesRegex(
        TypeError,
        'Sample transformation \'bad\' must have only one Tensor dtype'):
      targets.VectorModel(
          TestBadSampleTransformationModel(),
          flatten_sample_transformations=True)

  def testExample(self):
    base_model = targets.SyntheticItemResponseTheory()
    vec_model = targets.VectorModel(base_model)

    self.assertAllAssertsNested(
        self.assertEqual, {
            'mean_student_ability': tf.float32,
            'centered_student_ability': tf.float32,
            'question_difficulty': tf.float32,
        }, base_model.dtype)
    self.assertEqual(tf.float32, vec_model.dtype)
    self.assertAllAssertsNested(
        self.assertEqual, {
            'mean_student_ability': [],
            'centered_student_ability': [400],
            'question_difficulty': [100],
        },
        base_model.event_shape,
        shallow=base_model.dtype)
    self.assertEqual([501], list(vec_model.event_shape))


if __name__ == '__main__':
  tf.test.main()
