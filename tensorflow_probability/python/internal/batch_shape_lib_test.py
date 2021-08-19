# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for batch_shape."""

# Dependency imports

from absl import logging
from absl.testing import parameterized
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import batch_shape_lib
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import test_util

from tensorflow.python.platform import test as tf_test  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


class _MVNTriLWithDynamicParamNdims(tfd.MultivariateNormalTriL):

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        loc=parameter_properties.ParameterProperties(
            event_ndims=lambda _: None,
            event_ndims_tensor=(
                lambda _: tf1.placeholder_with_default(1, shape=None))),
        scale_tril=parameter_properties.ParameterProperties(
            event_ndims=lambda _: None,
            event_ndims_tensor=(
                lambda _: tf1.placeholder_with_default(2, shape=None))))


@test_util.test_graph_and_eager_modes
class BatchShapeInferenceTests(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': '_trivial',
       'value_fn': lambda: tfd.Normal(loc=0., scale=1.),
       'expected_batch_shape_parts': {'loc': [], 'scale': []},
       'expected_batch_shape': []},
      {'testcase_name': '_simple_tensor_broadcasting',
       'value_fn': lambda: tfd.MultivariateNormalDiag(  # pylint: disable=g-long-lambda
           loc=[0., 0.], scale_diag=tf.convert_to_tensor([[1., 1.], [1., 1.]])),
       'expected_batch_shape_parts': {'loc': [], 'scale_diag': [2]},
       'expected_batch_shape': [2]},
      {'testcase_name': '_rank_deficient_tensor_broadcasting',
       'value_fn': lambda: tfd.MultivariateNormalDiag(  # pylint: disable=g-long-lambda
           loc=0., scale_diag=tf.convert_to_tensor([[1., 1.], [1., 1.]])),
       'expected_batch_shape_parts': {'loc': [], 'scale_diag': [2]},
       'expected_batch_shape': [2]},
      {'testcase_name': '_dynamic_event_ndims',
       'value_fn': lambda: _MVNTriLWithDynamicParamNdims(  # pylint: disable=g-long-lambda
           loc=[[0., 0.], [1., 1.], [2., 2.]],
           scale_tril=[[1., 0.], [-1., 1.]]),
       'expected_batch_shape_parts': {'loc': [3], 'scale_tril': []},
       'expected_batch_shape': [3]},
      {'testcase_name': '_mixture_same_family',
       'value_fn': lambda: tfd.MixtureSameFamily(  # pylint: disable=g-long-lambda
           mixture_distribution=tfd.Categorical(
               logits=[[[1., 2., 3.],
                        [4., 5., 6.]]]),
           components_distribution=tfd.Normal(loc=0.,
                                              scale=[[[1., 2., 3.],
                                                      [4., 5., 6.]]])),
       'expected_batch_shape_parts': {'mixture_distribution': [1, 2],
                                      'components_distribution': [1, 2]},
       'expected_batch_shape': [1, 2]},
      {'testcase_name': '_deeply_nested',
       'value_fn': lambda: tfd.Independent(  # pylint: disable=g-long-lambda
           tfd.Independent(
               tfd.Independent(
                   tfd.Independent(
                       tfd.Normal(loc=0., scale=[[[[[[[[1.]]]]]]]]),
                       reinterpreted_batch_ndims=2),
                   reinterpreted_batch_ndims=0),
               reinterpreted_batch_ndims=1),
           reinterpreted_batch_ndims=1),
       'expected_batch_shape_parts': {'distribution': [1, 1, 1, 1]},
       'expected_batch_shape': [1, 1, 1, 1]},
      {'testcase_name': 'noparams',
       'value_fn': tfb.Exp,
       'expected_batch_shape_parts': {},
       'expected_batch_shape': []})
  @test_util.numpy_disable_test_missing_functionality('b/188002189')
  def test_batch_shape_inference_is_correct(
      self, value_fn, expected_batch_shape_parts, expected_batch_shape):
    value = value_fn()  # Defer construction until we're in the right graph.

    parts = batch_shape_lib.batch_shape_parts(value)
    self.assertAllEqualNested(
        parts,
        nest.map_structure_up_to(
            parts, tf.TensorShape, expected_batch_shape_parts))

    self.assertAllEqual(expected_batch_shape,
                        batch_shape_lib.inferred_batch_shape_tensor(value))

    batch_shape = batch_shape_lib.inferred_batch_shape(value)
    self.assertIsInstance(batch_shape, tf.TensorShape)
    self.assertTrue(batch_shape.is_compatible_with(expected_batch_shape))

  def test_bijector_event_ndims(self):
    bij = tfb.Sigmoid(low=tf.zeros([2]), high=tf.ones([3, 2]))
    self.assertAllEqual(batch_shape_lib.inferred_batch_shape(bij), [3, 2])
    self.assertAllEqual(batch_shape_lib.inferred_batch_shape_tensor(bij),
                        [3, 2])
    self.assertAllEqual(
        batch_shape_lib.inferred_batch_shape(bij, bijector_x_event_ndims=1),
        [3])
    self.assertAllEqual(
        batch_shape_lib.inferred_batch_shape_tensor(
            bij, bijector_x_event_ndims=1),
        [3])


class ParametersAsKwargsTest(test_util.TestCase):
  # This doesn't really deserve to be a separate test class, but is split
  # out from BatchShapeInferenceTests because the `test_graph_and_eager_modes`
  # decorator interacts poorly with the `tf_test.mock.patch.object` decorator.

  @test_util.numpy_disable_test_missing_functionality('tf_logging')
  @test_util.jax_disable_test_missing_functionality('tf_logging')
  @tf_test.mock.patch.object(logging, 'warning', autospec=True)
  def test_parameters_as_kwargs(self, mock_warning):
    dist = tfd.Normal(loc=tf.zeros([2]), scale=tf.ones([5, 1]))
    self.assertAllEqual(
        batch_shape_lib.inferred_batch_shape_tensor(dist), [5, 2])
    self.assertAllEqual(
        batch_shape_lib.inferred_batch_shape_tensor(dist, loc=tf.zeros([5])),
        [5, 5])
    self.assertAllEqual(
        batch_shape_lib.inferred_batch_shape_tensor(
            dist, scale=tf.zeros([5]), loc=tf.zeros([1, 1])),
        [1, 5])

    # Check that passing an unrecognized argument raises a warning.
    self.assertEqual(0, mock_warning.call_count)
    batch_shape_lib.inferred_batch_shape_tensor(dist, df=3.)
    self.assertEqual(1, mock_warning.call_count)

if __name__ == '__main__':
  tf.test.main()
