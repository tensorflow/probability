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
"""Restructure Tests."""

import collections

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import restructure
from tensorflow_probability.python.distributions import joint_distribution_named
from tensorflow_probability.python.distributions import joint_distribution_sequential
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class RestructureBijectorTest(test_util.TestCase):
  """Tests the correctness of the Y = Restructure({nested}) transformation."""

  def testListToStructure(self):
    bij = restructure.Restructure({'foo': [1, 2], 'bar': 0, 'baz': (3, 4)})

    x = [[1, 2, 3], [4, 5, 6], 7., 8., 9.]
    x_ndims = [1, 1, 0, 0, 0]

    y = {
        'foo': [[4, 5, 6], 7.],
        'bar': [1, 2, 3],
        'baz': (8., 9.),
    }
    y_ndims = {'foo': [1, 0], 'bar': 1, 'baz': (0, 0)}

    # Invert assertion arguments to infer structure from bijector output.
    self.assertAllEqualNested(bij.forward(x), y, check_types=True)
    self.assertAllEqualNested(bij.inverse(y), x, check_types=True)

    self.assertAllEqualNested(
        0., self.evaluate(bij.forward_log_det_jacobian(x, x_ndims)))
    self.assertAllEqualNested(
        0., self.evaluate(bij.inverse_log_det_jacobian(y, y_ndims)))

  def testDictToStructure(self):
    bij = restructure.Restructure({
        'foo': ['b', 'c'],
        'bar': 'a',
        'baz': ('d', 'e')
    })

    x = {'a': [1, 2, 3],
         'b': [4, 5, 6],
         'c': 7., 'd': 8., 'e': 9.}
    x_ndims = {'a': 1, 'b': 1, 'c': 0, 'd': 0, 'e': 0}

    y = {'foo': [[4, 5, 6], 7.],
         'bar': [1, 2, 3],
         'baz': (8., 9.)}
    y_ndims = {'foo': [1, 0], 'bar': 1, 'baz': (0, 0)}

    # Invert assertion arguments to infer structure from bijector output.
    self.assertAllEqualNested(bij.forward(x), y, check_types=True)
    self.assertAllEqualNested(bij.inverse(y), x, check_types=True)

    self.assertAllEqualNested(
        0., self.evaluate(bij.forward_log_det_jacobian(x, x_ndims)))
    self.assertAllEqualNested(
        0., self.evaluate(bij.inverse_log_det_jacobian(y, y_ndims)))

  def testSmartConstructors(self):
    x = collections.OrderedDict([
        ('a', [1, 2, 3]),
        ('b', [4, 5, 6]),
        ('c', 7.),
        ('d', 8.),
        ('e', 9.)])
    bij = restructure.tree_flatten(x)

    bij_2 = restructure.pack_sequence_as(x)

    # Invert assertion arguments to infer structure from bijector output.
    self.assertAllEqualNested(
        bij.forward(x), [1, 2, 3, 4, 5, 6, 7, 8, 9], check_types=True)
    self.assertAllEqualNested(
        bij_2.forward(bij.forward(x)), x, check_types=True)

  def testStructureToStructure(self):
    bij = restructure.Restructure(
        input_structure={
            'foo': [0, 1],
            'bar': 2,
            'baz': (3, 4)
        },
        output_structure={
            'zip': [1, 2, 3],
            'zap': 0,
            'zop': 4
        })

    x = {'foo': [0., [1.]],
         'bar': [[2.]],
         'baz': ([[[3.]]], [[[[4.]]]])}
    x_ndims = {'foo': [0, 1], 'bar': 2, 'baz': (3, 4)}

    y = {'zip': [[1.], [[2.]], [[[3.]]]],
         'zap': 0.,
         'zop': [[[[4.]]]]}
    y_ndims = {'zip': [1, 2, 3], 'zap': 0, 'zop': 4}

    # Invert assertion arguments to infer structure from bijector output.
    self.assertAllEqualNested(bij.forward(x), y, check_types=True)
    self.assertAllEqualNested(bij.inverse(y), x, check_types=True)

    self.assertAllEqualNested(
        0., self.evaluate(bij.forward_log_det_jacobian(x, x_ndims)))
    self.assertAllEqualNested(
        0., self.evaluate(bij.inverse_log_det_jacobian(y, y_ndims)))

  def testEventNdims(self):
    bij = restructure.Restructure(
        input_structure={
            'foo': [0, 1],
            'bar': 2,
            'baz': (3, 4)
        },
        output_structure={
            'zip': [1, 2, 3],
            'zap': 0,
            'zop': 4
        })

    x_ndims = {'foo': [10, 11], 'bar': 12, 'baz': (13, 14)}
    y_ndims = {'zip': [11, 12, 13], 'zap': 10, 'zop': 14}

    self.assertAllEqualNested(
        y_ndims, bij.forward_event_ndims(x_ndims), check_types=True)
    self.assertAllEqualNested(
        x_ndims, bij.inverse_event_ndims(y_ndims), check_types=True)

  def testPartsWithUnusedInternalStructure(self):
    dist = joint_distribution_sequential.JointDistributionSequential([
        joint_distribution_named.JointDistributionNamed(
            {'a': normal.Normal(0., 1.)}),
        joint_distribution_named.JointDistributionNamed(
            {'b': normal.Normal(1000., 1.)}),
    ])
    x = dist.sample(  # Shape: [{'a': []}, {'b': []}]
        seed=test_util.test_seed(sampler_type='stateless'))

    # Test that we can swap the outer list entries, even though they contain
    # internal structure (i.e., are themselves dicts).
    swap_elements = restructure.Restructure(
        input_structure=[1, 0], output_structure=[0, 1])
    self.assertAllEqualNested(swap_elements(x), [x[1], x[0]], check_types=True)

    swapped_dist = swap_elements(dist)
    self.assertAllEqualNested(swapped_dist.event_shape,
                              [dist.event_shape[1], dist.event_shape[0]],
                              check_types=True)
    self.assertEqual(swapped_dist.dtype,
                     [dist.dtype[1], dist.dtype[0]])

  def testCompositeTensor(self):
    bij = restructure.Restructure({'foo': [1, 2], 'bar': 0, 'baz': (3, 4)})

    x = [[1, 2, 3], [4, 5, 6], 7., 8., 9.]
    flat = tf.nest.flatten(bij, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(bij, flat, expand_composites=True)
    self.assertAllCloseNested(
        bij.forward(x),
        tf.function(lambda b_: b_.forward(x))(unflat))

  def testFloat64LDJ(self):
    bij = restructure.Restructure([0, 1])
    x = [tf.zeros([], tf.float64), tf.zeros([], tf.float64)]
    event_ndims = [0, 0]
    fldj = bij.forward_log_det_jacobian(x, event_ndims)
    ildj = bij.inverse_log_det_jacobian(x, event_ndims)
    self.assertEqual(tf.float64, fldj.dtype)
    self.assertEqual(tf.float64, ildj.dtype)


if __name__ == '__main__':
  test_util.main()
