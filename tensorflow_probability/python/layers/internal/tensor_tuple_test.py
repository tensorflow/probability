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
"""Tests for `TensorTuple`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.layers.internal import tensor_tuple

from tensorflow.python.framework import ops  # pylint: disable=g-direct-tensorflow-import


class MyTuple(object):
  """Pretend user-side class for `ConvertToCompositeTensorTest ."""

  def __init__(self, sequence):
    self._sequence = tuple(sequence)

  def __getitem__(self, key):
    return self._sequence[key]

  def __len__(self):
    return len(self._sequence)

  def __iter__(self):
    return iter(self._sequence)


tf.register_tensor_conversion_function(
    MyTuple, conversion_func=lambda x, *_, **__: tensor_tuple.TensorTuple(x))


@test_util.test_all_tf_execution_regimes
class CustomConvertToCompositeTensorTest(test_util.TestCase):

  def test_iter(self):
    x = MyTuple((1, [2., 3.], [[4, 5], [6, 7]]))
    y = ops.convert_to_tensor_or_composite(value=x)
    # TODO(jsimsa): The behavior of `is_tensor` for composite tensors have
    # changed (from True to False) and this check needs to be disabled so that
    # both TAP presubmits (running at HEAD) and Kokoro presubmit (using TF
    # nightly) pass. Re-enable this check when TF nightly picks up this change.
    # self.assertTrue(tf.is_tensor(y))
    self.assertIsInstance(y, tensor_tuple.TensorTuple)
    self.assertLen(y, 3)
    for x_, y_ in zip(x, y):
      self.assertIsInstance(y_, tf.Tensor)
      self.assertTrue(tf.is_tensor(y_))
      self.assertAllEqual(x_, tf.get_static_value(y_))

  def test_getitem(self):
    x = MyTuple((1, [2., 3.], [[4, 5], [6, 7]]))
    y = ops.convert_to_tensor_or_composite(value=x)
    self.assertLen(y, 3)
    for i in range(3):
      self.assertAllEqual(x[i], tf.get_static_value(y[i]))
    self.assertEqual(not(tf.executing_eagerly()), y._is_graph_tensor)

  def test_to_from(self):
    x = MyTuple((1, [2., 3.], [[4, 5], [6, 7]]))
    y = ops.convert_to_tensor_or_composite(value=x)
    self.assertIs(y._sequence, y._to_components())
    z = tensor_tuple.TensorTuple._from_components(y._sequence)
    self.assertIsInstance(z, tensor_tuple.TensorTuple)
    self.assertEqual(y._sequence, z._sequence)

  def test_str_repr(self):
    x = MyTuple((1, [2., 3.], [[4, 5], [6, 7]]))
    y = ops.convert_to_tensor_or_composite(value=x)

    if tf.executing_eagerly():
      expected = ('(<tf.Tensor: shape=(), dtype=int32, numpy=1>,'
                  ' <tf.Tensor: shape=(2,), dtype=float32,'
                  ' numpy=array([2.,3.], dtype=float32)>,'
                  ' <tf.Tensor: shape=(2,2), dtype=int32,'
                  ' numpy=array([[4,5],[6,7]], dtype=int32)>)')
      regexp2 = re.compile(r'\s+([\d\[\]])')
      def _strip(s):
        s = s.replace('\n', '')
        s = re.sub(regexp2, r'\1', s)
        return s
      self.assertEqual(expected, _strip(str(y)))
      self.assertEqual(expected, _strip(repr(y)))
    else:
      expected = ('(<tf.Tensor \'Const:0\' shape=() dtype=int32>,'
                  ' <tf.Tensor \'Const_1:0\' shape=(2,) dtype=float32>,'
                  ' <tf.Tensor \'Const_2:0\' shape=(2, 2) dtype=int32>)')
      self.assertEqual(expected, str(y))
      self.assertEqual(expected, repr(y))

  def test_shape_invariant(self):
    x = MyTuple((1, [2., 3.], [[4, 5], [6, 7]]))
    y = ops.convert_to_tensor_or_composite(value=x)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError, 'TensorTuple._shape_invariant_to_components'):
      y._shape_invariant_to_components()


if __name__ == '__main__':
  tf.test.main()
