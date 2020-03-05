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
"""Tests for nest_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import test_util


class LeafList(list):
  _tfp_nest_expansion_force_leaf = ()

  def __repr__(self):
    return 'LeafList' + super(LeafList, self).__repr__()


class LeafTuple(tuple):
  _tfp_nest_expansion_force_leaf = ()

  def __repr__(self):
    return 'LeafTuple' + super(LeafTuple, self).__repr__()


class LeafDict(dict):
  _tfp_nest_expansion_force_leaf = ()

  def __repr__(self):
    return 'LeafDict' + super(LeafDict, self).__repr__()


NamedTuple = collections.namedtuple('NamedTuple', 'x, y')

# Alias for readability.
Tensor = np.array  # pylint: disable=invalid-name


class LeafNamedTuple(
    collections.namedtuple('LeafNamedTuple', 'x, y')):
  _tfp_nest_expansion_force_leaf = ()


@test_util.test_all_tf_execution_regimes
class NestUtilTest(test_util.TestCase):

  @parameterized.parameters((1, [2, 2], [1, 1]),
                            ([1], [2, 2], [1, 1]),
                            (1, NamedTuple(2, 2), NamedTuple(1, 1)),
                            ([1, 2], NamedTuple(2, 2), [1, 2]),
                            (1, 1, 1))
  def testBroadcastStructure(self, from_structure, to_structure, expected):
    ret = nest_util.broadcast_structure(to_structure, from_structure)
    self.assertAllEqual(expected, ret)

  # pylint: disable=bad-whitespace
  @parameterized.parameters(
      # Input                Output
      # Directly convertible.
      (1,                    Tensor(1)),
      (LeafList([1]),        Tensor([1])),
      (LeafTuple([1]),       Tensor([1])),
      (LeafNamedTuple(1, 2), Tensor([1, 2])),
      # Leaves convertible
      (LeafDict({'a': 1}),   LeafDict({'a': Tensor(1)})),
      (NamedTuple(1, 2),     NamedTuple(Tensor(1), Tensor(2))),
      # Outer lists/tuples/dicts
      ([1],                  [Tensor(1)]),
      ([LeafList([1])],      [Tensor([1])]),
      [(1,),                 (Tensor(1),)],
      ([[1], [2]],           [Tensor([1]), Tensor([2])]),
      ({'a': 1},             {'a': Tensor(1)}),
      ({'a': [1, 2]},        {'a': Tensor([1, 2])}),
      ([[1, 2], [3, 4]],     [Tensor([1, 2]), Tensor([3, 4])]),
      ({'a': [1, 2], 'b': {'c': [[3, 4]]}},
       {'a': Tensor([1, 2]), 'b': {'c': Tensor([[3, 4]])}}),
      # Ragged lists.
      ([[[1], [2, 3]]],      [[Tensor([1]), Tensor([2, 3])]]),
      )
  # pylint: enable=bad-whitespace
  def testConvertArgsToTensor(self, args, expected_converted_args):
    # This tests that `args`, after conversion, has the same structure of
    # converted_args_struct and is filled with Tensors. This verifies that the
    # tf.convert_to_tensor was called at the appropriate times.
    converted_args = nest_util.convert_args_to_tensor(args)
    tf.nest.assert_same_structure(expected_converted_args, converted_args)
    tf.nest.map_structure(lambda e: self.assertIsInstance(e, tf.Tensor),
                          converted_args)
    converted_args_ = tf.nest.map_structure(self.evaluate, converted_args)
    tf.nest.map_structure(self.assertAllEqual, expected_converted_args,
                          converted_args_)

  @parameterized.parameters(
      # Input              DType             Output
      # Concrete dtypes.
      (1,                  tf.int64,         Tensor(1, dtype=np.int64)),
      ({'a': 1, 'b': 2},   {'a': tf.int64, 'b': tf.int64},
       {'a': Tensor(1, np.int64), 'b': Tensor(2, dtype=np.int64)}),
      # Override outer structure.
      ([1, 2],             tf.int32,         Tensor([1, 2])),
      (NamedTuple(1, 2),   tf.int32,         Tensor([1, 2])),
      # Override inner structure.
      ([[1, 2]],           [[None, None]],   [[Tensor(1), Tensor(2)]]),
      ([[1, [2]]],         [[None, [None]]], [[Tensor(1), [Tensor(2)]]]),
      # None with structured leaves.
      ([NamedTuple(1, 2)], [None], [NamedTuple(Tensor(1), Tensor(2))]),
      )
  def testConvertArgsToTensorWithDType(self, args, dtype,
                                       expected_converted_args):
    # Like the above test, but now with dtype hints.
    converted_args = nest_util.convert_args_to_tensor(args, dtype)
    tf.nest.assert_same_structure(expected_converted_args, converted_args)
    tf.nest.map_structure(lambda e: self.assertIsInstance(e, tf.Tensor),
                          converted_args)
    converted_args_ = tf.nest.map_structure(self.evaluate, converted_args)
    tf.nest.map_structure(self.assertAllEqual, expected_converted_args,
                          converted_args_)

  @parameterized.parameters(
      # Input              DType
      # Structure mismatch.
      ([1],                 [None, None]),
      ([1],                 (None,)),
      # Not even a Tensor.
      (np.array,            None),
      )
  def testConvertArgsToTensorErrors(self, args, dtype):
    with self.assertRaises((TypeError, ValueError)):
      nest_util.convert_args_to_tensor(args, dtype)

  @parameterized.parameters(
      (1,),
      ([1],),
      (NamedTuple(1, 2),),
      ({'arg': 1},))
  def testCallFnOneArg(self, arg):
    def fn(arg):
      return arg

    self.assertEqual(
        tf.nest.flatten(arg), tf.nest.flatten(nest_util.call_fn(fn, arg)))

  @parameterized.parameters((LeafDict({'arg': 1}),),
                            (LeafList([1, 2]),))
  def testCallFnLeafArgs(self, arg):
    def fn(arg):
      return arg
    self.assertEqual(arg, fn(arg))

  @parameterized.parameters(((1, 2),),
                            ([1, 2],),
                            ({'arg1': 1, 'arg2': 2},))
  def testCallFnTwoArgs(self, arg):
    def fn(arg1, arg2):
      return arg1 + arg2

    self.assertEqual(3, nest_util.call_fn(fn, arg))


if __name__ == '__main__':
  tf.test.main()
