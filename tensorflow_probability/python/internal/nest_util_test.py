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

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import half_cauchy
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import test_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


# TODO(b/159050264): Use `tf.TensorSpec` once there's a np equivalent.
class TensorSpec(object):
  """Stub for tf.TensorSpec to simplify tests in np/jax backends."""

  def __init__(self, shape, dtype=tf.float32, name=None):
    self.shape = tuple(shape)
    self.dtype = dtype
    self._name = name

  @classmethod
  def from_tensor(cls, tensor):
    try:
      name = tensor.op.name
    except AttributeError:
      name = None
    return cls(tensor.shape, tensor.dtype, name)

  @property
  def name(self):
    # Name is meaningless in Eager mode.
    if not tf.executing_eagerly():
      return self._name

  def __eq__(self, other):
    return (
        self.shape == other.shape
        and self.dtype == other.dtype
        and self.name == other.name)

  def __repr__(self):
    return 'TensorSpec(shape={}, dtype={}, name={})'.format(
        self.shape, self.dtype, self.name)


class VariableSpec(TensorSpec):
  """Stub for VariableSpec to simplify tests in np/jax backends."""

  def __init__(self, shape, dtype=tf.float32, trainable=True, name=None):
    super(VariableSpec, self).__init__(shape, dtype, name=name)
    self.trainable = trainable

  @classmethod
  def from_value(cls, variable):
    return cls(
        variable.shape, variable.dtype, variable.trainable, variable.name)

  def __eq__(self, other):
    return (super(VariableSpec, self).__eq__(other)
            and self.trainable == other.trainable)

  def __repr__(self):
    return 'VariableSpec(shape={}, dtype={}, trainable={}, name={})'.format(
        self.shape, self.dtype, self.trainable, self.name)


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


NamedTuple = collections.namedtuple('NamedTuple', 'x y')
NamedTupleYX = collections.namedtuple('NamedTuple', 'y x')

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

  def testMapStructureWithNamedArgs(self):
    x = {'a': [1., 2., (3., 4.)], 'b': -4., 'c': {'d': 10., 'e': (0., 0.)}}
    powers = {'a': [1., 1., (2., 3.)],
              'b': 3.,
              'c': {'d': 0.5, 'e': (1., 1.)}}
    coefs = {'a': [1., 1., (1., 1.)],
             'b': -1.,
             'c': {'d': 10., 'e': (10., 10.)}}
    intercepts = {'a': [-3., -6., (24., -19.)],
                  'b': 0.,
                  'c': {'d': 100., 'e': (-3., -9.)}}
    func = lambda x, power, coef, intercept: coef * (x**power + intercept)
    expected_result = tf.nest.map_structure(func, x, powers, coefs, intercepts)

    # Call with positional args.
    self.assertAllEqualNested(
        tf.nest.map_structure(func, x, powers, coefs, intercepts),
        nest_util.map_structure_with_named_args(
            func, x, powers, coefs, intercepts))

    # Call with named args only.
    self.assertAllEqualNested(
        expected_result,
        nest_util.map_structure_with_named_args(
            func, x=x, intercept=intercepts, coef=coefs, power=powers))

    # Call with both positional and named args.
    self.assertAllEqualNested(
        expected_result,
        nest_util.map_structure_with_named_args(
            func, x, powers, intercept=intercepts, coef=coefs))

    # Out-of-order arguments raise an error.
    with self.assertRaisesRegex(TypeError, 'got multiple values for argument'):
      nest_util.map_structure_with_named_args(
          func, powers, coefs, intercepts, x=x)

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

  @parameterized.named_parameters({
      'testcase_name': '_dtype_atom',
      'value': [1, 2],
      'dtype': tf.float32,
      'expected': TensorSpec([2], tf.float32, name='c2t')
  },{
      'testcase_name': '_dtype_seq',
      'value': [1, 2],
      'dtype': [tf.float32, None],
      'expected': [TensorSpec([], tf.float32, name='c2t/0'),
                   TensorSpec([], tf.int32, name='c2t/1')]
  },{
      'testcase_name': '_matching_dtype_and_hint',
      'value': [1, 2],
      'dtype': [tf.float32, None],
      'dtype_hint': [tf.int32, tf.int64],
      'expected': [TensorSpec([], tf.float32, name='c2t/0'),
                   TensorSpec([], tf.int64, name='c2t/1')]
  },{
      'testcase_name': '_broadcast_hint_to_dtype',
      'value': [1, 2],
      'dtype': [tf.float32, None],
      'dtype_hint': tf.int64,
      'expected': [TensorSpec([], tf.float32, name='c2t/0'),
                   TensorSpec([], tf.int64, name='c2t/1')]
  },{
      'testcase_name': '_broadcast_dtype_to_hint',
      'value': [1, 2],
      'dtype': tf.float32,
      'dtype_hint': [tf.int32, tf.int64],
      'expected': [TensorSpec([], tf.float32, name='c2t/0'),
                   TensorSpec([], tf.float32, name='c2t/1')]
  },{
      'testcase_name': '_dtype_dict',
      'value': {'a': 1, 'b': 2},
      'dtype': {'a': tf.float32, 'b': None},
      'expected': {'a': TensorSpec([], tf.float32, name='c2t/a'),
                   'b': TensorSpec([], tf.int32, name='c2t/b')}
  },{
      'testcase_name': '_dtype_dict_with_hint',
      'value': {'a': 1, 'b': 2},
      'dtype': {'a': tf.float32, 'b': None},
      'dtype_hint': tf.int64,
      'expected': {'a': TensorSpec([], tf.float32, name='c2t/a'),
                   'b': TensorSpec([], tf.int64, name='c2t/b')}
  },{
      'testcase_name': '_tensor_with_hint',
      'value': [TensorSpec([], tf.int32, name='tensor')],
      'dtype_hint': [tf.float32],
      'expected': [TensorSpec([], tf.int32, name='tensor')]
  },{
      'testcase_name': '_tensor_struct',
      'value': [TensorSpec([], tf.int32, name='tensor'),
                TensorSpec([], tf.float32, name='tensor')],
      'dtype_hint': [tf.float32, tf.float32],
      'expected': [TensorSpec([], tf.int32, name='tensor'),
                   TensorSpec([], tf.float32, name='tensor_1')]
  },{
      'testcase_name': '_deep_structure',
      'value': {'a': [{'b': 0}]},
      'dtype_hint': {'a': [{'b': tf.float32}]},
      'expected': {'a': [{'b': TensorSpec([], tf.float32, name='c2t/a.0.b')}]}
  },{
      'testcase_name': '_without_name',
      'value': [0., 1.],
      'dtype_hint': [None, None],
      'name': None,
      'expected': [TensorSpec([], tf.float32, name='Const'),
                   TensorSpec([], tf.float32, name='Const_1')]
  },{
      'testcase_name': '_tensor_and_variable_struct',
      'value': [TensorSpec([], tf.int32, name='tensor'),
                VariableSpec([], tf.float32, trainable=False, name='variable')],
      'dtype_hint': [tf.float32, tf.float32],
      'convert_ref': False,
      'expected': [
          TensorSpec([], tf.int32, name='tensor'),
          VariableSpec([], tf.float32, trainable=False, name='variable:0')]
  },{
      'testcase_name': '_tensor_and_variable_struct_convert_ref',
      'value': [VariableSpec([], tf.int32, name='variable'),
                TensorSpec([], tf.float32, name='tensor')],
      'dtype_hint': [tf.float32, tf.float32],
      'expected': [TensorSpec([], tf.int32, name='c2t/ReadVariableOp'),
                   TensorSpec([], tf.float32, name='tensor')]
  })
  def testConvertToNestedTensor(
      self, value, dtype=None, dtype_hint=None, name='c2t', convert_ref=True,
      expected=None):
    # Convert specs to tensors or variables.
    def maybe_spec_to_tensor(x):
      if isinstance(x, VariableSpec):
        return tf.Variable(
            tf.zeros(x.shape, x.dtype), trainable=x.trainable, name=x.name)
      if isinstance(x, TensorSpec):
        return tf.zeros(x.shape, x.dtype, name=x.name)
      return x
    value = nest.map_structure(maybe_spec_to_tensor, value)

    # Grab shape/dtype from convert_to_nested_tensor for comparison.
    def spec_from_value(x):
      if isinstance(x, tf.Variable):
        return VariableSpec.from_value(x)
      return TensorSpec.from_tensor(x)

    observed = nest.map_structure(
        spec_from_value,
        nest_util.convert_to_nested_tensor(
            value, dtype, dtype_hint, convert_ref=convert_ref, name=name))
    self.assertAllEqualNested(observed, expected)

  @parameterized.named_parameters({
      'testcase_name': '_seq_length',
      'value': [1, 2],
      'dtype': [tf.float32, tf.int32, tf.int64],
      'error': (ValueError, "The two structures don't have the same")
  },{
      'testcase_name': '_incompatible_dtype_and_hint',
      'value': [[1,2],[3,4]],
      'dtype': [[tf.float32, tf.int32], None],
      'dtype_hint': [None, [tf.float64, tf.int64]],
      'error': (ValueError, "The two structures don't have the same")
  },{
      'testcase_name': '_struct_of_tensors',
      'value': [TensorSpec([])],
      'dtype_hint': tf.float32,
      'error': (NotImplementedError, 'Cannot convert a structure of tensors')
  })
  def testConvertToNestedTensorRaises(
      self, value, dtype=None, dtype_hint=None, error=None):
    # Convert specs to tensors
    def maybe_spec_to_tensor(x):
      if isinstance(x, TensorSpec):
        return tf.zeros(x.shape, x.dtype, name='tensor')
      return x
    value = nest.map_structure(maybe_spec_to_tensor, value)

    # Structure must be exact.
    with self.assertRaisesRegex(*error):
      nest_util.convert_to_nested_tensor(
          value=value,
          dtype=dtype,
          dtype_hint=dtype_hint)

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      disable_jax=True,
      reason='`convert_to_tensor` happily coerces `np.ndarray`.')
  def testConvertToNestedTensorRaises_incompatible_dtype(self):
    # Dtype checks are strict if TF backend
    with self.assertRaisesRegex(ValueError,
                                'Tensor conversion requested dtype float64'):
      nest_util.convert_to_nested_tensor(
          value=tf.constant(1, tf.float32),
          dtype=tf.float64)

  @parameterized.named_parameters({
      'testcase_name': '_ntup_ntup',
      'target': NamedTupleYX(1, 2),
      'source': NamedTuple(3, 4),
      'expect': NamedTupleYX(y=4, x=3)
  },{
      'testcase_name': '_ntup_dict',
      'target': NamedTupleYX(1, 2),
      'source': {'x': 3, 'y': 4},
      'expect': NamedTupleYX(y=4, x=3)
  },{
      'testcase_name': '_ntup_list',
      'target': NamedTupleYX(1, 2),
      'source': [3, 4],
      'expect': NamedTupleYX(y=3, x=4)
  },{
      'testcase_name': '_list_ntup',
      'target': [1, 2],
      'source': NamedTupleYX(3, 4),
      'expect': [3, 4]
  },{
      'testcase_name': '_dict_ntup',
      'target': {'x': 1, 'y': 2},
      'source': NamedTupleYX(3, 4),
      'expect': {'x': 4, 'y': 3}
  },{
      'testcase_name': '_up_to',
      'target': NamedTupleYX(1, 2),
      'source': NamedTuple([3,4], [5,6]),
      'expect': NamedTupleYX(y=[5,6], x=[3,4])
  },{
      'testcase_name': '_deep_wrap',
      'target': {'foo': NamedTuple(1, 2), 'bar': NamedTupleYX(3, 4)},
      'source': {'foo': NamedTupleYX(5, 6), 'bar': NamedTuple(7, 8)},
      'expect': {'foo': NamedTuple(6, 5), 'bar': NamedTupleYX(8, 7)}
  })
  def testCoerceStructure(self, target, source, expect):
    coerced = nest_util.coerce_structure(target, source)
    self.assertAllEqualNested(expect, coerced)

  @parameterized.named_parameters({
      'testcase_name': '_seqlength',
      'target': NamedTuple(1, 2),
      'source': [3, 4, 5],
      'message': 'sequence length'
  },{
      'testcase_name': '_dict_list',
      'target': {'foo': 1, 'bar': 2},
      'source': [3, 4],
      'message': 'sequence type'
  },{
      'testcase_name': '_shallow_input',
      'target': [1, 2],
      'source': 3,
      'message': 'input must also be a sequence'
  })
  def testCoerceStructureRaises(self, target, source, message):
    with self.assertRaisesRegex((ValueError, TypeError), message):
      nest_util.coerce_structure(target, source)


def _product_generator(a, b):
  """Toy generator for testing `map_structure_coroutine`."""
  yield a * b


def _get_yielded_and_returned_values(gen, values_to_send=()):
  """Helper for testing generators and coroutines."""
  yvals = []
  try:
    yvals.append(next(gen))
    for s in values_to_send:
      yvals.append(gen.send(s))
    while True:  # Loop until `gen` returns, even if no more values to send.
      yvals.append(next(gen))
  except StopIteration as e:
    rvals = e.value
  return yvals, rvals


class MapStructureCoroutineTest(test_util.TestCase):

  def testYieldsInFlattenedOrder(self):
    a = {'y': 2., 'x': 3., 'z': 2.}
    b = {'y': 10., 'x': 0.1, 'z': -4.}
    expected_result = tf.nest.flatten(
        tf.nest.map_structure(lambda a, b: a * b, a, b))
    yvals, _ = _get_yielded_and_returned_values(
        nest_util.map_structure_coroutine(_product_generator, a, b=b))
    self.assertAllClose(expected_result, yvals)

  def testReturnsExpectedStructure(self):

    def _return_product_generator(a, b):
      return a * b
      yield  # pylint: disable=unreachable

    a = {'y': 2., 'x': 3., 'z': 2.}
    b = {'y': 10., 'x': 0.1, 'z': -4.}
    _, rvals = _get_yielded_and_returned_values(
        nest_util.map_structure_coroutine(_return_product_generator, a, b))
    self.assertAllCloseNested(rvals,
                              tf.nest.map_structure(lambda a, b: a * b, a, b))

  def testNamedAndPositionalArgs(self):
    a = {'y': 2., 'x': 3., 'z': 2.}
    b = {'y': 10., 'x': 0.1, 'z': -4.}
    yvals_positional, _ = _get_yielded_and_returned_values(
        nest_util.map_structure_coroutine(_product_generator, a, b))
    yvals_named, _ = _get_yielded_and_returned_values(
        nest_util.map_structure_coroutine(_product_generator, a=a, b=b))
    yvals_mixed, _ = _get_yielded_and_returned_values(
        nest_util.map_structure_coroutine(_product_generator, a, b=b))
    self.assertAllCloseNested(yvals_positional, yvals_named)
    self.assertAllCloseNested(yvals_positional, yvals_mixed)

    # Out-of-order arguments raise an error.
    with self.assertRaisesRegex(TypeError, 'got multiple values for argument'):
      _get_yielded_and_returned_values(
          nest_util.map_structure_coroutine(_product_generator, b, a=a))

  def testUpTo(self):
    a = {'y': 3., 'z': 2.}
    b = {'y': [10., -1.], 'z': [-4., 2.]}
    def g(a, b):
      b1, b2 = b
      yield a * b1
      return a * b2
    yvals, rvals = _get_yielded_and_returned_values(
        nest_util.map_structure_coroutine(g, a, b, _up_to=a))
    self.assertAllCloseNested(yvals, [30., -8.])
    self.assertAllCloseNested(rvals, {'y': -3., 'z': 4.})

  def testWithTuplePaths(self):
    def g(path, _):
      return path
      yield  # pylint: disable=unreachable
    a = {'y': 2., 'x': 3., 'z': 2.}
    _, rvals = _get_yielded_and_returned_values(
        nest_util.map_structure_coroutine(g, a, _with_tuple_paths=True))
    self.assertAllEqualNested(
        rvals,
        nest.map_structure_with_tuple_paths(lambda path, a: path, a))

  def testDocstringJDCExample(self):
    def _horseshoe(path, scale):
      # Auxiliary-variable representation of a horseshoe prior.
      name = ','.join(path)
      z = yield half_cauchy.HalfCauchy(loc=0, scale=scale, name=name + '_z')
      w_noncentered = yield normal.Normal(
          loc=0., scale=z, name=name + '_w_noncentered')
      return z * w_noncentered

    @jdab.JointDistributionCoroutineAutoBatched
    def model():
      weights = yield from nest_util.map_structure_coroutine(
          _horseshoe,
          scale={'a': tf.ones([5]) * 100., 'b': tf.ones([2]) * 1e-2},
          _with_tuple_paths=True)
      yield deterministic.Deterministic(
          tf.sqrt(tf.norm(weights['a'])**2 + tf.norm(weights['b'])**2),
          name='weights_norm')
    event_shape = model.event_shape
    print(event_shape)
    self.assertAllEqualNested(
        event_shape._asdict(),
        {'a_z': tf.constant([5]),
         'a_w_noncentered': tf.constant([5]),
         'b_z': tf.constant([2]),
         'b_w_noncentered': tf.constant([2]),
         'weights_norm': tf.constant([])})

if __name__ == '__main__':
  test_util.main()
