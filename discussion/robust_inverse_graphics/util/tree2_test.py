# Copyright 2024 The TensorFlow Probability Authors.
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
"""Tests for tree."""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import os
from typing import Any, NamedTuple

from absl.testing import parameterized
import flax
import immutabledict
import jax
import jax.numpy as jnp
import numpy as np

from discussion.robust_inverse_graphics.util import test_util
from discussion.robust_inverse_graphics.util import tree2
import tensorflow_probability.substrates.jax as tfp

global_registry = tree2.Registry(allow_unknown_types=True)


class UnregisteredNamedTuple(NamedTuple):
  x: Any
  y: Any


@global_registry.auto_register_type('test.RegisteredNamedTuple')
class RegisteredNamedTuple(NamedTuple):
  x: Any
  y: Any


@dataclasses.dataclass
class UnregisteredDataClass:
  x: Any
  y: Any


@global_registry.auto_register_type('test.RegisteredDataClass')
@dataclasses.dataclass
class RegisteredDataClass:
  x: Any
  y: Any


# IntEnum, so we can sort them.
@global_registry.auto_register_type('test.RegisteredEnum')
class RegisteredEnum(enum.IntEnum):
  X = enum.auto()
  Y = enum.auto()


class UnregisteredSequence(Sequence):

  def __init__(self, values):
    self._values = values

  def __getitem__(self, idx):
    return self._values[idx]

  def __len__(self):
    return len(self._values)


class UnregisteredMapping(Mapping):

  def __init__(self, values):
    self._values = values

  def __getitem__(self, idx):
    return self._values[idx]

  def __len__(self):
    return len(self._values)

  def __iter__(self):
    return iter(self._values.keys())


class NamedTupleV0(NamedTuple):
  x: Any
  y: Any


class NamedTupleV1(NamedTuple):
  x: Any


@dataclasses.dataclass
class DataClassV0:
  x: Any
  y: Any


@dataclasses.dataclass
class DataClassV1:
  x: Any


@dataclasses.dataclass
class DataClassMulti:
  x: Any


def make_structtuple():

  @tfp.distributions.JointDistributionCoroutine
  def model():
    yield tfp.distributions.Normal(0., 1., name='x')

  return model.sample(seed=jax.random.PRNGKey(0))


class TreeTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('scalar', 0),
      ('string', 'abc'),
      ('list', [1, 2]),
      ('tuple', (1, 2)),
      ('dict', {
          'a': 1,
          'b': 2
      }),
      ('immutabledict', immutabledict.immutabledict({
          'a': 1,
          'b': 2
      })),
      ('flax_frozendict', flax.core.frozen_dict.FrozenDict({
          'a': 1,
          'b': 2
      })),
      ('dict_int_keys', {
          1: 2,
          3: 4,
      }),
      ('dict_enum_keys', {
          RegisteredEnum.X: 1,
          RegisteredEnum.Y: 2,
      }),
      ('dict_tuple_keys', {
          (1, 2): 1,
          (3, 4): 2,
      }),
      ('set', set([1, 2])),
      ('frozenset', frozenset([1, 2])),
      ('namedtuple', RegisteredNamedTuple(1, 2)),
      ('dataclass', RegisteredDataClass(1, 2)),
      ('numpy_array', np.arange(3)),
      ('structtuple', make_structtuple, True, True),
      ('jax_array', lambda: jnp.arange(3), True),
      ('jax_array_b16', lambda: jnp.array([1., 2.], jnp.bfloat16), True),
      ('nested',
       RegisteredNamedTuple(
           RegisteredNamedTuple([1, 2], {'a': 3}), np.zeros(100))),
      ('nested_dataclass',
       RegisteredDataClass(RegisteredDataClass([1, 2], {'a': 3}), 1)),
      (
          'enum',
          RegisteredEnum.X,
      ),
  )
  def test_roundtrip(self,
                     tree,
                     need_numpy_lhs=False,
                     compare_type_names=False):
    if callable(tree):
      tree = tree()
    path = self.create_tempdir()
    tree_path = os.path.join(path, 'tree')

    global_registry.save_tree(tree, tree_path)
    out_tree = global_registry.load_tree(tree_path)

    if need_numpy_lhs:
      tree = jax.tree.map(np.asarray, tree)
    if compare_type_names:
      self.assertEqual(type(tree).__name__, type(out_tree).__name__)
    else:
      self.assertIs(type(tree), type(out_tree))
    self.assertAllEqualNested(tree, out_tree)

  def test_deferred(self):
    tree = {
        'a': np.arange(10),
        'b': np.arange(1000).reshape([20, 50]).astype(np.float32)
    }
    path = self.create_tempdir()
    tree_path = os.path.join(path, 'tree')

    global_registry.save_tree(tree, tree_path)
    out_tree = global_registry.load_tree(tree_path, {'defer_numpy': True})
    self.assertIsInstance(out_tree['a'], np.ndarray)
    self.assertIsInstance(out_tree['b'], tree2.DeferredNumpyArray)
    self.assertEqual(out_tree['b'].dtype, np.float32)
    self.assertEqual(out_tree['b'].shape, (20, 50))

    array = np.array(out_tree['b'])
    self.assertEqual(array[19, 49], 999)

  def test_unregistered_sequence(self):
    path = self.create_tempdir()
    tree_path = os.path.join(path, 'tree')
    tree = UnregisteredSequence([1, 2])

    global_registry.save_tree(tree, tree_path)
    out_tree = global_registry.load_tree(tree_path)

    self.assertIsInstance(out_tree, list)
    self.assertEqual(out_tree, list(tree))

  def test_unregistered_mapping(self):
    path = self.create_tempdir()
    tree_path = os.path.join(path, 'tree')
    tree = UnregisteredMapping({'x': 1, 'y': 2})

    global_registry.save_tree(tree, tree_path)
    out_tree = global_registry.load_tree(tree_path)

    self.assertIsInstance(out_tree, dict)
    self.assertEqual(out_tree, dict(tree))

  def test_unregistered_namedtuple(self):
    path = self.create_tempdir()
    tree_path = os.path.join(path, 'tree')
    tree = UnregisteredNamedTuple(x=1, y=2)

    global_registry.save_tree(tree, tree_path)
    out_tree = global_registry.load_tree(tree_path)

    self.assertAllEqualNested(out_tree, tree)

  def test_unregistered_dataclass(self):
    path = self.create_tempdir()
    tree_path = os.path.join(path, 'tree')
    tree = UnregisteredDataClass(x=1, y=2)

    global_registry.save_tree(tree, tree_path)
    out_tree = global_registry.load_tree(tree_path)

    self.assertAllEqualNested(
        dataclasses.asdict(out_tree), dataclasses.asdict(tree))

  def test_namedtuple_unknown_field(self):
    path = self.create_tempdir()
    tree_path = os.path.join(path, 'tree')

    reg1 = tree2.Registry()
    reg1.register_namedtuple_type('test.namedtuple')(NamedTupleV0)

    tree = NamedTupleV0(x=1, y=2)
    reg1.save_tree(tree, tree_path)

    # Simulate changing the definition of the type and loading an old serialized
    # copy.
    reg2 = tree2.Registry()
    reg2.register_namedtuple_type('test.namedtuple')(NamedTupleV1)
    out_tree = reg2.load_tree(tree_path)

    self.assertAllEqual({'x': 1}, out_tree._asdict())

  def test_dataclass_unknown_field(self):
    path = self.create_tempdir()
    tree_path = os.path.join(path, 'tree')

    reg1 = tree2.Registry()
    reg1.register_dataclass_type('test.dataclass')(DataClassV0)

    tree = DataClassV0(x=1, y=2)
    reg1.save_tree(tree, tree_path)

    # Simulate changing the definition of the type and loading an old serialized
    # copy.
    reg2 = tree2.Registry()
    reg2.register_dataclass_type('test.dataclass')(DataClassV1)
    out_tree = reg2.load_tree(tree_path)

    self.assertAllEqual({'x': 1}, dataclasses.asdict(out_tree))

  def test_multiple_tags(self):
    path = self.create_tempdir()
    tree_path = os.path.join(path, 'tree')

    reg1 = tree2.Registry()
    reg1.register_dataclass_type('test.multi_old')(DataClassMulti)

    tree = DataClassMulti(x=1)
    reg1.save_tree(tree, tree_path)

    # Simulate changing the tag, but keeping backwards loading compatibility.
    reg2 = tree2.Registry()
    reg2.register_dataclass_type(['test.multi_new', 'test.multi_old'])(
        DataClassMulti)
    out_tree = reg2.load_tree(tree_path)
    reg2.save_tree(tree, tree_path)

    # Verify that the tree got saved with `multi_new` tag.
    reg3 = tree2.Registry()
    reg3.register_dataclass_type('test.multi_new')(DataClassMulti)
    out_tree = reg3.load_tree(tree_path)

    self.assertAllEqual({'x': 1}, dataclasses.asdict(out_tree))

  def test_interactive_mode(self):
    reg1 = tree2.Registry()
    reg1.auto_register_type('test.UnregisteredNamedTuple')(
        UnregisteredNamedTuple
    )
    with self.assertRaisesRegex(TypeError, 'already registered'):
      reg1.auto_register_type('test.UnregisteredNamedTuple')(
          UnregisteredNamedTuple
      )
    reg1.interactive_mode = True
    reg1.auto_register_type('test.UnregisteredNamedTuple')(
        UnregisteredNamedTuple
    )

if __name__ == '__main__':
  test_util.main()
