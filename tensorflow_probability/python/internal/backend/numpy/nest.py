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
"""Reimplementation of `tensorflow.python.util.nest` using dm-tree.

This module defines aliases which allow to use `tree` as drop-in replacement
for TensorFlow's `nest`.

Usage:
```python
from tensorflow_probability.python.internal.backend.numpy import nest
nest.pack_sequence_as(..., ...)
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# pylint: disable=unused-import
from tree import _assert_shallow_structure
from tree import _IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ
from tree import _INPUT_TREE_SMALLER_THAN_SHALLOW_TREE
from tree import _is_attrs
from tree import _sequence_like
from tree import _SHALLOW_TREE_HAS_INVALID_KEYS
from tree import _STRUCTURES_HAVE_MISMATCHING_LENGTHS
from tree import _STRUCTURES_HAVE_MISMATCHING_TYPES
from tree import _yield_flat_up_to
from tree import _yield_value
from tree import assert_same_structure
from tree import flatten
from tree import flatten_up_to
from tree import flatten_with_path
from tree import flatten_with_path_up_to
from tree import is_nested
from tree import map_structure as dm_tree_map_structure
from tree import map_structure_up_to
from tree import map_structure_with_path
from tree import map_structure_with_path_up_to
from tree import unflatten_as
# pylint: enable=unused-import


JAX_MODE = False


def map_structure(func, *structure, **kwargs):
  """Add expand_composites support for JAX."""
  expand_composites = kwargs.pop('expand_composites', False)
  if expand_composites and JAX_MODE:
    from jax import tree_util  # pylint: disable=g-import-not-at-top
    leaves = [func(*leaves) for leaves in zip(
        *(tree_util.tree_leaves(struct) for struct in structure))]
    return tree_util.tree_unflatten(tree_util.tree_structure(structure[0]),
                                    leaves)
  return dm_tree_map_structure(func, *structure, **kwargs)


def _is_namedtuple(v):
  return isinstance(v, tuple) and hasattr(v, '_fields')


def apply_to_structure(branch_fn, leaf_fn, structure):
  """`apply_to_structure` applies branch_fn and leaf_fn to branches and leaves.

  This function accepts two separate callables depending on whether the
  structure is a sequence.

  Args:
    branch_fn: A function to call on a struct if is_nested(struct) is `True`.
    leaf_fn: A function to call on a struct if is_nested(struct) is `False`.
    structure: A nested structure containing arguments to be applied to.

  Returns:
    A nested structure of function outputs.

  Raises:
    TypeError: If `branch_fn` or `leaf_fn` is not callable.
    ValueError: If no structure is provided.
  """
  if not callable(leaf_fn):
    raise TypeError('leaf_fn must be callable, got: %s' % leaf_fn)

  if not callable(branch_fn):
    raise TypeError('branch_fn must be callable, got: %s' % branch_fn)

  if not is_nested(structure):
    return leaf_fn(structure)

  processed = branch_fn(structure)

  new_structure = [
      apply_to_structure(branch_fn, leaf_fn, value)
      for value in _yield_value(processed)
  ]
  return _sequence_like(processed, new_structure)


def assert_shallow_structure(shallow_tree, input_tree, check_types=True):
  return _assert_shallow_structure(shallow_tree, input_tree, check_types)


def flatten_dict_items(dictionary):
  """Returns a dictionary with flattened keys and values.

  This function flattens the keys and values of a dictionary, which can be
  arbitrarily nested structures, and returns the flattened version of such
  structures:

  >>> example_dictionary = {(4, 5, (6, 8)): ('a', 'b', ('c', 'd'))}
  >>> result = {4: 'a', 5: 'b', 6: 'c', 8: 'd'}
  >>> assert tree.flatten_dict_items(example_dictionary) == result

  The input dictionary must satisfy two properties:

  1. Its keys and values should have the same exact nested structure.
  2. The set of all flattened keys of the dictionary must not contain repeated
     keys.

  Args:
    dictionary: the dictionary to zip

  Returns:
    The zipped dictionary.

  Raises:
    TypeError: If the input is not a dictionary.
    ValueError: If any key and value do not have the same structure layout, or
      if keys are not unique.
  """
  if not isinstance(dictionary, (dict, collections.Mapping)):
    raise TypeError('input must be a dictionary')

  flat_dictionary = {}
  for i, v in dictionary.items():
    if not is_nested(i):
      if i in flat_dictionary:
        raise ValueError(
            'Could not flatten dictionary: key %s is not unique.' % i)
      flat_dictionary[i] = v
    else:
      flat_i = flatten(i)
      flat_v = flatten(v)
      if len(flat_i) != len(flat_v):
        raise ValueError(
            'Could not flatten dictionary. Key had %d elements, but value had '
            '%d elements. Key: %s, value: %s.'
            % (len(flat_i), len(flat_v), flat_i, flat_v))
      for new_i, new_v in zip(flat_i, flat_v):
        if new_i in flat_dictionary:
          raise ValueError(
              'Could not flatten dictionary: key %s is not unique.'
              % (new_i,))
        flat_dictionary[new_i] = new_v
  return flat_dictionary


def flatten_with_joined_string_paths(structure, separator='/'):
  """Returns a list of (string path, data element) tuples.

  The order of tuples produced matches that of `nest.flatten`. This allows you
  to flatten a nested structure while keeping information about where in the
  structure each data element was located. See `nest.yield_flat_paths`
  for more information.

  Args:
    structure: the nested structure to flatten.
    separator: string to separate levels of hierarchy in the results, defaults
      to '/'.

  Returns:
    A list of (string, data element) tuples.
  """
  flat_paths = yield_flat_paths(structure)
  def stringify_and_join(path_elements):
    return separator.join(str(path_element) for path_element in path_elements)
  flat_string_paths = [stringify_and_join(path) for path in flat_paths]
  return list(zip(flat_string_paths, flatten(structure)))


def flatten_with_tuple_paths(structure):
  return flatten_with_path(structure)


def flatten_with_tuple_paths_up_to(shallow_structure,
                                   input_structure,
                                   check_types=True):
  return flatten_with_path_up_to(shallow_structure, input_structure,
                                 check_types)


def map_structure_with_tuple_paths(func, *structures, **kwargs):
  return map_structure_with_path(func, *structures, **kwargs)


def map_structure_with_tuple_paths_up_to(func, *structures, **kwargs):
  return map_structure_with_path_up_to(func, *structures, **kwargs)


def pack_sequence_as(structure, flat_sequence):
  return unflatten_as(structure, flat_sequence)


def get_traverse_shallow_structure(traverse_fn, structure):
  """Generates a shallow structure from a `traverse_fn` and `structure`.

  `traverse_fn` must accept any possible subtree of `structure` and return
  a depth=1 structure containing `True` or `False` values, describing which
  of the top-level subtrees may be traversed.  It may also
  return scalar `True` or `False` 'traversal is OK / not OK for all subtrees.'

  Examples are available in the unit tests (nest_test.py).

  Args:
    traverse_fn: Function taking a substructure and returning either a scalar
      `bool` (whether to traverse that substructure or not) or a depth=1
      shallow structure of the same type, describing which parts of the
      substructure to traverse.
    structure: The structure to traverse.

  Returns:
    A shallow structure containing python bools, which can be passed to
    `map_up_to` and `flatten_up_to`.

  Raises:
    TypeError: if `traverse_fn` returns a sequence for a non-sequence input,
      or a structure with depth higher than 1 for a sequence input,
      or if any leaf values in the returned structure or scalar are not type
      `bool`.
  """
  to_traverse = traverse_fn(structure)
  if not is_nested(structure):
    if not isinstance(to_traverse, bool):
      raise TypeError('traverse_fn returned structure: %s for non-structure: %s'
                      % (to_traverse, structure))
    return to_traverse
  level_traverse = []
  if isinstance(to_traverse, bool):
    if not to_traverse:
      # Do not traverse this substructure at all.  Exit early.
      return False
    else:
      # Traverse the entire substructure.
      for branch in _yield_value(structure):
        level_traverse.append(
            get_traverse_shallow_structure(traverse_fn, branch))
  elif not is_nested(to_traverse):
    raise TypeError('traverse_fn returned a non-bool scalar: %s for input: %s'
                    % (to_traverse, structure))
  else:
    # Traverse some subset of this substructure.
    assert_shallow_structure(to_traverse, structure)
    for t, branch in zip(_yield_value(to_traverse), _yield_value(structure)):
      if not isinstance(t, bool):
        raise TypeError(
            'traverse_fn didn\'t return a depth=1 structure of bools.  saw: %s '
            ' for structure: %s' % (to_traverse, structure))
      if t:
        level_traverse.append(
            get_traverse_shallow_structure(traverse_fn, branch))
      else:
        level_traverse.append(False)
  return _sequence_like(structure, level_traverse)


def is_sequence(structure):
  return is_nested(structure)


def map_structure_with_paths(func, *structure, **kwargs):
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(path, x[0], x[1], ..., **kwargs)` where x[i] is an entry in
  `structure[i]` and `path` is the common path to x[i] in the structures.  All
  structures in `structure` must have the same arity, and the return value will
  contain the results with the same structure layout.

  Args:
    func: A callable with the signature func(path, *values, **kwargs) that is
      evaluated on the leaves of the structure.
    *structure: A variable number of compatible structures to process.
    **kwargs: Optional kwargs to be passed through to func. Special kwarg
      `check_types` is not passed to func, but instead determines whether the
      types of iterables within the structures have to be same (e.g.
      `map_structure(func, [1], (1,))` raises a `TypeError` exception).
      To allow this set this argument to `False`.

  Returns:
    A structure of the same form as the input structures whose leaves are the
    result of evaluating func on corresponding leaves of the input structures.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    TypeError: If `check_types` is not `False` and the two structures differ in
      the type of sequence in any of their substructures.
    ValueError: If no structures are provided.
  """
  def wrapper_func(tuple_path, *inputs, **kwargs):
    string_path = '/'.join(str(s) for s in tuple_path)
    return func(string_path, *inputs, **kwargs)

  return map_structure_with_path_up_to(structure[0], wrapper_func,
                                       *structure, **kwargs)


def yield_flat_paths(nest):
  """Yields paths for some nested structure.

  Paths are lists of objects which can be str-converted, which may include
  integers or other types which are used as indices in a dict.

  The flat list will be in the corresponding order as if you called
  `flatten` on the structure. This is handy for naming Tensors such
  the TF scope structure matches the tuple structure.

  E.g. if we have a tuple `value = Foo(a=3, b=Bar(c=23, d=42))`

  >>> Foo = collections.namedtuple('Foo', ['a', 'b'])
  >>> Bar = collections.namedtuple('Bar', ['c', 'd'])
  >>> value = Foo(a=3, b=Bar(c=23, d=42))

  >>> flatten(value)
  [3, 23, 42]

  >>> list(yield_flat_paths(value))
  [('a',), ('b', 'c'), ('b', 'd')]

  >>> list(yield_flat_paths({'a': [3]}))
  [('a', 0)]

  >>> list(yield_flat_paths({'a': 3}))
  [('a',)]

  Args:
    nest: the value to produce a flattened paths list for.

  Yields:
    Tuples containing index or key values which form the path to a specific
      leaf value in the nested structure.
  """
  for k, _ in _yield_flat_up_to(nest, nest):
    yield k
