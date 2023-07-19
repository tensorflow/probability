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

import collections
from collections import abc as collections_abc
import types
import tree as dm_tree

# pylint: disable=g-import-not-at-top
try:
  import wrapt
  ObjectProxy = wrapt.ObjectProxy
except ImportError:

  class ObjectProxy(object):
    """Stub-class for `wrapt.ObjectProxy``."""


JAX_MODE = False


_SHALLOW_TREE_HAS_INVALID_KEYS = (
    "The shallow_tree's keys are not a subset of the input_tree's keys. The "
    'shallow_tree has the following keys that are not in the input_tree: {}.'
)

_STRUCTURES_HAVE_MISMATCHING_TYPES = (
    "The two structures don't have the same sequence type. Input structure "
    'has type {input_type}, while shallow structure has type {shallow_type}.'
)

_STRUCTURES_HAVE_MISMATCHING_LENGTHS = (
    "The two structures don't have the same sequence length. Input "
    'structure has length {input_length}, while shallow structure has length '
    '{shallow_length}.'
)

_INPUT_TREE_SMALLER_THAN_SHALLOW_TREE = (
    'The input_tree has fewer elements than the shallow_tree. Input structure '
    'has length {input_size}, while shallow structure has length '
    '{shallow_size}.'
)

_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ = (
    'If shallow structure is a sequence, input must also be a sequence. '
    'Input has type: {}.'
)


assert_same_structure = dm_tree.assert_same_structure


def assert_shallow_structure(shallow_tree, input_tree, check_types=True):
  """Asserts that `shallow_tree` is a shallow structure of `input_tree`.

  That is, this function recursively tests if each key in shallow_tree has its
  corresponding key in input_tree.

  Examples:

  The following code will raise an exception:

  ```python
  shallow_tree = {"a": "A", "b": "B"}
  input_tree = {"a": 1, "c": 2}
  assert_shallow_structure(shallow_tree, input_tree)
  ```

  ```none
  ValueError: The shallow_tree's keys are not a subset of the input_tree's ...
  ```

  The following code will raise an exception:

  ```python
  shallow_tree = ["a", "b"]
  input_tree = ["c", ["d", "e"], "f"]
  assert_shallow_structure(shallow_tree, input_tree)
  ```

  ```
  ValueError: The two structures don't have the same sequence length.
  ```

  By setting check_types=False, we drop the requirement that corresponding
  nodes in shallow_tree and input_tree have to be the same type. Sequences
  are treated equivalently to Mappables that map integer keys (indices) to
  values. The following code will therefore not raise an exception:

  ```python
  assert_shallow_structure({0: "foo"}, ["foo"], check_types=False)
  ```

  Args:
    shallow_tree: an arbitrarily nested structure.
    input_tree: an arbitrarily nested structure.
    check_types: if `True` (default) the sequence types of `shallow_tree` and
      `input_tree` have to be the same.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`. Only raised if `check_types` is `True`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  if is_nested(shallow_tree):
    if not is_nested(input_tree):
      raise TypeError(
          _IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree))
      )

    if isinstance(shallow_tree, ObjectProxy):
      shallow_type = type(shallow_tree.__wrapped__)
    else:
      shallow_type = type(shallow_tree)

    if check_types and not isinstance(input_tree, shallow_type):
      # Duck-typing means that nest should be fine with two different
      # namedtuples with identical name and fields.
      shallow_is_namedtuple = _is_namedtuple(shallow_tree)
      input_is_namedtuple = _is_namedtuple(input_tree)
      if shallow_is_namedtuple and input_is_namedtuple:
        # pylint: disable=protected-access
        if type(shallow_tree) is not type(input_tree):
          raise TypeError(
              _STRUCTURES_HAVE_MISMATCHING_TYPES.format(
                  input_type=type(input_tree), shallow_type=shallow_type
              )
          )
        # pylint: enable=protected-access
      elif not (isinstance(shallow_tree, collections_abc.Mapping)
                and isinstance(input_tree, collections_abc.Mapping)):
        raise TypeError(
            _STRUCTURES_HAVE_MISMATCHING_TYPES.format(
                input_type=type(input_tree), shallow_type=shallow_type
            )
        )

    if len(input_tree) != len(shallow_tree):
      raise ValueError(
          _STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
              input_length=len(input_tree), shallow_length=len(shallow_tree)
          )
      )
    elif len(input_tree) < len(shallow_tree):
      raise ValueError(
          _INPUT_TREE_SMALLER_THAN_SHALLOW_TREE.format(
              input_size=len(input_tree), shallow_size=len(shallow_tree)
          )
      )

    shallow_iter = _yield_sorted_items(shallow_tree)
    input_iter = _yield_sorted_items(input_tree)

    def get_matching_input_branch(shallow_key):
      for input_key, input_branch in input_iter:
        if input_key == shallow_key:
          return input_branch

      raise ValueError(_SHALLOW_TREE_HAS_INVALID_KEYS.format([shallow_key]))

    for shallow_key, shallow_branch in shallow_iter:
      input_branch = get_matching_input_branch(shallow_key)
      assert_shallow_structure(
          shallow_branch, input_branch, check_types=check_types)


def flatten(structure, expand_composites=False):
  """Add expand_composites support for JAX."""
  if expand_composites and JAX_MODE:
    from jax import tree_util  # pylint: disable=g-import-not-at-top
    return tree_util.tree_leaves(structure)
  return dm_tree.flatten(structure)


def flatten_up_to(*args, expand_composites=False, **kwargs):
  # Internal `tree` does not accept check_types here; see b/198436438.
  # Apparently the open-source version of same still does.
#   kwargs.pop('check_types', None)  # DisableOnExport
  if expand_composites:
    raise NotImplementedError(
        '`expand_composites=True` is not supported in JAX.')
  return dm_tree.flatten_up_to(*args, **kwargs)


def flatten_with_joined_string_paths(structure, separator='/',
                                     expand_composites=False):
  """Returns a list of (string path, data element) tuples.

  The order of tuples produced matches that of `nest.flatten`. This allows you
  to flatten a nested structure while keeping information about where in the
  structure each data element was located. See `nest.yield_flat_paths`
  for more information.

  Args:
    structure: the nested structure to flatten.
    separator: string to separate levels of hierarchy in the results, defaults
      to '/'.
    expand_composites: Python bool included for compatibility; `True` values
      are not supported.

  Returns:
    A list of (string, data element) tuples.
  """

  def stringify_and_join(path_elements):
    return separator.join(str(path_element) for path_element in path_elements)
  if expand_composites:
    raise NotImplementedError(
        '`expand_composites=True` is not supported in JAX.')
  return [(stringify_and_join(pe), v)
          for pe, v in dm_tree.flatten_with_path(structure)]


def flatten_with_tuple_paths(structure, expand_composites=False):
  if expand_composites:
    raise NotImplementedError(
        '`expand_composites=True` is not supported in JAX.')
  return dm_tree.flatten_with_path(structure)


def flatten_with_tuple_paths_up_to(shallow_structure,
                                   input_structure,
                                   check_types=True,
                                   expand_composites=False):
  if expand_composites:
    raise NotImplementedError(
        '`expand_composites=True` is not supported in JAX.')
  return dm_tree.flatten_with_path_up_to(shallow_structure,
                                         input_structure,
                                         check_types)


_FALSE_SENTINEL = object()


def get_traverse_shallow_structure(traverse_fn, structure):
  """Generates a shallow structure from a `traverse_fn` and `structure`.

  `traverse_fn` must accept any possible subtree of `structure` and return
  a depth=1 structure containing `True` or `False` values, describing which
  of the top-level subtrees may be traversed.  It may also
  return scalar `True` or `False` 'traversal is OK / not OK for all subtrees.'

  Examples are available in the unit tests (nest_test.py).

  Args:
    traverse_fn: Function taking a substructure and returning either a scalar
      `bool` (whether to traverse that substructure or not) or a depth=1 shallow
      structure of the same type, describing which parts of the substructure to
      traverse.
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

  def outer_traverse_fn(subtree):
    res = traverse_fn(subtree)
    if is_nested(res):

      def inner_traverse_fn(do_traverse, subtree):
        if do_traverse:
          return dm_tree.traverse(outer_traverse_fn, subtree)
        else:
          return _FALSE_SENTINEL

      return dm_tree.map_structure_up_to(res, inner_traverse_fn, res, subtree)
    else:
      return None if res else _FALSE_SENTINEL

  return map_structure(lambda x: False if x is _FALSE_SENTINEL else True,
                       dm_tree.traverse(outer_traverse_fn, structure))


def _is_namedtuple(v):
  return isinstance(v, tuple) and hasattr(v, '_fields')


is_nested = dm_tree.is_nested


def map_structure(func, *structure, **kwargs):
  """Add expand_composites support for JAX."""
  expand_composites = kwargs.pop('expand_composites', False)
  if expand_composites and JAX_MODE:
    from jax import tree_util  # pylint: disable=g-import-not-at-top
    return tree_util.tree_map(func, *structure)
  return dm_tree.map_structure(func, *structure, **kwargs)


def map_structure_up_to(shallow_structure, func, *structures, **kwargs):
  return map_structure_with_tuple_paths_up_to(
      shallow_structure,
      lambda _, *args: func(*args),  # Discards path.
      *structures,
      **kwargs)


def map_structure_with_tuple_paths(func, *structures, **kwargs):
  return map_structure_with_tuple_paths_up_to(structures[0], func, *structures,
                                              **kwargs)


def map_structure_with_tuple_paths_up_to(shallow_structure, func, *structures,
                                         expand_composites=False, **kwargs):
  """Wraps nest.map_structure_with_path_up_to, with structure/type checking."""
  if not structures:
    raise ValueError('Cannot map over no sequences')

  # Internal `tree` does not accept check_types here; see b/198436438.
  check_types = kwargs.get('check_types', True)
  kwargs.pop('check_types', None)

  if expand_composites:
    raise NotImplementedError(
        '`expand_composites=True` is not supported in JAX.')

  for input_tree in structures:
    assert_shallow_structure(
        shallow_structure, input_tree, check_types=check_types)
  return dm_tree.map_structure_with_path_up_to(
      shallow_structure, func, *structures, **kwargs)


def pack_sequence_as(structure, flat_sequence, **kwargs):
  expand_composites = kwargs.pop('expand_composites', False)
  if expand_composites and JAX_MODE:
    from jax import tree_util  # pylint: disable=g-import-not-at-top
    return tree_util.tree_unflatten(
        tree_util.tree_structure(structure), flat_sequence)
  return dm_tree.unflatten_as(structure, flat_sequence)


def _sequence_like(instance, args):
  """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, `namedtuple`, `dict`, or
      `collections.OrderedDict`.
    args: elements to be converted to the `instance` type.

  Returns:
    `args` with the type of `instance`.
  """
  if isinstance(instance, (dict, collections_abc.Mapping)):
    # Pack dictionaries in a deterministic order by sorting the keys.
    # Notice this means that we ignore the original order of `OrderedDict`
    # instances. This is intentional, to avoid potential bugs caused by mixing
    # ordered and plain dicts (e.g., flattening a dict but using a
    # corresponding `OrderedDict` to pack it back).
    result = dict(zip(_sorted(instance), args))
    keys_and_values = ((key, result[key]) for key in instance)
    if isinstance(instance, collections.defaultdict):
      # `defaultdict` requires a default factory as the first argument.
      return type(instance)(instance.default_factory, keys_and_values)
    elif isinstance(instance, types.MappingProxyType):
      # MappingProxyType requires a dict to proxy to.
      return type(instance)(dict(keys_and_values))
    else:
      return type(instance)(keys_and_values)
  elif isinstance(instance, collections_abc.MappingView):
    # We can't directly construct mapping views, so we create a list instead
    return list(args)
  elif _is_namedtuple(instance):
    if isinstance(instance, ObjectProxy):
      instance_type = type(instance.__wrapped__)
    else:
      instance_type = type(instance)
    return instance_type(*args)
  elif isinstance(instance, ObjectProxy):
    # For object proxies, first create the underlying type and then re-wrap it
    # in the proxy type.
    return type(instance)(_sequence_like(instance.__wrapped__, args))
  else:
    # Not a namedtuple
    return type(instance)(args)


def yield_flat_paths(nest, expand_composites=False):
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
    expand_composites: Python bool included for compatibility; `True` values
      are not supported.
  Yields:
    Tuples containing index or key values which form the path to a specific
      leaf value in the nested structure.
  """
  for k, _ in flatten_with_tuple_paths(nest,
                                       expand_composites=expand_composites):
    yield k


def _yield_value(iterable):
  for _, v in _yield_sorted_items(iterable):
    yield v


def _sorted(dictionary):
  """Returns a sorted list of the dict keys, with error if keys not sortable."""
  try:
    return sorted(dictionary)
  except TypeError:
    raise TypeError('tree only supports dicts with sortable keys.')


def _yield_sorted_items(iterable):
  """Yield (key, value) pairs for `iterable` in a deterministic order.

  For Sequences, the key will be an int, the array index of a value.
  For Mappings, the key will be the dictionary key.
  For objects (e.g. namedtuples), the key will be the attribute name.

  In all cases, the keys will be iterated in sorted order.

  Args:
    iterable: an iterable.

  Yields:
    The iterable's (key, value) pairs, in order of sorted keys.
  """
  if not is_nested(iterable):
    raise ValueError(f'{iterable} is not an iterable')

  top_structure = dm_tree.traverse(lambda x: None  # pylint: disable=g-long-lambda
                                   if x is iterable else False, iterable)

  for p, v in dm_tree.flatten_with_path_up_to(top_structure, iterable):
    yield p[0], v
