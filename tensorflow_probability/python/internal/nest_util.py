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
"""Utilities dealing with nested structures."""

import collections

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'broadcast_structure',
    'call_fn',
    'cast_structure',
    'expand_as_args',
    'map_structure_with_named_args'
]

_is_namedtuple = nest._is_namedtuple  # pylint: disable=protected-access


UNSPECIFIED = object()


def broadcast_structure(to_structure, from_structure):
  """Broadcasts `from_structure` to `to_structure`.

  This is useful for downstream usage of `zip` or `tf.nest.map_structure`.

  If `from_structure` is a singleton, it is tiled to match the structure of
  `to_structure`. Note that the elements in `from_structure` are not copied if
  this tiling occurs.

  Args:
    to_structure: A structure.
    from_structure: A structure.

  Returns:
    new_from_structure: Same structure as `to_structure`.

  #### Example:

  ```python
  a_structure = ['a', 'b', 'c']
  b_structure = broadcast_structure(a_structure, 'd')
  # -> ['d', 'd', 'd']
  c_structure = tf.nest.map_structure(
      lambda a, b: a + b, a_structure, b_structure)
  # -> ['ad', 'bd', 'cd']
  ```
  """
  from_parts = tf.nest.flatten(from_structure)
  if len(from_parts) == 1:
    from_structure = tf.nest.map_structure(lambda _: from_parts[0],
                                           to_structure)
  return from_structure


def cast_structure(value, structure):
  """Cast a structure."""
  if tf.nest.is_nested(structure):
    if _is_namedtuple(structure):  # pylint: disable=protected-access
      return type(structure)(*value)
    else:
      return type(structure)(value)
  return value


def map_structure_with_named_args(func,
                                  *structures,
                                  _check_types=True,  # pylint: disable=invalid-name
                                  _expand_composites=False,  # pylint: disable=invalid-name
                                  _up_to=UNSPECIFIED,  # pylint: disable=invalid-name
                                  **named_structures):
  """Calls `nest.map_structure` with named args.

  Args:
    func: a callable that accepts one or more named arguments.
    *structures: Structures of arguments passed positionally to `func`.
    _check_types: Forwarded as `map_structure(..., check_types=_check_types)`.
    _expand_composites: Forwarded as
      `map_structure(..., expand_composites=_expand_composites)`.
    _up_to: Optional shallow structure to map up to. If provided,
      `nest.map_structure_up_to` is called rather than `nest.map_structure`.
      Default value: `UNSPECIFIED`.
    **named_structures: Structures of arguments passed by name to `func`.
  Returns:
    A new structure matching that of the input structures (or the shallow
      structure `_up_to`, if specified), in which each element is computed
      by applying `func` to the corresponding elements of the input structures.

  #### Examples

  ```python
  func = lambda x, y: 2 * x + 3 * y

  map_structure_with_named_args(func, [1, 2], [10, 11])
  # ==> [32, 37]

  map_structure_with_named_args(func, [1, 2], y=[10, 11])
  # ==> [32, 37]

  map_structure_with_named_args(func, x=[1, 2], y=[10, 11])
  # ==> [32, 37]

  map_structure_with_named_args(func, [10, 11], x=[1, 2])
  # ==> TypeError: <lambda>() got multiple values for argument 'x'.
  ```

  """
  names, named_values = (zip(*named_structures.items())
                         if named_structures else ((), ()))
  # Wrapper function that takes positional args and passes keyword args.
  def kwarg_passing_fn(*leaf_values):
    return func(*leaf_values[:len(structures)],
                **dict(zip(names, leaf_values[len(structures):])))

  map_fn = (nest.map_structure if _up_to is UNSPECIFIED
            else lambda *a, **kw: nest.map_structure_up_to(_up_to, *a, **kw))
  return map_fn(kwarg_passing_fn,
                *(structures + named_values),
                check_types=_check_types,
                expand_composites=_expand_composites)


def map_structure_coroutine(coroutine,
                            *structures,
                            _expand_composites=False,  # pylint: disable=invalid-name
                            _up_to=UNSPECIFIED,  # pylint: disable=invalid-name
                            _with_tuple_paths=False,  # pylint: disable=invalid-name
                            **named_structures):
  # pylint: disable=g-doc-return-or-yield
  """Invokes a coroutine multiple times with args from provided structures.

  This is semantically identical to `map_structure_with_named_args`, except
  that the first argument is a generator or coroutine (a callable whose body
  contains `yield` statements) rather than a function. This is invoked with
  arguments from the provided structure(s), thus defining an outer generator/
  coroutine that `yield`s values in sequence from each call to the inner
  `coroutine`.

  The argument structures are traversed, and the coroutine is invoked, in
  the order defined by `tf.nest.flatten`. A stripped-down implementation of
  the core logic is as follows:

  ```python
  def map_structure_coroutine(coroutine, *structures):
    flat_results = []
    for args in zip(*[tf.nest.flatten(s) for s in structures]):
      retval = yield from coroutine(*args)
      flat_results.append(retval)
    return tf.nest.pack_sequence_as(structures[0], flat_results)
  ```

  Args:
    coroutine: a generator/coroutine callable that accepts one or more named
      arguments.
    *structures: Structures of arguments passed positionally to `coroutine`.
    _expand_composites: Forwarded as
      `tf.nest.flatten(..., expand_composites=_expand_composites)`.
    _up_to: Optional shallow structure to map up to. If provided,
      `nest.map_structure_up_to` is called rather than `nest.map_structure`.
      Default value: `UNSPECIFIED`.
    _with_tuple_paths: Python bool. If `True`, the first argument to `coroutine`
      is a tuple path to the current leaf of the argument structure(s).
      Default value: `False`.
    **named_structures: Structures of arguments passed by name to `coroutine`.
  Yields:
    Values `yield`ed by each invocation of `coroutine`, with invocations in
      order corresponding to `tf.nest.flatten`.
  Returns:
    A new structure matching that of the input structures (or the shallow
      structure `_up_to`, if specified), in which each element is the return
      value from applying `coroutine` to the corresponding elements of the input
      structures.

  ## Examples

  A JointDistributionCoroutine may define a reusable submodel as its own
  coroutine, for example:

  ```python
  def horseshoe_prior(path, scale):
    # Auxiliary-variable representation of a horseshoe prior on sparse weights.
    name = ','.join(path)
    z = yield tfd.HalfCauchy(loc=0., scale=scale, name=name + '_z')
    w_noncentered = yield tfd.Normal(
        loc=0., scale=z, name=name + '_w_noncentered')
    return z * w_noncentered
  ```

  Note that this submodel yields two auxiliary random variables, and returns the
  sampled weight as a third value.

  Using `map_structure_coroutine` we can define a structure of such submodels,
  and collect their return values:

  ```
  @tfd.JointDistributionCoroutineAutoBatched
  def model():
    weights = yield from nest_util.map_structure_coroutine(
        horseshoe_prior,
        scale={'a': tf.ones([5]) * 100., 'b': tf.ones([2]) * 1e-2},
        _with_tuple_paths=True)
    # ==> `weights` is a dict of weight values.
    yield tfd.Deterministic(
        tf.sqrt(tf.norm(weights['a'])**2 + tf.norm(weights['b'])**2),
        name='weights_norm')

  print(model.event_shape)
  # ==> StructTuple(
  #       a_z=TensorShape([5]),
  #       a_w_noncentered=TensorShape([5]),
  #       b_z=TensorShape([2]),
  #       b_w_noncentered=TensorShape([2]),
  #       weights_norm=TensorShape([]))
  ```
  """
  # pylint: enable=g-doc-return-or-yield

  names, named_structure_values = (zip(*named_structures.items())
                                   if named_structures else ((), ()))
  all_structures = structures + named_structure_values
  result_structure = all_structures[0] if _up_to is UNSPECIFIED else _up_to
  flat_arg_structures = [
      nest.flatten_up_to(result_structure, s)
      for s in all_structures]

  if _with_tuple_paths:
    # Pass tuple paths as a first positional arg (before any provided args).
    flat_paths = nest.yield_flat_paths(result_structure,
                                       expand_composites=_expand_composites)
    flat_arg_structures = [list(flat_paths)] + flat_arg_structures
    num_positional_args = 1 + len(structures)
  else:
    num_positional_args = len(structures)

  flat_results = []
  for leaf_values in zip(*flat_arg_structures):
    result = yield from coroutine(
        *leaf_values[:num_positional_args],
        **dict(zip(names, leaf_values[num_positional_args:])))
    flat_results.append(result)

  return nest.pack_sequence_as(result_structure, flat_results)


def _force_leaf(struct):
  # Returns `True` if `struct` should be treated as a leaf, rather than
  # expanded/recursed into.
  return hasattr(struct, '_tfp_nest_expansion_force_leaf')


def _force_expand_as_args(struct):
  return hasattr(struct, '_tfp_nest_expansion_force_args')


def expand_as_args(args):
  """Returns `True` if `args` should be expanded as `*args`."""
  return ((isinstance(args, collections.abc.Sequence) and
           not _is_namedtuple(args) and not _force_leaf(args)) or
          _force_expand_as_args(args))


def _expand_as_kwargs(args):
  # Returns `True` if `args` should be expanded as `**args`.
  return isinstance(args, collections.abc.Mapping) and not _force_leaf(args)


def _maybe_convertible_to_tensor(struct):
  # Returns `True` if `struct` should be passed to `convert_to_tensor`.
  return not _is_namedtuple(struct) or _force_leaf(struct)


def _get_shallow_structure(struct):
  # Get a shallow version of struct where the children are replaced by
  # 'False'.
  return nest.get_traverse_shallow_structure(lambda s: s is struct, struct)


def _nested_convert_to_tensor(struct, dtype=None, name=None):
  """Eagerly converts struct to Tensor, recursing upon failure."""
  if dtype is not None or not tf.nest.is_nested(struct):
    return tf.convert_to_tensor(struct, dtype=dtype)

  if _maybe_convertible_to_tensor(struct):
    try:
      # Try converting the structure wholesale.
      return tf.convert_to_tensor(struct, name=name)
    except (ValueError, TypeError):
      # Unfortunately Eager/Graph mode don't agree on the error type.
      pass
  # Try converting all of its children.
  shallow_struct = _get_shallow_structure(struct)
  return nest.map_structure_up_to(
      shallow_struct, lambda s: _nested_convert_to_tensor(s, name=name), struct)


def convert_args_to_tensor(args, dtype=None, name=None):
  """Converts `args` to `Tensor`s.

  Use this when it is necessary to convert user-provided arguments that will
  then be passed to user-provided callables.

  When `dtype` is `None` this function behaves as follows:

  1A. If the top-level structure is a `list`/`tuple` but not a `namedtuple`,
      then it is left as is and only its elements are converted to `Tensor`s.

  2A. The sub-structures are converted to `Tensor`s eagerly. E.g. if `args` is
      `{'arg': [[1], [2]]}` it is converted to
      `{'arg': tf.constant([[1], [2]])}`. If the conversion fails, it will
      attempt to recurse into its children.

  When `dtype` is specified, it acts as both a structural and numeric type
  constraint. `dtype` can be a single `DType`, `None` or a nested collection
  thereof. The conversion rule becomes as follows:

  1B. The return value of this function will have the same structure as `dtype`.

  2B. If the leaf of `dtype` is a concrete `DType`, then the corresponding
      sub-structure in `args` is converted to a `Tensor`.

  3B. If the leaf of `dtype` is `None`, then the corresponding sub-structure is
      converted eagerly as described in the rule 2A above.

  Args:
    args: Arguments to convert to `Tensor`s.
    dtype: Optional structure/numeric type constraint.
    name: Optional name-scope to use.

  Returns:
    args: Converted `args`.

  #### Examples.

  This table shows some useful conversion cases. `T` means `Tensor`, `NT` means
  `namedtuple` and `CNT` means a `namedtuple` with a `Tensor`-conversion
  function registered.

  |     args     |    dtype   |       output       |
  |:------------:|:----------:|:------------------:|
  | `{"a": 1}`   | `None`     | `{"a": T(1)}`      |
  | `T(1)`       | `None`     | `T(1)`             |
  | `[1]`        | `None`     | `[T(1)]`           |
  | `[1]`        | `tf.int32` | `T([1])`           |
  | `[[T(1)]]`   | `None`     | `[T([1])]`         |
  | `[[T(1)]]`   | `[[None]]` | `[[T(1)]]`         |
  | `NT(1, 2)`   | `None`     | `NT(T(1), T(2))`   |
  | `NT(1, 2)`   | `tf.int32` | `T([1, 2])`        |
  | `CNT(1, 2)`  | `None`     | `T(...)`           |
  | `[[1, [2]]]` | `None`     | `[[T(1), T([2])]]` |

  """
  if dtype is None:
    if expand_as_args(args) or _expand_as_kwargs(args):
      shallow_args = _get_shallow_structure(args)
      return nest.map_structure_up_to(
          shallow_args, lambda s: _nested_convert_to_tensor(s, name=name), args)
    else:
      return _nested_convert_to_tensor(args, name=name)
  else:
    return nest.map_structure_up_to(
        dtype, lambda s, dtype: _nested_convert_to_tensor(s, dtype, name), args,
        dtype)


def call_fn(fn, args):
  """Calls `fn` with `args`, possibly expanding `args`.

  Use this function when calling a user-provided callable using user-provided
  arguments.

  The expansion rules are as follows:

  `fn(*args)` if `args` is a `list` or a `tuple`, but not a `namedtuple`.
  `fn(**args)` if `args` is a `dict`.
  `fn(args)` otherwise.

  Args:
    fn: A callable that takes either `args` as an argument(s).
    args: Arguments to `fn`.

  Returns:
    result: Return value of `fn`.
  """

  if expand_as_args(args):
    return fn(*args)
  elif _expand_as_kwargs(args):
    return fn(**args)
  else:
    return fn(args)


def convert_to_nested_tensor(value, dtype=None, dtype_hint=None,
                             allow_packing=False, as_shape_tensor=False,
                             convert_ref=True, name=None):
  """Converts the given `value` to a (structure of) `Tensor`.

  This function converts Python objects of various types to a (structure of)
  `Tensor` objects. It accepts `Tensor` objects, numpy arrays, Python lists, and
  Python scalars.

  Args:
    value: An object whose structure matches that of `dtype` and for which each
      leaf has a registered `Tensor` conversion function.
    dtype: Optional structure of dtypes defining the structure of outputs and
      the `dtype` argument for nested calls to `convert_to_tensor`. If not
      nested, will be broadcasted to match the structure of `dtype_hint`.
    dtype_hint: Optional structure of dtypes defining the structure of outputs
      and the `dtype_hint` argument for nested calls to `convert_to_tensor`. If
      not nested, will be broadcasted to match the structure of `dtype`.
    allow_packing: Python `bool`, default `False`. If `True`, allow
      `convert_to_nested_tensor` to stack nested lists of Tensors along the
      leading dimension. Otherwise, raise.
    as_shape_tensor: Optional boolean when if `True` uses
      `prefer_static.convert_to_shape_tensor` instead of `tf.convert_to_tensor`
      for JAX compatibility.
    convert_ref: Python `bool`, default `True`. If `True`, convert objects with
      reference semantics to Tensor.
    name: Optional name to use if a new `Tensor` is created. If inputs are
      structured, elements are named accoring to '{name}/{path}.{to}.{elem}'.

  Returns:
    tensor: A (structure of) `Tensor` based on `value`.
  """
  dtype_is_nested = nest.is_nested(dtype)
  hint_is_nested = nest.is_nested(dtype_hint)
  # If only one of dtype/dtype_hint is nested, broadcast the atom to match.
  if dtype_is_nested and hint_is_nested:
    nest.assert_same_structure(dtype, dtype_hint)
  elif dtype_is_nested:
    dtype_hint = broadcast_structure(dtype, dtype_hint)
  elif hint_is_nested:
    dtype = broadcast_structure(dtype_hint, dtype)

  # Call coerce_structure to force the argument structure to match dtype.
  value = coerce_structure(dtype, value)

  def convert_fn(path, value, dtype, dtype_hint, name=None):
    if not allow_packing and nest.is_nested(value) and any(
        # Treat arrays like Tensors for full parity in JAX backend.
        tf.is_tensor(x) or isinstance(x, np.ndarray)
        for x in nest.flatten(value)):
      raise NotImplementedError(('Cannot convert a structure of tensors to a '
                                 'single tensor. Saw {} at path {}.'
                                ).format(value, path))
    if as_shape_tensor:
      return ps.convert_to_shape_tensor(value, dtype, dtype_hint, name=name)
    elif 'KerasTensor' in str(type(value)):
      # This is a hack to detect symbolic Keras tensors to work around
      # b/206660667.  The issue was that symbolic Keras tensors would
      # break the Bijector cache on forward/inverse log det jacobian,
      # because tf.convert_to_tensor is not a no-op thereon.
      return value
    elif convert_ref:
      return tf.convert_to_tensor(value, dtype, dtype_hint, name=name)
    else:
      return tensor_util.convert_nonref_to_tensor(
          value, dtype, dtype_hint, name=name)

  ### The following branches only affect naming.
  # For unstructured calls, just use the provided name.
  if not nest.is_nested(dtype):
    return convert_fn((), value, dtype, dtype_hint, name=name)
  # For structured calls where name is provided, include a scope and name
  # members according to "{path}.{to}.{element}".
  elif name is not None:
    with tf.name_scope(name):
      convert_with_name = lambda path, *args: convert_fn(  # pylint: disable=g-long-lambda
          path, *args, name='.'.join(map(str, path)))
      return nest.map_structure_with_tuple_paths_up_to(
          dtype, convert_with_name, value, dtype, dtype_hint, check_types=False)
  # For structured calls without name, skip the scope and don't pass a
  # struct-path to convert-to-tensor.
  else:
    return nest.map_structure_with_tuple_paths_up_to(
        dtype, convert_fn, value, dtype, dtype_hint, check_types=False)


class _DotString(object):

  def __str__(self):
    return '.'

  def __repr__(self):
    return '.'


_DOT = _DotString()


# pylint: disable=protected-access
# TODO(b/173044916): Support namedtuple interop in nest and remove this method.
def coerce_structure(shallow_tree, input_tree):
  """Coerces the containers in `input_tree` to exactly match `shallow_tree`.

  This method largely parallels the behavior of `nest.assert_shallow_structure`,
  but allows `namedtuples` to be interpreted as either sequences or mappings.
  It returns a structure with the container-classes found in `shallow_tree`
  and the contents of `input_tree`, such that `shallow_tree` and `input_tree`
  may be used safely in downstream calls to `nest.map_structure_up_to`.

  Note: this method does not currently support `expand_composites`.

  Example Usage:
  ```python

  ab = collections.namedtuple('AB', 'a b')(0, 1)
  ba = collections.namedtuple('BA', 'b a')(2, 3)

  coerce_structure(ab, ba)
  # -> AB(a=3, b=2)
  ```

  Args:
    shallow_tree: A (shallow) structure to be populated.
    input_tree: A (parallel) structure of values.
  Returns:
    A structure with containers from shallow_tree and values from input_tree.
  Raises:
    ValueError: When nested sub-structures have differing lengths.
    ValueError: When nested sub-structures have different keys.
    TypeError: When `shallow_tree` is deeper than `input_tree`
    TypeError: When nested sub-structures are incompatible (e.g., list vs dict).
  """
  try:
    return _coerce_structure(shallow_tree, input_tree)
  except (ValueError, TypeError) as e:
    str1 = str(nest.map_structure(lambda _: _DOT, shallow_tree))
    str2 = str(nest.map_structure(lambda _: _DOT, input_tree))
    raise type(e)(('{}\n'
                   'Entire first structure:\n{}\n'
                   'Entire second structure:\n{}'
                   ).format(e, str1, str2))


def _coerce_structure(shallow_tree, input_tree):
  """Implementation of coerce_structure."""
  if not nest.is_nested(shallow_tree):
    return input_tree

  if not nest.is_nested(input_tree):
    raise TypeError(nest._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(
        type(input_tree)))

  if len(input_tree) != len(shallow_tree):
    raise ValueError(
        nest._STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
            input_length=len(input_tree),
            shallow_length=len(shallow_tree)))

  # Determine whether shallow_tree should be treated as a Mapping or a Sequence.
  # Namedtuples can be interpreted either way (but keys take precedence).
  _shallow_is_namedtuple = nest._is_namedtuple(shallow_tree)  # pylint: disable=invalid-name
  _shallow_is_mapping = isinstance(shallow_tree, collections.abc.Mapping)  # pylint: disable=invalid-name
  shallow_supports_keys = _shallow_is_namedtuple or _shallow_is_mapping
  shallow_supports_iter = _shallow_is_namedtuple or not _shallow_is_mapping

  # Branch-selection depends on both shallow and input container-classes.
  input_is_mapping = isinstance(input_tree, collections.abc.Mapping)
  if nest._is_namedtuple(input_tree):
    if shallow_supports_keys:
      lookup_branch = lambda k: getattr(input_tree, k)
    else:
      input_iter = nest._yield_value(input_tree)
      lookup_branch = lambda _: next(input_iter)
  elif shallow_supports_keys and input_is_mapping:
    lookup_branch = lambda k: input_tree[k]
  elif shallow_supports_iter and not input_is_mapping:
    input_iter = nest._yield_value(input_tree)
    lookup_branch = lambda _: next(input_iter)
  else:
    raise TypeError(nest._STRUCTURES_HAVE_MISMATCHING_TYPES.format(
        input_type=type(input_tree),
        shallow_type=(
            type(shallow_tree.__wrapped__)
            if hasattr(shallow_tree, '__wrapped__') else
            type(shallow_tree))))

  flat_coerced = []
  needs_wrapping = type(shallow_tree) is not type(input_tree)
  for shallow_key, shallow_branch in nest._yield_sorted_items(shallow_tree):
    try:
      input_branch = lookup_branch(shallow_key)
    except (KeyError, AttributeError):
      raise ValueError(
          nest._SHALLOW_TREE_HAS_INVALID_KEYS.format([shallow_key]))
    flat_coerced.append(_coerce_structure(shallow_branch, input_branch))
    # Keep track of whether nested elements have changed.
    needs_wrapping |= input_branch is not flat_coerced[-1]

  # Only create a new instance if containers differ or contents changed.
  return (nest._sequence_like(shallow_tree, flat_coerced)
          if needs_wrapping else input_tree)

# pylint: enable=protected-access
