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
"""Tree utilities."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, Generic, TypeVar

import jax

__all__ = [
    'DataclassView',
    'get_element',
    'update_element',
]

T = TypeVar('T')


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class DataclassView(Generic[T]):
  """Allows selecting which fields of a dataclass are visible to jax.tree_util.

  ```python
  e = Example(a=1, b=2)
  v_e = DataclassView(e, lambda n: n == 'a')
  v_e = jax.tree.map(lambda x: x + 1, v_e)
  assert Example(a=2, b=2) == v_e.value)
  ```

  Attributes:
    value: The dataclass.
    field_selector_fn: A callback that determines which fields are selected.
  """

  value: T
  field_selector_fn: Callable[[str], bool]

  def __post_init__(self):
    if not dataclasses.is_dataclass(self.value):
      raise TypeError(f'class_tree must be a dataclass: {self.value}.')

  # XXX(siege): This is very improper, since the contract for tree_util is that
  # the aux_data is hashable.
  def tree_flatten(self) -> tuple[list[Any], 'DataclassView[T]']:
    selected_fields = [
        getattr(self.value, f.name)
        for f in dataclasses.fields(self.value)
        if self.field_selector_fn(f.name)
    ]

    return selected_fields, self

  @classmethod
  def tree_unflatten(
      cls, aux_data: 'DataclassView[T]', children: list[Any]
  ) -> 'DataclassView[T]':
    selected_field_names = [
        f.name
        for f in dataclasses.fields(aux_data.value)
        if aux_data.field_selector_fn(f.name)
    ]
    selected_fields = dict(zip(selected_field_names, children))
    return cls(
        aux_data.value.replace(**selected_fields), aux_data.field_selector_fn
    )


def _handle_element(
    tree: Any,
    path: Sequence[Any],
    leaf_fn: Callable[[Any], Any],
    subtree_fn: Callable[[Any, Any, Any], Any],
) -> Any:
  """Implementation of get/update_element."""
  if not path:
    return leaf_fn(tree)

  cur, *rest = path

  if isinstance(tree, list):
    subtree = tree[cur]
  elif isinstance(tree, tuple) and not hasattr(tree, '_fields'):
    subtree = tree[cur]
  elif isinstance(tree, dict):
    subtree = tree[cur]
  elif dataclasses.is_dataclass(tree):
    subtree = getattr(tree, cur)
  elif hasattr(tree, cur):
    # Namedtuple
    subtree = getattr(tree, cur)
  else:
    raise TypeError(f'Cannot handle type: {type(tree)}')

  res = _handle_element(subtree, rest, leaf_fn, subtree_fn)
  return subtree_fn(tree, res, cur)


def _update_subtree(tree: Any, res: Any, cur: Any) -> Any:
  """Helper for update_element."""
  if isinstance(tree, list):
    new_tree = list(tree)
    new_tree[cur] = res
    return new_tree
  elif isinstance(tree, tuple) and not hasattr(tree, '_fields'):
    new_tree = list(tree)
    new_tree[cur] = res
    return tuple(new_tree)
  elif isinstance(tree, dict):
    new_tree = tree.copy()
    new_tree[cur] = res
    return new_tree
  elif dataclasses.is_dataclass(tree):
    return dataclasses.replace(tree, **{cur: res})
  elif hasattr(tree, cur):
    # Namedtuple
    return tree._replace(**{cur: res})
  else:
    raise TypeError(f'Cannot handle type: {type(tree)}')


def get_element(tree: Any, path: Sequence[Any]) -> Any:
  """Returns an element from a tree given by its path."""
  return _handle_element(tree, path, lambda x: x, lambda _tree, res, _cur: res)


def update_element(
    tree: Any, path: Sequence[Any], update_fn: Callable[[Any], Any]
) -> Any:
  """Updates an element from a tree given its path and returns a new tree."""
  return _handle_element(tree, path, update_fn, _update_subtree)
