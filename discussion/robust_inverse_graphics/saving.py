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
"""Saving/loading code."""

from typing import Any, BinaryIO, Mapping, TypeVar

import immutabledict
from discussion.robust_inverse_graphics.util import tree2
from fun_mc import using_jax as fun_mc


try:
  # This module doesn't exist at the time static analysis is done.
  # pylint: disable=g-import-not-at-top
  from fun_mc.dynamic.backend_jax import fun_mc_lib  # pytype: disable=import-error
except ImportError:
  pass

__all__ = [
    'enable_interactive_mode',
    'load',
    'register',
    'save',
]

_registry = tree2.Registry(allow_unknown_types=True)
_registry.auto_register_type('_TraceMaskHolder')(fun_mc_lib._TraceMaskHolder)  # pylint: disable=protected-access
_registry.auto_register_type('AdamState')(fun_mc.AdamState)
_registry.auto_register_type('InterruptibleTraceState')(
    fun_mc.InterruptibleTraceState
)


T = TypeVar('T')


def enable_interactive_mode():
  """Enables interactive mode (for notebook use)."""
  _registry.interactive_mode = True


def register(tree_type: type[T]) -> type[T]:
  """Registers a RobustVision type."""
  return _registry.auto_register_type(f'rig.{tree_type.__name__}')(tree_type)


def save(
    tree: Any,
    path: str | BinaryIO,
    options: Mapping[str, Any] = immutabledict.immutabledict({}),
):
  """Saves a tree."""
  _registry.save_tree(tree, path, options)


def load(
    path: str | BinaryIO,
    options: Mapping[str, Any] = immutabledict.immutabledict({}),
) -> Any:
  """Loads a tree."""
  return _registry.load_tree(path, options)
