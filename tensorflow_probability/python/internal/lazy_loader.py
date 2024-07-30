# Copyright 2020 The TensorFlow Probability Authors. All Rights Reserved.
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
# ==============================================================================

"""A LazyLoader class."""

import importlib
import types


__all__ = ['LazyLoader']


class LazyLoader(types.ModuleType):
  """Lazily import a module to avoid pulling in large deps, defer checks."""

  # The lint error here is incorrect.
  def __init__(self, local_name, parent_module_globals, name,
               on_first_access=None):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    self._on_first_access = on_first_access

    super(LazyLoader, self).__init__(name)

  def _load(self):
    """Load the module and insert it into the parent's globals."""
    if callable(self._on_first_access):
      self._on_first_access()
      self._on_first_access = None
    # Import the target module and insert it into the parent's namespace
    module = importlib.import_module(self.__name__)
    if self._parent_module_globals is not None:
      self._parent_module_globals[self._local_name] = module
      self._parent_module_globals = None

      # Update this object's dict so that if someone keeps a reference to the
      # LazyLoader, lookups are efficient (__getattr__ is only called on lookups
      # that fail).
      self.__dict__.update(module.__dict__)

    return module

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)

  def __dir__(self):
    module = self._load()
    return dir(module)

  def __reduce__(self):
    return importlib.import_module, (self.__name__,)
