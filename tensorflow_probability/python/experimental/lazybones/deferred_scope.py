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
# Lint as: python3
"""DeferredScope."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import threading

from tensorflow_probability.python.experimental.lazybones.utils import weak_container


__all__ = [
    'DeferredScope',
    'UNKNOWN',
]


class PrettyPrintTuple(tuple):
  """Helper to make easier to read tuples of Deferred."""

  __slots__ = ()

  def __repr__(self):
    return '(' + ', '.join([getattr(x, '_repr_no_eval', repr)()
                            for x in self]) + ')'

_NOT_FOUND = object()


class classproperty(property):  # pylint: disable=invalid-name
  """Like `property` but for `classmethod`s."""

  def __get__(self, cls, owner):
    return classmethod(self.fget).__get__(None, owner)()


class _Unknown(object):
  """No known value for deferred object in current `DeferredScope`."""

  __slots__ = ()

  def __repr__(self):
    return '[Unknown]'


UNKNOWN = _Unknown()


class DeferredScope(object):
  """Manages scope for Deferred objects."""

  __slots__ = ('_original_parent', '_parent', '_slots')

  # This class level variables will be "global" to all instances of
  # `DeferredScope`.  Making it a class variable is convenient for both
  # debugging and because it makes it possible for a user to redefine its
  # behavior.
  _thread_local = threading.local()

  def __init__(self):
    # `_original_parent` isnt really needed but might be useful for debugging.
    self._original_parent = self.current_scope
    self._parent = None
    self._slots = weak_container.WeakKeyDictionary()

  def __enter__(self):
    self._parent = self.current_scope
    self._thread_local.current_scope = self
    return self

  def __exit__(self, type_, value, traceback):
    self._thread_local.current_scope = self._parent

  def __getitem__(self, k):
    v = self._slots.get(k, _NOT_FOUND)
    if v is not _NOT_FOUND:
      return v
    if self._parent is None:
      return UNKNOWN, False
    return self._parent[k]

  def __setitem__(self, k, v):
    self._slots[k] = v

  def __delitem__(self, k):
    # Unlike "normal" __delitem__, ours has "delete if exists" semantics.
    v = self._slots.pop(k, _NOT_FOUND)
    if v is not _NOT_FOUND:
      return
    if self._parent is None:
      return
    del self._parent[k]

  @property
  def parent(self):
    return self._parent

  @property
  def slots(self):
    return self._slots

  @property
  def tracked(self):
    return PrettyPrintTuple(self._slots.keys())

  @classproperty
  def current_scope(cls):  # pylint: disable=no-self-argument
    """Threadlocal DeferredScope singleton of current Deferred scope context."""
    try:
      return cls._thread_local.current_scope
    except AttributeError:
      cls._thread_local.current_scope = None
      return cls().__enter__()  # pylint: disable=not-callable
