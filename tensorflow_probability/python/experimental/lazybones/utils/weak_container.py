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
"""Weak containers."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import collections
import weakref

collections_abc = getattr(collections, 'abc', collections)


__all__ = [
    'HashableWeakRef',
    'WeakKeyDictionary',
    'WeakSet',
]


class _WeakContainerCommon(object):
  """Methods common to both weak containers."""

  __slots__ = ('_store',)

  def __iter__(self):
    for x in self._store:
      x = x()
      if x is None:
        continue
      yield x

  def __len__(self):
    return len(self._store)

  def __contains__(self, key):
    return self.__key_transform_get__(key) in self._store

  def __key_transform_get__(self, key):
    if isinstance(key, HashableWeakRef):
      return key
    return HashableWeakRef(key)

  def __key_transform_set__(self, key):
    if isinstance(key, HashableWeakRef):
      key = key()
    return HashableWeakRef(key, self._store.pop)


class WeakSet(_WeakContainerCommon, collections_abc.MutableSet):
  """`WeakSet`."""

  def __init__(self, *args, **kwargs):
    try:
      self._store = collections.OrderedDict.fromkeys(*args, **kwargs)
    except TypeError:
      self._store = collections.OrderedDict(*args, **kwargs)

  def add(self, key):
    self._store[self.__key_transform_set__(key)] = None

  def discard(self, key):
    del self._store[self.__key_transform_get__(key)]

  def __repr__(self):
    return '{' + str(tuple(repr(x) for x in self))[1:-1] + '}'


class WeakKeyDictionary(_WeakContainerCommon, collections_abc.MutableMapping):
  """`WeakKeyDictionary`."""

  def __init__(self, *args, **kwargs):
    self._store = collections.OrderedDict(*args, **kwargs)

  def __getitem__(self, key):
    return self._store[self.__key_transform_get__(key)]

  def __setitem__(self, key, value):
    self._store[self.__key_transform_set__(key)] = value

  def __delitem__(self, key):
    del self._store[self.__key_transform_get__(key)]

  def __repr__(self):
    return repr(collections.OrderedDict((repr(k), v) for k, v in self.items()))


class HashableWeakRef(weakref.ref):
  """weakref.ref which makes wrapped object hashable.

  We take care to ensure that a hash can still be provided in the case that the
  ref has been cleaned up. This ensures that the WeakKeyDictionary doesn't
  suffer memory leaks by failing to clean up HashableWeakRef key objects whose
  referrents have gone out of scope and been destroyed.
  """

  __slots__ = ('_hash',)

  def __init__(self, referrent, callback=None):
    """weakref.ref which makes any object hashable.

    Args:
      referrent: Object that is being referred to.
      callback: Optional callback to invoke when object is GCed.
    """
    self._hash = self._get_hash(referrent)
    super(HashableWeakRef, self).__init__(referrent, callback)

  def __repr__(self):
    return '{}({})'.format(type(self).__name__,
                           super(HashableWeakRef, self).__repr__())

  def __hash__(self):
    return self._hash

  def __eq__(self, other, maybe_negate=lambda x: x):
    if maybe_negate(hash(self) == hash(other)):
      return True
    if isinstance(other, weakref.ref):
      other = other()
    return maybe_negate(self() is other)

  def __ne__(self, other):
    # weakref.ref overrides `!=` explicitly, so we must as well.
    return self.__eq__(other, lambda x: not(x))  # pylint: disable=superfluous-parens

  @staticmethod
  def _get_hash(referrent):
    try:
      return hash(referrent)
    except TypeError as e:
      if not str(e).startswith('unhashable type'):
        raise
      return id(referrent)
