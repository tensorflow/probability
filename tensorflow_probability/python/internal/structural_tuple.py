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
"""Implementation of structural tuples.

A structural tuple is like a regular namedtuple, except that it obeys structural
typing rules. Structural tuples with the same field names, in the same order,
are considered to be of the same type.
"""

import builtins
import collections
import keyword
import weakref

_TYPES = weakref.WeakValueDictionary()


def _concat_structtuples(a, b):
  tuple_result = tuple(a) + tuple(b)
  if not hasattr(a, '_fields') or not hasattr(b, '_fields'):
    return tuple_result

  new_fields = a._fields + b._fields
  return structtuple(new_fields)(*tuple_result)


def _validate_field_names(field_names):
  """Validate field names."""
  # Same logic as used for namedtuples.
  seen = set()
  for name in field_names:
    if not isinstance(name, str):
      raise TypeError('Field names must be strings: {} has type {}'.format(
          name, type(name)))
    if not name.isidentifier():
      raise ValueError('Field names must be valid identifiers: {}'.format(name))
    if keyword.iskeyword(name):
      raise ValueError('Field names cannot be a keyword: {}'.format(name))
    if name.startswith('_'):
      raise ValueError(
          'Field names cannot start with an underscore: {}'.format(name))
    if name in seen:
      raise ValueError('Encountered duplicate field name: {}'.format(name))
    seen.add(name)


def structtuple(field_names):
  """Return a StructTuple with specified fields.

  Calls to this function with the same field names will return the same type. To
  avoid memory leaks the cached types are stored by weak references, so there is
  a possibility that the id of the returned type can change. Other than storing
  the id separately, this should not otherwise be observable.

  Args:
    field_names: Python iterable of strings specifying the field names.

  Returns:
    structtuple: A StructTuple type.
  """
  _validate_field_names(field_names)

  key = ','.join(field_names)
  try:
    return _TYPES[key]
  except KeyError:
    pass

  # We can't use a regular namedtuple as a base class as that is restricted to
  # <= 255 field names.
  class StructTuple(tuple):
    """A structurally-typed tuple."""
    __slots__ = ()
    # Secret handshake with nest_util to make call_fn expand StructTuples as
    # *args.
    _tfp_nest_expansion_force_args = ()

    _fields = tuple(field_names)
    _field_name_to_index = dict([(k, idx) for idx, k in enumerate(field_names)])

    def __new__(_cls, *args, **kwargs):  # pylint: disable=bad-classmethod-argument
      # Values default to None.
      vals = [None] * len(_cls._fields)
      vals[:len(args)] = args
      for field_idx, field_name in enumerate(
          _cls._fields[len(args):], start=len(args)):
        vals[field_idx] = kwargs.pop(field_name, None)
        if not kwargs:
          break
      if kwargs:
        for k in kwargs:
          if k in _cls._fields:
            raise TypeError('Got multiple values for argument {}'.format(k))
          else:
            raise TypeError('Got an unexpected keyword argument {}'.format(k))
      return builtins.tuple.__new__(_cls, vals)

    @classmethod
    def _make(cls, iterable, new=tuple.__new__, len=len):  # pylint: disable=redefined-builtin
      """Make a new StrucTuple object from a sequence or iterable."""
      values = tuple(iterable)
      if len(values) > len(cls._fields):
        raise TypeError('Expected {} arguments or fewer, got {}.'.format(
            len(cls._fields), len(values)))
      if len(values) < len(cls._fields):
        values += (None,) * (len(cls._fields) - len(values))
      result = new(cls, values)
      return result

    def _replace(_self, **kwds):  # pylint: disable=invalid-name, no-self-argument
      """Return a new StructTuple replacing some fields with new values."""
      result = _self._make(kwds.pop(f, v) for f, v in zip(_self._fields, _self))
      if kwds:
        raise ValueError('Got unexpected field names: {}'.format(
            list(sorted(kwds))))
      return result

    def __repr__(self):
      return '{}(\n{}\n)'.format(
          type(self).__name__,
          ',\n'.join('  {}={}'.format(k,
                                      repr(v).replace('\n', '\n    '))
                     for (k, v) in self._asdict().items()))

    def _asdict(self):
      """Return a new OrderedDict which maps field names to their values."""
      return collections.OrderedDict(zip(self._fields, self))

    def __getnewargs__(self):
      """Return self as a plain tuple.  Used by copy and pickle."""
      return tuple(self)

    def __getattr__(self, attr):
      attr_idx = self._field_name_to_index.get(attr)
      if attr_idx is None:
        raise AttributeError('StructTuple has no attribute {}'.format(attr))
      return self[attr_idx]

    def __getitem__(self, index):
      tuple_result = tuple(self)[index]
      if isinstance(index, slice):
        new_fields = self._fields[index]
        return structtuple(new_fields)(*tuple_result)
      else:
        return tuple_result

    def __getslice__(self, i, j):
      return self.__getitem__(slice(i, j))

    def __add__(self, other):
      return _concat_structtuples(self, other)

    def __radd__(self, other):
      return _concat_structtuples(other, self)

  _TYPES[key] = StructTuple
  return StructTuple
