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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import weakref

_TYPES = weakref.WeakValueDictionary()


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
  key = ','.join(field_names)
  try:
    return _TYPES[key]
  except KeyError:

    class StructTuple(collections.namedtuple('StructTuple', list(field_names))):
      __slots__ = ()
      # Secret handshake with nest_util to make call_fn expand StructTuples as
      # *args.
      _tfp_nest_expansion_force_args = ()

    StructTuple.__new__.__defaults__ = (None,) * len(field_names)
    _TYPES[key] = StructTuple
    return StructTuple
