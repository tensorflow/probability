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
"""Utilities for accessing nested attributes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


__all__ = [
    'get_innermost',
    'get_nested_objs',
    'get_outermost',
    'has_nested',
    'replace_innermost',
    'replace_outermost',
    'UnnestingWrapper',
]


def get_nested_objs(obj,
                    nested_key='__nested_attrs__',
                    fallback_attrs=(
                        'inner_results',
                        'accepted_results',
                        'inner_kernel',
                    )):
  """Finds the list of nested objects inside an object's attributes.

  `get_nested_objs` proceeds as follow:

  1. If `hasattr(obj, nested_key)`, then set
     `nested_attrs = getattr(obj, nested_key)`.
  2. Otherwise set `nested_attrs = fallback_attrs`.
  3. In either case, `nested_attrs` should now be a string or collection of
     strings. Return the list `[(attr, getattr(obj, attr)) for attr in
     nested_attrs]` omitting missing attributes.

  `nested_key` is for class- or object-level customization, and `fallback_attrs`
  for invocation-level customization.

  Example:

  ```
  class Nest:
    __nested_attrs__ = ('inner1', 'inner2')

    def __init__(self, inner1, inner2, inner3):
      self.inner1 = inner1
      self.inner2 = inner2
      self.inner3 = inner3

  nest = Nest('x', 'y', 'z')

  # Search stops if `nested_key` is found.
  get_nested_objs(
      nest,
      nested_key='__nested_attrs__',
      fallback_attrs=('inner1', 'inner2', 'inner3')) == [
          ('inner1', 'x'), ('inner2', 'y')]  # True

  # If `nested_key` is not found, search in all of `fallback_attrs`.
  get_nested_objs(
      nest,
      nested_key='does_not_exist',
      fallback_attrs=('inner1', 'inner2', 'inner3')) == [
          ('inner1', 'x'), ('inner2', 'y'), ('inner3', 'z')]  # True

  # If nothing is found, empty list is returned.
  get_nested_objs(
      nest,
      nested_key='does_not_exist',
      fallback_attrs=('does_not_exist')) == []  # True

  # `getattr(obj, nested_key)` and `fallback_attrs` can be either strings or
  # collections of strings.
  nest2 = Nest('x', 'y', 'z')
  nest2.__nested_attrs__ = 'inner3'
  get_nested_objs(
      nest2,
      nested_key='__nested_attrs__',
      fallback_attrs=('inner1', 'inner2')) == [('inner3', 'z')]  # True
  get_nested_objs(
      nest2,
      nested_key='does_not_exist',
      fallback_attrs='inner2') == [('inner2', 'y')]  # True
  ```

  Args:
    obj: The object to find nested objects in.
    nested_key: A string that names an attribute on `obj` which contains the
      names of attributes to search for nested objects in. See function
      documentation for details. Default is `__nested_attrs__`.
    fallback_attrs: A string or collection of strings that name attributes to
      search for nested objects in, when `nested_key` is not present. See
      function documentation for details. Default is the tuple
      `('inner_results', 'accepted_results', 'inner_kernel')`, which works well
      for MCMC kernels and kernel results.

  Returns:
    pairs: Returns a (possibly empty) list of (field name, nested results).
  """
  if hasattr(obj, nested_key):
    attrs = getattr(obj, nested_key)
  else:
    attrs = fallback_attrs
  if isinstance(attrs, str):
    attrs = [attrs]
  return [(attr, getattr(obj, attr)) for attr in attrs
          if hasattr(obj, attr)]


def has_nested(obj, attr, nested_lookup_fn=get_nested_objs):
  """Check if the object has a (nested) attribute.

  Args:
    obj: The object to find (nested) attributes in.
    attr: A `string` attribute name to search for.
    nested_lookup_fn: A single-argument callable that returns a list of
      (attribute name, nested object) pairs. Defaults to `get_nested_objs`.

  Returns:
    has_nested: Boolean if the attribute was found or not.
  """
  if hasattr(obj, attr):
    return True
  for _, nested in nested_lookup_fn(obj):
    if has_nested(nested, attr, nested_lookup_fn=nested_lookup_fn):
      return True
  return False


SENTINEL = object()


def get_innermost(obj, attr, default=SENTINEL,
                  nested_lookup_fn=get_nested_objs):
  """Return a (nested) attribute value.

  The first attribute found is returned. Nested objects are traversed
  depth-first in post-order, with level-wise order determined by the list
  ordering returned from `nested_lookup_fn`.

  Args:
    obj: The object to find (nested) attributes in.
    attr: A `string` attribute name to search for.
    default: If `attr` does not exist in `obj` or nested objects, and `default`
      is set, return `default`.
    nested_lookup_fn: A single-argument callable that returns a list of
      (attribute name, nested object) pairs. Defaults to `get_nested_objs`.

  Returns:
    value: The (nested) attribute value, or `default` if it does not exist and
      `default` is set.

  Raises:
    AttributeError: if `attr` is not found and `default` is not specified.
  """
  for _, nested in nested_lookup_fn(obj):
    try:
      return get_innermost(nested, attr, nested_lookup_fn=nested_lookup_fn)
    except AttributeError:
      pass
  try:
    return getattr(obj, attr)
  except AttributeError:
    if default is not SENTINEL:
      return default
    raise AttributeError('No attribute `' + attr + '` in nested results of '
                         + str(obj.__class__))


def get_outermost(obj, attr, default=SENTINEL,
                  nested_lookup_fn=get_nested_objs):
  """Return a (nested) attribute value.

  The first attribute found is returned. Nested objects are traversed
  breadth-first, with level-wise order determined by the list ordering returned
  from `nested_lookup_fn`.

  Args:
    obj: The object to find (nested) attributes in.
    attr: A `string` attribute name to search for.
    default: If `attr` does not exist in `obj` or nested objects, and `default`
      is set, return `default`.
    nested_lookup_fn: A single-argument callable that returns a list of
      (attribute name, nested object) pairs. Defaults to `get_nested_objs`.

  Returns:
    value: The (nested) attribute value, or `default` if it does not exist and
      `default` is set.

  Raises:
    AttributeError: if `attr` is not found and `default` is not specified.
  """
  to_visit = collections.deque([obj])
  while to_visit:
    nested = to_visit.popleft()
    to_visit.extend((nested for _, nested in nested_lookup_fn(nested)))
    try:
      return getattr(nested, attr)
    except AttributeError:
      pass
  if default is not SENTINEL:
    return default
  raise AttributeError('No attribute `' + attr + '` in nested results of '
                       + str(obj.__class__))


def replace_innermost(ntuple, return_unused=False,
                      nested_lookup_fn=get_nested_objs, **kw):
  """Replace (nested) fields in a `namedtuple`.

  For each attribute-value update specified, this function only replaces the
  first matching attribute found. Nested objects are traversed depth-first in
  post-order, with level-wise order determined by the list ordering returned
  from `nested_lookup_fn`.

  Args:
    ntuple: A `namedtuple` to replace (nested) fields in.
    return_unused: If `True`, return the `dict` of attribute-value pairs in
     `**kw` that were not found and updated in `ntuple`.
    nested_lookup_fn: A single-argument callable that returns a list of
      (attribute name, nested object) pairs. Defaults to `get_nested_objs`.
    **kw: The attribute-value pairs to update.

  Returns:
    updated: A copy of `ntuple` with (nested) fields updated.
    unused: If `return_unused` is `True`, the dictionary of attribute-value
      pairs in `**kw` that were not found and updated in `ntuple`.

  Raises:
    ValueError: if `returne_unused=False` and attributes in `**kw` are not found
      in `ntuple`.
  """
  nested_updates = {}
  for fieldname, nested in nested_lookup_fn(ntuple):
    updated, kw = replace_innermost(
        nested, return_unused=True, nested_lookup_fn=nested_lookup_fn, **kw)
    nested_updates[fieldname] = updated
  kw.update(nested_updates)
  if not return_unused:
    return ntuple._replace(**kw)
  outer_updates = {attr: kw[attr] for attr in ntuple._fields
                   if attr in kw}
  extra = {attr: val for attr, val in kw.items()
           if attr not in outer_updates}
  return ntuple._replace(**outer_updates), extra


def replace_outermost(ntuple, return_unused=False,
                      nested_lookup_fn=get_nested_objs, **kw):
  """Replace (nested) fields in a `namedtuple`.

  For each attribute-value update specified, this function only replaces the
  first matching attribute found. Nested objects are traversed breadth-first,
  with level-wise order determined by the list ordering returned from
  `nested_lookup_fn`.

  Args:
    ntuple: A `namedtuple` to replace (nested) fields in.
    return_unused: If `True`, return the `dict` of attribute-value pairs in
     `**kw` that were not found and updated in `ntuple`.
    nested_lookup_fn: A single-argument callable that returns a list of
      (attribute name, nested object) pairs. Defaults to `get_nested_objs`.
    **kw: The attribute-value pairs to update.

  Returns:
    updated: A copy of `ntuple` with (nested) fields updated.
    unused: If `return_unused` is `True`, the dictionary of attribute-value
      pairs in `**kw` that were not found and updated in `ntuple`.

  Raises:
    ValueError: if `returne_unused=False` and attributes in `**kw` are not found
      in `ntuple`.
  """
  root = ntuple
  root_update = {k: kw[k] for k in root._fields if k in kw}
  kw = {k: v for k, v in kw.items() if k not in root_update}
  # Collect the updates to apply later by traversing breadth-first, but with
  # backlinks to parent updates.
  to_visit = collections.deque(
      [(root_update, field, child)
       for field, child in nested_lookup_fn(root)])
  inner_updates = []
  while to_visit and kw:
    parent_update, field, child = to_visit.popleft()
    child_update = {k: kw[k] for k in child._fields if k in kw}
    if child_update:
      kw = {k: v for k, v in kw.items() if k not in child_update}
      inner_updates.append((parent_update, field, child, child_update))
    to_visit.extend(
        [(child_update, child_field, child_child)
         for child_field, child_child in nested_lookup_fn(child)])
  # Now apply updates in reverse order, propogating up to root.
  for parent_update, field, child, child_update in reversed(inner_updates):
    parent_update[field] = child._replace(**child_update)
  root = root._replace(**root_update)
  if not return_unused:
    if kw:
      raise ValueError(
          'Got unexpected (nested) field names: {}'.format(list(kw)))
    return root
  return root, kw


class UnnestingWrapper:
  """For when you want to get (nested) fields by usual attribute access.

  Example usage:

  ```
  results = ...
  wrapped = UnnestingWrapper(results)
  wrapped.my_attr  # equivalent to `get_innermost(results, 'my_attr')

  # Use `_object` to get at the wrapped object.
  new_results = replace_innermost(wrapped._object, ...)
  ```
  """

  def __init__(self, obj, innermost=True):
    """Wraps objects so attribute access searches nested objects.

    Args:
      obj: The object to find nested objects in.
      innermost: Boolean. When `True`, attribute access uses `get_innermost`;
        otherwise uses `get_outermost`. Defaults to `True`.
    """
    self._object = obj
    self._innermost = innermost

  def __getattr__(self, attr):
    if self._innermost:
      return get_innermost(self._object, attr)
    else:
      return get_outermost(self._object, attr)

  def __repr__(self):
    return 'UnnestingWrapper(innermost={}):\n{}'.format(
        self._innermost,
        repr(self._object))
