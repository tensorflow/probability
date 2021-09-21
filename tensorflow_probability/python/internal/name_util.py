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
"""Utility functions for dealing with `tf.name_scope` names."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import re

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


__all__ = [
    'camel_to_lower_snake',
    'get_name_scope_name',
    'instance_scope'
]


_IN_INSTANCE_SCOPE = False


_valid_chars_re = re.compile(r'[^a-zA-Z0-9_]+')
_camel_snake_re = re.compile(r'((?<=[a-z0-9])[A-Z]|(?!^)(?<!_)[A-Z](?=[a-z]))')


def strip_invalid_chars(name):
  return re.sub(_valid_chars_re, r'_', name).strip('_') if name else ''


def camel_to_lower_snake(name):
  return (re.sub(_camel_snake_re, r'_\1', name).lower()
          if name else '')


def get_name_scope_name(name):
  """Returns the input name as a unique `tf.name_scope` name."""
  if name and name[-1] == '/':
    return name
  name = strip_invalid_chars(name)
  with tf.name_scope(name) as unique_name:
    pass
  return unique_name


@contextlib.contextmanager
def instance_scope(instance_name, constructor_name_scope):
  """Constructs a name scope for methods of a distribution (etc.) instance."""
  global _IN_INSTANCE_SCOPE
  with tf.name_scope(_instance_scope_name(instance_name,
                                          constructor_name_scope)
                     ) as name_scope:
    was_in_instance_scope = _IN_INSTANCE_SCOPE
    _IN_INSTANCE_SCOPE = True
    try:
      yield name_scope
    finally:
      _IN_INSTANCE_SCOPE = was_in_instance_scope


def _instance_scope_name(instance_name, constructor_name_scope):
  """Specifies a name scope for methods of a distribution (etc.) instance."""
  global _IN_INSTANCE_SCOPE
  current_parent_scope = _get_parent_scope(_name_scope_dry_run(instance_name))
  constructor_parent_scope = _get_parent_scope(constructor_name_scope)
  if current_parent_scope == constructor_parent_scope:
    # Reuse initial scope.
    return constructor_name_scope

  if _IN_INSTANCE_SCOPE:
    # Elide the constructor scope annotation when we're inside a method of a
    # higher-level distribution (which should itself have annotated its
    # constructor scope).
    constructor_scope_annotation = ''
  else:
    # Otherwise, include a reference to the sanitized constructor scope.
    constructor_scope_annotation = (
        '_CONSTRUCTED_AT_' + (strip_invalid_chars(constructor_parent_scope[:-1])
                              if constructor_parent_scope[:-1]
                              else 'top_level'))
  return (current_parent_scope +
          instance_name +
          constructor_scope_annotation + '/')


def _get_parent_scope(scope):
  """Removes the final leaf from a scope (`a/b/c/` -> `a/b/`)."""
  parts = scope.split('/')
  return '/'.join(parts[:-2] + parts[-1:])


def _name_scope_dry_run(name):
  """Constructs a scope like `tf.name_scope` but without marking it used."""
  if tf.executing_eagerly():
    # Names in eager mode are not unique, so we can just invoke name_scope
    # directly.
    with tf.name_scope(name) as name_scope:
      return name_scope

  graph = tf1.get_default_graph()
  if not name:
    name = ''
  elif name[-1] != '/':
    name = graph.unique_name(name, mark_as_used=False) + '/'
  return name
