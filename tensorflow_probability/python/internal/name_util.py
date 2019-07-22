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

import re

import tensorflow.compat.v2 as tf


__all__ = [
    'camel_to_lower_snake',
    'get_name_scope_name',
]


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
