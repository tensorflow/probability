# Copyright 2023 The TensorFlow Probability Authors.
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
"""Numpy stub for `type_spec_registry`."""

import re

__all__ = [
    'lookup',
    'register',
]


_TYPE_SPEC_TO_NAME = {}
_NAME_TO_TYPE_SPEC = {}
_REGISTERED_NAME_RE = re.compile(r'^(\w+\.)+\w+$')


def register(_):
  """No-op for registering a `tf.TypeSpec` for `saved_model`."""
  def decorator_fn(cls):
    return cls
  return decorator_fn


def lookup(_):
  # Raise ValueError instead of NotImplementedError to conform to TF.
  raise ValueError('`TypeSpec`s are not registered in Numpy/JAX.')
