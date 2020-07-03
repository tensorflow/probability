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
"""Stub implementation of tensorflow.python.util.deprecation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

# pylint: disable=unused-argument


def deprecated_alias(deprecated_name, name, func_or_class, warn_once=True):
  return func_or_class


def deprecated_endpoints(*args):
  return lambda func: func


def deprecated(date, instructions, warn_once=True):
  return lambda func: func


def deprecated_args(date, instrutions, *deprecated_arg_names_or_tuples,
                    **kwargs):
  return lambda func: func


def deprecated_arg_values(date, instructions, warn_once=True,
                          **deprecated_kwargs):
  return lambda func: func


def deprecated_argument_lookup(new_name, new_value, old_name, old_value):
  if old_value is not None:
    if new_value is not None:
      raise ValueError("Cannot specify both '%s' and '%s'" %
                       (old_name, new_name))
    return old_value
  return new_value


@contextlib.contextmanager
def silence():
  yield
