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
"""Contains utilities for collecting intermediate summary values."""
import functools
from oryx.core.interpreters import harvest

SUMMARY = 'summary'


def summary(value, *, name: str, mode: str = 'strict'):
  """Tags a value as a summary.

  Args:
    value: a JAX value to be tagged.
    name: a string name for the tagged value.
    mode: the harvest mode for the tagged value.
  Returns:
    The original value.
  """
  return harvest.sow(value, tag=SUMMARY, name=name, mode=mode)


def get_summaries(f):
  """Transforms a function into one that additionally output summaries.

  Args:
    f: a callable.
  Returns:
    A function that when called returns the original output of `f` and a
    dictionary mapping summary names to their values during execution.
  """
  return functools.partial(harvest.harvest(f, tag=SUMMARY), {})
