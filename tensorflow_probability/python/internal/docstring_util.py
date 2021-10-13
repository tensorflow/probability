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
"""Utilities for programmable docstrings."""

import inspect
import re
import six

__all__ = [
    'expand_docstring',
]


def expand_docstring(**kwargs):
  """Decorator to programmatically expand the docstring.

  Args:
    **kwargs: Keyword arguments to set. For each key-value pair `k` and `v`,
      the key is found as `${k}` in the docstring and replaced with `v`.

  Returns:
    Decorated function.
  """
  def _fn_wrapped(fn):
    """Original function with modified `__doc__` attribute."""
    doc = inspect.cleandoc(fn.__doc__)
    for k, v in six.iteritems(kwargs):
      # Capture each ${k} reference to replace with v.
      # We wrap the replacement in a function so no backslash escapes
      # are processed.
      pattern = r'\$\{' + str(k) + r'\}'
      doc = re.sub(pattern, lambda match: v, doc)  # pylint: disable=cell-var-from-loop
    fn.__doc__ = doc
    return fn
  return _fn_wrapped
