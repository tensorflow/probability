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
"""Module that provides kwargs utility functions."""
import inspect

__all__ = [
    'filter_kwargs',
    'check_in_kwargs',
]


def _argspec_and_keywords(func):
  argspec = inspect.getfullargspec(func)
  keywords = argspec.varkw
  return argspec, keywords


def filter_kwargs(func, kwargs):
  """Inspects a function's keyword arguments and filters an input dictionary to match.

  If `func` accepts variable keyword arguments, `filter_kwargs(func, kwargs)`
  returns `kwargs`. Otherwise, `filter_kwargs` will inspect `func`'s signature
  to determine what keyword arguments it accepts and remove any entries
  in kwargs that do not match those arguments.

  Args:
    func: a Python function.
    kwargs: a dictionary of keyword arguments to be passed to `func`.
  Returns:
    A filtered `kwargs` dictionary that excludes keywords
    not in func.
  """
  argspec, keywords = _argspec_and_keywords(func)
  accepts_all_kwargs = keywords is not None
  if accepts_all_kwargs:
    return kwargs
  defaults = argspec.defaults
  if defaults is None:
    return {}
  valid_kwargs = set(argspec.args[-len(defaults):])
  filtered_kwargs = {}
  for kw in kwargs:
    if kw in valid_kwargs:
      filtered_kwargs[kw] = kwargs[kw]
  return filtered_kwargs


def check_in_kwargs(func, key):
  """Checks whether a key is in a func kwargs."""
  argspec, keywords = _argspec_and_keywords(func)
  accepts_all_kwargs = keywords is not None
  if accepts_all_kwargs:
    return True
  defaults = argspec.defaults
  kwonly = argspec.kwonlyargs
  if defaults is None and not kwonly:
    return False
  kws = argspec.args[-len(defaults):] if defaults else ()
  return key in kws or key in kwonly
