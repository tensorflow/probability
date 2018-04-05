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
"""Interceptor for controlling the execution of programs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager

import functools
import threading

__all__ = [
    "get_interceptor",
    "interceptable",
    "interception",
]


class _InterceptorStack(threading.local):
  """A thread-local stack of interceptors."""

  def __init__(self):
    super(_InterceptorStack, self).__init__()
    self.stack = [lambda f, *args, **kwargs: f(*args, **kwargs)]


_interceptor_stack = _InterceptorStack()


@contextmanager
def interception(interceptor):
  """Python context manager for interception.

  Upon entry, an interception context manager pushes an interceptor onto a
  thread-local stack. Upon exiting, it pops the interceptor from the stack.

  Args:
    interceptor: Function which takes a callable `f` and inputs `*args`,
      `**kwargs`.

  Yields:
    None.

  #### Examples

  Interception controls the execution of Edward programs. Below we illustrate
  how to set the value of a specific random variable within a program.

  ```python
  from tensorflow_probability import edward2 as ed

  def model():
    return ed.Poisson(rate=1.5, name="y")

  def interceptor(f, *args, **kwargs):
    if kwargs.get("name") == "y":
      kwargs["value"] = 42
    return f(*args, **kwargs)

  with ed.interception(interceptor):
    y = model()

  with tf.Session() as sess:
    assert sess.run(y.value) == 42
  ```
  """
  try:
    _interceptor_stack.stack.append(interceptor)
    yield
  finally:
    _interceptor_stack.stack.pop()


def get_interceptor():
  """Returns the top-most (last) interceptor on the thread's stack.

  The bottom-most (first) interceptor in the stack is a function which takes
  `f, *args, **kwargs` as input and returns `f(*args, **kwargs)`. It is the
  default if no `interception` contexts have been entered.
  """
  return _interceptor_stack.stack[-1]


def interceptable(func):
  """Decorator that wraps `func` so that its execution is intercepted.

  The wrapper passes `func` to the interceptor for the current thread.

  Args:
    func: Function to wrap.

  Returns:
    The decorated function.
  """
  @functools.wraps(func)
  def func_wrapped(*args, **kwargs):
    return get_interceptor()(func, *args, **kwargs)
  return func_wrapped
