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

import collections
from contextlib import contextmanager

import functools
import threading

__all__ = [
    "get_next_interceptor",
    "interceptable",
    "interception",
    "tape",
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
    return interceptable(f)(*args, **kwargs)

  with ed.interception(interceptor):
    y = model()

  with tf.Session() as sess:
    assert sess.run(y.value) == 42
  ```

  Wrapping `f` as `interceptable` allows interceptors down the stack to
  additionally modify this operation. Since the operation `f()` is not wrapped
  by default, we could have called it directly. Refer also to the example in
  `get_next_interceptor()` for more details on nested interceptors.
  """
  try:
    _interceptor_stack.stack.append(interceptor)
    yield
  finally:
    _interceptor_stack.stack.pop()


@contextmanager
def get_next_interceptor():
  """Yields the top-most interceptor on the thread-local interceptor stack.

  Operations may be intercepted by multiple nested interceptors. Once reached,
  an operation can be forwarded through nested interceptors until resolved.
  To allow for nesting, implement interceptors by re-wrapping their first
  argument (`f`) as an `interceptable`. To avoid nesting, manipulate the
  computation without using `interceptable`.

  This function allows for nesting by manipulating the thread-local interceptor
  stack, so that operations are intercepted in the order of interceptor nesting.

  #### Examples

  ```python
  from tensorflow_probability import edward2 as ed

  def model():
    x = ed.Normal(loc=0., scale=1., name="x")
    y = ed.Normal(loc=x, scale=1., name="y")
    return x + y

  def double(f, *args, **kwargs):
    return 2. * interceptable(f)(*args, **kwargs)

  def set_y(f, *args, **kwargs):
    if kwargs.get("name") == "y":
      kwargs["value"] = 0.42
    return interceptable(f)(*args, **kwargs)

  with interception(double):
    with interception(set_y):
      z = model()
  ```

  This will firstly put `double` on the stack, and then `set_y`,
  resulting in the stack:
  (TOP) set_y -> double -> apply (BOTTOM)

  The execution of `model` is then (top lines are current stack state):
  1) (TOP) set_y -> double -> apply (BOTTOM);
  `ed.Normal(0., 1., "x")` is intercepted by `set_y`, and as the name is not "y"
  the operation is simply forwarded to the next interceptor on the stack.

  2) (TOP) double -> apply (BOTTOM);
  `ed.Normal(0., 1., "x")` is intercepted by `double`, to produce
  `2*ed.Normal(0., 1., "x")`, with the operation being forwarded down the stack.

  3) (TOP) apply (BOTTOM);
  `ed.Normal(0., 1., "x")` is intercepted by `apply`, which simply calls the
  constructor.

  (At this point, the nested calls to `get_next_interceptor()`, produced by
  forwarding operations, exit, and the current stack is again:
  (TOP) set_y -> double -> apply (BOTTOM))

  4) (TOP) set_y -> double -> apply (BOTTOM);
  `ed.Normal(0., 1., "y")` is intercepted by `set_y`,
  the value of `y` is set to 0.42 and the operation is forwarded down the stack.

  5) (TOP) double -> apply (BOTTOM);
  `ed.Normal(0., 1., "y")` is intercepted by `double`, to produce
  `2*ed.Normal(0., 1., "y")`, with the operation being forwarded down the stack.

  6) (TOP) apply (BOTTOM);
  `ed.Normal(0., 1., "y")` is intercepted by `apply`, which simply calls the
  constructor.

  The final values for `x` and `y` inside of `model()` are tensors where `x` is
  a random draw from Normal(0., 1.) doubled, and `y` is a constant 0.84, thus
  z = 2 * Normal(0., 1.) + 0.84.
  """
  try:
    interceptor = _interceptor_stack.stack.pop()
    yield interceptor
  finally:
    _interceptor_stack.stack.append(interceptor)


def interceptable(func):
  """Decorator that wraps `func` so that its execution is intercepted.

  The wrapper passes `func` to the interceptor for the current thread.

  If there is no next interceptor, we perform an "immediate" call to `func`.
  That is, `func` terminates without forwarding its execution to another
  interceptor.

  Args:
    func: Function to wrap.

  Returns:
    The decorated function.
  """
  @functools.wraps(func)
  def func_wrapped(*args, **kwargs):
    with get_next_interceptor() as interceptor:
      return interceptor(func, *args, **kwargs)

  return func_wrapped


@contextmanager
def tape():
  """Context manager for recording interceptable executions onto a tape.

  Similar to `tf.GradientTape`, operations are recorded if they are executed
  within this context manager. In addition, the operation must be registered
  (wrapped) as `ed.interceptable`.

  Yields:
    tape: OrderedDict where operations are recorded in sequence. Keys are
      the `name` keyword argument to the operation (typically, a random
      variable's `name`) and values are the corresponding output of the
      operation. If the operation has no name, it is not recorded.

  #### Examples

  ```python
  from tensorflow_probability import edward2 as ed

  def probabilistic_matrix_factorization():
    users = ed.Normal(0., 1., sample_shape=[5000, 128], name="users")
    items = ed.Normal(0., 1., sample_shape=[7500, 128], name="items")
    ratings = ed.Normal(loc=tf.matmul(users, items, transpose_b=True),
                        scale=0.1,
                        name="ratings")
    return ratings

  with ed.tape() as model_tape:
    ratings = probabilistic_matrix_factorization()

  assert model_tape["users"].shape == (5000, 128)
  assert model_tape["items"].shape == (7500, 128)
  assert model_tape["ratings"] == ratings
  ```

  """
  tape_data = collections.OrderedDict({})

  def record(f, *args, **kwargs):
    """Records execution to a tape."""
    name = kwargs.get("name")
    output = interceptable(f)(*args, **kwargs)
    if name:
      tape_data[name] = output
    return output

  with interception(record):
    yield tape_data
