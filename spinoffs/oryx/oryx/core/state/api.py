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
"""Module for single-dispatch functions for handling state.

This module defines single-dispatch functions that are used to construct
and use `Module`s. The main functions are `init`, `call_and_update` and `spec`.

They are all single-dispatch functions, meaning they have specific
implementations depending on the type of their first inputs. These
implementations can be provided from outside of this library, so they act as a
general API for handling state.

# Methods

## `init`
`init` converts an input object into an "initializer" function, i.e. one that
takes in a random PRNGKey and a set of inputs and returns a `Module`.
`function.py` registers Python functions with this transformations and another
potential application is neural network layers.

## `call_and_update`
`call_and_update` executes the computation associated with an input
object, returning the output and a copy of the object with updated state.
For example, for a `Module`, `call_and_update(module, ...)`
runs `module.call_and_update` but this behavior could be defined for arbitrary
objects. For example in `registrations.py` we provide some default registrations
for various Python data structures like lists and tuples.

We also provide a `call` and `update` function which are wrappers around
`call_and_update`.

## `spec`

`spec` has the same API as `init` without the PRNGKey and returns the shape
of the output that would result from calling the input object.

# Example:
```python
def f(x, init_key=None):
  w = module.variable(random.normal(init_key, x.shape), name='w')
  w = module.assign(w + 1., name='w')
  return np.dot(w, x)

api.spec(f)(random.PRNGKey(0), np.ones(5))  # ==> ArraySpec((), np.float32)

m = api.init(f)(random.PRNGKey(0), np.ones(5))
m.variables()  # ==> {'w': ...}

output, new_module = api.call_and_update(m, np.ones(5))
```
"""
import functools

from typing import Any, Callable, Optional

import jax.numpy as np

from oryx.core.state import module

__all__ = [
    'init',
    'ArraySpec',
    'call_and_update',
    'call',
    'update',
    'make_array_spec',
    'spec'
]

NestedSpec = Any


@functools.singledispatch
def init(obj, *, name: Optional[str] = None) -> Callable[..., module.Module]:
  """Transforms an object into a function that initializes a module.

  `init` is used to create `Module`s from other Python data structures or
  objects. For example, `init`-ing function would involve tracing
  it to find `variable` and `assign` primitives, which would then be used to
  define a module's `variables()` and its `call_and_update` method.
  Another example would be a neural network template object, which can be
  `init`-ed to return a function that returns a neural network layer module.

  Args:
    obj: The object to be initialized.
    name: A string name for the initialized module..
      Default value: `None`

  Returns:
    A function that when called, returns an initialized `Module` object.
  """
  raise NotImplementedError(f'`init` not implemented for {type(obj)}')


class ArraySpec:
  """Encapsulates shape and dtype of an abstract array."""

  def __init__(self, shape, dtype=np.float32):  # pylint: disable=redefined-outer-name
    """Constructor for ArraySpec object.

    Args:
      shape: an int or list/tuple representing the shape of an array.
      dtype: a Numpy dtype.
    """
    if not isinstance(shape, (list, tuple)):
      shape = (shape,)
    self.shape = shape
    self.dtype = dtype

  def __eq__(self, other):
    return (self.shape == other.shape) and (self.dtype == other.dtype)

  def __repr__(self):
    return 'ArraySpec({}, dtype={})'.format(
        self.shape, self.dtype)


@functools.singledispatch
def call_and_update(obj, *args, **kwargs) -> Any:
  raise NotImplementedError(
      f'`call_and_update` not implemented for {type(obj)}')


def call(obj, *args, **kwargs) -> Any:
  return call_and_update(obj, *args, **kwargs)[0]


def update(obj, *args, **kwargs) -> Any:
  return call_and_update(obj, *args, **kwargs)[1]


Shape = ArraySpec


def make_array_spec(x):
  return ArraySpec(x.shape, dtype=x.dtype)


@functools.singledispatch
def spec(obj) -> Callable[..., NestedSpec]:
  """A general purpose transformation for getting the output shape and dtype of an object.

  `spec` is used to query the output shapes and dtypes of objects like Modules
  or functions.

  Args:
    obj: The object whose `spec` will be queried.

  Returns:
    A function when passed in values or specs will return an output spec.
  """
  raise NotImplementedError(f'`spec` not implemented for {type(obj)}')
