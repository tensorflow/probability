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
"""Registers state dispatch functions with Python data structures.

In this module, we provide registrations for some Python data structures
for the single-dispatch functions in api.py. This is not necessarily meant for
public consumption, but ensures some reasonable behavior for common data
structures.

# Tuple
Conceptually, a tuple corresponds to "fanning-out" a computation. `init`-ing
a tuple initializes its components and an initializer for a tuple of `Module`s.
`call_and_update`-ing a tuple will recursively `call_and_update` each
element of the tuple all on the same input, returning a tuple of outputs
and a tuple of updated objects. `spec`-ing a tuple will return a tuple
of output `Spec`s.

### Example:
```python
obj = (lambda x: x, lambda x: x + 1., lambda x: x + 2.)
api.spec(obj)(api.Shape(()))  # ==> (ArraySpec(()), ArraySpec(()),
                              # ==>  ArraySpec(()))
m = api.init(obj)(random.PRNGKey(0), 1.)
type(m)  # ==> tuple
api.call(m, 1.)  #==> (1., 2., 3.)


# List
A list corresponds conceptually to a sequence of computations where the
output of one is input of the next one, i.e. function composition.
init`-ing a list corresponds to creating a list of `Module`s where the intent is
to compose them. `call_and_update` of a module will return the output of the
final object and a list of updated objects. `spec` will return the output
`Spec` for the final object in the list.

### Example:
```python
obj = [lambda x: x, lambda x: x + 1., lambda x: x + 2.]
api.spec(obj)(api.Shape(()))  # ==> ArraySpec(())
m = api.init(obj)(random.PRNGKey(0), 1.)
type(m)  # ==> list
api.call(m, 1.)  # ==> 4.
"""
from jax import random

from oryx.core.state import api

__all__ = []


@api.init.register(tuple)
def tuple_init(tupl, *, name=None):
  """Maps `init` across a tuple."""
  if not tupl:
    raise ValueError('Cannot `init` empty tuple.')
  def wrapped(init_key, *args, **kwargs):
    if init_key is None:
      init_keys = (None,) * len(tupl)
    else:
      init_keys = random.split(init_key, len(tupl))
    names = [None if name is None else f'{name}_{i}' for i
             in range(len(tupl))]
    modules = tuple(api.init(t, name=name)(init_key, *args, **kwargs)
                    for t, name, init_key in zip(tupl, names, init_keys))
    return tuple.__new__(tupl.__class__, modules)
  return wrapped


@api.call_and_update.register(tuple)
def tuple_call_and_update(tupl, *args, **kwargs):
  """Maps `call_and_update` across a tuple."""
  out, new_tupl = zip(*[api.call_and_update(t, *args, **kwargs) for t in tupl])
  return (
      tuple.__new__(tupl.__class__, out),
      tuple.__new__(tupl.__class__, new_tupl))


@api.spec.register(tuple)
def tuple_spec(tupl):
  assert tupl, 'Cannot spec an empty tuple.'
  def wrapped(*args, **kwargs):
    args = tuple(api.spec(t)(*args, **kwargs) for t in tupl)
    return tuple.__new__(tupl.__class__, args)
  return wrapped


@api.init.register(list)
def list_init(inits, *, name=None):
  """Maps `init` across a tuple."""
  def wrapped(init_key, *args, **kwargs):
    if init_key is not None:
      init_keys = random.split(init_key, len(inits))
    else:
      init_keys = (None,) * len(inits)
    modules = []
    for i, (elem_init, init_key) in enumerate(zip(inits, init_keys)):
      if isinstance(args, api.ArraySpec):
        args = (args,)
      elem_name = None if name is None else '{}_{}'.format(name, i)
      elem = api.init(elem_init, name=elem_name)(init_key, *args, **kwargs)  # pylint: disable=assignment-from-no-return
      args = api.spec(elem_init)(*args, **kwargs)  # pylint: disable=assignment-from-no-return
      modules.append(elem)
    return inits.__class__(modules)
  return wrapped


@api.call_and_update.register(list)
def list_call_and_update(modules, *args, **kwargs):
  """Maps `call_and_update` across a list."""
  modules_out = []
  for module in modules:
    if not isinstance(args, tuple):
      args = (args,)
    args, new_module = api.call_and_update(module, *args, **kwargs)  # pylint: disable=assignment-from-no-return
    modules_out.append(new_module)
  return args, modules_out


@api.spec.register(list)
def list_spec(inits):
  def wrapped(*specs, **kwargs):
    for elem_init in inits:
      if isinstance(specs, api.ArraySpec):
        specs = (specs,)
      specs = api.spec(elem_init)(*specs, **kwargs)  # pylint: disable=assignment-from-no-return
    return specs
  return wrapped
