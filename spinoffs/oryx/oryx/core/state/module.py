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
"""Contains highest-level abstractions for the stateful function API."""
import abc
from typing import Any, Dict, Tuple

from oryx.core import ppl
from oryx.core import primitive
from oryx.core import pytree
from oryx.core.interpreters import harvest
from oryx.core.interpreters import log_prob

__all__ = [
    'VARIABLE',
    'ASSIGN',
    'variable',
    'assign',
    'Module',
]


VARIABLE = 'variable'
ASSIGN = 'assign'


def variable(value, *, name: str, key=None, mode: str = 'strict'):
  """Tags a value as a variable.

  `variable` should be used to initialize state in stateful functions.
  Typically, `variable` will be called with a value downstream of an
  initialization key. The `init` transformation will then pull all values tagged
  as variables in a function body and store them in a `Module`.

  Args:
    value: JAX value to be tagged as variable.
    name: string name for the value.
    key: JAX value that is used to tie in `value`.
      Default value: `None`
    mode: string name for sow mode (see `harvest` documentation).
      Default value: `'strict'`

  Returns:
    The value that was passed in.
  """
  return harvest.sow(value, tag=VARIABLE, name=name, key=key, mode=mode)


def assign(value, *, name: str, key=None, mode: str = 'clobber'):
  """Assigns a value to a variable.

  In a stateful function, `assign` is used define state updates. In particular,
  when a function with an `assign` is transformed using `init`, it returns a
  Module whose `call_and_update` returns the values tagged as `assign` as its
  second output. `init` requires that an assigned value must have a matching
  variable (as defined by the `name`).

  Args:
    value: JAX value to be assigned.
    name: string name for the value.
    key: JAX value that is used to tie in `value`.
      Default value: `None`
    mode: string name for sow mode (see `harvest` documentation).
      Default value: `'clobber'`

  Returns:
    The value that was passed in.
  """
  return harvest.sow(value, tag=ASSIGN, name=name, key=key, mode=mode)


class Module(pytree.Pytree, metaclass=abc.ABCMeta):
  """Encapsulates a parameterized function, along with updates to its state.

  Modules have the dual purpose of acting as containers for state and as
  functions of that state.

  As containers, a Module's `variables()` method returns any encapsulated state,
  and its `flatten` and `unflatten` method are used to register it as a Pytree,
  so that it can be passed in and out of JAX-transformed functions. For example,
  if we have a neural network layer module `layer`, then `layer.variables()`
  returns the weights of the layer and `grad(some_f)(layer)` returns the
  gradient of `some_f` with respect to those weights.

  As functions, a `Module` has three methods: `call`, `update`, and
  `call_and_update`. Conceptually a `Module` represents a parameterized function
  `f_variables(inputs)`, and its `call` method computes the output of the
  function. A module's `update` method returns a new copy of the module for
  a set of inputs with potentially new state (variables). The `call_and_update`
  method returns both the output of the function and an updated module.

  The `__call__` method has some extra logic needed for composing stateful
  functions. If a module `m1` is called in the body of another module `m2`
  we would like `m2` to "inherit" the state in `m1`, so that either `m1`
  or `m1`'s variables appear in `m2`'s variables. If a module has a non-`None`
  `name`, then we'd like it to appear in `m2`'s variables with name `name`,
  and if not, we'd like variables in `m1` to appear in `m2` directly. To emulate
  this behavior, we have the `__call__` method call `assign` on the updated
  module (or its member variables) to appropriately update the state to an
  outer stateful context.
  """

  def __init__(self, *, name=None):
    self.name = name
    self.__name__ = name or 'module'

  @abc.abstractmethod
  def call(self, *args, **kwargs) -> Any:
    pass

  @abc.abstractmethod
  def update(self, *args, **kwargs) -> 'Module':
    pass

  @abc.abstractmethod
  def call_and_update(self, *args, **kwargs) -> Tuple[Any, 'Module']:
    pass

  def __call__(self, *args, **kwargs) -> Any:
    """Emulates a regular function call.

    A `Module`'s dunder call will ensure state is updated after the function
    call by calling `assign` on the updated state before returning the output of
    the function.

    Args:
      *args: The arguments to the module.
      **kwargs: The keyword arguments to the module.

    Returns:
      The output of the module.
    """
    out, new_module = self.call_and_update(*args, **kwargs)
    if self.name is not None:
      new_module = assign(new_module, name=self.name)
      out = primitive.tie_in(new_module, out)
    else:
      variables = {
          k: assign(val, name=k) for k, val in new_module.variables().items()
      }
      out = primitive.tie_in(variables, out)
    return out

  @abc.abstractmethod
  def flatten(self):
    pass

  @abc.abstractclassmethod
  def unflatten(cls, data, xs):
    pass

  @abc.abstractmethod
  def variables(self) -> Dict[str, Any]:
    raise NotImplementedError


@ppl.log_prob.register(Module)
def module_log_prob(module, *args, **kwargs):
  return log_prob.log_prob(module, *args, **kwargs)  # pytype: disable=wrong-arg-count
