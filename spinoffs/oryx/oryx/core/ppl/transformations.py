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
"""Module for probabilistic programming transformations.

## Probabilistic programs
A probabilistic program is defined as a JAX function that takes in a
`jax.random.PRNGKey` as its first input, and any number of subsequent
conditioning arguments. The output of the function is the output of the
probabilistic program.

### A simple program:
```python
def f(key):
  return random.normal(key)
```
In this program, we sample a random normal variable and return it. Conceptually,
this program represents the distribution `p(x) = Normal(0, 1)`.

### A conditional program:
```python
def f(key, z):
  return z * random.normal(key)
```

In this program we sample a distribution conditional on `z` (i.e. a distribution
`p(x | z)`).

## Function transformations

The goal of the probabilistic programming package is to enable writing simple
programs and to use program transformations to create complexity. Here we
outline some of the transformations available in the module.

### `random_variable`
`random_variable` is a general purpose function that can be used to 1) tag
values for use in downstream transforms and 2) convert objects into
probabilistic programs. In implementation, `random_variable` is a
single-dispatch function whose implementation for functions and objects is
already registered. By default, it will tag a value with a name and will only
work on JAX types (e.g. DeviceArrays and  tracers). We also register an
implementation for function types, where it returns the original function but
when provided the name, tags the output of the function. The registry enables
objects such as TensorFlow Probability distributions to register as a random
variable-like with Oryx.

Tagging a value in a probabilistic program as a random variable enables it to
be used by downstream transforms described below, such as `joint_sample`,
`conditional`, `intervene`, and `graph_replace`.

### `log_prob`
`log_prob` takes a probabilistic program and returns a function that computes
the log probability of a sample. It relies on the fact that certain sampling
primitives have been registered with the transformation. Specifically, it
returns a program that when provided an output from the program attempts to
compute the log-probability of *all* random samples in the program.

Examples:
```python
def f1(key):
  return random_normal(key)
log_prob(f1)(0.)  # ==> -0.9189385
log_prob(f1)(1.)  # ==> -1.4189385

def f2(key):
  k1, k2 = random.split(key)
  return [random_normal(k1), random_normal(k2)]
log_prob(f2)([0., 0.])  # ==> -1.837877
```

For programs that sample variables that aren't returned as part of the output of
the program (latent variables), the `log_prob` of the program will error,
because there is insufficient information from the output of the program to
compute the log probabilities of all the random samples in the program.

```python
def f(key):
  k1, k2 = random.split(key)
  z = random_normal(k1)
  return random_normal(k2) + z
log_prob(f)(0.)  # ==> Error!
```

In this case, we can use `joint_sample` to transform it into one that returns
values for all the latent variables, and `log_prob` will compute the joint
log-probability of the function.

`log_prob` is also able to invert bijective functions and compute the
change-of-variables formula for probabilistic programs. For more details,
see `oryx.core.interpreters.log_prob`.

```python
def f(key):
  return np.exp(random_normal(key))
log_prob(f)(np.exp(0.))  # ==> -0.9189385
log_prob(f)(np.exp(1.))  # ==> -2.4189386
```

### `trace`
`trace` takes a probabilistic program and returns a program that when executed
returns both the original program's output and a dictionary that includes the
latent random variables sampled during the original program's execution.

Example:
```python
def f(key):
  k1, k2 = random.split(key)
  z = random_variable(random.normal, name='z')(k1)
  return z + random_variable(random.normal, name='x')(k2)
trace(f)(random.PRNGKey(0))  # ==> (0.1435, {'z': -0.0495, 'x': 0.193})
```

### `joint_sample`
`joint_sample` takes a probabilistic program and returns another one that
returns a dictionary mapping named latent variables (tagged by
`random_variable`) to their latent values during execution.

Example:
```python
def f(key):
  k1, k2 = random.split(key)
  z = random_variable(random.normal, name='z')(k1)
  return z + random_variable(random.normal, name='x')(k2)
joint_sample(f)(random.PRNGKey(0))  # ==> {'z': -0.0495, 'x': 0.193}
```

### `joint_log_prob`
`joint_log_prob` takes a probabilistic program and returns a function that
computes a log probability of dictionary mapping names to values corresponding
to random variables during the program's execution. It is the composition of
`log_prob` and `joint_sample`.

Example:
```python
def f(key):
  k1, k2 = random.split(key)
  z = random_variable(random.normal, name='z')(k1)
  return z + random_variable(random.normal, name='x')(k2)
joint_log_prob(f)({'z': 0., 'x': 0.})  # ==> -1.837877
```

### `block`
`block` takes a probabilistic program and a sequence of string names and returns
the same program except that downstream transformations will ignore the provided
names.

Example:
```python
def f(key):
  k1, k2 = random.split(key)
  z = random_variable(random.normal, name='z')(k1)
  return z + random_variable(random.normal, name='x')(k2)
joint_sample(block(f, names=['x']))(random.PRNGKey(0))  # ==> {'z': -0.0495}
```


### `intervene`
`intervene` takes a probabilistic program and a dictionary mapping names to
values of intervened random variables, and returns a new probabilistic program.
The new program runs the original, but when sampling a tagged random variable
whose name is present in the dictionary, it instead substitutes in the provided
value.

```python
def f1(key):
  return random_variable(random.normal, name='x')(key)
intervene(f1, x=1.)(random.PRNGKey(0))  # => 1.

def f2(key):
  k1, k2 = random.split(key)
  z = random_variable(random.normal, name='z')(k1)
  return z + random_variable(random.normal, name='x')(k2)
intervene(f2, z=1., x=1.)(random.PRNGKey(0))  # => 2.
```


### `conditional`
`conditional` is similar to `intervene`, except instead of taking a dictionary
of observations, it takes a list of names and returns a conditional
probabilistic program which takes additional arguments corresponding to random
variables with the aforementioned list of names.

Example:
```python
def f(key):
  k1, k2 = random.split(key)
  z = random_variable(random.normal, name='z')(k1)
  return z + random_variable(random.normal, name='x')(k2)
conditional(f, ['z'])(random.PRNGKey(0), 0.)  # => -1.25153887
conditional(f, ['z'])(random.PRNGKey(0), 1.)  # => -0.25153887
conditional(f, ['z', 'x'])(random.PRNGKey(0), 1., 2.)  # => 3.
```


### `graph_replace`
`graph_replace` is a transformation that executes the original program but
with new inputs and outputs specified by random variable names. Input names
allow injecting values for random variables in the program, and the values of
random variables corresponding to output names are returned.

Example:
```python
def f(key):
  k1, k2, k3 = random.split(key, 3)
  z = random_variable(random_normal, name='z')(k1)
  x = random_variable(lambda key: random_normal(key) + z, name='x')(k2)
  y = random_variable(lambda key: random_normal(key) + x, name='y')(k3)
  return y
graph_replace(f, 'z', 'y') # returns a program p(y | z) with a latent variable x
graph_replace(f, 'z', 'x') # returns a program p(x | z)
graph_replace(f, 'x', 'y') # returns a program p(y | x)
```
"""
import functools
import types

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from jax import util as jax_util

from oryx.core import primitive
from oryx.core.interpreters import harvest
from oryx.core.interpreters import log_prob as lp
from oryx.core.ppl import plate_util

__all__ = [
    'block',
    'random_variable',
    'rv',
    'nest',
    'log_prob',
    'joint_sample',
    'joint_log_prob',
    'intervene',
    'conditional',
    'graph_replace',
]

Program = Callable[..., Any]
Scalar = Any
LogProbFunction = Callable[..., Scalar]

RANDOM_VARIABLE = 'random_variable'

safe_zip = jax_util.safe_zip

nest = harvest.nest


@functools.singledispatch
def random_variable(obj,
                    *,
                    name: Optional[str] = None,
                    plate: Optional[str] = None) -> Program:  # pylint: disable=redefined-outer-name
  """A single-dispatch function used to tag values and the outputs of programs.

  `random_variable` is a single-dispatch function that enables registering
  custom types. Its default implementation is to tag input value with a name
  and return it.

  Args:
    obj: A JAX type to be tagged.
    name: A string name to tag input value, cannot be `None`.
    plate: A string named axis for this random variable's plate.

  Returns:
    The input value.
  """
  if name is None:
    raise ValueError(f'Cannot call `random_variable` on {type(obj)} '
                     'without passing in a name.')
  if plate is not None:
    raise ValueError(f'Cannot call `random_variable` on {type(obj)} '
                     'with a plate.')
  return harvest.sow(obj, tag=RANDOM_VARIABLE, name=name, mode='strict')


def plate(f: Optional[Program] = None, name: Optional[str] = None):
  """Transforms a program into one that draws samples on a named axis.

  In graphical model parlance, a plate designates independent random variables.
  The `plate` transformation follows this idea, where a `plate`-ed program
  draws independent samples. Unlike `jax.vmap`-ing a program, which also
  produces independent samples with positional batch dimensions, `plate`
  produces samples with implicit named axes. Named axis support is useful for
  other JAX transformations like `pmap` and `xmap`.

  Specifically, a `plate`-ed program creates a different key for each axis
  of the named axis. `log_prob` reduces over the named axis to produce a single
  value.

  Example usage:
  ```python
  @ppl.plate(name='foo')
  def model(key):
    return random_variable(random.normal)(key)
  # We can't call model directly because there are implicit named axes present
  try:
    model(random.PRNGKey(0))
  except NameError:
    print('No named axis present!')
  # If we vmap with a named axis, we produce independent samples.
  vmap(model, axis_name='foo')(random.split(random.PRNGKey(0), 3))
  # ==> [0.58776844, -0.4009751, 0.01193586]
  ```

  Args:
    f: a `Program` to transform. If `f` is `None`, `plate` returns a decorator.
    name: a `str` name for the plate which can used as a name axis in JAX
      functions and transformations.

  Returns:
    A decorator if `f` is `None` or a transformed program if `f` is provided.
    The transformed program behaves produces independent across a named
    axis with name `name`.
  """

  def transform(f: Program) -> Program:
    return plate_util.make_plate(f, name=name)

  if f is not None:
    return transform(f)
  return transform


# Alias for random_variable
rv = random_variable


@random_variable.register(types.FunctionType)
@random_variable.register(functools.partial)
def function_random_variable(f: Program,
                             *,
                             name: Optional[str] = None,
                             plate: Optional[str] = None) -> Program:  # pylint: disable=redefined-outer-name
  """Registers functions with the `random_variable` single dispatch function.

  Args:
    f: A probabilistic program.
    name (str): A string name that is used to when tagging the output of `f`.
    plate (str): A string named axis for this random variable's plate.

  Returns:
    A probabilistic program whose output is tagged with `name`.
  """

  def wrapped(*args, **kwargs):
    fun = f
    if plate is not None:
      fun = plate_util.make_plate(fun, name=plate)
    if name is not None:
      return random_variable(nest(fun, scope=name)(*args, **kwargs), name=name)
    return fun(*args, **kwargs)

  return wrapped


@functools.singledispatch
def log_prob(obj: object) -> LogProbFunction:
  """Returns a function that computes the log probability of a sample."""
  raise NotImplementedError(f'`log_prob` not implemented for type: {type(obj)}')


@log_prob.register(types.FunctionType)
@log_prob.register(functools.partial)
def function_log_prob(f: Program) -> LogProbFunction:
  """Registers the `log_prob` for probabilistic programs.

  See `core.interpreters.log_prob` for details of this function's
  implementation.

  Args:
    f: A probabilitic program.

  Returns:
    A function that computes the log probability of a sample from the program.
  """
  return lp.log_prob(f)


def trace(f: Program) -> Program:
  """Returns a program that additionally outputs sampled latents."""
  return harvest.call_and_reap(f, tag=RANDOM_VARIABLE)


def trace_log_prob(f: Program) -> LogProbFunction:
  """Returns a function that computes the log probability program's output and its random variables."""
  return lambda obs, latents, *args: log_prob(trace(f))((obs, latents), *args)


def joint_sample(f: Program) -> Program:
  """Returns a program that outputs a dictionary of latent random variable samples."""
  return harvest.reap(f, tag=RANDOM_VARIABLE)


def joint_log_prob(f: Program) -> LogProbFunction:
  """Returns a function that computes the log probability of all of a program's random variables."""
  return log_prob(joint_sample(f))


def block(f: Program, names: Sequence[str]) -> Program:
  """Returns a program that removes the provided names from transformations."""

  def program(key, *args, **kwargs):
    return harvest.plant(
        f,
        tag=RANDOM_VARIABLE,
        blocklist=names)({}, key, *args, **kwargs)

  return program


def intervene(f: Program, **observations: Dict[str, Any]) -> Program:
  """Transforms a program into one where provided random variables are fixed.

  `intervene` is a probabilistic program transformation that fixes the values
  for certain random samples in an input program. A probabilistic program may
  sample intermediate latent random variables while computing its output.
  Observing those random variables converts them into deterministic constants
  that are just used in the forward computation.

  Random variables that are intervened are *no longer random variables*. This
  means that if a variable `x` is intervened , it will no longer appear in the
  `joint_sample` of a program and its `log_prob` will no longer be computed as
  part of a program's `log_prob`.

  ## Examples:

  ### Simple usage:
  ```python
  def model(key):
    return random_variable(random.normal, name='x')(key)
  intervene(model, x=1.)(random.PRNGKey(0))  # => 1.
  ```
  ### Multiple random variables:
  ```python
  def model(key):
    k1, k2 = random.split(key)
    z = random_variable(random.normal, name='z')(k1)
    return z + random_variable(random.normal, name='x')(k2)
  intervene(model, z=1., x=1.)(random.PRNGKey(0))  # => 2.
  ```

  Args:
    f: A probabilistic program.
    **observations: A dictionary mapping string names for random variables to
      values.

  Returns:
    A probabilistic program that executes its input program with provided
    variables fixed to their values.
  """

  def wrapped(*args, **kwargs):
    return harvest.plant(
        f, tag=RANDOM_VARIABLE)(observations, *args, **kwargs)

  return wrapped


def conditional(f: Program, names: Union[List[str], str]) -> Program:
  """Conditions a probabilistic program on random variables.

  `conditional` is a probabilistic program transformation that converts latent
  random variables into conditional inputs to the program. The random variables
  that are moved to the input are specified via a list of names that correspond
  to tagged random samples from the program. The final arguments to the output
  program correspond to the list of names passed into `conditional`.

  Random variables that are conditioned are *no longer random variables*. This
  means that if a variable `x` is conditioned on, it will no longer appear in
  the `joint_sample` of a program and its `log_prob` will no longer be computed
  as part of a program's `log_prob`.

  ## Example:
  ```python
  def model(key):
    k1, k2 = random.split(key)
    z = random_variable(random.normal, name='z')(k1)
    return z + random_variable(random.normal, name='x')(k2)
  conditional(model, ['z'])(random.PRNGKey(0), 0.)  # => -1.25153887
  conditional(model, ['z'])(random.PRNGKey(0), 1.)  # => -0.25153887
  conditional(model, ['z'. 'x'])(random.PRNGKey(0), 1., 2.)  # => 3.
  ```

  Args:
    f: A probabilistic program.
    names: A string or list of strings correspond to random variable names in
      `f`.

  Returns:
    A probabilistic program with additional conditional inputs.
  """
  if isinstance(names, str):
    names = [names]
  num_conditions = len(names)

  def wrapped(*args, **kwargs):
    if num_conditions > 0:
      args, condition_values = args[:-num_conditions], args[-num_conditions:]
      conditions = dict(safe_zip(names, condition_values))
    else:
      conditions = {}
    return intervene(f, **conditions)(*args, **kwargs)

  return wrapped


def graph_replace(f: Program, input_names: Union[List[str], str],
                  output_names: Union[List[str], str]) -> Program:
  """Transforms a program to one with new inputs and outputs.

  `graph_replace` enables redefining the inputs and outputs of a probabilistic
  program that samples latent random variables. It takes a program, along
  with a list of input names and output names, and returns a function from
  the random variables corresponding to the input names to the ones
  corresponding to the output names.

  Args:
    f: A probabilistic program.
    input_names: A string or list of strings that correspond to random
      variables.
    output_names: A string or list of strings that correspond to random
      variables.

  Returns:
    A probabilistic program that maps the random variables corresponding to the
    input names to those of the output names.
  """
  if isinstance(output_names, str):
    output_names = [output_names]
    single_output = True
  else:
    single_output = False

  def wrapped(*args, **kwargs):
    latents = harvest.reap(
        conditional(f, input_names), tag=RANDOM_VARIABLE)(*args, **kwargs)
    outputs = [latents[name] for name in output_names]
    latents = {
        name: harvest.sow(value, tag=RANDOM_VARIABLE, name=name, mode='strict')
        for name, value in latents.items()
        if name not in output_names
    }
    if single_output:
      outputs = outputs[0]
    return primitive.tie_in(latents, outputs)

  return wrapped
