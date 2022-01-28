# Copyright 2021 The TensorFlow Probability Authors.
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
"""Utilities for defining state variables with coroutines.

The utilities in this file support a simple state-management pattern that can be
used to implement 'trainable' objects (distributions, bijectors, etc.). The
pattern is to define state variables by yielding `Parameter` namedtuples,
for which the environment will return a `Tensor` value. For example, to define
a trainable normal distribution, we might write:

```python
def trainable_normal(shape):
  loc = yield trainable_state_util.Parameter(
    init_fn=functools.partial(tf.random.stateless_normal, shape=[]),
    name='loc')
  scale_bijector = tfb.Softplus()
  scale = yield trainable_state_util.Parameter(
    init_fn=(  # Initialize scale to a positive value.
      lambda seed: scale_bijector(tf.random.stateless_normal([], seed=seed))),
    constraining_bijector=scale_bijector,
    name='scale')
  return tfd.Normal(loc, scale=scale)
```

Because this is a generator, it can't be called directly. This module
provides wrappers that interpret this generator to build either a
*stateless* trainable distribution, represented by an `init_fn` and `apply_fn`,
or a *stateful* distribution, represented as a Distribution instance
parameterized by `tf.Variable`.

```python
build_trainable_normal = as_stateful_builder(trainable_normal)
build_trainable_normal_stateless = as_stateless_builder(
  trainable_normal)
```

This generator pattern is a spiritual cousin to the variable abstractions
provided by JAX-specific state libraries such as Flax/Linen or Haiku. Its main
advantage is that it inherits TFP's cross-compatibility with both TF and JAX,
allowing us to implement trainable distributions that work across all backends.
It is not currently intended for external use. In particular, JAX users who do
not need TF compatibility will likely be better served by one of the
JAX-specific libraries. (and TF users will likely prefer the more idiomatic
`tf.Variable` pattern).

### Stateless trainable distributions

A stateless trainable distribution (bijector, etc) is represented by an
`init_fn` and an `apply_fn`:

```
initial_parameters = init_fn(seed)
# ==> `initial_parameters` is a StructTuple of unconstrained Tensor values.

dist = apply_fn(*initial_parameters)
# Passing the list as a single arg is also supported:
dist = apply_fn(initial_parameters)
```

This supports optimizing over distribution parameters by differentiating through
the `apply_fn` :

```
import optax  # Requires JAX.
init_fn, apply_fn = build_trainable_normal_stateless(shape=[])

# Find the maximum likelihood distribution given observed data.
x_observed = [3., -2., 1.7]
mle_parameters, losses = tfp.math.minimize_stateless(
  loss_fn=lambda *params: -apply_fn(*params).log_prob(x_observed),
  init=init_fn(seed=seed),
  optimizer=optax.adam(0.1),
  num_steps=100)
mle_dist = apply_fn(mle_parameters)
print(f"Estimated normal distribution with mean {mle_dist.mean()} and "
      "stddev {mle_dist.stddev()}")
```

# Stateful trainable distributions (TF only)

TensorFlow also supports stateful objects with `tf.Variable` parameters:

```python
build_trainable_normal = make_stateful(trainable_normal)
trainable_dist = build_trainable_normal(shape=[], seed=seed)

# Find the maximum likelihood distribution given observed data.
x_observed = [3., -2., 1.7]
losses = tfp.math.minimize(
  loss_fn=lambda: -trainable_dist.log_prob(x_observed),
  optimizer=tf.optimizer.Adam(0.1),
  num_steps=100)

# Distribution was updated in-place.
print(f"Estimated normal distribution with mean {trainable_dist.mean()} and "
      "stddev {trainable_dist.stddev()}")
```

"""

import collections
import functools
import inspect
import re
import types

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import structural_tuple


JAX_MODE = False
NUMPY_MODE = False


__all__ = [
    'as_stateless_builder',
    'as_stateful_builder',
    'Parameter'
]


# Generic replacements for the `Yields` section of a generator docstring.
_STATEFUL_RETURNS_DOCSTRING = """
Returns:
  instance: instance parameterized by trainable `tf.Variable`s.

"""

_STATELESS_RETURNS_DOCSTRING = """
Returns:
  init_fn: Python callable with signature `initial_parameters = init_fn(seed)`.
  apply_fn: Python callable with signature `instance = apply_fn(*parameters)`.

"""


class Parameter(
    collections.namedtuple('Parameter',
                           ['init_fn', 'constraining_bijector', 'name'],
                           defaults=[None, 'parameter'])):
  """Specifies a trainable parameter.

  Elements:
    init_fn: Python `callable` that takes either no arguments, or a single
      argument `seed`, and returns a `Tensor` initial parameter value. If a
      `constraining_bijector` is specified, the initial value is in the
      constrained parameter space; the unconstrained 'raw' value is computed
      as `raw_parameter = constraining_bijector.inverse(init(seed))`.
    constraining_bijector: Optional `tfb.Bijector` instance transforming an
      unconstrained Tensor value into the parameter space.
      Default value: `None`.
    name: Optional Python `str` name for this parameter.
      Default value: 'parameter'.
  """
  __slots__ = ()


def _call_init_fn(init_fn, seed):
  """Calls `init_fn` with `seed` as a named or positional arg."""
  if not callable(init_fn):
    return tf.convert_to_tensor(init_fn)  # Non-callable initial value.
  try:
    return init_fn(seed=seed)  # Try passing seed as named arg.
  except TypeError:
    pass
  try:
    return init_fn()  # Maybe init_fn is deterministic?
  except (TypeError, ValueError):
    pass
  return init_fn(seed)  # Fall back to passing seed as positional arg.


def _get_unused_parameter_name(name, already_used_names):
  """Returns a string `name` different from all currently used names."""
  already_used_names = set(already_used_names)
  i = 1
  unique_name = name
  while unique_name in already_used_names:
    # Start at 0001 since the already-used bare `name` was implicitly 0000.
    unique_name = name + '_{:04d}'.format(i)
    i += 1
  return unique_name


def _initialize_parameters(generator, seed=None):
  """Samples initial values for all parameters yielded by a generator.

  Args:
    generator: Python generator that yields initialization callables
      (which take a `seed` and return a (structure of) `Tensor`(s)),
      returns a value, and has no side effects. See module description.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
  Returns:
    raw_parameters: Python list of `Tensor` (or structure of `Tensor`s) initial
      parameter values returned from the yielded callables.
  """
  gen = generator()
  if not isinstance(gen, types.GeneratorType):
    raise ValueError('Expected generator but saw function: {}. A generator '
                     'must contain at least one `yield` statement. To define a '
                     'trivial generator, which yields zero times, a `yield` '
                     'statement may be placed after `return`, but must still '
                     'be present.'.format(generator))

  raw_parameters = []
  parameter_names = []
  param_value = None
  try:
    while True:
      parameter = gen.send(param_value)
      if not hasattr(parameter, 'init_fn'):
        raise ValueError('Expected generator to yield a '
                         'trainable_state_util.Parameter namedtuple, but saw '
                         '{} instead.'.format(parameter))
      seed, local_seed = samplers.split_seed(seed, n=2)
      # Note: this framework guarantees that the `init_fn` is only ever called
      # here, immediately after being yielded before control is returned
      # to the coroutine. This allows the coroutine to safely incorporate
      # loop-dependent state in the closure of `init_fn` if desired.
      param_value = _call_init_fn(parameter.init_fn, seed=local_seed)
      raw_value = (param_value if parameter.constraining_bijector is None
                   else parameter.constraining_bijector.inverse(param_value))
      raw_parameters.append(raw_value)
      parameter_names.append(
          _get_unused_parameter_name(
              parameter.name or 'parameter', parameter_names))
  except StopIteration:
    pass
  return structural_tuple.structtuple(parameter_names)(*raw_parameters)


def _sanitize_parameters(*raw_parameters):
  if len(raw_parameters) == 1 and isinstance(raw_parameters[0], (list, tuple)):
    # Do the right thing if called with an explicit parameters list as the:
    # sole arg (`_apply_parameters(generator, raw_parameters)`).
    # This is unambiguous for single-parameter generators as long as
    # their parameters returned by `init_fn` are Tensor-like values rather
    # than not lists or tuples.
    raw_parameters = raw_parameters[0]
  return raw_parameters


def _apply_parameters(generator, *raw_parameters):
  """Runs the generator with the given parameter values and returns the result.

  Args:
    generator: Python generator that yields initialization callables
      (which take a `seed` and return a (structure of) `Tensor`(s)),
      returns a value, and has no side effects. See module description.
    *raw_parameters: iterable of `Tensor` (or structure of `Tensor`) parameter
      values, of length corresponding to the number of `yield` invocations
      made by the generator. Alternately, a length-1 tuple containing such
      an iterable.
  Returns:
    retval: the value returned by the generator when run with the given
      parameters.
  """
  raw_parameters = _sanitize_parameters(*raw_parameters)
  gen = generator()
  if not isinstance(gen, types.GeneratorType):
    raise ValueError('Expected generator but saw function: {}. A generator '
                     'must contain at least one `yield` statement. To define a '
                     'trivial generator, which yields zero times, a `yield` '
                     'statement may be placed after `return`, but must still '
                     'be present.'.format(generator))
  try:
    parameter = next(gen)
    for param_value in raw_parameters:
      if parameter.constraining_bijector is not None:
        param_value = parameter.constraining_bijector.forward(
            # Disable bijector cache so that gradients are defined.
            tf.nest.map_structure(tf.identity, param_value))
      parameter = gen.send(param_value)
  except StopIteration as e:
    return e.value
  raise ValueError('Insufficient parameters provided for generator {}. Saw '
                   'parameters: {}.'.format(gen, raw_parameters))


def as_stateless_builder(generator):
  """Wraps a generator to build a stateless init_fn/apply_fn pair."""

  @functools.wraps(generator)
  def build_stateless_trainable(*args, **kwargs):
    g = functools.partial(generator, *args, **kwargs)
    init_fn = lambda seed=None: _initialize_parameters(g, seed)
    apply_fn = lambda *parameters: _apply_parameters(g, *parameters)
    return init_fn, apply_fn

  # Replace `Yields` section of docstring with `Returns` init_fn/apply_fn.
  if generator.__doc__:
    doc = inspect.cleandoc(generator.__doc__)
    doc = re.sub(
        r'Yields:\n(.+\n)+\n', _STATELESS_RETURNS_DOCSTRING, doc + '\n\n')
    build_stateless_trainable.__doc__ = doc

  return build_stateless_trainable


def as_stateful_builder(generator):
  """Wraps a generator to build trainable objects parameterized by Variables."""

  def error_no_variables(*args, **kwargs):
    """Raises error under JAX and Numpy backends."""
    raise ValueError('TensorFlow is required for `tf.Variable`s. Only '
                     'stateless representations are supported under the JAX '
                     'and Numpy backends. ')
  if JAX_MODE or NUMPY_MODE:
    return error_no_variables

  # TF-specific imports.
  from tensorflow_probability.python.experimental.util import deferred_module  # pylint: disable=g-import-not-at-top
  def build_stateful_trainable(*args, seed=None, **kwargs):
    g = functools.partial(generator, *args, **kwargs)
    params = _initialize_parameters(g, seed=seed)
    params_as_variables = []
    for name, value in params._asdict().items():
      # Params may themselves be structures, in which case there's no 1:1
      # mapping between param names and variable names. Currently we just give
      # the same name to all variables in a param structure and let TF sort
      # things out.
      params_as_variables.append(
          tf.nest.map_structure(
              lambda t, n=name: t if t is None else tf.Variable(t, name=n),
              value, expand_composites=True))
    return deferred_module.DeferredModule(
        functools.partial(_apply_parameters, g),
        *params_as_variables,
        also_track=tf.nest.flatten((args, kwargs)))

  # Update docstring.
  if generator.__doc__:
    doc = inspect.cleandoc(generator.__doc__)
    # Add `seed` to end of args list.
    seed_str = 'seed: PRNG seed; see `tfp.random.sanitize_seed` for details.'
    doc = re.sub(r'\nYields:', '  ' + seed_str + '\nYields:', doc)
    # Replace `Yields` section with `Returns` a trainable instance.
    doc = re.sub(r'Yields:\n(.+\n)+\n',
                 _STATEFUL_RETURNS_DOCSTRING,
                 doc + '\n\n')
    build_stateful_trainable.__doc__ = doc

  return build_stateful_trainable
