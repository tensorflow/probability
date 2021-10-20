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
"""Contains the `Template` and `Layer` API for Oryx.

`Module`s are an abstraction provided by Oryx that enable encapsulating both
state and functionality, and some basic neural network layers could be
implemented with the stateful function API. However, we want a few extra pieces
of functionality that important to writing practical neural networks beyond
what exists in `Module`:

1) Handling keyword arguments. It is common to have neural networks have
behavior conditional on a flag, like whether or not we are training the
neural network. We can handle this using a custom Jaxpr interpreter.

2) Custom batch semantics. We can't implement a layer like batch normalization
without having custom behavior for batches. We can accomplish this by using
a batch rule for a custom JAX primitive associated with layers.

3) Special combinators. Building a custom `Module` abstraction enables
overloading operators like `>>` to build complex architectures, along with
handling explicit threading of `PRNGKey`s.

We implement these additions with the `Template` and `Layer` classes.

# `Template`

A `Template` is an object registered with `core.state.init` that can
be initialized into a `Layer`. For example, for the template `nn.Dense(20)`
can be initialized into a `Layer` by calling
`core.state.init(nn.Dense(20))(random.PRNGKey(...), ...)`, just like a
stateful function. In most ways, `Template`s behave like stateful functions,
in that you can call them, i.e. `nn.Dense(20)(x, init_key=...)` and it will
execute a dense layer initialized with `init_key`. However, `Template`s have
some extra functionality. We can use the `>>` operator to chain `Template`s
together, i.e. `nn.Dense(200) >> nn.Relu()` is a new `Template` that composes
a dense layer with a ReLU activation. It will appropriately split and thread
the `init_key` to all its inner `Template`s when initialized.

# `Layer`

A `Layer` is a subclass of `Module` with some additional functionality. Like
`Module`s, `Layer`s have a `variables()` method that returns a dictionary
mapping names to state values. It also has a `call_and_update` function that
returns the output of a computation and a new `Layer` with updated state.
Underneath the hood, `Layers` do a couple extra things beyond `Module`s.

The first is that a `Layer`'s `call_and_update` is associated with a JAX
primitive: `layer_cau_p`. This primitive serves multiple purposes. The first is
that it enables custom behavior when being `vmap`-ed. For example, a
`BatchNorm` layer has different behavior when being `vmap`-ed over many
examples vs. a single example. When a `Layer` implements a
`_call_and_update_batched` method, the `layer_cau_p` primitive knows to use that
method instead of mapping over the default `_call_and_update` method. This
enables an inheritance pattern for custom behavior under JAX transformations.
The `layer_cau_p` primitive also enables threading keyword arguments from the
`Layer`'s call site (like if we did `layer(x, training=True)`). This allows
`Layer`s to be implemented with keyword arguments without worrying if they are
being traced.


## `Template`s vs `Layer`s

`Layer`s cannot be constructed directly. In fact, we override their `__new__`
method to construct a `Template` instead. So, despite `nn.Dense` being
a subclass of `nn.Layer`, `nn.Dense(20)` will return an instance of `Template`.
The `Template` acts as a factory class for `Layer`s and using `core.state.init`
will actually build the `Layer` by calling special methods in the `Layer`.

## `initialize`/`spec`
All `Layer`s must implement the `initialize` and `spec` class methods. These
enable a `Template` to know the shapes and correctly initialize the parameters
in a `Layer`. `initialize` is responsible for parameter initialization and
`spec` is responsible for shape inference.

`initialize` must return a `LayerParams` object, which represents the state
of a `Layer`, broken down into three components:
1) `params` - what we traditionally consider the "weights" of a layer, i.e.
quantities we'd like to differentiate with respect to.
2) `state` - refers to numerical values that are part of a layer that we would
*not* like to differentiate with respect to, such as running averages. These
quantities are automatically stop-gradiented before running the forward pass.
2) `info` - refers to metadata that is not numerical, such as configuration
strings.

## `_call`

All `Layer`s must implement a `_call` method, which executes their forward pass.
They can refer to the `LayerParams` returned in `initialize`. The `_call` method
can accept keyword arguments such as `training`. We specially handle the
keyword argument `rng` to ensure it is traced and split properly so it can be
used for stochasticity in the forward pass.

## `_update`

If a `Layer` wants to optionally update its state, it can implement an `_update`
method, which has the same input arguments as `_call` but instead returns a
copy of the `Layer` with updated state. By default, it just returns `self`.

## `_call_and_update_batched`

If a `Layer` would like a custom batching rule, it can implement
`_call_and_update_batched`, which assumes all the input arguments have a leading
batch dimension. It must return the batched outputs and an unbatched, updated
`Layer`.
"""
import abc
import collections
import itertools as it

import jax
from jax import lax
from jax import linear_util as lu
from jax import tree_util
from jax.example_libraries import stax
from jax.interpreters import batching

from oryx.core import kwargs_util
from oryx.core import primitive
from oryx.core import state

__all__ = [
    'LayerParams',
    'Layer',
    'Template',
    'create_parameter',
]


class LayerParams(collections.namedtuple(
    'LayerParams',
    ['params',  # Layer parameters, can compute gradients on them.
     'info',  # Layer auxiliary info needed to perform computation.
     'state',  # Layer state needed to perform computation.
    ])):
  """LayerParams holds params and info of Layers."""

LayerParams.__new__.__defaults__ = ((), (), ())


LAYER_IDS = {}


class Layer(state.Module, metaclass=abc.ABCMeta):
  """Base class for neural network layers.

  A `Layer` is a subclass of `Module` with some additional functionality. Like
  `Module`s, `Layer`s have a `variables()` method that returns a dictionary
  mapping names to state values. It also has a `call_and_update` function that
  returns the output of a computation and a new `Layer` with updated state.
  Underneath the hood, `Layers` do a couple extra things beyond `Module`s.
  """

  def __init__(self, layer_params, name=None):
    self._params = layer_params.params
    self._state = layer_params.state
    self._info = layer_params.info
    super().__init__(name=name)

  @classmethod
  def new(cls, layer_params, name=None):
    """Creates Layer given a LayerParams namedtuple.

    Args:
      layer_params: LayerParams namedtuple that defines the Layer.
      name: a string name for the Layer.
    Returns:
      A `Layer` object.
    """
    layer = object.__new__(cls)
    layer.__init__(layer_params, name=name)
    return layer

  def variables(self):
    """Returns the variables dictionary for this `Layer`."""
    return dict(params=self.params, state=self.state)

  @property
  def params(self):
    """Returns the parameters of this `Layer`."""
    return self._params

  @property
  def state(self):
    """Returns the state of this `Layer`."""
    return self._state

  @property
  def info(self):
    """Returns the info for this `Layer`."""
    return self._info

  def __new__(cls, *args, **kwargs):
    """Returns a Template factory that creates instances of this `Layer`."""
    return Template(cls, *args, **kwargs)

  def call(self, *args, **kwargs):
    """Calls the `Layer`'s `call_and_update` and returns the first result."""
    return self.call_and_update(*args, **kwargs)[0]

  @abc.abstractmethod
  def _call(self, x):
    """Executes the forward pass for this `Layer`.

    This function must be implemented for subclasses of `Layer`.

    Args:
      x: Inputs to the layer.

    Returns:
      outputs: Outputs of the Layer.
    """

  def update(self, *args, **kwargs):
    """Calls the `Layer`'s `call_and_update` and returns the second result."""
    return self.call_and_update(*args, **kwargs)[1]

  def _update(self, *args, **kwargs):
    """Returns a copy of the layer with updated internal state.

    By default, it is the identity function and returns `self`.

    Args:
      *args: Inputs to the layer
      **kwargs: Keyword arguments to the layer

    Returns:
      A `Layer` object with updated state.
    """
    del args, kwargs
    return self  # Default behavior for stateless layers

  def call_and_update(self, *args, rng=None, **kwargs):
    """Uses the `layer_cau` primitive to call `self._call_and_update."""
    if rng is None:
      has_rng = False
    else:
      # The layer_cau primitive expects RNG as the first argument if `has_rng`
      # kwarg is True.
      args = (rng,) + args
      has_rng = True
    kwargs = dict(kwargs, has_rng=has_rng)
    return primitive.initial_style_bind(layer_cau_p, kwargs=kwargs)(
        Layer._call_and_update)(self, *args, **kwargs)

  def _call_and_update(self, *args, has_rng=False, **kwargs):
    """Runs and returns the `Layer`'s `_call` and `_update` functions."""
    if has_rng:
      rng, args = args[0], args[1:]
      kwargs = dict(kwargs, rng=rng)
    call_kwargs = kwargs_util.filter_kwargs(self._call, kwargs)
    update_kwargs = kwargs_util.filter_kwargs(self._update, kwargs)
    layer = self.replace(state=lax.stop_gradient(self.state))
    return (layer._call(*args, **call_kwargs),  # pylint: disable=protected-access
            layer._update(*args, **update_kwargs))  # pylint: disable=protected-access

  @abc.abstractclassmethod
  def initialize(cls, init_key, in_spec):
    """Initializes a `Layer` from an `init_key` and input specification."""

  def __repr__(self):
    """String representation of the Layer."""
    return '{}(params={}, info={})'.format(
        self.__class__.__name__, self.params, self.info)

  def __str__(self):
    """String representation of the Layer."""
    return '{}()'.format(self.__class__.__name__)

  def flatten(self):
    """Converts the Layer to a tuple suitable for PyTree."""
    data = (self.info, self.name)
    xs = (self.params, self.state)
    return xs, data

  @classmethod
  def unflatten(cls, data, xs):
    """Reconstruct the Layer from a flattened version."""
    layer_params = LayerParams(params=xs[0],
                               info=data[0],
                               state=xs[1])
    return cls.new(layer_params, name=data[1])

  def replace(self, params=None, state=None, info=None):  # pylint: disable=redefined-outer-name
    """Returns a copy of the layer with replaced properties."""
    params = self.params if params is None else params
    state = self.state if state is None else state
    info = self.info if info is None else info
    new_layer_params = LayerParams(params, info, state)
    return self.__class__.new(new_layer_params, name=self.name)


def template_build(cls, init_key, *args, name=None, **kwargs):
  """Instantiates layer object from RNG and layer specifications."""
  if init_key is None:
    raise ValueError('Cannot initialize template with `None` PRNGKey.')
  layer_params = cls.initialize(init_key, *args, **kwargs)
  if init_key is not None:
    new_params = tree_util.tree_map(lambda x: primitive.tie_in(init_key, x),
                                    (layer_params.params, layer_params.state))
    layer_params = LayerParams(params=new_params[0], state=new_params[1],
                               info=layer_params.info)
  return cls.new(layer_params, name=name)


class Template(object):
  """Template class used by neural network layers."""

  def __init__(self, cls, *init_args, **init_kwargs):
    self.cls = cls
    self.init_args = init_args
    self.init_kwargs = init_kwargs

  def __call__(self, *args, **kwargs):
    return self.call(*args, **kwargs)

  def call(self, *args, **kwargs):
    return self._call(*args, **kwargs)

  def _call(self, *args, **kwargs):
    init_key = kwargs.pop('init_key', None)
    layer = self.init(init_key, *args, **kwargs)
    return layer(*args, **kwargs)

  def init(self, init_key, *args, name=None, **kwargs):
    """Initializes a Template into a Layer."""
    specs = jax.tree_map(state.make_array_spec, args)
    kwargs = dict(
        cls=self.cls,
        specs=specs,
        init_args=self.init_args,
        init_kwargs=self.init_kwargs,
    )
    layer = primitive.initial_style_bind(template_init_p)(
        _template_build)(init_key, name=name, **kwargs)
    if name is not None:
      layer = state.variable(layer, name=name)
    else:
      layer_params = {k: state.variable(v, name=k)
                      for k, v in layer.variables().items()}
      layer = layer.replace(**layer_params)
    return layer

  def spec(self, *specs):
    return template_spec(self)(*specs)

  def _spec(self, *specs):
    return self.cls.spec(*it.chain(specs, self.init_args), **self.init_kwargs)

  def compose(self, template):
    from oryx.experimental.nn import Serial  # pylint: disable=g-import-not-at-top,import-outside-toplevel #  pytype: disable=import-error
    return Serial([self, template])

  def __rshift__(self, template):
    return self.compose(template)


def _get_layer_from_args(*args, **kwargs):
  """Unflattens a layer and handles rng in kwargs."""
  kwargs = kwargs.copy()
  in_tree = kwargs.pop('in_tree')
  num_weights = kwargs.pop('num_weights')
  flattened_layer, args = args[:num_weights], args[num_weights:]
  # Special handling of the `rng` kwarg
  # to make sure it is traced
  has_rng = kwargs.pop('has_rng', False)
  if has_rng:
    kwargs['rng'], args = args[0], args[1:]
  layer = tree_util.tree_unflatten(in_tree, flattened_layer)
  return layer, args, kwargs


def _layer_to_args(layer, *args, **kwargs):
  """Flattens a layer and handles rng in kwargs."""
  kwargs = kwargs.copy()
  flattened_layer, in_tree = tree_util.tree_flatten(layer)
  kwargs['num_weights'] = len(flattened_layer)
  kwargs['in_tree'] = in_tree
  rng = kwargs.pop('rng', None)
  kwargs['has_rng'] = has_rng = rng is not None
  if has_rng:
    args = (rng,) + args
  return flattened_layer, args, kwargs


layer_cau_p = primitive.InitialStylePrimitive('layer_cau')


class NoneProxy:
  pass
not_mapped = NoneProxy()


def custom_layer_cau_batch(vals, dims, *, num_consts, in_tree, out_tree, kwargs,
                           **params):
  """Batching rule for layer_cau primitive to handle custom layers."""
  if all(dim is batching.not_mapped for dim in dims):
    return layer_cau_p.bind(*vals, num_consts=num_consts, in_tree=in_tree,
                            out_tree=out_tree, kwargs=kwargs, **params)
  orig_vals, orig_dims = vals, dims
  vals, dims = vals[num_consts:], dims[num_consts:]
  args = tree_util.tree_unflatten(in_tree, vals)
  dims_ = [not_mapped if dim is None else dim for dim in dims]
  layer, args = args[0], args[1:]
  if hasattr(layer, '_call_and_update_batched'):
    num_params = len(tree_util.tree_leaves(layer))
    layer_dims, arg_dims = dims_[:num_params], dims_[num_params:]
    if kwargs['has_rng']:
      rng, args = args[0], args[1:]
      rng_dim, arg_dims = arg_dims[0], arg_dims[1:]
    mapping_over_layer = all(layer_dim is not not_mapped for
                             layer_dim in layer_dims)
    mapping_over_args = all(arg_dim is not not_mapped for
                            arg_dim in arg_dims)
    assert mapping_over_layer or mapping_over_args, (layer_dims, arg_dims)
    if not mapping_over_layer and mapping_over_args:
      if kwargs['has_rng']:
        if rng_dim is not not_mapped:
          arg_dims = tuple(None if dim is not_mapped else dim
                           for dim in arg_dims)
          map_fun = jax.vmap(
              lambda layer, rng, *args: _layer_cau_batched(layer, rng, *args,  # pylint: disable=unnecessary-lambda, g-long-lambda
                                                           **kwargs),
              in_axes=(None, rng_dim) + (None,) * len(arg_dims))
        else:
          map_fun = lambda layer, *args: _layer_cau_batched(layer, *args,  # pylint: disable=unnecessary-lambda, g-long-lambda
                                                            **kwargs)
        vals_out, update_out = map_fun(layer, rng, *args)
      else:
        vals_out, update_out = _layer_cau_batched(layer, *args,
                                                  **kwargs)
      vals_out = tree_util.tree_leaves(vals_out)
      update_out = tree_util.tree_leaves(update_out)
      assert all(dim == 0 for dim in arg_dims)
      # Assume dimensions out are consistent
      dims_out = (0,) * len(vals_out)
      dims_update = (None,) * len(update_out)
      assert len(vals_out) == len(dims_out)
      assert len(update_out) == len(dims_update)
      return vals_out + update_out, dims_out + dims_update
  batched, out_dims = primitive.batch_fun(lu.wrap_init(
      layer_cau_p.impl, dict(params, num_consts=num_consts, in_tree=in_tree,
                             out_tree=out_tree,
                             kwargs=kwargs)), orig_dims)
  return batched.call_wrapped(*orig_vals), out_dims()
batching.primitive_batchers[layer_cau_p] = custom_layer_cau_batch


def _layer_cau_batched(layer, *args, **kwargs):
  kwargs = kwargs.copy()
  has_rng = kwargs.pop('has_rng', False)
  layer = layer.replace(state=lax.stop_gradient(layer.state))
  if has_rng:
    rng, args = args[0], args[1:]
    kwargs['rng'] = rng
  kwargs = kwargs_util.filter_kwargs(layer._call_and_update_batched, kwargs)  # pylint: disable=protected-access
  return layer._call_and_update_batched(*args, **kwargs)  # pylint: disable=protected-access


# Registrations


@state.spec.register(Layer)
def layer_spec(layer):
  def wrapped(*args, **kwargs):
    in_specs = tree_util.tree_map(state.make_array_spec, args)
    out_specs = layer.spec(*in_specs, **kwargs)
    return tree_util.tree_map(state.make_array_spec, out_specs)
  return wrapped


@state.call_and_update.register(Layer)
def layer_call_and_update(layer, *args, **kwargs):
  return layer.call_and_update(*args, **kwargs)  # pylint: disable=protected-access


@state.call_and_update.register(Template)
def template_call_and_update(template, *args, **kwargs):
  return template._call(*args, **kwargs), template  # pylint: disable=protected-access


template_init_p = primitive.InitialStylePrimitive('template_init')


def _template_build(init_key, *, cls, specs, init_args, init_kwargs, name=None):
  return template_build(
      cls, init_key, *it.chain(specs, init_args), name=name, **init_kwargs)


@state.init.register(Template)
def template_init(template, *, name=None):
  """Initialize a Template into a Layer."""
  def wrapped(init_key, *args, **kwargs):
    return template.init(init_key, *args, **kwargs, name=name)
  return wrapped


@state.spec.register(Template)
def template_spec(template):
  def wrapped(*args, **kwargs):
    in_specs = tree_util.tree_map(state.make_array_spec, args)
    kwargs = kwargs_util.filter_kwargs(template._spec, kwargs)  # pylint: disable=protected-access
    out_specs = template._spec(*in_specs, **kwargs)  # pylint: disable=protected-access
    return tree_util.tree_map(state.make_array_spec, out_specs)
  return wrapped


def create_parameter(rng, spec, init=stax.glorot()):
  return init(rng, spec)
