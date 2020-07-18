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
"""Wraps TFP bijectors for use with Jax."""
from jax import tree_util
from jax import util as jax_util
import jax.numpy as np

from six.moves import zip
from oryx.core import primitive
from oryx.core.interpreters import inverse
from oryx.core.interpreters.inverse import slice as slc
from tensorflow_probability.substrates import jax as tfp

__all__ = [
    'make_type',
]

safe_map = jax_util.safe_map
tf = tfp.tf2jax
tfb = tfp.bijectors

_registry = {}

InverseAndILDJ = inverse.core.InverseAndILDJ
NDSlice = slc.NDSlice


class _JaxBijectorTypeSpec(object):
  """TypeSpec for flattening/unflattening bijectors."""

  __slots__ = ('_clsid', '_kwargs', '_param_specs')

  def __init__(self, clsid, param_specs, kwargs):
    self._clsid = clsid
    self._kwargs = kwargs
    self._param_specs = param_specs

  @property
  def value_type(self):
    return _registry[self._clsid]

  def _to_components(self, obj):
    components = {
        'args': obj._args  # pylint: disable=protected-access
    }
    for k, v in sorted(obj._kwargs.items()):  # pylint: disable=protected-access
      if k in self._param_specs:  # pylint: disable=protected-access
        components[k] = v
    return components

  def _from_components(self, components):
    kwargs = dict(self._kwargs)  # pylint: disable=protected-access
    kwargs.update(components)
    args = kwargs.pop('args')
    return self.value_type(*args, **kwargs)  # pylint: disable=not-callable

  @property
  def _component_specs(self):
    return self._param_specs

  def _serialize(self):
    # Include default version 1 for now
    return 1, self._clsid, self._param_specs, self._kwargs

  @classmethod
  def _deserialize(cls, encoded):
    version, clsid, param_specs, kwargs = encoded
    if version != 1: raise ValueError
    if clsid not in _registry: raise ValueError(clsid)
    return cls(clsid, param_specs, kwargs)


bijector_p = primitive.HigherOrderPrimitive('bijector')


class _CellProxy:
  """Used for avoid recursing into cells when doing Pytree flattening/unflattening."""

  def __init__(self, cell):
    self.cell = cell


def bijector_ildj_rule(incells, outcells, **params):
  """Inverse/ILDJ rule for bijectors."""
  incells = incells[1:]
  num_consts = len(incells) - params['num_args']
  const_incells, flat_incells = jax_util.split_list(incells, [num_consts])
  flat_inproxies = safe_map(_CellProxy, flat_incells)
  in_tree = params['in_tree']
  bijector_proxies, inproxy = tree_util.tree_unflatten(in_tree,
                                                       flat_inproxies)
  flat_bijector_cells = [proxy.cell for proxy
                         in tree_util.tree_leaves(bijector_proxies)]
  if any(not cell.top() for cell in flat_bijector_cells):
    return const_incells + flat_incells, outcells, False, None
  bijector = tree_util.tree_multimap(lambda x: x.cell.val, bijector_proxies)
  direction = params['direction']
  if direction == 'forward':
    forward_func = bijector.forward
    inv_func = bijector.inverse
    ildj_func = bijector.inverse_log_det_jacobian
  elif direction == 'inverse':
    forward_func = bijector.inverse
    inv_func = bijector.forward
    ildj_func = bijector.forward_log_det_jacobian
  else:
    raise ValueError('Bijector direction must be '
                     '"forward" or "inverse".')

  outcell, = outcells
  incell = inproxy.cell
  if incell.bottom() and not outcell.bottom():
    val, ildj = outcell.val, outcell.ildj
    inildj = ildj + ildj_func(val, np.ndim(val))
    ndslice = NDSlice.new(inv_func(val), inildj)
    flat_incells = [
        InverseAndILDJ(incell.aval, [ndslice])
    ]
    new_outcells = outcells
  elif outcell.is_unknown() and not incell.is_unknown():
    new_outcells = [InverseAndILDJ.new(forward_func(incell.val))]
  new_incells = flat_bijector_cells + flat_incells
  return const_incells + new_incells, new_outcells, None
inverse.core.ildj_registry[bijector_p] = bijector_ildj_rule


def make_wrapper_type(cls):
  """Creates new Bijector type that can be flattened/unflattened and is lazy."""

  clsid = (cls.__module__, cls.__name__)

  def bijector_bind(bijector, x, **kwargs):
    return primitive.call_bind(
        bijector_p, direction=kwargs['direction'])(_bijector)(
            bijector, x, **kwargs)

  def _bijector(bij, x, **kwargs):
    direction = kwargs.pop('direction', 'forward')
    if direction == 'forward':
      return cls.forward(bij, x, **kwargs)
    elif direction == 'inverse':
      return cls.inverse(bij, x, **kwargs)
    else:
      raise ValueError('Bijector direction must be "forward" or "inverse".')

  if clsid not in _registry:
    class _WrapperType(cls):
      """Oryx bijector wrapper type."""

      def __init__(self, *args, **kwargs):
        self.use_primitive = kwargs.pop('use_primitive', True)
        self._args = args
        self._kwargs = kwargs

      def forward(self, x, **kwargs):
        if self.use_primitive:
          return bijector_bind(self, x, direction='forward',
                               **kwargs)
        return cls.forward(self, x, **kwargs)

      def inverse(self, x, **kwargs):
        if self.use_primitive:
          return bijector_bind(self, x, direction='inverse',
                               **kwargs)
        return cls.inverse(self, x, **kwargs)

      def _get_instance(self):
        obj = object.__new__(cls)
        cls.__init__(obj, *self._args, **self._kwargs)
        return obj

      def __getattr__(self, key):
        if key not in ('_args', '_kwargs', 'parameters', '_type_spec'):
          return getattr(self._get_instance(), key)
        return object.__getattribute__(self, key)

      @property
      def parameters(self):
        return self._get_instance().parameters

      @property
      def _type_spec(self):
        kwargs = dict(self._kwargs)
        param_specs = {}
        event_ndims = {}
        for k in event_ndims:
          if k in kwargs and kwargs[k] is not None:
            elem = kwargs.pop(k)
            if type(elem) == object:  # pylint: disable=unidiomatic-typecheck
              param_specs[k] = object
            elif tf.is_tensor(elem):
              param_specs[k] = (elem.shape, elem.dtype)
            else:
              param_specs[k] = type(elem)
        for k, v in list(kwargs.items()):
          if isinstance(v, tfb.Bijector):
            param_specs[k] = kwargs.pop(k)
        return _JaxBijectorTypeSpec(
            clsid, param_specs, kwargs)

      def __str__(self):
        return repr(self)

      def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    _WrapperType.__name__ = cls.__name__ + 'Wrapper'

    def to_tree(obj):
      type_spec = obj._type_spec  # pylint: disable=protected-access
      components = type_spec._to_components(obj)  # pylint: disable=protected-access
      keys, values = list(zip(*sorted(components.items())))
      return values, (keys, type_spec)

    def from_tree(info, xs):
      keys, type_spec = info
      components = dict(list(zip(keys, xs)))
      return type_spec._from_components(components)  # pylint: disable=protected-access

    tree_util.register_pytree_node(
        _WrapperType,
        to_tree,
        from_tree
    )

    _registry[clsid] = _WrapperType
  return _registry[clsid]


def make_type(dist):
  """Entry point for wrapping distributions."""

  class _JaxBijectorType(dist):

    def __new__(cls, *args, **kwargs):
      type_ = make_wrapper_type(dist)
      obj = object.__new__(type_)
      obj.__init__(*args, **kwargs)
      return obj

  _JaxBijectorType.__name__ = dist.__name__
  return _JaxBijectorType
