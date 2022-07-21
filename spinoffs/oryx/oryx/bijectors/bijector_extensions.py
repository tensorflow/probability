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
"""Wraps TFP bijectors for use with Jax."""

import functools
import inspect

from jax import tree_util
from jax import util as jax_util
import jax.numpy as np

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


bijector_p = primitive.InitialStylePrimitive('bijector')


class _CellProxy:
  """Used for avoid recursing into cells when doing Pytree flattening/unflattening."""

  def __init__(self, cell):
    self.cell = cell


def bijector_ildj_rule(incells, outcells, *, in_tree, num_consts, direction,
                       num_bijector, **_):
  """Inverse/ILDJ rule for bijectors."""
  const_incells, flat_incells = jax_util.split_list(incells, [num_consts])
  flat_bijector_cells, arg_incells = jax_util.split_list(
      flat_incells, [num_bijector])
  if any(not cell.top() for cell in flat_bijector_cells):
    return (const_incells + flat_incells, outcells, None)
  flat_inproxies = safe_map(_CellProxy, flat_incells)
  _, inproxy = tree_util.tree_unflatten(in_tree, flat_inproxies)
  bijector_vals = [cell.val for cell in flat_bijector_cells]
  bijector, _ = tree_util.tree_unflatten(
      in_tree, bijector_vals + [None] * len(arg_incells))
  if direction == 'forward':
    forward_func = bijector.forward
    inv_func = bijector.inverse
    ildj_func = bijector.inverse_log_det_jacobian
  elif direction == 'inverse':
    forward_func = bijector.inverse
    inv_func = bijector.forward
    ildj_func = bijector.forward_log_det_jacobian
  else:
    raise ValueError('Bijector direction must be ' '"forward" or "inverse".')

  outcell, = outcells
  incell = inproxy.cell
  if incell.bottom() and not outcell.bottom():
    val, ildj = outcell.val, outcell.ildj
    inildj = ildj + ildj_func(val, np.ndim(val))
    ndslice = NDSlice.new(inv_func(val), inildj)
    flat_incells = [InverseAndILDJ(incell.aval, [ndslice])]
    new_outcells = outcells
  elif outcell.is_unknown() and not incell.is_unknown():
    new_outcells = [InverseAndILDJ.new(forward_func(incell.val))]
  new_incells = flat_bijector_cells + flat_incells
  return (const_incells + new_incells, new_outcells, None)


inverse.core.ildj_registry[bijector_p] = bijector_ildj_rule


def make_wrapper_type(cls):
  """Creates new Bijector type that can be flattened/unflattened and is lazy."""

  clsid = (cls.__module__, cls.__name__)

  def bijector_bind(bijector, x, **kwargs):
    return primitive.initial_style_bind(
        bijector_p,
        direction=kwargs['direction'],
        num_bijector=len(tree_util.tree_leaves(bijector)),
        bijector_name=bijector.__class__.__name__)(_bijector)(bijector, x,
                                                              **kwargs)

  def _bijector(bij, x, **kwargs):
    direction = kwargs.pop('direction', 'forward')
    if direction == 'forward':
      return cls.forward(bij, x, **kwargs)
    elif direction == 'inverse':
      return cls.inverse(bij, x, **kwargs)
    else:
      raise ValueError('Bijector direction must be "forward" or "inverse".')

  def unify_signature(f):
    """Unify __init__ signature for _Wrapper, cls for auto Pytree conversion."""

    @functools.wraps(f)
    def with_fixed_signature(*args, **kwargs):
      return f(*args, **kwargs)

    old_init_sig = inspect.signature(f)
    new_init_sig = inspect.signature(cls.__init__)
    sig = old_init_sig.replace(
        parameters=(tuple(new_init_sig.parameters.values()) +
                    (old_init_sig.parameters['use_primitive'],)))
    f.__signature__ = sig
    return with_fixed_signature

  if clsid not in _registry:

    class _WrapperType(cls):
      """Oryx bijector wrapper type."""

      @unify_signature
      def __init__(self, *args, use_primitive=True, **kwargs):
        self.use_primitive = use_primitive
        self._args = args
        self._kwargs = kwargs

      def forward(self, x, **kwargs):
        if self.use_primitive:
          return bijector_bind(self, x, direction='forward', **kwargs)
        return cls.forward(self, x, **kwargs)

      def inverse(self, x, **kwargs):
        if self.use_primitive:
          return bijector_bind(self, x, direction='inverse', **kwargs)
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

      def __str__(self):
        return repr(self)

      def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    _WrapperType.__name__ = cls.__name__ + 'Wrapper'
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
