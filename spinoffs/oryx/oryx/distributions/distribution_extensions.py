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
"""Wraps TFP distributions for use with Jax."""
from typing import Optional

from jax import tree_util
from jax import util as jax_util
from six.moves import zip
from oryx.core import ppl
from oryx.core import primitive
from oryx.core.interpreters import inverse
from oryx.core.interpreters import log_prob
from oryx.core.interpreters import unzip
from tensorflow_probability.substrates import jax as tfp

tf = tfp.tf2jax
tfd = tfp.distributions


InverseAndILDJ = inverse.core.InverseAndILDJ

_registry = {}


class _JaxDistributionTypeSpec(object):
  """TypeSpec for flattening/unflattening distributions."""

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


random_variable_p = primitive.HigherOrderPrimitive('random_variable')
unzip.block_registry.add(random_variable_p)


def random_variable_log_prob_rule(flat_incells, flat_outcells, **params):
  """Registers Oryx distributions with the log_prob transformation."""
  del params
  # First incell is the call primitive function
  return flat_incells[1:], flat_outcells, None
log_prob.log_prob_rules[random_variable_p] = random_variable_log_prob_rule


def random_variable_log_prob(flat_incells, val, **params):
  """Registers Oryx distributions with the log_prob transformation."""
  num_consts = len(flat_incells) - params['num_args']
  _, flat_incells = jax_util.split_list(flat_incells, [num_consts])
  _, dist = tree_util.tree_unflatten(params['in_tree'], flat_incells)
  if any(not cell.top() for cell in flat_incells[1:]
         if isinstance(val, InverseAndILDJ)):
    return None
  return dist.log_prob(val)


log_prob.log_prob_registry[
    random_variable_p] = random_variable_log_prob


def _sample_distribution(key, dist):
  return dist.sample(seed=key)


@ppl.random_variable.register(tfd.Distribution)
def distribution_random_variable(dist: tfd.Distribution, *,
                                 name: Optional[str] = None):
  def wrapped(key):
    result = primitive.call_bind(random_variable_p)(_sample_distribution)(
        key, dist)
    if name is not None:
      result = ppl.random_variable(result, name=name)
    return result
  return wrapped


@ppl.log_prob.register(tfd.Distribution)
def distribution_log_prob(dist: tfd.Distribution):
  def wrapped(value):
    return dist.log_prob(value)
  return wrapped


def make_wrapper_type(cls):
  """Creates a flattenable Distribution type."""

  clsid = (cls.__module__, cls.__name__)

  if clsid not in _registry:
    class _WrapperType(cls):
      """Oryx distribution wrapper type."""

      def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._instance = object.__new__(cls)
        cls.__init__(self._instance, *self._args, **self._kwargs)

      def __getattr__(self, key):
        if key not in ('_args', '_kwargs', '_type_spec', '_instance'):
          return getattr(self._instance, key)
        return object.__getattribute__(self, key)

      @property
      def _type_spec(self):
        kwargs = dict(self._kwargs)
        param_specs = {}
        try:
          event_ndims = self._params_event_ndims()
        except NotImplementedError:
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
          if isinstance(v, tfd.Distribution):
            param_specs[k] = kwargs.pop(k)
        return _JaxDistributionTypeSpec(
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

  class _JaxDistributionType(dist):

    def __new__(cls, *args, **kwargs):
      type_ = make_wrapper_type(dist)
      obj = object.__new__(type_)
      obj.__init__(*args, **kwargs)
      return obj

  _JaxDistributionType.__name__ = dist.__name__
  return _JaxDistributionType
