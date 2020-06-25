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
from tensorflow_probability.python.experimental.substrates import jax as tfp
from oryx import core
from oryx.core import ppl
from oryx.core.interpreters import inverse
tf = tfp.tf2jax
tfd = tfp.distributions


InverseAndILDJ = inverse.InverseAndILDJ

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


random_variable_p = core.HigherOrderPrimitive('random_variable')
core.interpreters.unzip.block_registry.add(random_variable_p)


def random_variable_log_prob_rule(flat_invals, flat_outvals, **params):
  """Registers Oryx distributions with the log_prob transformation."""
  flat_invals = flat_invals[1:]
  num_consts = len(flat_invals) - params['num_args']
  const_invals, flat_invals = jax_util.split_list(flat_invals, [num_consts])
  arg_vals = tree_util.tree_unflatten(params['in_tree'], flat_invals)
  seed_val, dist_val = arg_vals[0], arg_vals[1]
  if not (seed_val.is_unknown() or all(
      v.is_unknown() for v in tree_util.tree_flatten(dist_val)[0])):
    dist = tree_util.tree_map(lambda x: x.val, dist_val)
    s = dist.sample(seed=seed_val.val, **params['kwargs'])
    return const_invals + flat_invals, [InverseAndILDJ.new(s)], True, None
  elif not all(val.is_unknown() for val in flat_outvals):
    return const_invals + flat_invals, flat_outvals, True, None
  return const_invals + flat_invals, flat_outvals, False, None
core.interpreters.log_prob.log_prob_rules[
    random_variable_p] = random_variable_log_prob_rule


def random_variable_log_prob(flat_invals, val, **params):
  """Registers Oryx distributions with the log_prob transformation."""
  num_consts = len(flat_invals) - params['num_args']
  _, flat_invals = jax_util.split_list(flat_invals, [num_consts])
  _, dist = tree_util.tree_unflatten(params['in_tree'], flat_invals)
  if any(val.is_unknown() for val in flat_invals[1:]
         if isinstance(val, InverseAndILDJ)):
    return None
  return dist.log_prob(val)


core.interpreters.log_prob.log_prob_registry[
    random_variable_p] = random_variable_log_prob


def _sample_distribution(key, dist):
  return dist.sample(seed=key)


@ppl.random_variable.register(tfd.Distribution)
def distribution_random_variable(dist: tfd.Distribution, *,
                                 name: Optional[str] = None):
  def wrapped(key):
    result = core.call_bind(random_variable_p)(_sample_distribution)(key, dist)
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
  """Creates a flattenable Distribution type that has lazily evaluated attributes."""

  clsid = (cls.__module__, cls.__name__)

  if clsid not in _registry:
    class _WrapperType(cls):
      """Oryx distribution wrapper type."""

      def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

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
