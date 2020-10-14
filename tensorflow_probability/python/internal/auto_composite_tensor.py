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
"""Turns arbitrary objects into tf.CompositeTensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import warnings

import six
import tensorflow.compat.v2 as tf

from tensorflow.python.framework.composite_tensor import CompositeTensor  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.saved_model import nested_structure_coder  # pylint: disable=g-direct-tensorflow-import

__all__ = ['auto_composite_tensor']


_registry = {}  # Mapping from (python pkg, class name) -> class.

_SENTINEL = object()


def _mk_err_msg(clsid, obj, suffix=''):
  msg = ('Unable to expand "{}", derived from type `{}.{}`, to its Tensor '
         'components. Email `tfprobability@tensorflow.org` or file an issue on '
         'github if you would benefit from this working. {}'.format(
             obj, clsid[0], clsid[1], suffix))
  warnings.warn(msg)
  return msg


def _kwargs_from(clsid, obj, limit_to=None):
  """Extract constructor kwargs to reconstruct `obj`."""
  if six.PY3:
    argspec = inspect.getfullargspec(obj.__init__)
    invalid_spec = bool(argspec.varargs or argspec.varkw)
    params = argspec.args + argspec.kwonlyargs
  else:
    argspec = inspect.getargspec(obj.__init__)  # pylint: disable=deprecated-method
    invalid_spec = bool(argspec.varargs or argspec.keywords)
    params = argspec.args
  if invalid_spec:
    raise NotImplementedError(
        _mk_err_msg(
            clsid, obj,
            '*args and **kwargs are not supported. Found `{}`'.format(argspec)))
  keys = [p for p in params if p != 'self']
  if limit_to is not None:
    keys = [k for k in keys if k in limit_to]
  kwargs = {k: getattr(obj, k, getattr(obj, '_' + k, _SENTINEL)) for k in keys}
  for k, v in kwargs.items():
    if v is _SENTINEL:
      raise ValueError(
          _mk_err_msg(
              clsid, obj,
              'Object did not have getter for constructor argument {k}. (Tried '
              'both `obj.{k}` and obj._{k}`).'.format(k=k)))
  return kwargs


class _AutoCompositeTensorTypeSpec(tf.TypeSpec):
  """A tf.TypeSpec for `AutoCompositeTensor` objects."""

  __slots__ = ('_clsid', '_param_specs', '_kwargs')

  def __init__(self, clsid, param_specs, kwargs):
    self._clsid = clsid
    self._param_specs = param_specs
    self._kwargs = kwargs

  @property
  def value_type(self):
    return _registry[self._clsid]

  def _to_components(self, obj):
    params = _kwargs_from(self._clsid, obj, limit_to=list(self._param_specs))
    return params

  def _from_components(self, components):
    kwargs = dict(self._kwargs, **components)
    return self.value_type(**kwargs)  # pylint: disable=not-callable

  @property
  def _component_specs(self):
    return self._param_specs

  def _serialize(self):
    return 1, self._clsid, self._param_specs, self._kwargs

  @classmethod
  def _deserialize(cls, encoded):
    version, clsid, param_specs, kwargs = encoded
    if version != 1:
      raise ValueError('Unexpected version')
    if clsid not in _registry:
      raise ValueError(
          'Unable to identify AutoCompositeTensor type for {}. Make sure the '
          'class is decorated with `@tfp.experimental.auto_composite_tensor` '
          'and its module is imported before calling '
          '`tf.saved_model.load`.'.format(clsid))
    return cls(clsid, param_specs, kwargs)


_TypeSpecCodec = nested_structure_coder._TypeSpecCodec  # pylint: disable=protected-access
_TypeSpecCodec.TYPE_SPEC_CLASS_FROM_PROTO[321584790] = (
    _AutoCompositeTensorTypeSpec)
_TypeSpecCodec.TYPE_SPEC_CLASS_TO_PROTO[_AutoCompositeTensorTypeSpec] = (
    321584790)
del _TypeSpecCodec


def auto_composite_tensor(cls=None, omit_kwargs=()):
  """Automagically create a `CompositeTensor` class for `cls`.

  `CompositeTensor` objects are able to pass in and out of `tf.function`,
  `tf.while_loop` and serve as part of the signature of a TF saved model.

  The basic contract is that all args must have public attributes (or
  properties) or private attributes corresponding to each argument to
  `__init__`. Each of these is inspected to determine whether it is a Tensor
  or non-Tensor metadata. Lists and tuples of objects are supported provided
  all items therein are all either Tensor/CompositeTensor, or all are not.

  ## Example

  ```python
  @tfp.experimental.auto_composite_tensor(omit_kwargs=('name',))
  class Adder(object):
    def __init__(self, x, y, name=None):
      with tf.name_scope(name or 'Adder') as name:
        self._x = tf.convert_to_tensor(x)
        self._y = tf.convert_to_tensor(y)
        self._name = name

    def xpy(self):
      return self._x + self._y

  def body(obj):
    return Adder(obj.xpy(), 1.),

  result, = tf.while_loop(
      cond=lambda _: True,
      body=body,
      loop_vars=(Adder(1., 1.),),
      maximum_iterations=3)

  result.xpy()  # => 5.
  ```

  Args:
    cls: The class for which to create a CompositeTensor subclass.
    omit_kwargs: Optional sequence of kwarg names to be omitted from the spec.

  Returns:
    ctcls: A subclass of `cls` and TF CompositeTensor.
  """
  if cls is None:
    return functools.partial(auto_composite_tensor, omit_kwargs=omit_kwargs)
  clsid = (cls.__module__, cls.__name__, omit_kwargs)

  # Also check for subclass if retrieving from the _registry, in case the user
  # has redefined the class (e.g. in a REPL/notebook).
  if clsid in _registry and issubclass(_registry[clsid], cls):
    return _registry[clsid]

  class _AutoCompositeTensor(cls, CompositeTensor):
    """A per-`cls` subclass of `CompositeTensor`."""

    @property
    def _type_spec(self):
      kwargs = _kwargs_from(clsid, self)
      param_specs = {}
      # Heuristically identify the tensor parts, and separate them.
      for k, v in list(kwargs.items()):  # We might pop in the loop body.

        if k in omit_kwargs:
          kwargs.pop(k)
          continue

        def reduce(v):
          has_tensors = False
          if tf.is_tensor(v):
            v = tf.TensorSpec.from_tensor(v)
            has_tensors = True
          if isinstance(v, CompositeTensor):
            v = v._type_spec  # pylint: disable=protected-access
            has_tensors = True
          if isinstance(v, (list, tuple)):
            reduced = [reduce(v_) for v_ in v]
            has_tensors = any(ht for (_, ht) in reduced)
            if has_tensors != all(ht for (_, ht) in reduced):
              raise NotImplementedError(
                  _mk_err_msg(
                      clsid, self,
                      'Found `{}` with both Tensor and non-Tensor parts: {}'
                      .format(type(v), v)))
            v = type(v)([spec for (spec, _) in reduced])
          return v, has_tensors

        v, has_tensors = reduce(v)
        if has_tensors:
          kwargs.pop(k)
          param_specs[k] = v
        # Else, we assume this entry is not a Tensor (bool, str, etc).

      # Construct the spec.
      spec = _AutoCompositeTensorTypeSpec(
          clsid, param_specs=param_specs, kwargs=kwargs)
      # Verify the spec serializes.
      struct_coder = nested_structure_coder.StructureCoder()
      try:
        struct_coder.encode_structure(spec)
      except nested_structure_coder.NotEncodableError as e:
        raise NotImplementedError(
            _mk_err_msg(clsid, self,
                        '(Unable to serialize: {})'.format(str(e))))
      return spec

  _AutoCompositeTensor.__name__ = '{}_AutoCompositeTensor'.format(cls.__name__)
  _registry[clsid] = _AutoCompositeTensor
  return _AutoCompositeTensor
