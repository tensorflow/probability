# Copyright 2019 The TensorFlow Probability Authors.
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
"""Use `tfp.distributions.Distribution`s as `tf.CompositeTensor`s."""

import inspect
import six

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.framework.composite_tensor import CompositeTensor  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.saved_model import nested_structure_coder  # pylint: disable=g-direct-tensorflow-import

__all__ = ['as_composite', 'register_composite']

_registry = {}  # Mapping from (python pkg, class name) -> class.


class _TFPTypeSpec(tf.TypeSpec):
  """A tf.TypeSpec for `tfp.distributions.Distribution` and related objects."""

  __slots__ = ('_clsid', '_kwargs', '_param_specs')

  def __init__(self, clsid, param_specs, kwargs):
    self._clsid = clsid
    self._kwargs = kwargs
    self._param_specs = param_specs

  @property
  def value_type(self):
    return _registry[self._clsid]

  def _to_components(self, obj):
    return {
        k: getattr(obj, k, obj.parameters[k]) for k in sorted(self._param_specs)
    }

  def _from_components(self, components):
    kwargs = dict(self._kwargs)
    kwargs.update(components)
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
    if _find_clsid(clsid) is None:
      raise ValueError(
          'Unable to identify distribution type for {}. For user-defined '
          'distributions (not in TFP), make sure the distribution is decorated '
          'with `@tfp.experimental.register_composite` and its module is '
          'imported before calling `tf.saved_model.load`.'.format(clsid))
    return cls(clsid, param_specs, kwargs)


_TypeSpecCodec = nested_structure_coder._TypeSpecCodec  # pylint: disable=protected-access
_TypeSpecCodec.TYPE_SPEC_CLASS_FROM_PROTO[275837168] = _TFPTypeSpec
_TypeSpecCodec.TYPE_SPEC_CLASS_TO_PROTO[_TFPTypeSpec] = 275837168
del _TypeSpecCodec


def _make_convertible(cls):
  """Makes a subclass of `cls` that also subclasses `CompositeTensor`."""

  clsid = (cls.__module__, cls.__name__)

  if clsid in _registry:
    return _registry[clsid]

  class _CompositeTensorDist(cls, CompositeTensor):
    """A per-`cls` subclass of `CompositeTensor`."""

    def _parameter_control_dependencies(self, is_init):
      # We are forced by the CompositeTensor contract (no graph operations in
      # `_to_components`, `_from_components`) to defer the
      # `_initial_parameter_control_dependencies` to point-of-use.
      if is_init:
        return ()

      result = tuple(
          super(_CompositeTensorDist,
                self)._parameter_control_dependencies(is_init=True))
      result += tuple(
          super(_CompositeTensorDist,
                self)._parameter_control_dependencies(is_init=False))
      return result

    @property
    def _type_spec(self):
      def get_default_args(fn_or_object):
        fn = type(fn_or_object) if isinstance(fn_or_object,
                                              object) else fn_or_object
        return {
            k: v.default
            for k, v in inspect.signature(fn).parameters.items()
            if v.default is not inspect.Parameter.empty
        }

      if six.PY3:
        default_kwargs = get_default_args(self)
        missing = object()
        kwargs = {
            k: v
            for k, v in self.parameters.items()
            if default_kwargs.get(k, missing) is not v
        }  # non-default kwargs only
      else:
        kwargs = dict(self.parameters)
      param_specs = {}
      try:
        composite_tensor_params = self._composite_tensor_params  # pylint: disable=protected-access
      except (AttributeError, NotImplementedError):
        composite_tensor_params = ()
      for k in composite_tensor_params:
        if k in kwargs and kwargs[k] is not None:
          v = kwargs.pop(k)
          def composite_helper(v):
            if isinstance(v, CompositeTensor):
              return v._type_spec  # pylint: disable=protected-access
            elif tf.is_tensor(v):
              return tf.TensorSpec.from_tensor(v)
          param_specs[k] = tf.nest.map_structure(composite_helper, v)
      for k, v in list(kwargs.items()):
        if isinstance(v, CompositeTensor):
          param_specs[k] = v._type_spec  # pylint: disable=protected-access
          kwargs.pop(k)
        elif callable(v):
          raise NotImplementedError(
              'Unable to make CompositeTensor including callable argument.' + k)
      return _TFPTypeSpec(
          clsid, param_specs=param_specs, kwargs=kwargs)

  _CompositeTensorDist.__name__ = '{}CT'.format(cls.__name__)
  _registry[clsid] = _CompositeTensorDist
  return _CompositeTensorDist


# Lazy-cache into `_registry` so that `tf.saved_model.load` will work.
def _find_clsid(clsid):
  pkg, cls = clsid
  if clsid not in _registry:
    if pkg.startswith('tensorflow_probability.') and '.distributions' in pkg:
      dist_cls = getattr(distributions, cls)
      if (inspect.isclass(dist_cls) and
          issubclass(dist_cls, distributions.Distribution)):
        _make_convertible(dist_cls)
  return _registry[clsid] if clsid in _registry else None


def as_composite(obj):
  """Returns a `CompositeTensor` equivalent to the given object.

  Note that the returned object will have any `Variable`,
  `tfp.util.DeferredTensor`, or `tfp.util.TransformedVariable` references it
  closes over converted to tensors at the time this function is called. The
  type of the returned object will be a subclass of both `CompositeTensor` and
  `type(obj)`.  For this reason, one should be careful about using
  `as_composite()`, especially for `tf.Module` objects.

  For example, when the composite tensor is created even as part of a
  `tf.Module`, it "fixes" the values of the `DeferredTensor` and `tf.Variable`
  objects it uses:

  ```python
  class M(tf.Module):
    def __init__(self):
      self._v = tf.Variable(1.)
      self._d = tfp.distributions.Normal(
        tfp.util.DeferredTensor(self._v, lambda v: v + 1), 10)
      self._dct = tfp.experimental.as_composite(self._d)

    @tf.function
    def mean(self):
      return self._dct.mean()

  m = M()
  m.mean()
  >>> <tf.Tensor: numpy=2.0>
  m._v.assign(2.)  # Doesn't update the CompositeTensor distribution.
  m.mean()
  >>> <tf.Tensor: numpy=2.0>
  ```

  If, however, the creation of the composite is deferred to a method
  call, then the Variable and DeferredTensor will be properly captured
  and respected by the Module and its `SavedModel` (if it is serialized).

  ```python
  class M(tf.Module):
    def __init__(self):
      self._v = tf.Variable(1.)
      self._d = tfp.distributions.Normal(
        tfp.util.DeferredTensor(self._v, lambda v: v + 1), 10)

    @tf.function
    def d(self):
      return tfp.experimental.as_composite(self._d)

  m = M()
  m.d().mean()
  >>> <tf.Tensor: numpy=2.0>
  m._v.assign(2.)
  m.d().mean()
  >>> <tf.Tensor: numpy=3.0>
  ```

  Note: This method is best-effort and based on a heuristic for what the
  tensor parameters are and what the non-tensor parameters are. Things might be
  broken, especially for meta-distributions like `TransformedDistribution` or
  `Independent`. (We try to raise NotImplementedError in such cases.) If you'd
  benefit from better coverage, please file an issue on github or send an email
  to `tfprobability@tensorflow.org`.

  Args:
    obj: A `tfp.distributions.Distribution`.

  Returns:
    obj: A `tfp.distributions.Distribution` that extends `CompositeTensor`.
  """
  if isinstance(obj, CompositeTensor):
    return obj
  cls = _make_convertible(type(obj))
  kwargs = dict(obj.parameters)

  def mk_err_msg(suffix=''):
    return ('Unable to make a CompositeTensor for "{}" of type `{}`. Email '
            '`tfprobability@tensorflow.org` or file an issue on github if you '
            'would benefit from this working. {}'.format(
                obj, type(obj), suffix))

  try:
    composite_tensor_params = obj._composite_tensor_params  # pylint: disable=protected-access
  except (AttributeError, NotImplementedError):
    composite_tensor_params = ()
  for k in composite_tensor_params:
    # Use dtype inference from ctor.
    if k in kwargs and kwargs[k] is not None:
      v = getattr(obj, k, kwargs[k])
      try:
        kwargs[k] = tf.convert_to_tensor(v, name=k)
      except (ValueError, TypeError) as e:
        kwargs[k] = v
  for k, v in kwargs.items():
    def composite_helper(v):
      # If we have a parameters attribute, then we may be able to convert to
      # a composite tensor by guessing which of the parameters are tensors.  In
      # essence, we duck-type based on this attribute.
      if hasattr(v, 'parameters'):
        return as_composite(v)
      return v
    kwargs[k] = tf.nest.map_structure(composite_helper, v)
    # Unfortunately, tensor_util.is_ref(v) returns true for a
    # tf.linalg.LinearOperator even though that is not ideal behavior.
    if tensor_util.is_ref(v) and not isinstance(v, tf.linalg.LinearOperator):
      try:
        kwargs[k] = tf.convert_to_tensor(v, name=k)
      except TypeError as e:
        raise NotImplementedError(
            mk_err_msg('(Unable to convert dependent entry \'{}\' of object '
                       '\'{}\': {})'.format(k, obj, str(e))))
  result = cls(**kwargs)
  try:
    nested_structure_coder.encode_structure(result._type_spec)  # pylint: disable=protected-access
  except nested_structure_coder.NotEncodableError as e:
    raise NotImplementedError(
        mk_err_msg('(Unable to serialize: {})'.format(str(e))))
  return result


def register_composite(cls):
  """A decorator that registers a TFP object as composite-friendly.

  This registration is not required to call `as_composite` on instances
  of a given distribution (or bijector or other TFP object), but it *is*
  required if a `SavedModel` with functions accepting or returning composite
  wrappers of this object will be loaded in python (without having called
  `as_composite` already).

  Example:

  ```python
  class MyDistribution(tfp.distributions.Distribution):
     ...

  # This will fail to load.
  model = tf.saved_model.load(
      '/path/to/sm_with_funcs_returning_composite_tensor_MyDistribution')
  ```

  Instead:
  ```python
  @tfp.experimental.register_composite
  class MyDistribution(tfp.distributions.Distribution):
     ...

  # This will load.
  model = tf.saved_model.load(
      '/path/to/sm_with_funcs_returning_composite_tensor_MyDistribution')
  ```

  Args:
    cls: A subclass of `Distribution`.

  Returns:
    The input, with the side-effect of registering it as a composite-friendly
    distribution.

  Raises:
    TypeError: If `cls` does not have _composite_tensor_params, or if
      registration fails (`cls` is not convertible).
    NotImplementedError: If registration fails (`cls` is not convertible).
  """
  if not hasattr(cls, '_composite_tensor_params'):
    raise TypeError('Expected cls to have property "_composite_tensor_params".')
  _make_convertible(cls)
  return cls
