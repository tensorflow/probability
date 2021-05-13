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

import contextlib
import functools
import threading

import numpy as np
import tensorflow.compat.v2 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import tf_inspect
# pylint: enable=g-direct-tensorflow-import

__all__ = [
    'auto_composite_tensor',
    'AutoCompositeTensor',
    'is_deferred_assertion_context',
]

_DEFERRED_ASSERTION_CONTEXT = threading.local()
_DEFERRED_ASSERTION_CONTEXT.is_deferred = False


def is_deferred_assertion_context():
  return getattr(_DEFERRED_ASSERTION_CONTEXT, 'is_deferred', False)


@contextlib.contextmanager
def _deferred_assertion_context(is_deferred=True):
  was_deferred = getattr(_DEFERRED_ASSERTION_CONTEXT, 'is_deferred', False)
  _DEFERRED_ASSERTION_CONTEXT.is_deferred = is_deferred
  try:
    yield
  finally:
    _DEFERRED_ASSERTION_CONTEXT.is_deferred = was_deferred


_registry = {}  # Mapping from (python pkg, class name) -> class.

_SENTINEL = object()

_AUTO_COMPOSITE_TENSOR_VERSION = 2

# Cache maps __init__ method to signature
_sig_cache = {}


def _cached_signature(f):
  if f not in _sig_cache:
    _sig_cache[f] = tf_inspect.signature(f)
  return _sig_cache[f]


def _extract_init_kwargs(obj, omit_kwargs=(), limit_to=None,
                         prefer_static_value=()):
  """Extract constructor kwargs to reconstruct `obj`."""
  # If `obj` inherits its constructor from `AutoCompositeTensor` (which inherits
  # its constructor from `object`) return an empty dictionary to avoid
  # triggering the error below due to *args and **kwargs in the constructor.
  if type(obj).__init__ is AutoCompositeTensor.__init__:
    return {}

  sig = _cached_signature(type(obj).__init__)
  if any(v.kind in (tf_inspect.Parameter.VAR_KEYWORD,
                    tf_inspect.Parameter.VAR_POSITIONAL)
         for v in sig.parameters.values()):
    raise ValueError(
        '*args and **kwargs are not supported. Found `{}`'.format(sig))

  keys = [p for p in sig.parameters if p != 'self' and p not in omit_kwargs]
  if limit_to is not None:
    keys = [k for k in keys if k in limit_to]

  kwargs = {}
  not_found = object()
  for k in keys:
    src1 = getattr(obj, k, not_found)
    if src1 is not not_found:
      kwargs[k] = src1
    else:
      src2 = getattr(obj, '_' + k, not_found)
      if src2 is not not_found:
        kwargs[k] = src2
      else:
        src3 = getattr(obj, 'parameters', {}).get(k, not_found)
        if src3 is not not_found:
          kwargs[k] = src3
        else:
          raise ValueError(
              f'Could not determine an appropriate value for field `{k}` in'
              f' object `{obj}`. Looked for \n'
              f' 1. an attr called `{k}`,\n'
              f' 2. an attr called `_{k}`,\n'
              f' 3. an entry in `obj.parameters` with key "{k}".')
    if k in prefer_static_value and kwargs[k] is not None:
      if tf.is_tensor(kwargs[k]):
        static_val = tf.get_static_value(kwargs[k])
        if static_val is not None:
          kwargs[k] = static_val
    if isinstance(kwargs[k], (np.ndarray, np.generic)):
      # Generally, these are shapes or int, but may be other parameters such as
      # `power` for `tfb.PowerTransform`.
      kwargs[k] = kwargs[k].tolist()
  return kwargs


def _extract_type_spec_recursively(value):
  """Return (collection of) TypeSpec(s) for `value` if it includes `Tensor`s.

  If `value` is a `Tensor` or `CompositeTensor`, return its `TypeSpec`. If
  `value` is a collection containing `Tensor` values, recursively supplant them
  with their respective `TypeSpec`s in a collection of parallel stucture.

  If `value` is nont of the above, return it unchanged.

  Args:
    value: a Python `object` to (possibly) turn into a (collection of)
    `tf.TypeSpec`(s).

  Returns:
    spec: the `TypeSpec` or collection of `TypeSpec`s corresponding to `value`
    or `value`, if no `Tensor`s are found.
  """
  if isinstance(value, composite_tensor.CompositeTensor):
    return value._type_spec  # pylint: disable=protected-access
  if isinstance(value, tf.Variable):
    return resource_variable_ops.VariableSpec(
        value.shape, dtype=value.dtype, trainable=value.trainable)
  if tf.is_tensor(value):
    return tf.TensorSpec(value.shape, value.dtype)
  if isinstance(value, (list, tuple)):
    specs = [_extract_type_spec_recursively(v) for v in value]
    was_tensor = list([a is not b for a, b in zip(value, specs)])
    has_tensors = any(was_tensor)
    has_only_tensors = all(was_tensor)
    if has_tensors:
      if has_tensors != has_only_tensors:
        raise NotImplementedError(
            'Found `{}` with both Tensor and non-Tensor parts: {}'
            .format(type(value), value))
      return type(value)(specs)
  return value


class _AutoCompositeTensorTypeSpec(tf.TypeSpec):
  """A tf.TypeSpec for `AutoCompositeTensor` objects."""

  __slots__ = ('_param_specs', '_non_tensor_params', '_omit_kwargs',
               '_prefer_static_value', '_callable_params', '_serializable',
               '_comparable')

  def __init__(self, param_specs, non_tensor_params, omit_kwargs,
               prefer_static_value, callable_params=None):
    """Initializes a new `_AutoCompositeTensorTypeSpec`.

    Args:
      param_specs: Python `dict` of `tf.TypeSpec` instances that describe
        kwargs to the `AutoCompositeTensor`'s constructor that are `Tensor`-like
        or `CompositeTensor` subclasses.
      non_tensor_params: Python `dict` containing non-`Tensor` and non-
        `CompositeTensor` kwargs to the `AutoCompositeTensor`'s constructor.
      omit_kwargs: Python `tuple` of strings corresponding to the names of
        kwargs to the `AutoCompositeTensor`'s constructor that should be omitted
        from the `_AutoCompositeTensorTypeSpec`'s serialization, equality/
        compatibility checks, and rebuilding of the `AutoCompositeTensor` from
        `Tensor` components.
      prefer_static_value: Python `tuple` of strings corresponding to the names
        of `Tensor`-like kwargs to the `AutoCompositeTensor`s constructor that
        may be stored as static values, if known. These are typically shapes or
        axis values.
      callable_params: Python `dict` of callable kwargs to the
        `AutoCompositeTensor`'s constructor that do not subclass
        `CompositeTensor`, or `None`. If `callable_params` is a non-empty
        `dict`, then serialization of the `_AutoCompositeTensorTypeSpec` is not
         supported. Defaults to `None`, which is converted to an empty `dict`.
    """
    self._param_specs = param_specs
    self._non_tensor_params = non_tensor_params
    self._omit_kwargs = omit_kwargs
    self._prefer_static_value = prefer_static_value
    self._callable_params = {} if callable_params is None else callable_params

    self._serializable = (
        _AUTO_COMPOSITE_TENSOR_VERSION,
        self._param_specs,
        self._non_tensor_params,
        self._omit_kwargs,
        self._prefer_static_value)

    # TODO(b/182603117): Distinguish between `omit_kwargs_from_constructor`
    # and `omit_kwargs_for_comparison`.
    self._comparable = self._serializable + (
        tf.nest.map_structure(id, self._callable_params),)

  @classmethod
  def from_instance(cls, instance, omit_kwargs=()):
    cls_value_type = cls.value_type.fget(None)
    if type(instance) is not cls_value_type:  # pylint: disable=unidiomatic-typecheck
      raise ValueError(f'`{type(instance).__name__}` has inherited the '
                       f'`_type_spec` of `{cls_value_type.__name__}`. It '
                       f'should define its own, either directly, or by '
                       f'applying `auto_composite_tensor` to '
                       f'`{type(instance).__name__}.`')
    prefer_static_value = tuple(
        getattr(instance, '_composite_tensor_shape_params', ()))
    kwargs = _extract_init_kwargs(instance, omit_kwargs=omit_kwargs,
                                  prefer_static_value=prefer_static_value)

    non_tensor_params = {}
    param_specs = {}
    callable_params = {}
    for k, v in list(kwargs.items()):
      # If v contains no Tensors, this will just be v
      type_spec_or_v = _extract_type_spec_recursively(v)
      if type_spec_or_v is not v:
        param_specs[k] = type_spec_or_v
      elif callable(v):
        callable_params[k] = v
      else:
        non_tensor_params[k] = v

    # Construct the spec.
    return cls(param_specs=param_specs,
               non_tensor_params=non_tensor_params,
               omit_kwargs=omit_kwargs,
               prefer_static_value=prefer_static_value,
               callable_params=callable_params)

  def _to_components(self, obj):
    return _extract_init_kwargs(obj, limit_to=list(self._param_specs))

  def _from_components(self, components):
    kwargs = dict(
        self._non_tensor_params, **self._callable_params, **components)
    with _deferred_assertion_context():
      return self.value_type(**kwargs)

  @property
  def _component_specs(self):
    return self._param_specs

  def _serialize(self):
    if self._callable_params:
      raise ValueError(
          f'Cannot serialize object with callable parameters that are not '
          f'`CompositeTensor`s: {self._callable_params.keys()}.')
    return self._serializable

  @classmethod
  def _deserialize(cls, encoded):
    version = encoded[0]
    if version == 1:
      encoded = encoded + ((),)
      version = 2
    if version != _AUTO_COMPOSITE_TENSOR_VERSION:
      raise ValueError(f'Expected version {_AUTO_COMPOSITE_TENSOR_VERSION},'
                       f' but got {version}.')
    return cls(*encoded[1:])

  def most_specific_compatible_type(self, other):
    """Returns the most specific TypeSpec compatible with `self` and `other`.

    Args:
      other: A `TypeSpec`.

    Raises:
      ValueError: If there is no TypeSpec that is compatible with both `self`
        and `other`.
      ValueError: If the `_callable_params` attributes of `self` and `other` are
        not equal.
    """
    if type(self) is not type(other):
      raise ValueError(
          f'No TypeSpec is compatible with both {self} and {other}.')
    # pylint: disable=protected-access
    if self._callable_params != other._callable_params:
      raise ValueError(f'Callable parameters must be identical. Saw '
                       f'{self._callable_params} and {other._callable_params}.')
    merged = self._TypeSpec__most_specific_compatible_type_serialization(
        self._comparable[:-1], other._comparable[:-1])
    # pylint: enable=protected-access
    return type(self)(*merged[1:], self._callable_params)

  def is_compatible_with(self, spec_or_value):
    """Returns true if `spec_or_value` is compatible with this TypeSpec."""
    if not isinstance(spec_or_value, tf.TypeSpec):
      spec_or_value = type_spec.type_spec_from_value(spec_or_value)
    if type(self) is not type(spec_or_value):
      return False
    return self._TypeSpec__is_compatible(
        self._comparable, spec_or_value._comparable)  # pylint: disable=protected-access

  def _with_tensor_ranks_only(self):
    """Returns a TypeSpec compatible with `self`, with tensor shapes relaxed.

    Returns:
      A `TypeSpec` that is compatible with `self`, where any `TensorShape`
      information has been relaxed to include only tensor rank (and not
      the dimension sizes for individual axes).
    """
    def relax(value):
      if isinstance(value, tf.TypeSpec):
        return value._with_tensor_ranks_only()  # pylint: disable=protected-access
      elif (isinstance(value, tf.TensorShape) and
            value.rank is not None):
        return tf.TensorShape([None] * value.rank)
      else:
        return value

    return type(self)(
        tf.nest.map_structure(relax, self._param_specs),
        self._non_tensor_params,
        self._omit_kwargs,
        self._prefer_static_value,
        self._callable_params)

  def __get_cmp_key(self):
    return (type(self), self._TypeSpec__make_cmp_key(self._comparable))

  def __repr__(self):
    return '%s%r' % (
        type(self).__name__, self._serializable + (self._callable_params,))

  def __reduce__(self):
    if self._callable_params:
      raise ValueError(
          f'Cannot serialize object with callable parameters that are not '
          f'`CompositeTensor`s: {self._callable_params.keys()}.')
    super(_AutoCompositeTensorTypeSpec, self).__reduce__()

  def __eq__(self, other):
    return (type(other) is type(self) and
            self.__get_cmp_key() == other.__get_cmp_key())  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self == other

  def __hash__(self):
    return hash(self.__get_cmp_key())


class AutoCompositeTensor(composite_tensor.CompositeTensor):
  """Recommended base class for `@auto_composite_tensor`-ified classes.

  See details in `tfp.experimental.auto_composite_tensor` description.
  """

  @property
  def _type_spec(self):
    # This property will be overwritten by the `@auto_composite_tensor`
    # decorator. However, we need it so that a valid subclass of the `ABCMeta`
    # class `CompositeTensor` can be constructed and passed to the
    # `@auto_composite_tensor` decorator
    pass


def auto_composite_tensor(cls=None, omit_kwargs=(), module_name=None):
  """Automagically generate `CompositeTensor` behavior for `cls`.

  `CompositeTensor` objects are able to pass in and out of `tf.function` and
  `tf.while_loop`, or serve as part of the signature of a TF saved model.

  The contract of `auto_composite_tensor` is that all __init__ args and kwargs
  must have corresponding public or private attributes (or properties). Each of
  these attributes is inspected (recursively) to determine whether it is (or
  contains) `Tensor`s or non-`Tensor` metadata. `list` and `tuple` attributes
  are supported, but must either contain *only* `Tensor`s (or lists, etc,
  thereof), or *no* `Tensor`s. E.g.,
    - object.attribute = [1., 2., 'abc']                        # valid
    - object.attribute = [tf.constant(1.), [tf.constant(2.)]]   # valid
    - object.attribute = ['abc', tf.constant(1.)]               # invalid

  If the attribute is a callable, serialization of the `TypeSpec`, and therefore
  interoperability with `tf.saved_model`, is not currently supported. As a
  workaround, callables that do not contain or close over `Tensor`s may be
  expressed as functors that subclass `AutoCompositeTensor` and used in place of
  the original callable arg:

  ```python
  @auto_composite_tensor(module_name='my.module')
  class F(AutoCompositeTensor):

    def __call__(self, *args, **kwargs):
      return original_callable(*args, **kwargs)
  ```

  Callable objects that do contain or close over `Tensor`s should either
  (1) subclass `AutoCompositeTensor`, with the `Tensor`s passed to the
  constructor, (2) subclass `CompositeTensor` and implement their own
  `TypeSpec`, or (3) have a conversion function registered with
  `type_spec.register_type_spec_from_value_converter`.

  If the object has a `_composite_tensor_shape_parameters` field (presumed to
  have `tuple` of `str` value), the flattening code will use
  `tf.get_static_value` to attempt to preserve shapes as static metadata, for
  fields whose name matches a name specified in that field. Preserving static
  values can be important to correctly propagating shapes through a loop.
  Note that the Distribution and Bijector base classes provide a
  default implementation of `_composite_tensor_shape_parameters`, populated by
  `parameter_properties` annotations.

  If the decorated class `A` does not subclass `CompositeTensor`, a *new class*
  will be generated, which mixes in `A` and `CompositeTensor`.

  To avoid this extra class in the class hierarchy, we suggest inheriting from
  `auto_composite_tensor.AutoCompositeTensor`, which inherits from
  `CompositeTensor` and implants a trivial `_type_spec` @property. The
  `@auto_composite_tensor` decorator will then overwrite this trivial
  `_type_spec` @property. The trivial one is necessary because `_type_spec` is
  an abstract property of `CompositeTensor`, and a valid class instance must be
  created before the decorator can execute -- without the trivial `_type_spec`
  property present, `ABCMeta` will throw an error! The user may thus do any of
  the following:

  #### `AutoCompositeTensor` base class (recommended)
  ```python
  @tfp.experimental.auto_composite_tensor
  class MyClass(tfp.experimental.AutoCompositeTensor):
    ...

  mc = MyClass()
  type(mc)
  # ==> MyClass
  ```

  #### No `CompositeTensor` base class (ok, but changes expected types)
  ```python
  @tfp.experimental.auto_composite_tensor
  class MyClass(object):
    ...

  mc = MyClass()
  type(mc)
  # ==> MyClass_AutoCompositeTensor
  ```

  #### `CompositeTensor` base class, requiring trivial `_type_spec`
  ```python
  from tensorflow.python.framework import composite_tensor
  @tfp.experimental.auto_composite_tensor
  class MyClass(composite_tensor.CompositeTensor):
    @property
    def _type_spec(self):  # will be overwritten by @auto_composite_tensor
      pass
    ...

  mc = MyClass()
  type(mc)
  # ==> MyClass
  ```

  ## Full usage example

  ```python
  @tfp.experimental.auto_composite_tensor(omit_kwargs=('name',))
  class Adder(tfp.experimental.AutoCompositeTensor):
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
    module_name: The module name with which to register the `TypeSpec`. If
      `None`, defaults to `cls.__module__`.

  Returns:
    composite_tensor_subclass: A subclass of `cls` and TF CompositeTensor.
  """
  if cls is None:
    return functools.partial(auto_composite_tensor,
                             omit_kwargs=omit_kwargs,
                             module_name=module_name)

  if module_name is None:
    module_name = cls.__module__

  type_spec_class_name = f'{cls.__name__}_ACTTypeSpec'
  type_spec_name = f'{module_name}.{type_spec_class_name}'

  try:
    ts = type_spec.lookup(type_spec_name)
    return ts.value_type.fget(None)
  except ValueError:
    pass

  # If the declared class is already a CompositeTensor subclass, we can avoid
  # affecting the actual type of the returned class. Otherwise, we need to
  # explicitly mix in the CT type, and hence create and return a newly
  # synthesized type.
  if issubclass(cls, composite_tensor.CompositeTensor):

    @type_spec.register(type_spec_name)
    class _AlreadyCTTypeSpec(_AutoCompositeTensorTypeSpec):

      @property
      def value_type(self):
        return cls

    _AlreadyCTTypeSpec.__name__ = type_spec_class_name

    cls._type_spec = property(  # pylint: disable=protected-access
        lambda self: _AlreadyCTTypeSpec.from_instance(self, omit_kwargs))
    return cls

  clsid = (cls.__module__, cls.__name__, omit_kwargs)

  # Check for subclass if retrieving from the _registry, in case the user
  # has redefined the class (e.g. in a REPL/notebook).
  if clsid in _registry and issubclass(_registry[clsid], cls):
    return _registry[clsid]

  @type_spec.register(type_spec_name)
  class _GeneratedCTTypeSpec(_AutoCompositeTensorTypeSpec):

    @property
    def value_type(self):
      return _registry[clsid]

  _GeneratedCTTypeSpec.__name__ = type_spec_class_name

  class _AutoCompositeTensor(cls, composite_tensor.CompositeTensor):
    """A per-`cls` subclass of `CompositeTensor`."""

    @property
    def _type_spec(self):
      return _GeneratedCTTypeSpec.from_instance(self, omit_kwargs)

  _AutoCompositeTensor.__name__ = cls.__name__
  _registry[clsid] = _AutoCompositeTensor
  return _AutoCompositeTensor
