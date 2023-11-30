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

import contextlib
import functools
import threading

import numpy as np
import tensorflow.compat.v2 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
# pylint: enable=g-direct-tensorflow-import

JAX_MODE = False

__all__ = [
    'auto_composite_tensor',
    'AutoCompositeTensor',
    'is_deferred_assertion_context',
]

_DEFERRED_ASSERTION_CONTEXT = threading.local()
_DEFERRED_ASSERTION_CONTEXT.is_deferred = False


def is_composite_tensor(value):
  """Returns True for CTs and non-CT custom pytrees in JAX mode.

  Args:
    value: A TFP component (e.g. a distribution or bijector instance) or object
      that behaves as one.

  Returns:
    value_is_composite: bool, True if `value` is a `CompositeTensor` in TF mode
      or a non-leaf pytree in JAX mode.
  """
  if isinstance(value, composite_tensor.CompositeTensor):
    return True
  if JAX_MODE:
    from jax import tree_util  # pylint: disable=g-import-not-at-top
    # If `value` is not a pytree leaf, then it must be an instance of a class
    # that was specially registered as a pytree or that inherits from a class
    # representing a nested structure.
    treedef = tree_util.tree_structure(value)
    return not tree_util.treedef_is_leaf(treedef)
  return False


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

_AUTO_COMPOSITE_TENSOR_VERSION = 3

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

  If `value` is none of the above, return it unchanged.

  Args:
    value: a Python `object` to (possibly) turn into a (collection of)
    `tf.TypeSpec`(s).

  Returns:
    spec: the `TypeSpec` or collection of `TypeSpec`s corresponding to `value`;
    `value`, if no `Tensor`s are found; or `None` to indicate that `value` is
    registered as a JAX pytree.
  """
  if isinstance(value, composite_tensor.CompositeTensor):
    return value._type_spec  # pylint: disable=protected-access
  if isinstance(value, tf.Variable):
    return resource_variable_ops.VariableSpec(
        value.shape, dtype=value.dtype, trainable=value.trainable)
  if tf.is_tensor(value):
    return tf.TensorSpec(value.shape, value.dtype)
  if tf.nest.is_nested(value):
    specs = tf.nest.map_structure(_extract_type_spec_recursively, value)
    was_tensor = tf.nest.flatten(
        tf.nest.map_structure(lambda a, b: a is not b, value, specs))
    has_tensors = any(was_tensor)
    has_only_tensors = all(was_tensor)
    if has_tensors:
      if has_tensors != has_only_tensors:
        raise NotImplementedError(
            'Found `{}` with both Tensor and non-Tensor parts: {}'.format(
                type(value), value))
      return specs
  elif JAX_MODE:  # Handle custom pytrees.
    from jax import tree_util  # pylint: disable=g-import-not-at-top
    treedef = tree_util.tree_structure(value)
    # Return None so that the object identity comparison in
    # `_AutoCompositeTensorTypeSpec.from_instance` is False, indicating that
    # `value` should be treated as a "Tensor" param.
    if not tree_util.treedef_is_leaf(treedef):
      return None
  return value


class _AutoCompositeTensorTypeSpec(type_spec.BatchableTypeSpec):
  """A tf.TypeSpec for `AutoCompositeTensor` objects."""

  __slots__ = ('_param_specs', '_non_tensor_params', '_omit_kwargs',
               '_prefer_static_value', '_callable_params', '_serializable',
               '_comparable')

  def __init__(self, param_specs, non_tensor_params, omit_kwargs,
               prefer_static_value, non_identifying_kwargs,
               callable_params=None):
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
      non_identifying_kwargs: Python `tuple` of strings corresponding to the
        names of kwargs to the `AutoCompositeTensor`s constructor whose values
        are not relevant to the unique identification of the
        `_AutoCompositeTensorTypeSpec` instance. Equality/comparison checks and
        `__hash__` do not depend on these kwargs.
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
    self._non_identifying_kwargs = non_identifying_kwargs
    self._callable_params = {} if callable_params is None else callable_params

    self._serializable = (
        _AUTO_COMPOSITE_TENSOR_VERSION,
        self._param_specs,
        self._non_tensor_params,
        self._omit_kwargs,
        self._prefer_static_value,
        self._non_identifying_kwargs)

    def remove_kwargs(d):
      return {k: v for k, v in d.items()
              if k not in self._non_identifying_kwargs}

    self._comparable = (
        _AUTO_COMPOSITE_TENSOR_VERSION,
        remove_kwargs(self._param_specs),
        remove_kwargs(self._non_tensor_params),
        self._omit_kwargs,
        self._prefer_static_value,
        self._non_identifying_kwargs,
        tf.nest.map_structure(id, remove_kwargs(self._callable_params)))

  @classmethod
  def from_instance(cls, instance, omit_kwargs=(), non_identifying_kwargs=()):
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
               non_identifying_kwargs=non_identifying_kwargs,
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
    if version == 2:
      encoded = encoded + ((),)
      version = 3
    if version != _AUTO_COMPOSITE_TENSOR_VERSION:
      raise ValueError(f'Expected version {_AUTO_COMPOSITE_TENSOR_VERSION},'
                       f' but got {version}.')
    return cls(*encoded[1:])

  def is_subtype_of(self, other):
    """Returns True if `self` is subtype of `other`.

    Args:
      other: A `TypeSpec`.
    """
    # pylint: disable=protected-access
    if type(self) is not type(
        other) or self._callable_params != other._callable_params:
      return False

    try:
      tf.nest.assert_same_structure(self._comparable[:-1],
                                    other._comparable[:-1])
    except (TypeError, ValueError):
      return False

    self_elements = tf.nest.flatten(self._comparable[:-1])
    other_elements = tf.nest.flatten(other._comparable[:-1])

    def is_subtype_or_equal(a, b):
      try:
        return a.is_subtype_of(b)
      except AttributeError:
        return a == b

    return all(
        is_subtype_or_equal(self_element, other_element)
        for (self_element, other_element) in zip(self_elements, other_elements))

  def most_specific_common_supertype(self, others):
    """Returns the most specific supertype of `self` and `others`.

    Args:
      others: A Sequence of `TypeSpec`.

    Returns `None` if a supertype does not exist.
    """
    # pylint: disable=protected-access
    if not all(
        type(self) is type(other) and
        self._callable_params == other._callable_params for other in others):
      return None

    try:
      for other in others:
        tf.nest.assert_same_structure(self._comparable[:-1],
                                      other._comparable[:-1])
    except (TypeError, ValueError):
      return None

    self_elements = tf.nest.flatten(self._comparable[:-1])
    others_elements = [
        tf.nest.flatten(other._comparable[:-1]) for other in others
    ]

    def common_supertype_or_equal(a, bs):
      try:
        return a.most_specific_common_supertype(bs)
      except AttributeError:
        return a if all(a == b for b in bs) else None

    common_elements = [None] * len(self_elements)
    for i, self_element in enumerate(self_elements):
      common_elements[i] = common_supertype_or_equal(
          self_element,
          [other_elements[i] for other_elements in others_elements])
      if self_element is not None and common_elements[i] is None:
        return None
    common_comparable = tf.nest.pack_sequence_as(self._comparable[:-1],
                                                 common_elements)

    return type(self)(*common_comparable[1:], self._callable_params)

  def is_compatible_with(self, spec_or_value):
    """Returns true if `spec_or_value` is compatible with this TypeSpec."""
    if not isinstance(spec_or_value, tf.TypeSpec):
      spec_or_value = type_spec.type_spec_from_value(spec_or_value)
    if type(self) is not type(spec_or_value):
      return False
    return self._TypeSpec__is_compatible(
        self._comparable, spec_or_value._comparable)  # pylint: disable=protected-access

  def _copy(self, **overrides):
    kwargs = {
        'param_specs': self._param_specs,
        'non_tensor_params': self._non_tensor_params,
        'omit_kwargs': self._omit_kwargs,
        'prefer_static_value': self._prefer_static_value,
        'non_identifying_kwargs': self._non_identifying_kwargs,
        'callable_params': self._callable_params}
    kwargs.update(overrides)
    return type(self)(**kwargs)

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
    return self._copy(
        param_specs=tf.nest.map_structure(relax, self._param_specs))

  def _without_tensor_names(self):
    """Returns a TypeSpec compatible with `self`, with tensor names removed.

    Returns:
      A `TypeSpec` that is compatible with `self`, where the name of any
      `TensorSpec` is set to `None`.
    """
    def rename(value):
      if isinstance(value, tf.TypeSpec):
        return value._without_tensor_names()  # pylint: disable=protected-access
      else:
        return value
    return self._copy(
        param_specs=tf.nest.map_structure(rename, self._param_specs))

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
    return super(_AutoCompositeTensorTypeSpec, self).__reduce__()

  def __eq__(self, other):
    return (type(other) is type(self) and
            self.__get_cmp_key() == other.__get_cmp_key())  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self == other

  def __hash__(self):
    return hash(self.__get_cmp_key())

  def _batch(self, batch_size):
    """Returns a TypeSpec representing a batch of objects with this TypeSpec."""
    # This method recursively adds a batch dimension to all parameter Tensors.
    # Note that this may result in parameter shapes that do not broadcast. You
    # may wish to first call
    # `dist = dist._broadcast_parameters_with_batch_shape(tf.ones_like(
    # `dist.batch_shape_tensor()))` to ensure that the parameters of a
    # Distribution or analogous object will continue to broadcast after
    # batching.
    return self._copy(
        param_specs=tf.nest.map_structure(
            lambda spec: spec._batch(batch_size),  # pylint: disable=protected-access
            self._param_specs))

  def _unbatch(self):
    """Returns a TypeSpec representing a single element of this TypeSpec."""
    return self._copy(
        param_specs=tf.nest.map_structure(
            lambda spec: spec._unbatch(),  # pylint: disable=protected-access
            self._param_specs))


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


def type_spec_register(name, allow_overwrite=True):
  """Decorator used to register a unique name for a TypeSpec subclass.

  Unlike TensorFlow's `type_spec_registry.register`, this function allows a new
  `TypeSpec` to be registered with a `name` that already appears in the
  registry (overwriting the `TypeSpec` already registered with that name). This
  allows for re-definition of `AutoCompositeTensor` subclasses in test
  environments and iPython.

  Args:
    name: The name of the type spec. Must have the form
    `"{project_name}.{type_name}"`.  E.g. `"my_project.MyTypeSpec"`.
    allow_overwrite: `bool`, if `True` then the entry in the `TypeSpec` registry
      keyed by `name` will be overwritten if it exists. If `False`, then
      behavior is the same as `type_spec.register`.

  Returns:
    A class decorator that registers the decorated class with the given name.
  """
  # pylint: disable=protected-access
  if allow_overwrite and name in type_spec_registry._NAME_TO_TYPE_SPEC:
    type_spec_registry._TYPE_SPEC_TO_NAME.pop(
        type_spec_registry._NAME_TO_TYPE_SPEC.pop(name))
  return type_spec_registry.register(name)


def auto_composite_tensor(
    cls=None, omit_kwargs=(), non_identifying_kwargs=(), module_name=None):
  """Automagically generate `CompositeTensor` behavior for `cls`.

  `CompositeTensor` objects are able to pass in and out of `tf.function` and
  `tf.while_loop`, or serve as part of the signature of a TF saved model.

  The contract of `auto_composite_tensor` is that all __init__ args and kwargs
  must have corresponding public or private attributes (or properties). Each of
  these attributes is inspected (recursively) to determine whether it is (or
  contains) `Tensor`s or non-`Tensor` metadata. Nested (`list`, `tuple`, `dict`,
  etc) attributes are supported, but must either contain *only* `Tensor`s (or
  lists, etc, thereof), or *no* `Tensor`s. E.g.,
    - object.attribute = [1., 2., 'abc']                        # valid
    - object.attribute = [tf.constant(1.), [tf.constant(2.)]]   # valid
    - object.attribute = ['abc', tf.constant(1.)]               # invalid

  All `__init__` args that may be `ResourceVariable`s must also admit `Tensor`s
  (or else `_convert_variables_to_tensors` must be overridden).

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
    non_identifying_kwargs: Optional sequence of kwarg names to be omitted from
      equality/comparison checks and the `__hash__` method of the spec.
    module_name: The module name with which to register the `TypeSpec`. If
      `None`, defaults to `cls.__module__`.

  Returns:
    composite_tensor_subclass: A subclass of `cls` and TF CompositeTensor.
  """
  if cls is None:
    return functools.partial(auto_composite_tensor,
                             omit_kwargs=omit_kwargs,
                             non_identifying_kwargs=non_identifying_kwargs,
                             module_name=module_name)

  if JAX_MODE:
    from jax import tree_util  # pylint: disable=g-import-not-at-top
    tree_util.register_pytree_node(
        cls, pytree_flatten, functools.partial(pytree_unflatten, cls))

  if module_name is None:
    module_name = cls.__module__

  type_spec_class_name = f'{cls.__name__}_ACTTypeSpec'
  type_spec_name = f'{module_name}.{type_spec_class_name}'

  # If the declared class is already a CompositeTensor subclass, we can avoid
  # affecting the actual type of the returned class. Otherwise, we need to
  # explicitly mix in the CT type, and hence create and return a newly
  # synthesized type.
  if issubclass(cls, composite_tensor.CompositeTensor):

    @type_spec_register(type_spec_name)
    class _AlreadyCTTypeSpec(_AutoCompositeTensorTypeSpec):

      @property
      def value_type(self):
        return cls

    _AlreadyCTTypeSpec.__name__ = type_spec_class_name

    def _type_spec(obj):
      return _AlreadyCTTypeSpec.from_instance(
          obj, omit_kwargs, non_identifying_kwargs)

    # pylint: disable=protected-access
    cls._type_spec = property(_type_spec)
    cls._convert_variables_to_tensors = convert_variables_to_tensors
    # pylint: enable=protected-access
    return cls

  clsid = (cls.__module__, cls.__name__, omit_kwargs,
           non_identifying_kwargs)

  # Check for subclass if retrieving from the _registry, in case the user
  # has redefined the class (e.g. in a REPL/notebook).
  if clsid in _registry and issubclass(_registry[clsid], cls):
    return _registry[clsid]

  class _GeneratedCTTypeSpec(_AutoCompositeTensorTypeSpec):

    @property
    def value_type(self):
      return _registry[clsid]

  _GeneratedCTTypeSpec.__name__ = type_spec_class_name

  class _AutoCompositeTensor(cls, composite_tensor.CompositeTensor):
    """A per-`cls` subclass of `CompositeTensor`."""

    @property
    def _type_spec(self):
      return _GeneratedCTTypeSpec.from_instance(
          self, omit_kwargs, non_identifying_kwargs)

    def _convert_variables_to_tensors(self):
      return convert_variables_to_tensors(self)

  _AutoCompositeTensor.__name__ = cls.__name__
  _registry[clsid] = _AutoCompositeTensor
  type_spec_register(type_spec_name)(_GeneratedCTTypeSpec)
  return _AutoCompositeTensor


def convert_variables_to_tensors(obj):
  """Recursively converts Variables in the AutoCompositeTensor to Tensors.

  This method flattens `obj` into a nested structure of `Tensor`s or
  `CompositeTensor`s, converts any `ResourceVariable`s (which are
  `CompositeTensor`s) to `Tensor`s, and rebuilds `obj` with `Tensor`s in place
  of `ResourceVariable`s.

  The usage of `obj._type_spec._from_components` violates the contract of
  `CompositeTensor`, since it is called on a different nested structure
  (one containing only `Tensor`s) than `obj.type_spec` specifies (one that may
  contain `ResourceVariable`s). Since `AutoCompositeTensor`'s
  `_from_components` method passes the contents of the nested structure to
  `__init__` to rebuild the TFP object, and any TFP object that may be
  instantiated with `ResourceVariables` may also be instantiated with
  `Tensor`s, this usage is valid.

  Args:
    obj: An `AutoCompositeTensor` instance.

  Returns:
    tensor_obj: `obj` with all internal `ResourceVariable`s converted to
      `Tensor`s.
  """
  # pylint: disable=protected-access
  components = obj._type_spec._to_components(obj)
  tensor_components = variable_utils.convert_variables_to_tensors(
      components)
  return obj._type_spec._from_components(tensor_components)
  # pylint: enable=protected-access


def pytree_flatten(obj):
  """Flatten method for JAX pytrees."""
  # pylint: disable=protected-access
  components = obj._type_spec._to_components(obj)

  if components:
    keys, values = zip(*components.items())
  else:
    keys, values = (), ()

  metadata = dict(non_tensor_params=obj._type_spec._non_tensor_params,
                  callable_params=obj._type_spec._callable_params)
  return values, (keys, metadata)


def pytree_unflatten(cls, aux_data, children):
  """Unflatten method for JAX pytrees."""
  keys, metadata = aux_data
  parameters = dict(list(zip(keys, children)),
                    **metadata['non_tensor_params'],
                    **metadata['callable_params'])
  return cls(**parameters)
