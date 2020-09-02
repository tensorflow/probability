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
"""Utilities for bijector-caches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import weakref

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'BijectorCache',
    'BijectorCacheWithGreedyAttrs'
]


def hashable_structure(struct):
  """Hashes a possibly mutable structure of `Tensor`s."""
  def make_hashable(obj):
    if isinstance(obj, (HashableWeakRef, WeakStructRef, ObjectIdentityWrapper)):
      return obj
    elif isinstance(obj, np.ndarray):
      obj_hash = hash(str(obj.__array_interface__) + str(id(obj)))
      return ObjectIdentityWrapper(obj, object_hash=obj_hash)
    elif tf.is_tensor(obj):
      return ObjectIdentityWrapper(obj)
    try:
      hash(obj)
      return obj
    except TypeError:
      return ObjectIdentityWrapper(obj)
  # Flatten structs into a tuple of tuples to make mutable containers hashable.
  return tuple((k, make_hashable(v))
               for k, v in nest.flatten_with_tuple_paths(struct))


class ObjectIdentityWrapper(object):
  """Wraps an object, mapping __eq__ on wrapper to 'is' on wrapped."""

  __slots__ = ['_wrapped', '_hash']

  def __init__(self, wrapped, object_hash=None):
    if object_hash is None:
      object_hash = id(wrapped)
    self._hash = object_hash
    self._wrapped = wrapped

  def __eq__(self, other):
    if not isinstance(other, ObjectIdentityWrapper):
      return False
    return self._wrapped is other._wrapped  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return self._hash


class HashableWeakRef(weakref.ref):
  """weakref.ref which makes tf.Tensor and np.array objects hashable.

  We take care to ensure that a hash can still be provided in the case that the
  ref has been cleaned up. This ensures that the WeakKeyDefaultDict doesn't
  suffer memory leaks by failing to clean up HashableWeakRef key objects whose
  referrents have gone out of scope and been destroyed (as in
  https://github.com/tensorflow/probability/issues/647).
  """
  __slots__ = ('_hash',)

  def __init__(self, referrent, callback=None):
    """weakref.ref which makes tf.Tensor and np.array objects hashable.

    Arguments:
      referrent: Object that is being referred to.
      callback: Optional callback to invoke when object is GCed.
    """
    if isinstance(referrent, np.generic):
      raise ValueError('Unable to weakref np.generic')
    elif isinstance(referrent, np.ndarray):
      referrent.flags.writeable = False
    super(HashableWeakRef, self).__init__(referrent, callback)
    self._hash = hash(hashable_structure(referrent))

  def __repr__(self):
    return '{}({})'.format(
        self.__class__.__name__,
        super(HashableWeakRef, self).__repr__())

  def __hash__(self):
    return self._hash

  def __eq__(self, other):
    if other is None:
      return False
    # Check hashes and ids. This is the same for both strong and weak ref types.
    # Note: JAX does not (officially) support referential equality.
    # This may break in the future!
    return (hash(self) == hash(other)) and (self() is other())

  def __ne__(self, other):
    # weakref.ref overrides `!=` explicitly, so we must as well.
    return not self == other


class WeakStructRef(object):
  """weakref.ref object for tensors, np.ndarray, and structures.

  Invokes the callback when any element of the struct's ref-count drops to 0.
  """
  __slots__ = ('_struct', '_subkey', '_hash', '_callback', '_alive')

  def __init__(self, struct, subkey=None, callback=None):
    wrap = lambda x: HashableWeakRef(x, self._cleanup)
    self._struct = nest.map_structure(wrap, struct)
    # Copy subkey to avoid mutations.
    self._subkey = nest.map_structure((lambda x: x), subkey)
    # Because `struct` and `subkey` are both immutable containers,
    # we assume that their hashes are stable and unchanging.
    # Compute the hash and store it with the object.
    self._hash = hash(hashable_structure((self._struct, self._subkey)))
    self._callback = callback
    self._alive = True

  @property
  def alive(self):
    return self._alive

  def cancel(self):
    self._callback = None

  def _cleanup(self, _):
    if self._alive:
      if self._callback:
        self._callback(self)
        self._callback = None
      self._alive = False

  @property
  def ref_set(self):
    """Returns the set of unique references."""
    return set(nest.flatten(self._struct))

  @property
  def subkey(self):
    """Returns the subkey associated with the struct."""
    # Potentially unsafe to return mutable subkey! If this is ever exposed
    # outside of `cache_util`, consider returning a copy.
    return self._subkey

  def __call__(self):
    """Unwraps the tensor reference."""
    return nest.map_structure(lambda x: x(), self._struct)

  def __hash__(self):
    """Returns the cached hash of this structure."""
    return self._hash

  def __eq__(self, other):
    if isinstance(other, WeakStructRef):
      return (self.alive == other.alive
              and hash(self) == hash(other)
              and self._subkey == other._subkey   # pylint: disable=protected-access
              and self._struct == other._struct)  # pylint: disable=protected-access
    return False

  def __repr__(self):
    return '{clazz}({referrent})'.format(
        clazz=self.__class__.__name__,
        referrent=self._struct)


# This only affects numpy/jax backends, where non-tensor values can be returned.
def _containerize(value):
  """Converts values to types supported by weakref."""
  if isinstance(value, (int, float, np.generic)):
    return np.array(value)
  return value


# Private container for _CachedDirectedFunc state.
_DirectedFunctionInvocation = collections.namedtuple(
    'DirectedFunctionInvocation', [
        'input_ref',    # WeakRef input to the directed function
        'output_ref',   # WeakRef output of the directed function
        'attributes',   # Metadata associated with the input/output pair
        'dependencies'  # Optional references to arguments, preventing GC
    ])


class EphemeralDict(dict):
  """`dict` that enables unbound weakref callback."""

  def __init__(self, *args, **kwargs):
    super(EphemeralDict, self).__init__(*args, **kwargs)

    weakself = weakref.ref(self)
    def maybe_del(key):
      self = weakself()
      if self is not None:
        del self[key]
    self.maybe_del = maybe_del


class BijectorCache(object):
  """Caches a weak chain of transformations for all bijector instances.

  On a cache miss, the other side of the transformation is produced
  and both sides are cached before returning a result. The input object
  is cached with a strong reference, while the transformed output is
  held weakly. When there are no more references to the result, both
  directions of the mapping are cleared simulataneously.

  The general contracts are:
    1) The inputs used on cache misses will always be retained
       until their corresponding outputs are abandoned.
    2) `forward` will return a cached result if and only if
       `inverse` would have returned a cached result.

  Retaining results in such a manner guarantees that there will be no
  circular references, allowing the garbage collector to remove "dead"
  links from the cache.

  Example usage:

  ```py
  bij = MyBijector()
  cache = BijectorCache(bijector=bij)

  x = tf.constant(1.)
  ref = weakref.ref(x)

  for i in range(10):
    assert (len(cache.weak_keys(direction='forward'))
            == len(cache.weak_keys(direction='inverse'))
            == i)
    x = cache.forward(x)

  for i in range(10):
    assert len(cache.weak_keys(direction='inverse')) == 10 - i
    x = cache.inverse(x)
    assert len(cache.weak_keys(direction='forward')) == 10 - i

  assert len(cache.weak_keys(direction='inverse')) == 0
  assert ref() == x

  x = None
  assert len(cache.weak_keys(direction='forward')) == 0
  ```
  """

  def __init__(
      self,
      forward_name='_forward',
      inverse_name='_inverse',
      bijector=None,
      bijector_class=None,
      storage=None):
    """Constructs the BijectorCache.

    Args:
      forward_name: `str`, name of the bijectors' forward transformations.
      inverse_name: `str`, Name of the bijectors' inverse transformations.
      bijector: `tfb.Bijector` instance, or None for an unbound cache.
      bijector_class: Type of `bijector`, or None for an unbound cache.
      storage: `EphemeralDict` instance or None.
    """
    self._forward_name = forward_name
    self._inverse_name = inverse_name
    self._bijector = bijector
    self._bijector_class = bijector_class
    self.storage = EphemeralDict() if storage is None else storage

  def __get__(self, bijector, bijector_class=None):
    """Defines cache behavior when accessed as a bijector attribute.

    We define `__get__` so that the object can be used as a `Bijector` class
    attribute and still access the (eventual) `Bijector` instance without
    requiring that the `Bijector` instance pass `self` back into this cache.

    Args:
      bijector: a `tfb.Bijector` instance to which the cache is bound.
      bijector_class: Type of `bijector`.

    Returns:
      cache: A `BijectorCache` instance bound to `bijector`.
    """
    return type(self)(
        forward_name=self._forward_name,
        inverse_name=self._inverse_name,
        bijector=bijector,
        bijector_class=bijector_class,
        storage=self.storage)

  @property
  def bijector(self):
    return self._bijector

  @property
  def bijector_class(self):
    return self._bijector_class

  def forward(self, x, **kwargs):
    """Invokes the 'forward' transformation, or looks up previous results.

    Arguments:
      x: The singular argument passed to `bijector._forward`.
      **kwargs: Any auxiliary arguments passed to the function.
        These reflect shared context to the function, and are associated
        with both the input and output cache-keys.
    Returns:
      The output of the bijector's `_forward` method, or a cached result.
    """
    return self._lookup(x, self._forward_name, self._inverse_name, **kwargs)

  def inverse(self, y, **kwargs):
    """Invokes the 'inverse' transformation, or looks up previous results.

    Arguments:
      y: The singular argument passed to `bijector._inverse`.
      **kwargs: Any auxiliary arguments passed to the function.
        These reflect shared context to the function, and are associated
        with both the input and output cache-keys.
    Returns:
      The output of the bijector's `_inverse` method, or a cached result.
    """
    return self._lookup(y, self._inverse_name, self._forward_name, **kwargs)

  def forward_attributes(self, x, **kwargs):
    return self._attributes(x, self._forward_name, **kwargs)

  def inverse_attributes(self, y, **kwargs):
    return self._attributes(y, self._inverse_name, **kwargs)

  def weak_keys(self, bijector=None, bijector_class=None, direction=None):
    """Returns the keys in input_cache."""
    bijector = bijector or self.bijector
    bijector_class = bijector_class or self.bijector_class

    if direction == 'forward':
      fn_name = self._forward_name
    elif direction == 'inverse':
      fn_name = self._inverse_name
    elif direction is None:
      fn_name = None
    else:
      raise ValueError('`direction` must be `"forward"`, `"inverse"`, or '
                       '`None`. Saw: {}'.format(direction))
    out = []
    for k in self.storage.keys():
      bijector_key, bijector_class_key, fn_key, _ = k.subkey
      if (k.alive and (bijector is None or bijector_key == bijector)
          and (bijector_class is None or bijector_class == bijector_class_key)
          and (fn_name is None or fn_key == fn_name)):
        out.append(k)
    return out

  def items(self, bijector=None, bijector_class=None, direction=None):
    return [(k(), self.storage[k].output_ref())
            for k in self.weak_keys(bijector, bijector_class, direction)]

  # pylint: disable=redefined-builtin
  def _attributes(self, input, fn_name, **kwargs):
    """Looks up user-defined attributes associated with this argument.

    Attributes are dictionaries that store arbitrary information about the
    transformation; for example, its log Jacobian determinant. Each cache lookup
    will also return the corresponding `attributes` dictionary, which the user
    may populate however they want. Attributes are "bidirectional", such that
    `cache.forward_attributes(x)` will return the same dictionary instance as
    `cache.inverse_attributes(cache.forward(x))`.

    This method is "lazy", in the sense that it doesn't invoke the wrapped
    transformation. Whether `attributes` or `__call__` are invoked first does
    not affect behavior.

    Examples:

    ```python
    ## Basic Usage
    bijector = tfb.Exp()
    cache = BijectorCache(bijector=bijector)
    x = tf.random.normal()

    # Compute some attribute for x, and add it to the cache.
    ildj = -bijector.forward_log_det_jacobian(x)
    cache.forward_attributes(x)['ildj'] = ildj

    # Apply the forward transformation, and lookup the attribute by `y`.
    y = cache.forward(x)
    assert ildj is cache.inverse_attributes(y)['ildj']

    ## Garbage Collection
    # Attributes remain cached until refs to both `x` and `y` are collected.
    x = None
    assert ildj is cache.inverse_attributes(y)['ildj']

    x, y = cache.inverse(y), None
    assert ildj is cache.forward_attributes(x)['ildj']

    # When both refs are collected, attributes are cleared.
    x = y = None
    assert (len(cache.weak_keys(direction='forward'))
            == len(cache.weak_keys(direction='inverse'))
            == 0)
    ```

    Arguments:
      input: The singular ordered argument passed to the wrapped function.
      fn_name: `str`, name of the directed function to which `input` is an arg
        (typically `'_forward'` or `'_inverse'`).
      **kwargs: Any auxiliary arguments passed to the function.
        These reflect shared context to the function, and are associated
        with both the input and output cache-keys.
    Returns:
      The dictionary of attributes associated with the transformation.
    """
    if input is None:
      raise ValueError('Input must not be None.')
    return self._get_or_create_edge(input, fn_name, kwargs).attributes

  def _lookup(self, input, forward_name, inverse_name, **kwargs):
    """Retrieves or computes `out` and `attrs` for the argument.

    When called, this method checks the input->output cache for previously
    computed values. On cache miss, it invokes the forward function and
    inserts references from input->output and output->input in the corresponding
    caches.

    The reference to the output is weak, such that both sides of the cache will
    be cleared symmetrically if the output is GCed. In contract, the reference
    to the argument is strong: it can be recovered as long as a reference to the
    output is held.

    Examples:

    ```python
    cache = BijectorCache()
    x = tf.random.normal()
    y, attrs = cache._lookup(x, '_forward', '_inverse')
    attrs['foo'] = 'bar'

    assert cache.inverse._lookup(y, '_inverse', '_forward') == (x, attrs)
    ```

    Arguments:
      input: The singular ordered argument passed to the wrapped function.
      forward_name: `str`, the name of the function implementing the bijector's
        forward transformation (typically `'_forward'`).
      inverse_name: `str`, the name of the function implementing the bijector's
        inverse transformation (typically `'_inverse'`).
      **kwargs: Additional arguments associated with the transformation.
        These arguments are interpreted as inputs to both `x->y` and `y->x`.
    Returns:
      A tuple containing `output`, and a dictionary of `attributes`.
      Like `kwargs`, attributes are associated with both the `x->y` and `y->x`
      transformations. Unlike kwargs, they do not affect the behavior of
      either function. It's left to the user to add things to attributes.
    """
    if input is None:
      raise ValueError('Input must not be None.')
    input_ref, output_ref, attrs, _ = self._get_or_create_edge(
        input, forward_name, kwargs)
    output = output_ref() if output_ref else None
    if output is None:
      # Get the output structure, and declare a
      # weakref to clear it from the cache once it gets GCed
      output = nest.map_structure(
          _containerize,
          self._invoke(input, forward_name, kwargs, attrs))
      output_ref = WeakStructRef(
          output,
          subkey=(self.bijector, self.bijector_class, inverse_name, kwargs),
          callback=self.storage.maybe_del)
      # Set the input->output mapping.
      self.storage[input_ref] = _DirectedFunctionInvocation(
          input_ref, output_ref, attributes=attrs, dependencies=None)
      # As long as outputs are around keep strong references to inputs,
      # allowing us to recover the argument on request.
      to_retain = tuple(x() for x in input_ref.ref_set - output_ref.ref_set)
      self.storage[output_ref] = _DirectedFunctionInvocation(
          output_ref, input_ref, attributes=attrs, dependencies=to_retain)
    return output

  def __len__(self):
    """Returns the length of the input cache.

    This may differ from the output cache when lazily-defined attributes exist
    """
    return len(self.weak_keys(direction='forward'))

  def _get_or_create_edge(self, input, forward_name, kwargs):
    """Gets the _Edge associated with an input."""
    test_ref = WeakStructRef(
        input,
        subkey=(self.bijector, self.bijector_class, forward_name, kwargs),
        callback=self.storage.maybe_del)
    default = _DirectedFunctionInvocation(
        test_ref, None, attributes={}, dependencies=None)
    cached = self.storage.setdefault(test_ref, default)
    # If test_ref was in the cache, then the callback is already registered.
    # Cancel the callback we just created to prevent KeyErrors.
    if cached is not default:
      test_ref.cancel()
    return cached

  def _invoke(self, input, fn_name, kwargs, attributes):  # pylint: disable=unused-argument
    """Invokes the wrapped function. Override to customize behavior."""
    return getattr(self.bijector, fn_name)(input, **kwargs)

  def clear(self):
    """Clears cached values.

    If the cache is bound to a bijector instance, all entries in the cache keyed
    by the hash of the bijector instance are cleared. If the cache is bound to
    a bijector class, all entries keyed by the class are cleared. If the cache
    is unbound, the cache storage is re-instantiated as an empty EphemeralDict.
    """
    if self.bijector is None:
      if self.bijector_class is None:
        self.storage = EphemeralDict()
      else:
        for k in self.weak_keys():
          if k.subkey[1] == self.bijector_class:
            k.cancel()
            del self.storage[k]
    else:
      for k in self.weak_keys():
        if k.subkey[0] == self.bijector:
          k.cancel()
          del self.storage[k]


class BijectorCacheWithGreedyAttrs(BijectorCache):
  """A CachedDirectedFunction that updates attributes when called.

  Expects the wrapped function to return an `(out, attributes)` tuple.
  Updates the cached attributes based on the second element of the response.
  """

  def _invoke(self, input, fn_name, kwargs, attributes):  # pylint: disable=redefined-builtin
    # Update `attributes` with the attrs returned by `_func`.
    output, greedy_attrs = getattr(self.bijector, fn_name)(input, **kwargs)
    if greedy_attrs is not None:
      attributes.update(greedy_attrs)
    return output
