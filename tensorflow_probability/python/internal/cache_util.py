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
import functools
import weakref

# Dependency imports
import numpy as np

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'BijectorCache',
    'CachedDirectedFunction',
    'CachedDirectedFunctionWithGreedyAttrs',
]


class weak_binding(object):  # pylint: disable=invalid-name
  """Wrapper class for weakrefs to instance methods.

  Decorates a method defined on the class. When the method is accessed on an
  instance, it is replaced with a "weak" version that becomes a no-op after
  the parent object is GCed. The wrapped method is bound to the instance so it
  doesn't have to be recreated on every access.
  """

  def __init__(self, func):
    self.func = func

  def __get__(self, instance, owner):
    # Return the unbound method.
    if instance is None:
      return self.func
    # Construct a weakly bound method.
    bound_method = self._bind(self.func, weakref.ref(instance))
    # Don't redefine the bound method every time it's accessed.
    setattr(instance, self.func.__name__, bound_method)
    return bound_method

  @staticmethod
  def _bind(func, instref):
    """Returns a bound method that is only invoked while the instance lives."""
    @functools.wraps(func)
    def bound_method(*args, **kwargs):
      # Only invoke the wrapped function if "self" is alive.
      instance = instref()
      if instance is not None:
        return func(instance, *args, **kwargs)
    return bound_method


class _IdentityHash(object):
  """Wraps an object to return `id(obj)` in place of `hash(obj)`."""
  __slots__ = ('obj',)

  def __init__(self, obj):
    self.obj = obj

  def __hash__(self):
    return id(self.obj)


def hash_structure(struct):
  """Hashes a possibly mutable structure of tensors."""
  def make_hashable(obj):
    if isinstance(obj, (HashableWeakRef, WeakStructRef)):
      return obj
    elif isinstance(obj, np.ndarray):
      return str(obj.__array_interface__) + str(id(obj))
    else:
      return _IdentityHash(obj)
  # Flatten structs into a tuple of tuples to make mutable containers hashable.
  flat_pairs = nest.flatten_with_tuple_paths(struct)
  hashable = ((k, make_hashable(v)) for k, v in flat_pairs)
  return hash(tuple(hashable))


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
    self._hash = hash_structure(referrent)

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
    self._hash = hash_structure((self._struct, self._subkey))
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


class CachedDirectedFunction(object):
  """Handles one half of an invertible cache.

  Wraps a function, input->output dict, and an output->input dict.
  When called, this method checks the input->output cache for previously
  computed values. On cache miss, it invokes the wrapped function and
  inserts references from input->output and output->input in the corresponding
  caches.

  The reference to the output is weak, such that both sides of the cache will
  be cleared symmetrically if the output is GCed. In contract, the reference
  to the argument is strong: it can be recovered as long as a reference to the
  output is held.
  """

  def __init__(self, func, input_cache=None, output_cache=None):
    """Handles one half of an invertible cache.

    Arguments:
      func: The function to wrap.
      input_cache: A dictionary used to store input->output mappings.
      output_cache: A dictionary used to store output->output mappings.
    """
    self._func = func
    self._input_cache = input_cache
    self._output_cache = output_cache

  # pylint: disable=redefined-builtin
  def __call__(self, input, **kwargs):
    """Invokes the wrapped transformation, or looks up previous results.

    Arguments:
      input: The singular argument passed to the wrapped function.
      **kwargs: Any auxiliary arguments passed to the function.
        These reflect shared context to the function, and are associated
        with both the input and output cache-keys.
    Returns:
      The output of the wrapped function, or a cached result.
    """
    return self.lookup(input, **kwargs)[0]

  def attributes(self, input, **kwargs):
    """Looks up user-defined attributes associated with this argument.

    Attributes are dictionaries that store arbitrary information about the
    transformation; for example, its log Jacobian determinant. Each cache lookup
    will also return the corresponding `attributes` dictionary, which the user
    may populate however they want. Attributes are "bidirectional", such that
    `cache.forward.attributes(x)` will return the same dictionary instance as
    `cache.inverse.attributes(cache.forward(x))`.

    This method is "lazy", in the sense that it doesn't invoke the wrapped
    transformation. Whether `attributes` or `__call__` are invoked first does
    not affect behavior.

    Examples:

    ```python
    ## Basic Usage
    cache = BijectorCache(foward_impl, inverse_impl)
    x = tf.random.normal()

    # Compute some attribute for x, and add it to the cache.
    ildj = -forward_log_det_jacobian_impl(x)
    cache.forward.attributes(x)['ildj'] = ildj

    # Apply the forward transformation, and lookup the attribute by `y`.
    y = cache.forward(x)
    assert ildj is cache.inverse.attributes(y)['ildj']

    ## Garbage Collection
    # Attributes remain cached until refs to both `x` and `y` are collected.
    x = None
    assert ildj is cache.inverse.attributes(y)['ildj']

    x, y = cache.inverse(y), None
    assert ildj is cache.forward.attributes(x)['ildj']

    # When both refs are collected, attributes are cleared.
    x = y = None
    assert len(cache.forward) == len(cache.inverse) == 0
    ```

    Arguments:
      input: The singular ordered argument passed to the wrapped function.
      **kwargs: Any auxiliary arguments passed to the function.
        These reflect shared context to the function, and are associated
        with both the input and output cache-keys.
    Returns:
      The dictionary of attributes associated with the transformation.
    """
    if input is None:
      raise ValueError('Input must not be None.')
    return self._get_or_create_edge(input, kwargs).attributes

  def lookup(self, input, **kwargs):
    """Retrieves or computes `out` and `attrs` for the argument.

    Examples:

    ```python
    cache = BijectorCache(forward_impl, inverse_impl)
    x = tf.random.normal()
    y, attrs = cache.lookup(x)
    attrs['foo'] = 'bar'

    assert cache.inverse.lookup(y) == (x, attrs)
    ```

    Arguments:
      input: The singular ordered argument passed to the wrapped function.
      **kwargs: Additional arguments associated with the transformation.
        These arguments are interpreted as inputs to both `x->y` and `y->x`.
    Returns:
      A tuple containing `output`, and a dictionary of `attributes`.
      Like `kwargs`, attributes are associated with both the `x->y` and `y->x`
      transformations. Unlike kwargs, they do not affect the behavior of the
      either function. It's left to the user to add things to attributes.
    """
    if input is None:
      raise ValueError('Input must not be None.')
    input_ref, output_ref, attrs, _ = self._get_or_create_edge(input, kwargs)
    output = output_ref() if output_ref else None
    if output is None:
      # Get the output structure, and declare a
      # weakref to clear it from the cache once it gets GCed
      output = nest.map_structure(_containerize,
                                  self._invoke(input, kwargs, attrs))
      output_ref = WeakStructRef(output, kwargs, self._cleanup_output)
      # Set the input->output mapping.
      self._input_cache[input_ref] = _DirectedFunctionInvocation(
          input_ref, output_ref, attributes=attrs, dependencies=None)
      # As long as outputs are around keep strong references to inputs,
      # allowing us to recover the argument on request.
      to_retain = tuple(x() for x in input_ref.ref_set - output_ref.ref_set)
      self._output_cache[output_ref] = _DirectedFunctionInvocation(
          output_ref, input_ref, attributes=attrs, dependencies=to_retain)
    return output, attrs

  def __len__(self):
    """Returns the length of the input cache.

    This may differ from the output cache when lazily-defined attributes exist.
    """
    return len(self._input_cache)

  def weak_keys(self):
    """Returns the keys in input_cache."""
    return self._input_cache.keys()

  def _get_or_create_edge(self, input, kwargs):
    """Gets the _Edge associated with an input."""
    test_ref = WeakStructRef(input, kwargs, self._cleanup_input)
    default = _DirectedFunctionInvocation(
        test_ref, None, attributes={}, dependencies=None)
    cached = self._input_cache.setdefault(test_ref, default)
    # If test_ref was in the cache, then the callback is already registered.
    # Cancel the callback we just created to prevent KeyErrors.
    if cached is not default:
      test_ref.cancel()
    return cached

  def _invoke(self, input, kwargs, attributes):  # pylint: disable=unused-argument
    """Invokes the wrapped function. Override to customize behavior."""
    return self._func(input, **kwargs)

  # Cleanup methods must ignore callbacks after `self` has been GCed.
  # These are invoked independently so outputs can be cleared without
  # necessarily losing cached `attributes`.

  @weak_binding
  def _cleanup_input(self, input_ref):
    """Pops a key from the input->output cache."""
    del self._input_cache[input_ref]

  @weak_binding
  def _cleanup_output(self, output_ref):
    """Pops a key from the output->input cache."""
    del self._output_cache[output_ref]


class CachedDirectedFunctionWithGreedyAttrs(CachedDirectedFunction):
  """A CachedDirectedFunction that updates attributes when called.

  Expects the wrapped function to return an `(out, attributes)` tuple.
  Updates the cached attributes based on the second element of the response.
  """

  def _invoke(self, input, kwargs, attributes):  # pylint: disable=redefined-builtin
    # Update `attributes` with the attrs returned by `_func`.
    output, greedy_attrs = self._func(input, **kwargs)
    if greedy_attrs is not None:
      attributes.update(greedy_attrs)
    return output


class BijectorCache(object):
  """Class that caches a weak chain of bidirectional transformations.

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
  cache = BijectorCache(bij._forward, bij._inverse)

  x = tf.constant(1.)
  ref = weakref.ref(x)

  for i in range(10):
    assert len(cache.forward) == len(cache.inverse) == i
    x = cache.forward(x)

  for i in range(10):
    assert len(cache.inverse) == 10 - i
    x = cache.inverse(x)
    assert len(cache.forward) == 10 - i

  assert len(cache.inverse) == 0
  assert ref() == x

  x = None
  assert len(cache.forward) == 0
  ```
  """

  def __init__(self,
               forward_impl,
               inverse_impl,
               cache_type=CachedDirectedFunction):
    # Shared dictionaries mean each CachedDirectedFunction can look up
    # inverse results that were populated by the other.
    x2y, y2x = {}, {}
    self.forward = cache_type(forward_impl, x2y, y2x)
    self.inverse = cache_type(inverse_impl, y2x, x2y)

  def reset(self):
    """Clears all cached values."""
    self.__init__(self.forward._func, self.inverse._func,  # pylint: disable=protected-access
                  type(self.forward), type(self.inverse))
