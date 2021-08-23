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
"""Tests for cache_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import cache_util
from tensorflow_probability.python.internal import test_util


class HashableRefTest(test_util.TestCase):

  @parameterized.parameters(
      (cache_util.HashableWeakRef,),
      (cache_util.WeakStructRef,))
  def testReferentialEquality(self, ref_type):
    tensors = (
        tf.constant([1., 2., 3.], dtype=tf.float32, name='alice'),
        tf.constant([1., 2., 3.], dtype=tf.float32, name='alice'))
    ref1, ref2 = map(ref_type, tensors)
    self.assertNotEqual(ref1, ref2)
    self.assertNotEqual(hash(ref1), hash(ref2))

  def testStructRefChecksSubkey(self):
    tensor = tf.constant([1., 2., 3.], dtype=tf.float32, name='alice')
    ref1 = cache_util.WeakStructRef(tensor, subkey='a')
    ref2 = cache_util.WeakStructRef(tensor, subkey='b')
    self.assertNotEqual(ref1, ref2)
    self.assertNotEqual(hash(ref1), hash(ref2))

  def testStructRefIsWeak(self):
    tensor_struct = {
        'a': tf.constant(1., name='alice'),
        'b': tf.constant(2., name='bob'),
        'c': tf.constant(3., name='carol')}

    weak_ref = cache_util.WeakStructRef(tensor_struct)
    another_weak_ref = cache_util.WeakStructRef(tensor_struct)

    self.assertTrue(weak_ref.alive)
    del tensor_struct

    # In eager mode, references get cleaned up.
    if tf.executing_eagerly():
      self.assertFalse(weak_ref.alive)
      self.assertFalse(another_weak_ref.alive)
    # In graph mode, references stay alive.
    else:
      self.assertTrue(weak_ref.alive)
      self.assertTrue(another_weak_ref.alive)

  def testStructRefCallbackFiresOnce(self):
    tensor_struct = {
        'a': tf.constant(1., name='alice'),
        'b': tf.constant(2., name='bob'),
        'c': tf.constant(3., name='carol')}

    callback_keys = []
    def callback(key):
      callback_keys.append(key)
    struct_ref = cache_util.WeakStructRef(tensor_struct, callback=callback)

    if tf.executing_eagerly():
      self.assertTrue(struct_ref.alive)
      self.assertEqual(callback_keys, [])
      del tensor_struct['a']  # Goodbye, Alice!
      self.assertFalse(struct_ref.alive)
      self.assertEqual(callback_keys, [struct_ref])
      del tensor_struct['b']  # Goodbye, Bob!
      self.assertEqual(callback_keys, [struct_ref])
      del tensor_struct  # Goodbye, everybody!
      self.assertEqual(callback_keys, [struct_ref])

    else:
      self.assertTrue(struct_ref.alive)
      del tensor_struct
      self.assertTrue(struct_ref.alive)

  def testWeakStructRespectsContainerTypes(self):
    a = tf.constant(1., name='alice')
    b = tf.constant(2., name='bob')
    # Different lists are equal.
    self.assertEqual(cache_util.WeakStructRef([a, b]),
                     cache_util.WeakStructRef([a, b]))
    # List and tuple with same contents are not equal.
    self.assertNotEqual(cache_util.WeakStructRef([a, b]),
                        cache_util.WeakStructRef((a, b)))

  def testWeakStructCopiesContainers(self):
    a = tf.constant(1., name='alice')
    b = tf.constant(2., name='bob')
    c = tf.constant(3., name='carol')
    tensor_struct = [a, {'x': b, 'y': c}]

    # Reference the original struct.
    weak_ref = cache_util.WeakStructRef(tensor_struct)

    # Copy the structure, then mutate the original inplace.
    struct_copy = tf.nest.map_structure(lambda x: x, tensor_struct)
    tensor_struct[:] = tensor_struct[::-1]

    self.assertEqual(weak_ref(), struct_copy)
    with self.assertRaises(ValueError):  # pylint: disable=g-error-prone-assert-raises
      tf.nest.assert_same_structure(weak_ref(), tensor_struct)


class _BareBonesBijector(object):
  """Minimal specification of a `Bijector`."""

  def __init__(self, forward_impl, inverse_impl):
    self.forward_call_count = 0
    self.inverse_call_count = 0
    self._forward_impl = forward_impl
    self._inverse_impl = inverse_impl

  def _forward(self, x):
    self.forward_call_count += 1
    return self._forward_impl(x)

  def _inverse(self, y):
    self.inverse_call_count += 1
    return self._inverse_impl(y)


class CacheTestBase(object):
  """Common tests for CachedDirectedFunction."""

  def setUp(self):
    if not tf.executing_eagerly():
      self.skipTest('Not interesting in graph mode.')

    self.bijector = _BareBonesBijector(self.forward_impl, self.inverse_impl)
    self.cache = cache_util.BijectorCache(
        bijector=self.bijector, bijector_class=type(self.bijector))
    super(CacheTestBase, self).setUp()

  ### Abstract methods

  def forward_impl(self, x, **kwds):
    raise NotImplementedError()

  def inverse_impl(self, y, **kwds):
    raise NotImplementedError()

  def test_arg(self):
    raise NotImplementedError()

  def forward(self, x):
    return self.cache.forward(x)

  def inverse(self, y):
    return self.cache.inverse(y)

  @property
  def forward_keys(self):
    return self.cache.weak_keys(direction='forward')

  @property
  def inverse_keys(self):
    return self.cache.weak_keys(direction='inverse')


class CacheTestCommon(CacheTestBase):
  """Common tests for single-part and multi-part bijectors."""

  def testInputsAreCached(self, steps=4):
    struct = self.test_arg()
    weak_start = cache_util.WeakStructRef(struct)

    # Call forward a few times.
    for _ in range(steps):
      struct = self.forward(struct)

    print(type(struct), struct)
    self.assertLen(self.forward_keys, steps)
    self.assertEqual(self.bijector.forward_call_count, steps)
    self.assertLen(self.cache.items(direction='forward'), steps)
    self.assertLen(self.inverse_keys, steps)
    self.assertEqual(self.bijector.inverse_call_count, 0)
    self.assertLen(self.cache.items(direction='inverse'), steps)

    # Now invert our calls
    for _ in range(steps):
      struct = self.inverse(struct)

    self.assertLen(self.forward_keys, 1)  # Has cached attrs.
    self.assertEqual(self.bijector.forward_call_count, steps)    # No new calls.
    self.assertLen(self.inverse_keys, 0)  # Refs are all gone.
    self.assertEqual(self.bijector.inverse_call_count, 0)      # All cache hits.

    # Original is recoverable. Contents are referentially equal.
    self.assertTrue(weak_start.alive)
    tf.nest.map_structure(self.assertIs, struct, weak_start())

    struct = None
    self.assertFalse(weak_start.alive)
    self.assertLen(self.forward_keys, 0)
    self.assertLen(self.inverse_keys, 0)

  def testOutputsWaterfall(self, steps=4):
    struct = self.test_arg()

    # Call forward a few times.
    for i in range(steps):
      self.assertLen(self.forward_keys, i)
      self.assertLen(self.inverse_keys, i)
      struct = self.forward(struct)

    # Grab a reference, and call forward some more.
    mid_ref = struct
    for i in range(steps):
      self.assertLen(self.forward_keys, steps + i)
      self.assertLen(self.inverse_keys, steps + i)
      struct = self.forward(struct)

    # Clear strong references to the final output
    del struct
    self.assertLen(self.inverse_keys, steps)
    self.assertLen(self.forward_keys, steps + 1)

    # Clear strong references to the midpoint
    del mid_ref
    self.assertLen(self.forward_keys, 0)
    self.assertLen(self.inverse_keys, 0)

  def testAttrsGetCleanedUp(self):
    x = self.test_arg()
    x_attrs = self.cache.forward_attributes(x)
    x_attrs['foo'] = 'bar'

    # Attributes only exist in from_x
    self.assertEqual(x_attrs, {'foo': 'bar'})
    self.assertLen(self.forward_keys, 1)  # Holding attrs
    self.assertLen(self.inverse_keys, 0)  # from_x not called; nothing here.

    x = None
    # When x goes out of scope, it's cleared from the cache.
    self.assertLen(self.forward_keys, 0)
    self.assertLen(self.inverse_keys, 0)
    # But external references to attrs may be retained.
    self.assertEqual(x_attrs, {'foo': 'bar'})

  def testAttributesAreLazilyPropagated(self):
    x = self.test_arg()
    attrs = self.cache.forward_attributes(x)
    attrs['foo'] = 'bar'

    # Attributes only exist in from_x
    self.assertEqual(attrs, {'foo': 'bar'})
    self.assertLen(self.forward_keys, 1)
    self.assertLen(self.inverse_keys, 0)

    # Once we call from_x, they get shared.
    y = self.forward(x)
    attrs = self.cache.inverse_attributes(y)
    self.assertEqual(attrs, {'foo': 'bar'})
    self.assertLen(self.forward_keys, 1)
    self.assertLen(self.inverse_keys, 1)

    # Add some attrs to y, and clear the reference.
    y = None
    attrs['xxx'] = 'yyy'
    self.assertLen(self.inverse_keys, 0)
    self.assertLen(self.forward_keys, 1)

    # Cached attributes still exist for `x`
    self.assertEqual(self.cache.forward_attributes(x),
                     {'foo': 'bar', 'xxx': 'yyy'})

    # Clear x, and the rest of the attrs are gone.
    x = None
    self.assertLen(self.forward_keys, 0)
    self.assertLen(self.inverse_keys, 0)

  def testClear(self):

    bijector_2 = _BareBonesBijector(self.forward_impl, self.inverse_impl)
    cache_2 = cache_util.BijectorCache(
        bijector=bijector_2,
        bijector_class=_BareBonesBijector,
        storage=self.cache.storage)
    class_cache = cache_util.BijectorCache(
        bijector=None,
        bijector_class=_BareBonesBijector,
        storage=self.cache.storage)

    x = self.test_arg()
    _ = self.forward(x)
    _ = cache_2.forward(x)

    y = self.test_arg()
    _ = self.forward(y)

    # Class cache has three entries
    self.assertLen(class_cache.weak_keys(direction='forward'), 3)

    # Clear entry associated with bijector_2
    cache_2.clear()
    self.assertLen(class_cache.weak_keys(direction='forward'), 2)

    # Clear class cache
    class_cache.clear()
    self.assertLen(class_cache.weak_keys(direction='forward'), 0)


class SinglePartCacheTest(CacheTestCommon, test_util.TestCase):
  """Cache tests for single-part bijectors."""

  def setUp(self):
    if not tf.executing_eagerly():
      self.skipTest('Not interesting in graph mode.')
    super(SinglePartCacheTest, self).setUp()

  def forward_impl(self, x):
    return x + 1

  def inverse_impl(self, y):
    return y - 1

  def test_arg(self):
    return tf.constant(0.)


class MultiPartCacheTest(CacheTestCommon, test_util.TestCase):
  """Cache tests for multi-part bijectors."""

  def forward_impl(self, struct):
    x, y = struct
    return x, y * x

  def inverse_impl(self, struct):
    x, z = struct
    return x, z / x

  def test_arg(self):
    return tf.constant(2.), tf.constant(1.)


class IdentityCacheTest(CacheTestBase):
  """Cache tests for bijectors that don't return new things."""

  def forward_impl(self, struct):
    x, y = struct
    return y, x

  def inverse_impl(self, struct):
    y, x = struct
    return x, y

  def test_arg(self):
    return tf.constant(1.), tf.constant(2.)

  def testCircularReferencesDontLeak(self):
    struct = self.test_arg()

    # Transform and add to cache.
    struct = self.forward(struct)
    self.assertLen(self.forward_keys, 1)
    self.assertLen(self.inverse_keys, 1)

    # Outputs are the same as inputs; nothing's been cleared or added.
    struct = self.inverse(struct)
    self.assertLen(self.forward_keys, 1)
    self.assertLen(self.inverse_keys, 1)

    # We haven't called inverse on the original yet!
    struct = self.inverse(struct)
    self.assertLen(self.forward_keys, 2)
    self.assertLen(self.inverse_keys, 2)

    # Cache is emptied without leaks.
    struct = None
    self.assertLen(self.forward_keys, 0)
    self.assertLen(self.inverse_keys, 0)


if __name__ == '__main__':
  test_util.main()
