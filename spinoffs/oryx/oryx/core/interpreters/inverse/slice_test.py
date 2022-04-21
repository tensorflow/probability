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
"""Tests for tensorflow_probability.spinoffs.oryx.core.interpreters.inverse.slice."""
from absl.testing import absltest

import jax.numpy as jnp

from oryx.core.interpreters.inverse import slice as slc
from oryx.internal import test_util

Slice = slc.Slice
NDSlice = slc.NDSlice

# pylint: disable=g-generic-assert


class SliceTest(test_util.TestCase):

  def assertNotLess(self, left, right):
    self.assertFalse(left < right, msg=f'{left} is less than {right}.')

  def test_slice_less_than_slices_that_contain_it(self):
    slice_ = Slice(2, 5)
    self.assertNotLess(slice_, Slice(2, 5))
    self.assertNotLess(slice_, Slice(3, 5))
    self.assertNotLess(slice_, Slice(2, 4))
    self.assertLess(slice_, Slice(0, 5))
    self.assertLess(slice_, Slice(0, 6))

  def test_slices_equal_when_boundaries_are_identical(self):
    slice_ = Slice(2, 5)
    self.assertEqual(slice_, Slice(2, 5))
    self.assertNotEqual(slice_, Slice(3, 5))
    self.assertNotEqual(slice_, Slice(2, 4))

  def test_1d_slice_less_than_should_behave_like_slice(self):
    value = jnp.ones(5)
    ndslice = NDSlice.new(value, jnp.zeros_like(value))
    self.assertLess(NDSlice(value, jnp.zeros_like(value), Slice(2, 3)), ndslice)
    self.assertLess(NDSlice(value, jnp.zeros_like(value), Slice(2, 5)), ndslice)
    self.assertNotLess(
        NDSlice(value, jnp.zeros_like(value), Slice(0, 5)), ndslice)

  def test_2d_slice_less_check_any_dimension(self):
    value = jnp.ones((4, 5))
    ndslice = NDSlice.new(value, jnp.zeros_like(value))
    self.assertLess(
        NDSlice(value, jnp.zeros_like(value), Slice(0, 4), Slice(2, 3)),
        ndslice)
    self.assertLess(
        NDSlice(value, jnp.zeros_like(value), Slice(2, 4), Slice(0, 5)),
        ndslice)
    self.assertNotLess(
        NDSlice(value, jnp.zeros_like(value), Slice(0, 4), Slice(0, 5)),
        ndslice)
    self.assertNotLess(
        NDSlice(value, jnp.zeros_like(value), Slice(0, 4), Slice(3, 5)),
        NDSlice(value, jnp.zeros_like(value), Slice(2, 3), Slice(0, 5)))

  def test_2d_slice_can_concatenate_works(self):
    value = jnp.ones((4, 5))
    ndslice = NDSlice(value, jnp.ones((4, 5)), Slice(0, 4), Slice(0, 3))
    self.assertTrue(
        ndslice.can_concatenate(
            NDSlice(value, jnp.zeros_like(value), Slice(0, 4), Slice(3, 5)), 1))
    self.assertFalse(
        ndslice.can_concatenate(
            NDSlice(value, jnp.zeros_like(value), Slice(0, 4), Slice(3, 5)), 0))
    self.assertFalse(
        ndslice.can_concatenate(
            NDSlice(value, jnp.zeros_like(value), Slice(0, 4), Slice(3, 5)), 0))

  def test_2d_slice_concatenate_works(self):
    value = jnp.ones((4, 3))
    ndslice = NDSlice(value, jnp.zeros((4, 3)), Slice(0, 4), Slice(0, 3))
    out_slice = ndslice.concatenate(
        NDSlice(jnp.ones((4, 2)), jnp.zeros((4, 2)), Slice(0, 4), Slice(3, 5)),
        1)
    self.assertTupleEqual(out_slice.value.shape, (4, 5))
    self.assertTupleEqual(out_slice.slices, (Slice(0, 4), Slice(0, 5)))


if __name__ == '__main__':
  absltest.main()
