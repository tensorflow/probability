# Copyright 2018 The TensorFlow Probability Authors.
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
"""Tests for distribution slicing helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions.internal import slicing
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


class _MakeSlices(object):

  def __getitem__(self, slices):
    return slices if isinstance(slices, tuple) else (slices,)


make_slices = _MakeSlices()


@test_util.test_all_tf_execution_regimes
class SlicingTest(test_util.TestCase):

  def test_single_param_slice_withstep_broadcastdim(self):
    event_dim = 3
    sliced = slicing._slice_single_param(
        tf.zeros([1, 1, event_dim]),
        param_event_ndims=1,
        slices=make_slices[44:-52:-3, -94::],
        dist_batch_shape=tf.constant([2, 7], dtype=tf.int32))
    self.assertAllEqual((1, 1, event_dim), self.evaluate(sliced).shape)

  def test_single_param_slice_stop_leadingdim(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:2],
        dist_batch_shape=tf.constant([7, 6, 5], dtype=tf.int32))
    self.assertAllEqual((2, 6, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_stop_trailingdim(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[..., :2],
        dist_batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 6, 2, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_stop_broadcastdim(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 1, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, :2],
        dist_batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 1, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_newaxis_leading(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, tf.newaxis],
        dist_batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 1, 6, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_newaxis_trailing(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[..., tf.newaxis, :],
        dist_batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 6, 1, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_start(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, 2:],
        dist_batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 4, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_start_broadcastdim(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 1, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, 2:],
        dist_batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 1, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_int(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, 2],
        dist_batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_int_broadcastdim(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 1, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, 2],
        dist_batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_tensor(self):
    param = tf1.placeholder_with_default(
        tf.zeros([7, 6, 5, 4, 3]), shape=None)
    idx = tf1.placeholder_with_default(
        tf.constant(2, dtype=tf.int32), shape=[])
    sliced = slicing._slice_single_param(
        param,
        param_event_ndims=2,
        slices=make_slices[:, idx],
        dist_batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_tensor_broadcastdim(self):
    param = tf1.placeholder_with_default(
        tf.zeros([7, 1, 5, 4, 3]), shape=None)
    idx = tf1.placeholder_with_default(
        tf.constant(2, dtype=tf.int32), shape=[])
    sliced = slicing._slice_single_param(
        param,
        param_event_ndims=2,
        slices=make_slices[:, idx],
        dist_batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_broadcast_batch(self):
    if not tf.executing_eagerly():
      return
    sliced = slicing._slice_single_param(
        tf.zeros([4, 3, 1]),  # batch = [4, 3], event = [1]
        param_event_ndims=1,
        slices=make_slices[..., tf.newaxis, 2:, tf.newaxis],
        dist_batch_shape=tf.constant([7, 4, 3]))
    self.assertAllEqual(
        list(tf.zeros([1, 4, 3])[..., tf.newaxis, 2:, tf.newaxis].shape) + [1],
        self.evaluate(sliced).shape)

  def test_single_param_slice_broadcast_batch_leading_newaxis(self):
    if not tf.executing_eagerly():
      return
    sliced = slicing._slice_single_param(
        tf.zeros([4, 3, 1]),  # batch = [4, 3], event = [1]
        param_event_ndims=1,
        slices=make_slices[tf.newaxis, ..., tf.newaxis, 2:, tf.newaxis],
        dist_batch_shape=tf.constant([7, 4, 3]))
    expected = tensorshape_util.as_list((
        tf.zeros([1, 4, 3])[tf.newaxis, ..., tf.newaxis, 2:, tf.newaxis]
        ).shape) + [1]
    self.assertAllEqual(expected, self.evaluate(sliced).shape)

  def test_single_param_multi_ellipsis(self):
    with self.assertRaisesRegexp(ValueError, 'Found multiple `...`'):
      slicing._slice_single_param(
          tf.zeros([7, 6, 5, 4, 3]),
          param_event_ndims=2,
          slices=make_slices[:, ..., 2, ...],
          dist_batch_shape=tf.constant([7, 6, 5]))

  def test_single_param_too_many_slices(self):
    with self.assertRaises(
        (IndexError, ValueError, tf.errors.InvalidArgumentError)):
      slicing._slice_single_param(
          tf.zeros([7, 6, 5, 4, 3]),
          param_event_ndims=2,
          slices=make_slices[:, :3, ..., -2:, :],
          dist_batch_shape=tf.constant([7, 6, 5]))


if __name__ == '__main__':
  tf.test.main()
