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
"""Tests for slicing helper."""

import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import scale_matvec_diag
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import split
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import mixture_same_family
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import slicing
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
        batch_shape=tf.constant([2, 7], dtype=tf.int32))
    self.assertAllEqual((1, 1, event_dim), self.evaluate(sliced).shape)

  def test_single_param_slice_stop_leadingdim(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:2],
        batch_shape=tf.constant([7, 6, 5], dtype=tf.int32))
    self.assertAllEqual((2, 6, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_stop_trailingdim(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[..., :2],
        batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 6, 2, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_stop_broadcastdim(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 1, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, :2],
        batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 1, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_newaxis_leading(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, tf.newaxis],
        batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 1, 6, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_newaxis_trailing(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[..., tf.newaxis, :],
        batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 6, 1, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_start(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, 2:],
        batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 4, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_start_broadcastdim(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 1, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, 2:],
        batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 1, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_int(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 6, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, 2],
        batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_int_broadcastdim(self):
    sliced = slicing._slice_single_param(
        tf.zeros([7, 1, 5, 4, 3]),
        param_event_ndims=2,
        slices=make_slices[:, 2],
        batch_shape=tf.constant([7, 6, 5]))
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
        batch_shape=tf.constant([7, 6, 5]))
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
        batch_shape=tf.constant([7, 6, 5]))
    self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_broadcast_batch(self):
    if not tf.executing_eagerly():
      return
    sliced = slicing._slice_single_param(
        tf.zeros([4, 3, 1]),  # batch = [4, 3], event = [1]
        param_event_ndims=1,
        slices=make_slices[..., tf.newaxis, 2:, tf.newaxis],
        batch_shape=tf.constant([7, 4, 3]))
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
        batch_shape=tf.constant([7, 4, 3]))
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
          batch_shape=tf.constant([7, 6, 5]))

  def test_single_param_too_many_slices(self):
    with self.assertRaises(
        (IndexError, ValueError, tf.errors.InvalidArgumentError)):
      slicing._slice_single_param(
          tf.zeros([7, 6, 5, 4, 3]),
          param_event_ndims=2,
          slices=make_slices[:, :3, ..., -2:, :],
          batch_shape=tf.constant([7, 6, 5]))

  def test_slice_single_param_distribution(self):
    sliced = slicing._slice_single_param(
        normal.Normal(
            loc=tf.zeros([4, 3, 1]),  # batch = [4, 3], event = [2]
            scale=tf.ones([2])),
        param_event_ndims=1,
        slices=make_slices[..., tf.newaxis, 2:, tf.newaxis],
        batch_shape=tf.constant([7, 4, 3]))
    self.assertAllEqual(
        list(tf.zeros([1, 4, 3])[..., tf.newaxis, 2:, tf.newaxis].shape),
        sliced.batch_shape_tensor()[:-1])

  def test_slice_single_param_atomic(self):
    sliced = slicing._slice_single_param(
        identity.Identity(),
        param_event_ndims=0,
        slices=make_slices[..., tf.newaxis, 2:, tf.newaxis],
        batch_shape=tf.constant([7, 4, 3]))
    self.assertAllEqual([], sliced.experimental_batch_shape_tensor())

  def test_slice_single_param_bijector_composition(self):
    sliced = slicing._slice_single_param(
        joint_map.JointMap({
            'a': chain.Chain([invert.Invert(scale.Scale(tf.ones([4, 3, 1])))])
        }),
        param_event_ndims={'a': 1},
        slices=make_slices[..., tf.newaxis, 2:, tf.newaxis],
        batch_shape=tf.constant([7, 4, 3]))
    self.assertAllEqual(
        list(tf.zeros([1, 4, 3])[..., tf.newaxis, 2:, tf.newaxis].shape),
        sliced.experimental_batch_shape_tensor(x_event_ndims={'a': 1}))

  def test_jitted_slices(self):
    self.skip_if_no_xla()
    shp = [7, 6, 5, 4]
    t = tf.cast(tf.reshape(tf.range(np.prod(shp)), shp), tf.float32)
    @tf.function(jit_compile=True)
    def f(ix):
      return slicing._slice_params_to_dict(
          mvn_diag.MultivariateNormalDiag(t, tf.ones([shp[-1]])),
          slices=make_slices[..., ix, :])
    self.assertAllEqual(t[:, 3], f(tf.constant(3))['loc'])

  def test_slice_transformed_distribution_with_chain(self):
    dist = transformed_distribution.TransformedDistribution(
        distribution=mvn_diag.MultivariateNormalDiag(
            loc=tf.zeros([4]), scale_diag=tf.ones([1, 4])),
        bijector=chain.Chain([
            joint_map.JointMap(
                [identity.Identity(),
                 shift.Shift(tf.ones([4, 3, 2]))]),
            split.Split(2),
            scale_matvec_diag.ScaleMatvecDiag(tf.ones([5, 1, 3, 4])),
            exp.Exp()
        ]))
    self.assertAllEqual(dist.batch_shape_tensor(), [5, 4, 3])
    self.assertAllEqualNested(
        tf.nest.map_structure(lambda x: x.shape,
                              dist.sample(seed=test_util.test_seed())),
        [[5, 4, 3, 2], [5, 4, 3, 2]])

    sliced = dist[tf.newaxis, ..., 0, :, :-1]
    self.assertAllEqual(sliced.batch_shape_tensor(), [1, 4, 2])
    self.assertAllEqualNested(
        tf.nest.map_structure(lambda x: x.shape,
                              sliced.sample(seed=test_util.test_seed())),
        [[1, 4, 2, 2], [1, 4, 2, 2]])

  def test_slice_nested_mixture(self):
    dist = mixture_same_family.MixtureSameFamily(
        categorical.Categorical(logits=tf.zeros([2])),
        mixture_same_family.MixtureSameFamily(
            categorical.Categorical(logits=tf.zeros([2])),
            bernoulli.Bernoulli(logits=tf.zeros([1, 2, 2]))))
    self.assertAllEqual(dist[0, ...].batch_shape_tensor(), [])
    self.assertAllEqual(dist[0, ..., tf.newaxis].batch_shape_tensor(), [1])
    self.assertAllEqual(dist[..., tf.newaxis].batch_shape_tensor(), [1, 1])

  def test_slicing_does_not_modify_the_sliced_distribution(self):
    dist = exponential.Exponential(tf.ones((5, 2, 3)))
    sliced = dist[:4, :, 2]
    self.assertAllEqual([2], sliced[-1].batch_shape_tensor())
    self.assertAllEqual([3], sliced[:-1, 1].batch_shape_tensor())


if __name__ == '__main__':
  test_util.main()
