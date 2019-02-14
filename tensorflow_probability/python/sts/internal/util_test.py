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
"""Tests for StructuralTimeSeries utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.sts.internal import util as sts_util

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class MultivariateNormalUtilsTest(tf.test.TestCase):

  def test_factored_joint_mvn_diag_full(self):
    batch_shape = [3, 2]

    mvn1 = tfd.MultivariateNormalDiag(
        loc=tf.zeros(batch_shape + [3]),
        scale_diag=tf.ones(batch_shape + [3]))

    mvn2 = tfd.MultivariateNormalFullCovariance(
        loc=tf.ones(batch_shape + [2]),
        covariance_matrix=(tf.ones(batch_shape + [2, 2]) *
                           [[5., -2], [-2, 3.1]]))

    joint = sts_util.factored_joint_mvn([mvn1, mvn2])
    self.assertEqual(self.evaluate(joint.event_shape_tensor()),
                     self.evaluate(mvn1.event_shape_tensor() +
                                   mvn2.event_shape_tensor()))

    joint_mean_ = self.evaluate(joint.mean())
    self.assertAllEqual(joint_mean_[..., :3], self.evaluate(mvn1.mean()))
    self.assertAllEqual(joint_mean_[..., 3:], self.evaluate(mvn2.mean()))

    joint_cov_ = self.evaluate(joint.covariance())
    self.assertAllEqual(joint_cov_[..., :3, :3],
                        self.evaluate(mvn1.covariance()))
    self.assertAllEqual(joint_cov_[..., 3:, 3:],
                        self.evaluate(mvn2.covariance()))

  def test_factored_joint_mvn_broadcast_batch_shape(self):
    # Test that combining MVNs with different but broadcast-compatible
    # batch shapes yields an MVN with the correct broadcast batch shape.
    random_with_shape = (
        lambda shape: np.random.standard_normal(shape).astype(np.float32))

    event_shape = [3]
    # mvn with batch shape [2]
    mvn1 = tfd.MultivariateNormalDiag(
        loc=random_with_shape([2] + event_shape),
        scale_diag=tf.exp(random_with_shape([2] + event_shape)))

    # mvn with batch shape [3, 2]
    mvn2 = tfd.MultivariateNormalDiag(
        loc=random_with_shape([3, 2] + event_shape),
        scale_diag=tf.exp(random_with_shape([1, 2] + event_shape)))

    # mvn with batch shape [1, 2]
    mvn3 = tfd.MultivariateNormalDiag(
        loc=random_with_shape([1, 2] + event_shape),
        scale_diag=tf.exp(random_with_shape([2] + event_shape)))

    joint = sts_util.factored_joint_mvn([mvn1, mvn2, mvn3])
    self.assertAllEqual(self.evaluate(joint.batch_shape_tensor()), [3, 2])

    joint_mean_ = self.evaluate(joint.mean())
    broadcast_means = tf.ones_like(joint.mean()[..., 0:1])
    self.assertAllEqual(joint_mean_[..., :3],
                        self.evaluate(broadcast_means * mvn1.mean()))
    self.assertAllEqual(joint_mean_[..., 3:6],
                        self.evaluate(broadcast_means * mvn2.mean()))
    self.assertAllEqual(joint_mean_[..., 6:9],
                        self.evaluate(broadcast_means * mvn3.mean()))

    joint_cov_ = self.evaluate(joint.covariance())
    broadcast_covs = tf.ones_like(joint.covariance()[..., :1, :1])
    self.assertAllEqual(joint_cov_[..., :3, :3],
                        self.evaluate(broadcast_covs * mvn1.covariance()))
    self.assertAllEqual(joint_cov_[..., 3:6, 3:6],
                        self.evaluate(broadcast_covs * mvn2.covariance()))
    self.assertAllEqual(joint_cov_[..., 6:9, 6:9],
                        self.evaluate(broadcast_covs * mvn3.covariance()))

  def test_sum_mvns(self):
    batch_shape = [4, 2]
    random_with_shape = (
        lambda shape: np.random.standard_normal(shape).astype(np.float32))

    mvn1 = tfd.MultivariateNormalDiag(
        loc=random_with_shape(batch_shape + [3]),
        scale_diag=np.exp(random_with_shape(batch_shape + [3])))
    mvn2 = tfd.MultivariateNormalDiag(
        loc=random_with_shape(batch_shape + [3]),
        scale_diag=np.exp(random_with_shape(batch_shape + [3])))

    sum_mvn = sts_util.sum_mvns([mvn1, mvn2])
    self.assertAllClose(self.evaluate(sum_mvn.mean()),
                        self.evaluate(mvn1.mean() + mvn2.mean()))
    self.assertAllClose(self.evaluate(sum_mvn.covariance()),
                        self.evaluate(mvn1.covariance() + mvn2.covariance()))

  def test_sum_mvns_broadcast_batch_shape(self):
    random_with_shape = (
        lambda shape: np.random.standard_normal(shape).astype(np.float32))

    event_shape = [3]
    mvn1 = tfd.MultivariateNormalDiag(
        loc=random_with_shape([2] + event_shape),
        scale_diag=np.exp(random_with_shape([2] + event_shape)))
    mvn2 = tfd.MultivariateNormalDiag(
        loc=random_with_shape([1, 2] + event_shape),
        scale_diag=np.exp(random_with_shape([3, 2] + event_shape)))
    mvn3 = tfd.MultivariateNormalDiag(
        loc=random_with_shape([3, 2] + event_shape),
        scale_diag=np.exp(random_with_shape([2] + event_shape)))

    sum_mvn = sts_util.sum_mvns([mvn1, mvn2, mvn3])
    self.assertAllClose(self.evaluate(sum_mvn.mean()),
                        self.evaluate(mvn1.mean() + mvn2.mean() + mvn3.mean()))
    self.assertAllClose(self.evaluate(sum_mvn.covariance()),
                        self.evaluate(mvn1.covariance() +
                                      mvn2.covariance() +
                                      mvn3.covariance()))


@test_util.run_all_in_graph_and_eager_modes
class UtilityTests(tf.test.TestCase):

  def test_broadcast_batch_shape_static(self):

    batch_shapes = ([2], [3, 2], [1, 2])
    distributions = [
        tfd.Normal(loc=tf.zeros(batch_shape), scale=tf.ones(batch_shape))
        for batch_shape in batch_shapes
    ]
    self.assertEqual(sts_util.broadcast_batch_shape(distributions), [3, 2])

  def test_broadcast_batch_shape_dynamic(self):
    # Run in graph mode only, since eager mode always takes the static path
    if tf.executing_eagerly(): return

    batch_shapes = ([2], [3, 2], [1, 2])
    distributions = [
        tfd.Normal(
            loc=tf.compat.v1.placeholder_with_default(
                input=tf.zeros(batch_shape), shape=None),
            scale=tf.compat.v1.placeholder_with_default(
                input=tf.ones(batch_shape), shape=None))
        for batch_shape in batch_shapes
    ]

    self.assertAllEqual(
        [3, 2], self.evaluate(sts_util.broadcast_batch_shape(distributions)))

  def test_maybe_expand_trailing_dim(self):

    # static inputs
    self.assertEqual(
        sts_util.maybe_expand_trailing_dim(tf.zeros([4, 3])).shape,
        tf.TensorShape([4, 3, 1]))
    self.assertEqual(
        sts_util.maybe_expand_trailing_dim(tf.zeros([4, 3, 1])).shape,
        tf.TensorShape([4, 3, 1]))

    # dynamic inputs
    for shape_in, static_shape, expected_shape_out in [
        # pyformat: disable
        ([4, 3], None, [4, 3, 1]),
        ([4, 3, 1], None, [4, 3, 1]),
        ([4], [None], [4, 1]),
        ([1], [None], [1]),
        ([4, 3], [None, None], [4, 3, 1]),
        ([4, 1], [None, None], [4, 1]),
        ([4, 1], [None, 1], [4, 1])
        # pyformat: enable
    ]:
      shape_out = self.evaluate(
          sts_util.maybe_expand_trailing_dim(
              tf.compat.v1.placeholder_with_default(
                  input=tf.zeros(shape_in), shape=static_shape))).shape
      self.assertAllEqual(shape_out, expected_shape_out)

if __name__ == "__main__":
  tf.test.main()
