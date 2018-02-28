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
"""Tests for Trainable Distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
from tensorflow_probability.python import trainable_distributions as tfp_td
from tensorflow.python.framework import test_util


tfd = tf.contrib.distributions


class TestMVNTriL(tf.test.TestCase):

  def setUp(self):
    np.random.seed(142)

  @test_util.run_in_graph_and_eager_modes()
  def testDefaultsYieldCorrectShapesAndValues(self):
    batch_shape = [4, 3]
    in_dims = 3
    out_dims = 5
    x_ = np.random.randn(*np.concatenate([batch_shape, [in_dims]]))

    x = tf.constant(x_)
    mvn = tfp_td.multivariate_normal_tril(x, dims=out_dims)
    scale = mvn.scale.to_dense()
    scale_upper = tf.matrix_set_diag(
        tf.matrix_band_part(scale, num_lower=0, num_upper=-1),
        tf.zeros(np.concatenate([batch_shape, [out_dims]]), scale.dtype))
    scale_diag = tf.matrix_diag_part(scale)

    self.evaluate(tf.global_variables_initializer())
    [
        batch_shape_,
        event_shape_,
        scale_diag_,
        scale_upper_,
    ] = self.evaluate([
        mvn.batch_shape_tensor(),
        mvn.event_shape_tensor(),
        scale_diag,
        scale_upper,
    ])

    self.assertAllEqual(batch_shape, mvn.batch_shape)
    self.assertAllEqual([out_dims], mvn.event_shape)

    self.assertAllEqual(batch_shape, batch_shape_)
    self.assertAllEqual([out_dims], event_shape_)

    self.assertAllEqual(np.ones_like(scale_diag_, dtype=np.bool),
                        scale_diag_ > 0.)
    self.assertAllEqual(np.zeros_like(scale_upper_), scale_upper_)

  @test_util.run_in_graph_and_eager_modes()
  def testNonDefaultsYieldCorrectShapesAndValues(self):
    batch_shape = [4, 3]
    in_dims = 3
    out_dims = 5
    x_ = np.random.randn(*np.concatenate([batch_shape, [in_dims]]))

    x = tf.constant(x_)
    mvn = tfp_td.multivariate_normal_tril(
        x,
        dims=out_dims,
        loc_fn=tf.zeros_like,
        scale_fn=lambda x: tfd.fill_triangular(tf.ones_like(x)))
    scale = mvn.scale.to_dense()
    expected_scale = tf.matrix_band_part(
        tf.ones(np.concatenate([batch_shape, [out_dims, out_dims]]),
                scale.dtype),
        num_lower=-1,
        num_upper=0)

    self.evaluate(tf.global_variables_initializer())
    [
        batch_shape_,
        event_shape_,
        loc_,
        scale_,
        expected_scale_,
    ] = self.evaluate([
        mvn.batch_shape_tensor(),
        mvn.event_shape_tensor(),
        mvn.loc,
        scale,
        expected_scale,
    ])

    self.assertAllEqual(batch_shape, mvn.batch_shape)
    self.assertAllEqual([out_dims], mvn.event_shape)

    self.assertAllEqual(batch_shape, batch_shape_)
    self.assertAllEqual([out_dims], event_shape_)

    self.assertAllEqual(np.zeros_like(loc_), loc_)
    self.assertAllEqual(expected_scale_, scale_)


if __name__ == '__main__':
  tf.test.main()
