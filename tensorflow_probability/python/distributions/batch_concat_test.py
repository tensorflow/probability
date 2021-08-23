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
"""Tests for BatchConcat."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import batch_concat
from tensorflow_probability.python.internal import test_util


class _BatchConcatTest(object):
  batch_dim_1 = [1, 3, 4]
  batch_dim_2 = [2, 2, 4]
  batch_dim_3 = [2, 1, 1]
  event_dim_1 = [2]
  event_dim_2 = [2]
  event_dim_3 = [2]
  axis = 1
  dtype = tf.float32

  def get_distributions(self, validate_args=False):
    self.dist1 = tfd.MultivariateNormalDiag(
        loc=self.maybe_static(
            tf.zeros(self.batch_dim_1 + self.event_dim_1, dtype=self.dtype),
            self.is_static),
        scale_diag=self.maybe_static(
            tf.ones(self.batch_dim_1 + self.event_dim_1, dtype=self.dtype),
            self.is_static)
    )

    self.dist2 = tfd.OneHotCategorical(
        logits=self.maybe_static(
            tf.zeros(self.batch_dim_2 + self.event_dim_2),
            self.is_static),
        dtype=self.dtype)

    self.dist3 = tfd.Dirichlet(
        self.maybe_static(
            tf.zeros(self.batch_dim_3 + self.event_dim_3, dtype=self.dtype),
            self.is_static)
        )
    return batch_concat.BatchConcat(
        distributions=[self.dist1, self.dist2, self.dist3], axis=self.axis,
        validate_args=validate_args)

  def test_scalar_distributions(self):
    self.dist1 = tfd.Normal(
        loc=self.maybe_static(
            tf.zeros(self.batch_dim_1, dtype=self.dtype),
            self.is_static),
        scale=self.maybe_static(
            tf.ones(self.batch_dim_1, dtype=self.dtype),
            self.is_static)
    )
    self.dist2 = tfd.Logistic(
        loc=self.maybe_static(
            tf.zeros(self.batch_dim_2, dtype=self.dtype),
            self.is_static),
        scale=self.maybe_static(
            tf.ones(self.batch_dim_2, dtype=self.dtype),
            self.is_static)
    )
    self.dist3 = tfd.Exponential(
        rate=self.maybe_static(
            tf.ones(self.batch_dim_3, dtype=self.dtype),
            self.is_static)
    )
    concat_dist = batch_concat.BatchConcat(
        distributions=[self.dist1, self.dist2, self.dist3], axis=1,
        validate_args=False)
    self.assertAllEqual(
        self.evaluate(concat_dist.batch_shape_tensor()),
        [2, 6, 4])

    seed = test_util.test_seed()
    samples = concat_dist.sample(seed=seed)
    self.assertAllEqual(self.evaluate(tf.shape(samples)), [2, 6, 4])

  def test_batch_shape_tensor(self):
    concat_dist = self.get_distributions()
    self.assertAllEqual(
        self.evaluate(concat_dist.batch_shape_tensor()),
        [2, 6, 4])

  def test_batch_shape(self):
    concat_dist = self.get_distributions()
    concat_dist = self.get_distributions()
    if tf.executing_eagerly() or self.is_static:
      self.assertAllEqual(concat_dist.batch_shape,
                          [2, 6, 4])
    else:
      self.assertEqual(concat_dist.batch_shape, tf.TensorShape(None))

  def test_event_shape_tensor(self):
    concat_dist = self.get_distributions()
    self.assertAllEqual(self.evaluate(concat_dist.event_shape_tensor()), [2])

  def test_event_shape(self):
    concat_dist = self.get_distributions()
    if tf.executing_eagerly() or self.is_static:
      self.assertAllEqual(concat_dist.event_shape, [2])
    else:
      self.assertAllEqual(concat_dist.event_shape.as_list(), [None])

  def test_sample(self):
    concat_dist = self.get_distributions()
    seed = test_util.test_seed()
    samples = concat_dist.sample(seed=seed)
    self.assertAllEqual(self.evaluate(tf.shape(samples)), [2, 6, 4, 2])
    samples = concat_dist.sample([12, 20], seed=seed)
    self.assertAllEqual(self.evaluate(tf.shape(samples)), [12, 20, 2, 6, 4, 2])

  def test_sample_and_log_prob(self):
    concat_dist = self.get_distributions()
    seed = test_util.test_seed()
    samples, lp = concat_dist.experimental_sample_and_log_prob(seed=seed)
    self.assertAllEqual(self.evaluate(tf.shape(samples)), [2, 6, 4, 2])
    self.assertAllClose(lp, concat_dist.log_prob(samples))

  def test_split_sample(self):
    concat_dist = self.get_distributions()
    x_sample = tf.ones([2, 6, 4, 2])
    sample_shape_size, x_split = concat_dist._split_sample(x_sample)
    self.assertEqual(len(x_split), 3)
    if tf.executing_eagerly() or self.is_static:
      self.assertEqual(sample_shape_size, 0)
    else:
      self.assertEqual(self.evaluate(sample_shape_size), 0)

    self.assertAllEqual(self.evaluate(tf.shape(x_split[0])), [2, 3, 4, 2])
    self.assertAllEqual(self.evaluate(tf.shape(x_split[1])), [2, 2, 4, 2])
    self.assertAllEqual(self.evaluate(tf.shape(x_split[2])), [2, 1, 4, 2])

  def test_log_prob(self):
    concat_dist = self.get_distributions()
    x_sample = tf.ones([32, 128, 2, 6, 4, 2])
    log_prob = concat_dist.log_prob(x_sample)
    self.assertAllEqual(self.evaluate(tf.shape(log_prob)), [32, 128, 2, 6, 4])
    self.assertAllEqual(log_prob[:, :, :, :3],
                        self.dist1.log_prob(tf.ones([32, 128, 2, 3, 4, 2])))
    self.assertAllEqual(log_prob[:, :, :, 3:5],
                        self.dist2.log_prob(tf.ones([32, 128, 2, 2, 4, 2])))
    self.assertAllEqual(log_prob[:, :, :, 5:],
                        self.dist3.log_prob(tf.ones([32, 128, 2, 1, 4, 2])))

  def test_log_prob_no_batch(self):
    concat_dist = self.get_distributions()
    x_sample = tf.ones([2])
    if tf.executing_eagerly() or self.is_static:
      with self.assertRaises(Exception):
        concat_dist.log_prob(x_sample)
    else:
      log_prob = concat_dist.log_prob(x_sample)
      with self.assertRaises(Exception):
        self.evaluate(log_prob)

  def test_log_prob_broadcast_batch(self):
    concat_dist = self.get_distributions()
    x_sample = tf.ones([1, 1, 1, 2])
    if tf.executing_eagerly() or self.is_static:
      with self.assertRaises(Exception):
        concat_dist.log_prob(x_sample)
    else:
      log_prob = concat_dist.log_prob(x_sample)
      with self.assertRaises(Exception):
        self.evaluate(log_prob)

  def test_mean(self):
    concat_dist = self.get_distributions()
    means = concat_dist.mean()
    self.assertAllEqual(self.evaluate(tf.shape(means)), [2, 6, 4, 2])
    self.assertAllEqual(means[:, :3],
                        tf.zeros([2, 3, 4, 2], dtype=tf.float32))
    self.assertAllEqual(means[:, 3:5],
                        tf.ones([2, 2, 4, 2], dtype=tf.float32) * 0.5)

  def test_incompatible_batch_shape(self):
    batch_dims = [[1, 3, 5], [1, 3, 4, 10]]
    for batch_dim in batch_dims:
      self.batch_dim_1 = batch_dim
      if tf.executing_eagerly() or self.is_static:
        with self.assertRaises(Exception):
          concat_dist = self.get_distributions()
      else:
        concat_dist = self.get_distributions()
        with self.assertRaises(Exception):
          self.evaluate(concat_dist.mean())

  def test_incompatible_event_shape(self):
    self.event_dim_2 = [3]
    if tf.executing_eagerly() or self.is_static:
      with self.assertRaises(Exception):
        concat_dist = self.get_distributions()
    else:
      concat_dist = self.get_distributions()
      with self.assertRaises(Exception):
        self.evaluate(concat_dist.mean())

  def test_batch_concat_of_concat(self):
    concat_dist_1 = self.get_distributions()
    concat_dist_2 = self.get_distributions()
    concat_concat = batch_concat.BatchConcat(
        [concat_dist_1, concat_dist_2], axis=0)
    self.assertAllEqual(
        self.evaluate(concat_concat.batch_shape_tensor()),
        [4, 6, 4])
    x_sample = tf.zeros([32, 4, 6, 4, 2])
    self.assertAllEqual(
        self.evaluate(tf.shape(concat_concat.log_prob(x_sample))),
        [32, 4, 6, 4])

  def test_axis_negative(self):
    self.axis = -1
    with self.assertRaises(Exception):
      self.get_distributions()

  def test_axis_out_of_range(self):
    self.axis = 10
    if tf.executing_eagerly() or self.is_static:
      with self.assertRaises(Exception):
        self.get_distributions()
    else:
      concat_dist = self.get_distributions()
      with self.assertRaises(Exception):
        self.evaluate(concat_dist.mean())


@test_util.test_all_tf_execution_regimes
class StaticBatchConcatTest(_BatchConcatTest, test_util.TestCase):
  is_static = True


@test_util.test_all_tf_execution_regimes
class DynamicBatchConcatTest(_BatchConcatTest, test_util.TestCase):
  is_static = False


if __name__ == '__main__':
  test_util.main()
