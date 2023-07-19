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

# Dependency imports

from absl.testing import parameterized

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import batch_concat
from tensorflow_probability.python.distributions import dirichlet
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import logistic
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import onehot_categorical
from tensorflow_probability.python.internal import test_util


def maybe_static(x, is_static):
  return x if is_static else tf1.placeholder_with_default(x, shape=None)


def get_scalar_distributions(batch_dim_1=(1, 3, 4),
                             batch_dim_2=(2, 2, 4),
                             batch_dim_3=(2, 1, 1),
                             axis=1,
                             dtype=tf.float32,
                             validate_args=False,
                             is_static=False):
  dist1 = normal.Normal(
      loc=maybe_static(tf.zeros(batch_dim_1, dtype=dtype), is_static),
      scale=maybe_static(tf.ones(batch_dim_1, dtype=dtype), is_static))

  dist2 = exponential.Exponential(
      rate=maybe_static(tf.ones(batch_dim_2), is_static))

  dist3 = logistic.Logistic(
      loc=maybe_static(tf.zeros(batch_dim_3, dtype=dtype), is_static),
      scale=maybe_static(tf.ones(batch_dim_3, dtype=dtype), is_static))
  return ([dist1, dist2, dist3],
          batch_concat.BatchConcat(
              distributions=[dist1, dist2, dist3],
              axis=axis,
              validate_args=validate_args))


def get_vector_distributions(batch_dim_1=(1, 3, 4),
                             batch_dim_2=(2, 2, 4),
                             batch_dim_3=(2, 1, 1),
                             axis=1,
                             dtype=tf.float32,
                             validate_args=False,
                             is_static=False):
  dist1 = mvn_diag.MultivariateNormalDiag(
      loc=maybe_static(tf.zeros(batch_dim_1 + (2,), dtype=dtype), is_static),
      scale_diag=maybe_static(
          tf.ones(batch_dim_1 + (2,), dtype=dtype), is_static))

  dist2 = onehot_categorical.OneHotCategorical(
      logits=maybe_static(tf.zeros(batch_dim_2 + (2,)), is_static), dtype=dtype)

  dist3 = dirichlet.Dirichlet(
      maybe_static(tf.zeros(batch_dim_3 + (2,), dtype=dtype), is_static))
  return ([dist1, dist2, dist3],
          batch_concat.BatchConcat(
              distributions=[dist1, dist2, dist3],
              axis=axis,
              validate_args=validate_args))


def get_matrix_distributions(batch_dim_1=(1, 3, 4),
                             batch_dim_2=(2, 2, 4),
                             batch_dim_3=(2, 1, 1),
                             axis=1,
                             dtype=tf.float32,
                             validate_args=False,
                             is_static=False):
  # 2x2 matrix-valued distributions
  dist1 = independent.Independent(
      exponential.Exponential(
          rate=maybe_static(
              tf.ones(batch_dim_1 + (2, 2), dtype=dtype), is_static)),
      reinterpreted_batch_ndims=2)

  dist2 = independent.Independent(
      logistic.Logistic(
          loc=maybe_static(
              tf.zeros(batch_dim_2 + (2, 2), dtype=dtype), is_static),
          scale=maybe_static(
              tf.ones(batch_dim_2 + (2, 2), dtype=dtype), is_static)),
      reinterpreted_batch_ndims=2)

  dist3 = independent.Independent(
      normal.Normal(
          loc=maybe_static(
              tf.zeros(batch_dim_3 + (2, 2), dtype=dtype), is_static),
          scale=maybe_static(
              tf.ones(batch_dim_3 + (2, 2), dtype=dtype), is_static)),
      reinterpreted_batch_ndims=2)

  return ([dist1, dist2, dist3],
          batch_concat.BatchConcat(
              distributions=[dist1, dist2, dist3],
              axis=axis,
              validate_args=validate_args))


@test_util.test_all_tf_execution_regimes
class BatchConcatTest(test_util.TestCase):

  @parameterized.product(
      is_static=[True, False],
      get_distributions=[
          get_scalar_distributions, get_vector_distributions,
          get_matrix_distributions
      ])
  def test_batch_shape_tensor(self, is_static, get_distributions):
    _, concat_dist = get_distributions(is_static=is_static)
    self.assertAllEqual(
        self.evaluate(concat_dist.batch_shape_tensor()),
        [2, 6, 4])

  @parameterized.product(
      is_static=[True, False],
      get_distributions=[
          get_scalar_distributions, get_vector_distributions,
          get_matrix_distributions
      ])
  def test_batch_shape(self, is_static, get_distributions):
    _, concat_dist = get_distributions(is_static=is_static)
    if tf.executing_eagerly() or is_static:
      self.assertAllEqual(concat_dist.batch_shape,
                          [2, 6, 4])
    else:
      self.assertEqual(concat_dist.batch_shape, tf.TensorShape(None))

  @parameterized.product(
      [dict(is_static=True), dict(is_static=False)], [
          dict(get_distributions=get_scalar_distributions, event_shape=[]),
          dict(get_distributions=get_vector_distributions, event_shape=[2]),
          dict(get_distributions=get_matrix_distributions, event_shape=[2, 2])
      ])
  def test_event_shape_tensor(self, is_static, get_distributions, event_shape):
    _, concat_dist = get_distributions(is_static=is_static)
    self.assertAllEqual(
        self.evaluate(concat_dist.event_shape_tensor()), event_shape)

  @parameterized.product(
      [dict(is_static=True), dict(is_static=False)], [
          dict(
              get_distributions=get_scalar_distributions,
              static_event_shape=[],
              dynamic_event_shape=[]),
          dict(
              get_distributions=get_vector_distributions,
              static_event_shape=[2],
              dynamic_event_shape=[None]),
          dict(
              get_distributions=get_matrix_distributions,
              static_event_shape=[2, 2],
              dynamic_event_shape=[None, None])
      ])
  def test_event_shape(self, is_static, get_distributions, static_event_shape,
                       dynamic_event_shape):
    _, concat_dist = get_distributions(is_static=is_static)
    if tf.executing_eagerly() or is_static:
      self.assertAllEqual(concat_dist.event_shape, static_event_shape)
    else:
      self.assertAllEqual(concat_dist.event_shape.as_list(),
                          dynamic_event_shape)

  @parameterized.product(
      [dict(is_static=True), dict(is_static=False)], [
          dict(get_distributions=get_scalar_distributions, event_shape=[]),
          dict(get_distributions=get_vector_distributions, event_shape=[2]),
          dict(get_distributions=get_matrix_distributions, event_shape=[2, 2])
      ])
  def test_sample(self, is_static, get_distributions, event_shape):
    _, concat_dist = get_distributions(is_static=is_static)
    seed = test_util.test_seed()
    samples = concat_dist.sample(seed=seed)
    self.assertAllEqual(
        self.evaluate(tf.shape(samples)), [2, 6, 4] + event_shape)
    samples = concat_dist.sample([12, 20], seed=seed)
    self.assertAllEqual(
        self.evaluate(tf.shape(samples)), [12, 20, 2, 6, 4] + event_shape)

  @parameterized.product(
      [dict(is_static=True), dict(is_static=False)], [
          dict(get_distributions=get_scalar_distributions, event_shape=[]),
          dict(get_distributions=get_vector_distributions, event_shape=[2]),
          dict(get_distributions=get_matrix_distributions, event_shape=[2, 2])
      ])
  def test_sample_and_log_prob(self, is_static, get_distributions, event_shape):
    _, concat_dist = get_distributions(is_static=is_static)
    seed = test_util.test_seed()
    samples, lp = concat_dist.experimental_sample_and_log_prob(seed=seed)
    self.assertAllEqual(
        self.evaluate(tf.shape(samples)), [2, 6, 4] + event_shape)
    self.assertAllClose(lp, concat_dist.log_prob(samples))

  @parameterized.product(
      [dict(is_static=True), dict(is_static=False)], [
          dict(get_distributions=get_scalar_distributions, event_shape=[]),
          dict(get_distributions=get_vector_distributions, event_shape=[2]),
          dict(get_distributions=get_matrix_distributions, event_shape=[2, 2])
      ])
  def test_split_sample(self, is_static, get_distributions, event_shape):
    _, concat_dist = get_distributions(is_static=is_static)
    x_sample = tf.ones([2, 6, 4] + event_shape)
    sample_shape_size, x_split = concat_dist._split_sample(x_sample)
    self.assertEqual(len(x_split), 3)
    if tf.executing_eagerly() or is_static:
      self.assertEqual(sample_shape_size, 0)
    else:
      self.assertEqual(self.evaluate(sample_shape_size), 0)

    self.assertAllEqual(
        self.evaluate(tf.shape(x_split[0])), [2, 3, 4] + event_shape)
    self.assertAllEqual(
        self.evaluate(tf.shape(x_split[1])), [2, 2, 4] + event_shape)
    self.assertAllEqual(
        self.evaluate(tf.shape(x_split[2])), [2, 1, 4] + event_shape)

  @parameterized.product(
      [dict(is_static=True), dict(is_static=False)], [
          dict(get_distributions=get_scalar_distributions, event_shape=[]),
          dict(get_distributions=get_vector_distributions, event_shape=[2]),
          dict(get_distributions=get_matrix_distributions, event_shape=[2, 2])
      ])
  def test_split_sample_with_sample_shape(self, is_static, get_distributions,
                                          event_shape):
    _, concat_dist = get_distributions(is_static=is_static)
    x_sample = concat_dist.sample(7, seed=test_util.test_seed())
    sample_shape_size, x_split = concat_dist._split_sample(x_sample)
    self.assertEqual(len(x_split), 3)
    if tf.executing_eagerly() or is_static:
      self.assertEqual(sample_shape_size, 1)
    else:
      self.assertEqual(self.evaluate(sample_shape_size), 1)

    self.assertAllEqual(
        self.evaluate(tf.shape(x_split[0])), [7, 2, 3, 4] + event_shape)
    self.assertAllEqual(
        self.evaluate(tf.shape(x_split[1])), [7, 2, 2, 4] + event_shape)
    self.assertAllEqual(
        self.evaluate(tf.shape(x_split[2])), [7, 2, 1, 4] + event_shape)

  @parameterized.product(
      is_static=[True, False],
      get_distributions=[
          get_scalar_distributions, get_vector_distributions,
          get_matrix_distributions
      ])
  def test_log_prob(self, is_static, get_distributions):
    dists, concat_dist = get_distributions(is_static=is_static)
    x_sample = concat_dist.sample([32, 128], seed=test_util.test_seed())
    log_prob = concat_dist.log_prob(x_sample)
    self.assertAllEqual(self.evaluate(tf.shape(log_prob)), [32, 128, 2, 6, 4])
    self.assertAllEqual(log_prob[:, :, :, :3],
                        dists[0].log_prob(x_sample[:, :, :, :3, ...]))
    self.assertAllEqual(log_prob[:, :, :, 3:5],
                        dists[1].log_prob(x_sample[:, :, :, 3:5, ...]))
    self.assertAllEqual(log_prob[:, :, :, 5:],
                        dists[2].log_prob(x_sample[:, :, :, 5:, ...]))

  @parameterized.product(
      [dict(is_static=True), dict(is_static=False)], [
          dict(get_distributions=get_scalar_distributions, event_shape=[]),
          dict(get_distributions=get_vector_distributions, event_shape=[2]),
          dict(get_distributions=get_matrix_distributions, event_shape=[2, 2])
      ])
  def test_log_prob_no_batch(self, is_static, get_distributions, event_shape):
    _, concat_dist = get_distributions(is_static=is_static)
    x_sample = tf.ones(event_shape)
    if tf.executing_eagerly() or is_static:
      with self.assertRaises(Exception):
        concat_dist.log_prob(x_sample)
    else:
      with self.assertRaises(Exception):
        log_prob = concat_dist.log_prob(x_sample)
        self.evaluate(log_prob)

  @parameterized.product(
      [dict(is_static=True), dict(is_static=False)], [
          dict(get_distributions=get_scalar_distributions, event_shape=[]),
          dict(get_distributions=get_vector_distributions, event_shape=[2]),
          dict(get_distributions=get_matrix_distributions, event_shape=[2, 2])
      ])
  def test_log_prob_broadcast_batch(self, is_static, get_distributions,
                                    event_shape):
    _, concat_dist = get_distributions(is_static=is_static)
    x_sample = tf.ones([1, 1, 1] + event_shape)
    if tf.executing_eagerly() or is_static:
      with self.assertRaises(Exception):
        concat_dist.log_prob(x_sample)
    else:
      log_prob = concat_dist.log_prob(x_sample)
      with self.assertRaises(Exception):
        self.evaluate(log_prob)

  @parameterized.product(
      [dict(is_static=True), dict(is_static=False)], [
          dict(get_distributions=get_scalar_distributions, event_shape=[]),
          dict(get_distributions=get_vector_distributions, event_shape=[2]),
          dict(get_distributions=get_matrix_distributions, event_shape=[2, 2])
      ])
  def test_mean(self, is_static, get_distributions, event_shape):
    dists, concat_dist = get_distributions(is_static=is_static)
    means = concat_dist.mean()
    self.assertAllEqual(self.evaluate(tf.shape(means)), [2, 6, 4] + event_shape)
    self.assertAllEqual(
        means[:, :3], tf.broadcast_to(dists[0].mean(), [2, 3, 4] + event_shape))
    self.assertAllClose(
        means[:, 3:5], tf.broadcast_to(dists[1].mean(),
                                       [2, 2, 4] + event_shape))
    self.assertAllClose(
        means[:, 5:], tf.broadcast_to(dists[2].mean(), [2, 1, 4] + event_shape))

  @parameterized.product(
      is_static=[True, False],
      get_distributions=[
          get_scalar_distributions, get_vector_distributions,
          get_matrix_distributions
      ])
  def test_incompatible_batch_shape(self, is_static, get_distributions):
    batch_dims = [(1, 3, 5), (1, 3, 4, 10)]
    for batch_dim in batch_dims:
      if tf.executing_eagerly() or is_static:
        with self.assertRaises(Exception):
          _, concat_dist = get_distributions(
              batch_dim_1=batch_dim, is_static=is_static)
      else:
        _, concat_dist = get_distributions(
            batch_dim_1=batch_dim, is_static=is_static)
        with self.assertRaises(Exception):
          self.evaluate(concat_dist.mean())

  @parameterized.product(is_static=[True, False])
  def test_incompatible_event_shape(self, is_static):
    norm = normal.Normal(
        loc=self.maybe_static(tf.zeros([2, 3]), is_static),
        scale=self.maybe_static(tf.ones([2, 3]), is_static))
    mvn = mvn_diag.MultivariateNormalDiag(
        loc=self.maybe_static(tf.zeros([2, 3, 2]), is_static),
        scale_diag=self.maybe_static(tf.ones([2, 3, 2]), is_static))

    if tf.executing_eagerly() or is_static:
      with self.assertRaises(Exception):
        _ = batch_concat.BatchConcat([norm, mvn])
    else:
      concat_dist = batch_concat.BatchConcat([norm, mvn], axis=0)
      with self.assertRaises(Exception):
        self.evaluate(concat_dist.mean())

  @parameterized.product(
      is_static=[True, False],
      get_distributions=[
          get_scalar_distributions, get_vector_distributions,
          get_matrix_distributions
      ])
  def test_batch_concat_of_concat(self, is_static, get_distributions):
    _, concat_dist_1 = get_distributions(is_static=is_static)
    _, concat_dist_2 = get_distributions(is_static=is_static)
    concat_concat = batch_concat.BatchConcat(
        [concat_dist_1, concat_dist_2], axis=0)
    self.assertAllEqual(
        self.evaluate(concat_concat.batch_shape_tensor()),
        [4, 6, 4])
    x_sample = concat_concat.sample(32, seed=test_util.test_seed())
    self.assertAllEqual(
        self.evaluate(tf.shape(concat_concat.log_prob(x_sample))),
        [32, 4, 6, 4])

  @parameterized.product(
      is_static=[True, False],
      get_distributions=[
          get_scalar_distributions, get_vector_distributions,
          get_matrix_distributions
      ])
  def test_axis_negative(self, is_static, get_distributions):
    with self.assertRaises(Exception):
      _ = get_distributions(axis=-1, is_static=is_static)

  @parameterized.product(
      is_static=[True, False],
      get_distributions=[
          get_scalar_distributions, get_vector_distributions,
          get_matrix_distributions
      ])
  def test_axis_out_of_range(self, is_static, get_distributions):
    if tf.executing_eagerly() or is_static:
      with self.assertRaises(Exception):
        _ = get_distributions(axis=10, is_static=is_static)
    else:
      _, concat_dist = get_distributions(axis=10, is_static=is_static)
      with self.assertRaises(Exception):
        self.evaluate(concat_dist.mean())


if __name__ == '__main__':
  test_util.main()
