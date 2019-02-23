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
"""Tests for BatchNorm Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow_probability.python import distributions
from tensorflow_probability.python.internal import test_util
from tensorflow.python.framework import test_util as tf_test_util  # pylint: disable=g-direct-tensorflow-import


@tf_test_util.run_all_in_graph_and_eager_modes
class BatchNormTest(test_util.VectorDistributionTestHelpers,
                    parameterized.TestCase,
                    tf.test.TestCase):

  def _reduction_axes(self, input_shape, event_dims):
    if isinstance(event_dims, int):
      event_dims = [event_dims]
    ndims = len(input_shape)
    # Convert event_dims to non-negative indexing.
    event_dims = list(event_dims)
    for idx, x in enumerate(event_dims):
      if x < 0:
        event_dims[idx] = ndims + x
    return tuple(i for i in range(ndims) if i not in event_dims)

  @parameterized.parameters(
      ((5*2, 4), [-1], False),
      ((5, 2, 4), [-1], False),
      ((5, 2, 4), [1, 2], False),
      ((5, 2, 4), [0, 1], False),
      ((5*2, 4), [-1], True),
      ((5, 2, 4), [-1], True),
      ((5, 2, 4), [1, 2], True),
      ((5, 2, 4), [0, 1], True))
  def testForwardInverse(self, input_shape, event_dims, training):
    """Tests forward and backward passes with different event shapes and axes.

    Args:
      input_shape: Tuple of shapes for input tensor.
      event_dims: Tuple of dimension indices that will be normalized.
      training: Boolean of whether bijector runs in training or inference mode.
    """
    x_ = np.arange(5 * 4 * 2).astype(np.float32).reshape(input_shape)
    x = tf.compat.v1.placeholder_with_default(
        x_, input_shape if 0 in event_dims else (None,) + input_shape[1:])
    # When training, memorize the exact mean of the last
    # minibatch that it normalized (instead of moving average assignment).
    layer = tf.compat.v1.layers.BatchNormalization(
        axis=event_dims, momentum=0., epsilon=0.)
    batch_norm = tfb.BatchNormalization(
        batchnorm_layer=layer, training=training)
    # Minibatch statistics are saved only after norm_x has been computed.
    norm_x = batch_norm.inverse(x)
    with tf.control_dependencies(batch_norm.batchnorm.updates):
      moving_mean = tf.identity(batch_norm.batchnorm.moving_mean)
      moving_var = tf.identity(batch_norm.batchnorm.moving_variance)
      denorm_x = batch_norm.forward(tf.identity(norm_x))
      fldj = batch_norm.forward_log_det_jacobian(
          x, event_ndims=len(event_dims))
      # Use identity to invalidate cache.
      ildj = batch_norm.inverse_log_det_jacobian(
          tf.identity(denorm_x), event_ndims=len(event_dims))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    # Update variables.
    norm_x_ = self.evaluate(norm_x)
    [
        norm_x_,
        moving_mean_,
        moving_var_,
        denorm_x_,
        ildj_,
        fldj_,
    ] = self.evaluate([
        norm_x,
        moving_mean,
        moving_var,
        denorm_x,
        ildj,
        fldj,
    ])
    self.assertEqual("batch_normalization", batch_norm.name)

    reduction_axes = self._reduction_axes(input_shape, event_dims)
    keepdims = len(event_dims) > 1

    expected_batch_mean = np.mean(
        x_, axis=reduction_axes, keepdims=keepdims)
    expected_batch_var = np.var(x_, axis=reduction_axes, keepdims=keepdims)

    if training:
      # When training=True, values become normalized across batch dim and
      # original values are recovered after de-normalizing.
      zeros = np.zeros_like(norm_x_)
      self.assertAllClose(np.mean(zeros, axis=reduction_axes),
                          np.mean(norm_x_, axis=reduction_axes))

      self.assertAllClose(expected_batch_mean, moving_mean_)
      self.assertAllClose(expected_batch_var, moving_var_)
      self.assertAllClose(x_, denorm_x_, atol=1e-5)
      # Since moving statistics are set to batch statistics after
      # normalization, ildj and -fldj should match.
      self.assertAllClose(ildj_, -fldj_)
      # ildj is computed with minibatch statistics.
      expected_ildj = np.sum(np.log(1.) - .5 * np.log(
          expected_batch_var + batch_norm.batchnorm.epsilon))
      self.assertAllClose(expected_ildj, np.squeeze(ildj_))
    else:
      # When training=False, moving_mean, moving_var remain at their
      # initialized values (0., 1.), resulting in no scale/shift (a small
      # shift occurs if epsilon > 0.)
      self.assertAllClose(x_, norm_x_)
      self.assertAllClose(x_, denorm_x_, atol=1e-5)
      # ildj is computed with saved statistics.
      expected_ildj = np.sum(
          np.log(1.) - .5 * np.log(1. + batch_norm.batchnorm.epsilon))
      self.assertAllClose(expected_ildj, np.squeeze(ildj_))

  @parameterized.named_parameters(
      ("2d_event_ndims", (10, 4), [-1], False),
      ("1d_event_ndims", 2, [-1], False))
  def testLogProb(self, event_shape, event_dims, training):
    training = tf.compat.v1.placeholder_with_default(training, (), "training")
    layer = tf.compat.v1.layers.BatchNormalization(axis=event_dims, epsilon=0.)
    batch_norm = tfb.BatchNormalization(batchnorm_layer=layer,
                                        training=training)
    base_dist = distributions.MultivariateNormalDiag(
        loc=np.zeros(np.prod(event_shape), dtype=np.float32))
    # Reshape the events.
    if isinstance(event_shape, int):
      event_shape = [event_shape]
    base_dist = distributions.TransformedDistribution(
        distribution=base_dist,
        bijector=tfb.Reshape(event_shape_out=event_shape))
    dist = distributions.TransformedDistribution(
        distribution=base_dist,
        bijector=batch_norm,
        validate_args=True)
    samples = dist.sample(int(1e5))
    # No volume distortion since training=False, bijector is initialized
    # to the identity transformation.
    base_log_prob = base_dist.log_prob(samples)
    dist_log_prob = dist.log_prob(samples)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    base_log_prob_, dist_log_prob_ = self.evaluate(
        [base_log_prob, dist_log_prob])
    self.assertAllClose(base_log_prob_, dist_log_prob_)

  def testMutuallyConsistent(self):
    # BatchNorm bijector is only mutually consistent when training=False.
    dims = 4
    training = tf.compat.v1.placeholder_with_default(False, (), "training")
    layer = tf.compat.v1.layers.BatchNormalization(epsilon=0.)
    batch_norm = tfb.BatchNormalization(batchnorm_layer=layer,
                                        training=training)
    dist = distributions.TransformedDistribution(
        distribution=distributions.Normal(loc=0., scale=1.),
        bijector=batch_norm,
        event_shape=[dims],
        validate_args=True)
    self.run_test_sample_consistent_log_prob(
        sess_run_fn=self.evaluate,
        dist=dist,
        num_samples=int(1e5),
        radius=2.,
        center=0.,
        rtol=0.02)

  def testInvertMutuallyConsistent(self):
    # BatchNorm bijector is only mutually consistent when training=False.
    dims = 4
    training = tf.compat.v1.placeholder_with_default(False, (), "training")
    layer = tf.compat.v1.layers.BatchNormalization(epsilon=0.)
    batch_norm = tfb.Invert(
        tfb.BatchNormalization(batchnorm_layer=layer, training=training))
    dist = distributions.TransformedDistribution(
        distribution=distributions.Normal(loc=0., scale=1.),
        bijector=batch_norm,
        event_shape=[dims],
        validate_args=True)
    self.run_test_sample_consistent_log_prob(
        sess_run_fn=self.evaluate,
        dist=dist,
        num_samples=int(1e5),
        radius=2.,
        center=0.,
        rtol=0.02)


if __name__ == "__main__":
  tf.test.main()
