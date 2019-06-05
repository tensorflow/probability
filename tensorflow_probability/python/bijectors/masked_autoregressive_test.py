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
"""Tests for tfb.MaskedAutoregressiveFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import six
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from tensorflow_probability.python.bijectors.masked_autoregressive import _gen_mask
from tensorflow_probability.python.internal import test_util as tfp_test_util

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

tfk = tf.keras
tfkl = tf.keras.layers


def masked_autoregressive_2d_template(base_template, event_shape):

  def wrapper(x):
    x_flat = tf.reshape(
        x, tf.concat([tf.shape(input=x)[:-len(event_shape)], [-1]], -1))
    x_shift, x_log_scale = base_template(x_flat)
    return tf.reshape(x_shift, tf.shape(input=x)), tf.reshape(
        x_log_scale, tf.shape(input=x))

  return wrapper


def _masked_autoregressive_shift_and_log_scale_fn(hidden_units,
                                                  shift_only=False,
                                                  activation="relu",
                                                  name=None,
                                                  **kwargs):
  params = 1 if shift_only else 2
  layer = tfb.AutoregressiveNetwork(params, hidden_units=hidden_units,
                                    activation=activation, name=name, **kwargs)

  if shift_only:
    return lambda x: (layer(x)[..., 0], None)

  return lambda x: tf.unstack(layer(x), axis=-1)


@test_util.run_all_in_graph_and_eager_modes
class GenMaskTest(tf.test.TestCase):

  def test346Exclusive(self):
    expected_mask = np.array(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 0, 0]])
    mask = _gen_mask(num_blocks=3, n_in=4, n_out=6, mask_type="exclusive")
    self.assertAllEqual(expected_mask, mask)

  def test346Inclusive(self):
    expected_mask = np.array(
        [[1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 0]])
    mask = _gen_mask(num_blocks=3, n_in=4, n_out=6, mask_type="inclusive")
    self.assertAllEqual(expected_mask, mask)


@test_util.run_all_in_graph_and_eager_modes
class MaskedAutoregressiveFlowTest(tfp_test_util.VectorDistributionTestHelpers,
                                   tf.test.TestCase):

  event_shape = [4]

  @property
  def _autoregressive_flow_kwargs(self):
    return {
        "shift_and_log_scale_fn":
            tfb.masked_autoregressive_default_template(
                hidden_layers=[2], shift_only=False),
        "is_constant_jacobian":
            False,
    }

  def testNonBatchedBijector(self):
    x_ = np.arange(np.prod(self.event_shape)).astype(
        np.float32).reshape(self.event_shape)
    ma = tfb.MaskedAutoregressiveFlow(
        validate_args=True, **self._autoregressive_flow_kwargs)
    x = tf.constant(x_)
    forward_x = ma.forward(x)
    # Use identity to invalidate cache.
    inverse_y = ma.inverse(tf.identity(forward_x))
    forward_inverse_y = ma.forward(inverse_y)
    fldj = ma.forward_log_det_jacobian(x, event_ndims=len(self.event_shape))
    # Use identity to invalidate cache.
    ildj = ma.inverse_log_det_jacobian(
        tf.identity(forward_x), event_ndims=len(self.event_shape))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    [
        forward_x_,
        inverse_y_,
        forward_inverse_y_,
        ildj_,
        fldj_,
    ] = self.evaluate([
        forward_x,
        inverse_y,
        forward_inverse_y,
        ildj,
        fldj,
    ])
    self.assertStartsWith(ma.name, "masked_autoregressive_flow")
    self.assertAllClose(forward_x_, forward_inverse_y_, rtol=1e-6, atol=0.)
    self.assertAllClose(x_, inverse_y_, rtol=1e-5, atol=0.)
    self.assertAllClose(ildj_, -fldj_, rtol=1e-6, atol=0.)

  def testBatchedBijector(self):
    x_ = np.arange(4 * np.prod(self.event_shape)).astype(
        np.float32).reshape([4] + self.event_shape) / 10.
    ma = tfb.MaskedAutoregressiveFlow(
        validate_args=True, **self._autoregressive_flow_kwargs)
    x = tf.constant(x_)
    forward_x = ma.forward(x)
    # Use identity to invalidate cache.
    inverse_y = ma.inverse(tf.identity(forward_x))
    forward_inverse_y = ma.forward(inverse_y)
    fldj = ma.forward_log_det_jacobian(x, event_ndims=len(self.event_shape))
    # Use identity to invalidate cache.
    ildj = ma.inverse_log_det_jacobian(
        tf.identity(forward_x), event_ndims=len(self.event_shape))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    [
        forward_x_,
        inverse_y_,
        forward_inverse_y_,
        ildj_,
        fldj_,
    ] = self.evaluate([
        forward_x,
        inverse_y,
        forward_inverse_y,
        ildj,
        fldj,
    ])
    self.assertStartsWith(ma.name, "masked_autoregressive_flow")
    self.assertAllClose(forward_x_, forward_inverse_y_, rtol=1e-6, atol=1e-6)
    self.assertAllClose(x_, inverse_y_, rtol=1e-4, atol=1e-4)
    self.assertAllClose(ildj_, -fldj_, rtol=1e-6, atol=1e-6)

  def testMutuallyConsistent(self):
    maf = tfb.MaskedAutoregressiveFlow(
        validate_args=True, **self._autoregressive_flow_kwargs)
    base = tfd.Independent(
        tfd.Normal(loc=tf.zeros(self.event_shape), scale=1.),
        reinterpreted_batch_ndims=len(self.event_shape))
    reshape = tfb.Reshape(
        event_shape_out=[np.prod(self.event_shape)],
        event_shape_in=self.event_shape)
    bijector = tfb.Chain([reshape, maf])
    dist = tfd.TransformedDistribution(
        distribution=base, bijector=bijector, validate_args=True)
    self.run_test_sample_consistent_log_prob(
        sess_run_fn=self.evaluate,
        dist=dist,
        num_samples=int(1e6),
        radius=1.,
        center=0.,
        rtol=0.025)

  def testInvertMutuallyConsistent(self):
    maf = tfb.Invert(
        tfb.MaskedAutoregressiveFlow(
            validate_args=True, **self._autoregressive_flow_kwargs))
    base = tfd.Independent(
        tfd.Normal(loc=tf.zeros(self.event_shape), scale=1.),
        reinterpreted_batch_ndims=len(self.event_shape))
    reshape = tfb.Reshape(
        event_shape_out=[np.prod(self.event_shape)],
        event_shape_in=self.event_shape)
    bijector = tfb.Chain([reshape, maf])
    dist = tfd.TransformedDistribution(
        distribution=base, bijector=bijector, validate_args=True)

    self.run_test_sample_consistent_log_prob(
        sess_run_fn=self.evaluate,
        dist=dist,
        num_samples=int(1e6),
        radius=1.,
        center=0.,
        rtol=0.025)


@test_util.run_all_in_graph_and_eager_modes
class MaskedAutoregressiveFlowShiftOnlyTest(MaskedAutoregressiveFlowTest):

  @property
  def _autoregressive_flow_kwargs(self):
    return {
        "shift_and_log_scale_fn":
            tfb.masked_autoregressive_default_template(
                hidden_layers=[2], shift_only=True),
        "is_constant_jacobian":
            True,
    }


@test_util.run_all_in_graph_and_eager_modes
class MaskedAutoregressiveFlowShiftOnlyLayerTest(MaskedAutoregressiveFlowTest):

  @property
  def _autoregressive_flow_kwargs(self):
    return {
        "shift_and_log_scale_fn":
            _masked_autoregressive_shift_and_log_scale_fn(
                hidden_units=[2], shift_only=True),
        "is_constant_jacobian":
            True,
    }


@test_util.run_all_in_graph_and_eager_modes
class MaskedAutoregressiveFlowUnrollLoopTest(MaskedAutoregressiveFlowTest):

  @property
  def _autoregressive_flow_kwargs(self):
    return {
        "shift_and_log_scale_fn":
            tfb.masked_autoregressive_default_template(
                hidden_layers=[2], shift_only=False),
        "is_constant_jacobian":
            False,
        "unroll_loop":
            True,
    }


@test_util.run_all_in_graph_and_eager_modes
class MaskedAutoregressiveFlowUnrollLoopLayerTest(MaskedAutoregressiveFlowTest):

  @property
  def _autoregressive_flow_kwargs(self):
    return {
        "shift_and_log_scale_fn":
            _masked_autoregressive_shift_and_log_scale_fn(
                hidden_units=[10, 10], activation="relu"),
        "is_constant_jacobian":
            False,
        "unroll_loop":
            True,
    }


@test_util.run_all_in_graph_and_eager_modes
class MaskedAutoregressive2DTest(MaskedAutoregressiveFlowTest):
  event_shape = [3, 2]

  @property
  def _autoregressive_flow_kwargs(self):
    return {
        "shift_and_log_scale_fn":
            masked_autoregressive_2d_template(
                tfb.masked_autoregressive_default_template(
                    hidden_layers=[np.prod(self.event_shape)],
                    shift_only=False), self.event_shape),
        "is_constant_jacobian":
            False,
        "event_ndims":
            2,
    }


@test_util.run_all_in_graph_and_eager_modes
class MaskedAutoregressive2DLayerTest(MaskedAutoregressiveFlowTest):
  event_shape = [3, 2]

  @property
  def _autoregressive_flow_kwargs(self):
    return {
        "shift_and_log_scale_fn":
            masked_autoregressive_2d_template(
                _masked_autoregressive_shift_and_log_scale_fn(
                    hidden_units=[np.prod(self.event_shape)],
                    shift_only=False), self.event_shape),
        "is_constant_jacobian":
            False,
        "event_ndims":
            2,
    }


@test_util.run_all_in_graph_and_eager_modes
class AutoregressiveNetworkTest(tf.test.TestCase):

  def _count_trainable_params(self, layer):
    ret = 0
    for w in layer.trainable_weights:
      ret += np.prod(w.shape)
    return ret

  def assertIsAutoregressive(self, f, event_size, order):
    input_order = None
    if isinstance(order, six.string_types):
      if order == "left-to-right":
        input_order = range(event_size)
      elif order == "right-to-left":
        input_order = range(event_size - 1, -1, -1)
    elif np.all(np.sort(order) == np.arange(1, event_size + 1)):
      input_order = list(np.array(order) - 1)
    if input_order is None:
      raise ValueError("Invalid input order: '{}'.".format(order))

    # Test that if we change dimension `i` of the input, then the only changed
    # dimensions `j` of the output are those with larger `input_order[j]`.
    # (We could also do this by examining gradients.)
    diff = []
    mask = []
    for i in range(event_size):
      x = np.random.randn(event_size)
      delta = np.zeros(event_size)
      delta[i] = np.random.randn()
      diff = self.evaluate(f(x + delta) - f(x))
      mask = [[input_order[i] >= input_order[j]] for j in range(event_size)]
      self.assertAllClose(np.zeros_like(diff), mask * diff, atol=0., rtol=1e-6)

  def test_layer_right_to_left_float64(self):
    made = tfb.AutoregressiveNetwork(
        params=3, event_shape=4, activation=None, input_order="right-to-left",
        dtype=tf.float64, hidden_degrees="random", hidden_units=[10, 7, 10])
    self.assertEqual((4, 3), made(np.zeros(4, dtype=np.float64)).shape)
    self.assertEqual(5 * 10 + 11 * 7 + 8 * 10 + 11 * 12,
                     self._count_trainable_params(made))
    if not tf.executing_eagerly():
      self.evaluate(
          tf.compat.v1.initializers.variables(made.trainable_variables))
    self.assertIsAutoregressive(made, event_size=4, order="right-to-left")

  def test_layer_callable_activation(self):
    made = tfb.AutoregressiveNetwork(
        params=2, activation=tf.math.exp, input_order="random",
        kernel_regularizer=tfk.regularizers.l2(0.1), bias_initializer="ones",
        hidden_units=[9], hidden_degrees="equal")
    self.assertEqual((3, 5, 2), made(np.zeros((3, 5))).shape)
    self.assertEqual(6 * 9 + 10 * 10, self._count_trainable_params(made))
    if not tf.executing_eagerly():
      self.evaluate(
          tf.compat.v1.initializers.variables(made.trainable_variables))
    self.assertIsAutoregressive(made, event_size=5, order=made._input_order)

  def test_layer_smaller_hidden_layers_than_input(self):
    made = tfb.AutoregressiveNetwork(
        params=1, event_shape=9, activation="relu", use_bias=False,
        bias_regularizer=tfk.regularizers.l1(0.5), bias_constraint=tf.math.abs,
        input_order="right-to-left", hidden_units=[5, 5])
    self.assertEqual((9, 1), made(np.zeros(9)).shape)
    self.assertEqual(9 * 5 + 5 * 5 + 5 * 9, self._count_trainable_params(made))
    if not tf.executing_eagerly():
      self.evaluate(
          tf.compat.v1.initializers.variables(made.trainable_variables))
    self.assertIsAutoregressive(made, event_size=9, order="right-to-left")

  def test_layer_no_hidden_units(self):
    made = tfb.AutoregressiveNetwork(
        params=4, event_shape=3, use_bias=False, hidden_degrees="random",
        kernel_constraint="unit_norm")
    self.assertEqual((2, 2, 5, 3, 4), made(np.zeros((2, 2, 5, 3))).shape)
    self.assertEqual(3 * 12, self._count_trainable_params(made))
    if not tf.executing_eagerly():
      self.evaluate(
          tf.compat.v1.initializers.variables(made.trainable_variables))
    self.assertIsAutoregressive(made, event_size=3, order="left-to-right")

  def test_layer_v2_kernel_initializer(self):
    init = tf.compat.v2.keras.initializers.GlorotNormal()
    made = tfb.AutoregressiveNetwork(
        params=2, event_shape=4, activation="relu",
        hidden_units=[5, 5], kernel_initializer=init)
    self.assertEqual((4, 2), made(np.zeros(4)).shape)
    self.assertEqual(5 * 5 + 6 * 5 + 6 * 8, self._count_trainable_params(made))
    if not tf.executing_eagerly():
      self.evaluate(
          tf.compat.v1.initializers.variables(made.trainable_variables))
    self.assertIsAutoregressive(made, event_size=4, order="left-to-right")

  def test_doc_string(self):
    # Generate data.
    n = 2000
    x2 = np.random.randn(n).astype(dtype=np.float32) * 2.
    x1 = np.random.randn(n).astype(dtype=np.float32) + (x2 * x2 / 4.)
    data = np.stack([x1, x2], axis=-1)

    # Density estimation with MADE.
    made = tfb.AutoregressiveNetwork(params=2, hidden_units=[10, 10])

    distribution = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=tfb.MaskedAutoregressiveFlow(
            lambda x: tf.unstack(made(x), num=2, axis=-1)),
        event_shape=[2])

    # Construct and fit model.
    x_ = tfkl.Input(shape=(2,), dtype=tf.float32)
    log_prob_ = distribution.log_prob(x_)
    model = tfk.Model(x_, log_prob_)

    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                  loss=lambda _, log_prob: -log_prob)

    batch_size = 25
    model.fit(x=data,
              y=np.zeros((n, 0), dtype=np.float32),
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=1,  # Usually `n // batch_size`.
              shuffle=True,
              verbose=True)

    # Use the fitted distribution.
    self.assertAllEqual((3, 1, 2), distribution.sample((3, 1)).shape)
    self.assertAllEqual(
        (3,), distribution.log_prob(np.ones((3, 2), dtype=np.float32)).shape)

  def test_doc_string_images_case_1(self):
    # Generate fake images.
    images = np.random.choice([0, 1], size=(100, 8, 8, 3))
    n, width, height, channels = images.shape

    # Reshape images to achieve desired autoregressivity.
    event_shape = [width * height * channels]
    reshaped_images = np.reshape(images, [n, width * height * channels])

    made = tfb.AutoregressiveNetwork(params=1, event_shape=event_shape,
                                     hidden_units=[20, 20], activation="relu")

    # Density estimation with MADE.
    #
    # NOTE: Parameterize an autoregressive distribution over an event_shape of
    # [width * height * channels], with univariate Bernoulli conditional
    # distributions.
    distribution = tfd.Autoregressive(
        lambda x: tfd.Independent(  # pylint: disable=g-long-lambda
            tfd.Bernoulli(logits=tf.unstack(made(x), axis=-1)[0],
                          dtype=tf.float32),
            reinterpreted_batch_ndims=1),
        sample0=tf.zeros(event_shape, dtype=tf.float32))

    # Construct and fit model.
    x_ = tfkl.Input(shape=event_shape, dtype=tf.float32)
    log_prob_ = distribution.log_prob(x_)
    model = tfk.Model(x_, log_prob_)

    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                  loss=lambda _, log_prob: -log_prob)

    batch_size = 10
    model.fit(x=reshaped_images,
              y=np.zeros((n, 0), dtype=np.float32),
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=1,  # Usually `n // batch_size`.
              shuffle=True,
              verbose=True)

    # Use the fitted distribution.
    self.assertAllEqual(event_shape, distribution.sample().shape)
    self.assertAllEqual((n,), distribution.log_prob(reshaped_images).shape)

  def test_doc_string_images_case_2(self):
    # Generate fake images.
    images = np.random.choice([0, 1], size=(100, 8, 8, 3))
    n, width, height, channels = images.shape

    # Reshape images to achieve desired autoregressivity.
    reshaped_images = np.transpose(
        np.reshape(images, [n, width * height, channels]),
        axes=[0, 2, 1])

    made = tfb.AutoregressiveNetwork(params=1, event_shape=[width * height],
                                     hidden_units=[20, 20], activation="relu")

    # Density estimation with MADE.
    #
    # NOTE: Parameterize an autoregressive distribution over an event_shape of
    # [channels, width * height], with univariate Bernoulli conditional
    # distributions.
    distribution = tfd.Autoregressive(
        lambda x: tfd.Independent(  # pylint: disable=g-long-lambda
            tfd.Bernoulli(logits=tf.unstack(made(x), axis=-1)[0],
                          dtype=tf.float32),
            reinterpreted_batch_ndims=2),
        sample0=tf.zeros([channels, width * height], dtype=tf.float32))

    # Construct and fit model.
    x_ = tfkl.Input(shape=(channels, width * height), dtype=tf.float32)
    log_prob_ = distribution.log_prob(x_)
    model = tfk.Model(x_, log_prob_)

    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                  loss=lambda _, log_prob: -log_prob)

    batch_size = 10
    model.fit(x=reshaped_images,
              y=np.zeros((n, 0), dtype=np.float32),
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=1,  # Usually `n // batch_size`.
              shuffle=True,
              verbose=True)

    # Use the fitted distribution.
    self.assertAllEqual((7, channels, width * height),
                        distribution.sample(7).shape)
    self.assertAllEqual((n,), distribution.log_prob(reshaped_images).shape)


if __name__ == "__main__":
  tf.test.main()
