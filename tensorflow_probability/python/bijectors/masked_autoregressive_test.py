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
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from tensorflow_probability.python.bijectors.masked_autoregressive import _gen_mask
from tensorflow_probability.python.internal import test_util


def masked_autoregressive_2d_template(base_template, event_shape):

  def wrapper(x):
    x_flat = tf.reshape(x, tf.concat([tf.shape(x)[:-len(event_shape)], [-1]],
                                     -1))
    x_shift, x_log_scale = base_template(x_flat)
    return tf.reshape(x_shift, tf.shape(x)), tf.reshape(x_log_scale,
                                                        tf.shape(x))

  return wrapper


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


class MaskedAutoregressiveFlowTest(test_util.VectorDistributionTestHelpers,
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
    self.evaluate(tf.global_variables_initializer())
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
    self.assertEqual("masked_autoregressive_flow", ma.name)
    self.assertAllClose(forward_x_, forward_inverse_y_, rtol=1e-6, atol=0.)
    self.assertAllClose(x_, inverse_y_, rtol=1e-5, atol=0.)
    self.assertAllClose(ildj_, -fldj_, rtol=1e-6, atol=0.)

  def testBatchedBijector(self):
    x_ = np.arange(4 * np.prod(self.event_shape)).astype(
        np.float32).reshape([4] + self.event_shape)
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
    self.evaluate(tf.global_variables_initializer())
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
    self.assertEqual("masked_autoregressive_flow", ma.name)
    self.assertAllClose(forward_x_, forward_inverse_y_, rtol=1e-6, atol=0.)
    self.assertAllClose(x_, inverse_y_, rtol=1e-5, atol=0.)
    self.assertAllClose(ildj_, -fldj_, rtol=1e-6, atol=0.)

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
        rtol=0.02)

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
        rtol=0.02)


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


if __name__ == "__main__":
  tf.test.main()
