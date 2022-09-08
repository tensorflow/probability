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
"""Tests for RealNVP."""

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import inline
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import real_nvp
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import shift as shift_lib
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class RealNVPTestBase(
    test_util.VectorDistributionTestHelpers,
    test_util.TestCase):

  @parameterized.named_parameters(
      ('BatchedNumMasked', 4, None, (3,)),
      ('BatchedFractionmasked', None, 4. / 8., (3,)),
      ('NonBatchedNumMasked', 4, None, ()),
      ('NonBatchedFractionmasked', None, 4. / 8., ()),
  )
  def testRegularMask(self, num_masked, fraction_masked, batch_shape):
    x_ = np.random.normal(0., 1., batch_shape + (8,)).astype(np.float32)
    nvp = real_nvp.RealNVP(
        num_masked=num_masked,
        fraction_masked=fraction_masked,
        validate_args=True,
        **self._real_nvp_kwargs,
    )
    x = tf.constant(x_)
    forward_x = nvp.forward(x)
    # Use identity to invalidate cache.
    inverse_y = nvp.inverse(tf.identity(forward_x))
    forward_inverse_y = nvp.forward(inverse_y)
    fldj = nvp.forward_log_det_jacobian(x, event_ndims=1)
    # Use identity to invalidate cache.
    ildj = nvp.inverse_log_det_jacobian(tf.identity(forward_x), event_ndims=1)
    self.evaluate(tf1.global_variables_initializer())
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
    self.assertStartsWith(nvp.name, 'real_nvp')
    self.assertAllClose(forward_x_, forward_inverse_y_, rtol=1e-4, atol=0.)
    self.assertAllClose(x_, inverse_y_, rtol=1e-4, atol=0.)
    self.assertAllClose(ildj_, -fldj_, rtol=1e-6, atol=0.)

  @parameterized.named_parameters(
      ('BatchedNumMasked', -5, None, (3,)),
      ('BatchedFractionmasked', None, -5. / 8., (3,)),
      ('NonBatchedNumMasked', -5, None, ()),
      ('NonBatchedFractionmasked', None, -5. / 8., ()),
  )
  def testReverseMask(self, num_masked, fraction_masked, batch_shape):
    input_depth = 8
    x_ = np.random.normal(0., 1.,
                          batch_shape + (input_depth,)).astype(np.float32)
    flip_nvp = real_nvp.RealNVP(
        num_masked=num_masked,
        fraction_masked=fraction_masked,
        validate_args=True,
        **self._real_nvp_kwargs,
    )
    x = tf.constant(x_)

    forward_x = flip_nvp.forward(x)

    expected_num_masked = (
        num_masked if num_masked is not None else np.floor(input_depth *
                                                           fraction_masked))

    self.assertEqual(flip_nvp._masked_size, expected_num_masked)

    _, x2_ = np.split(x_, [input_depth - abs(flip_nvp._masked_size)], axis=-1)  # pylint: disable=unbalanced-tuple-unpacking

    # Check latter half is the same after passing thru reversed mask RealNVP.
    _, forward_x2 = tf.split(
        forward_x, [
            input_depth - abs(flip_nvp._masked_size),
            abs(flip_nvp._masked_size)
        ],
        axis=-1)
    self.evaluate(tf1.global_variables_initializer())
    forward_x2_ = self.evaluate(forward_x2)

    self.assertAllClose(forward_x2_, x2_, rtol=1e-4, atol=0.)

  def testMutuallyConsistent(self):
    dims = 4
    nvp = real_nvp.RealNVP(
        num_masked=3, validate_args=True, **self._real_nvp_kwargs)
    dist = transformed_distribution.TransformedDistribution(
        distribution=sample.Sample(normal.Normal(0., 1.), [dims]),
        bijector=nvp,
        validate_args=True)
    self.run_test_sample_consistent_log_prob(
        sess_run_fn=self.evaluate,
        dist=dist,
        num_samples=int(1e6),
        seed=54819,
        radius=1.,
        center=0.,
        rtol=0.2  # TODO(b/161840573): decrease once test is deterministic.
        )

  def testInvertMutuallyConsistent(self):
    dims = 4
    nvp = invert.Invert(
        real_nvp.RealNVP(
            num_masked=3, validate_args=True, **self._real_nvp_kwargs))
    dist = transformed_distribution.TransformedDistribution(
        distribution=sample.Sample(normal.Normal(0., 1.), [dims]),
        bijector=nvp,
        validate_args=True)
    self.run_test_sample_consistent_log_prob(
        sess_run_fn=self.evaluate,
        dist=dist,
        num_samples=int(1e6),
        seed=22197,
        radius=1.,
        center=0.,
        rtol=0.1)


@test_util.test_all_tf_execution_regimes
class TrivialTransformTest(RealNVPTestBase):

  @property
  def _real_nvp_kwargs(self):
    return {
        'shift_and_log_scale_fn':
            lambda x, output_dims: (x[..., :output_dims], x[..., :output_dims]),
        'is_constant_jacobian':
            False,
    }


@test_util.numpy_disable_test_missing_functionality('tf.make_template')
@test_util.jax_disable_test_missing_functionality('tf.make_template')
@test_util.test_all_tf_execution_regimes
class RealNVPTest(RealNVPTestBase):

  @property
  def _real_nvp_kwargs(self):
    return {
        'shift_and_log_scale_fn':
            real_nvp.real_nvp_default_template(
                hidden_layers=[3], shift_only=False),
        'is_constant_jacobian':
            False,
    }


@test_util.numpy_disable_test_missing_functionality('tf.make_template')
@test_util.jax_disable_test_missing_functionality('tf.make_template')
@test_util.test_all_tf_execution_regimes
class NICETest(RealNVPTestBase):

  @property
  def _real_nvp_kwargs(self):
    return {
        'shift_and_log_scale_fn':
            real_nvp.real_nvp_default_template(
                hidden_layers=[2], shift_only=True),
        'is_constant_jacobian':
            True,
    }


@test_util.test_all_tf_execution_regimes
class ConstantShiftScaleTest(RealNVPTestBase):

  @property
  def _real_nvp_kwargs(self):

    def constant_shift_log_scale_fn(x0, output_units):
      del x0, output_units
      shift = tf.constant([0.1])
      log_scale = tf.constant([0.5])
      return shift, log_scale

    return {
        'shift_and_log_scale_fn': constant_shift_log_scale_fn,
        'is_constant_jacobian': True,
    }


def _make_gated_bijector_fn():
  def _bijector_fn(x, output_units):
    if tensorshape_util.rank(x.shape) == 1:
      x = x[tf.newaxis, ...]
      reshape_output = lambda x: x[0]
    else:
      reshape_output = lambda x: x

    out = tf1.layers.dense(inputs=x, units=2 * output_units)
    shift, logit_gate = tf.split(out, 2, axis=-1)
    shift = reshape_output(shift)
    logit_gate = reshape_output(logit_gate)
    gate = tf.nn.sigmoid(logit_gate)
    return shift_lib.Shift(shift=(1. - gate) * shift)(scale.Scale(scale=gate))
  return tf1.make_template('gated_bijector', _bijector_fn)


@test_util.numpy_disable_test_missing_functionality('tf.make_template')
@test_util.jax_disable_test_missing_functionality('tf.make_template')
@test_util.test_all_tf_execution_regimes
class GatedTest(RealNVPTestBase):

  @property
  def _real_nvp_kwargs(self):
    return {
        'bijector_fn': _make_gated_bijector_fn(),
    }


class RealNVPTestCommon(test_util.TestCase):

  def testMatrixBijectorRaises(self):
    with self.assertRaisesRegexp(
        ValueError,
        'Bijectors with `forward_min_event_ndims` > 1 are not supported'):

      def bijector_fn(*args, **kwargs):
        del args, kwargs
        return inline.Inline(forward_min_event_ndims=2)

      rnvp = real_nvp.RealNVP(1, bijector_fn=bijector_fn, validate_args=True)
      rnvp.forward([1., 2.])

  def testRankChangingBijectorRaises(self):
    with self.assertRaisesRegexp(
        ValueError, 'Bijectors which alter `event_ndims` are not supported.'):

      def bijector_fn(*args, **kwargs):
        del args, kwargs
        return inline.Inline(
            forward_min_event_ndims=1, inverse_min_event_ndims=0)

      rnvp = real_nvp.RealNVP(1, bijector_fn=bijector_fn, validate_args=True)
      rnvp.forward([1., 2.])

  def testNonIntegerNumMaskedRaises(self):
    with self.assertRaisesRegexp(TypeError, '`num_masked` must be an integer'):
      real_nvp.RealNVP(
          num_masked=0.5, shift_and_log_scale_fn=lambda x, _: (x, x))

  def testNonFloatFractionMaskedRaises(self):
    with self.assertRaisesRegexp(TypeError,
                                 '`fraction_masked` must be a float'):
      real_nvp.RealNVP(
          fraction_masked=1, shift_and_log_scale_fn=lambda x, _: (x, x))

  @parameterized.named_parameters(
      ('TooLarge', 1.1),
      ('TooNegative', -1.1),
      ('LowerBoundary', -1.),
      ('UpperBoundary', 1.),
  )
  def testBadFractionRaises(self, fraction_masked):
    with self.assertRaisesRegexp(ValueError, '`fraction_masked` must be in'):
      real_nvp.RealNVP(
          fraction_masked=fraction_masked,
          shift_and_log_scale_fn=lambda x, _: (x, x))

  @parameterized.named_parameters(
      ('TooLarge', 2),
      ('TooNegative', -2),
      ('LowerBoundary', -1),
      ('UpperBoundary', 1),
  )
  def testBadNumMaskRaises(self, num_masked):
    with self.assertRaisesRegexp(
        ValueError,
        'Number of masked units {} must be smaller than the event size 1'
        .format(num_masked)):
      rnvp = real_nvp.RealNVP(
          num_masked=num_masked, shift_and_log_scale_fn=lambda x, _: (x, x))
      rnvp.forward(np.zeros(1))

  def testBijectorConditionKwargs(self):
    batch_size = 3
    x_ = np.linspace(-1.0, 1.0, (batch_size * 4 * 2)).astype(
        np.float32).reshape((batch_size, 4 * 2))

    conditions = {
        'a': np.random.normal(size=(batch_size, 4)).astype(np.float32),
        'b': np.random.normal(size=(batch_size, 4)).astype(np.float32),
    }

    def _condition_shift_and_log_scale_fn(x0, output_units, a, b):
      del output_units
      return x0 + a, x0 + b

    nvp = real_nvp.RealNVP(
        num_masked=4,
        validate_args=True,
        is_constant_jacobian=False,
        shift_and_log_scale_fn=_condition_shift_and_log_scale_fn)

    x = tf.constant(x_)

    forward_x = nvp.forward(x, **conditions)
    # Use identity to invalidate cache.
    inverse_y = nvp.inverse(tf.identity(forward_x), **conditions)
    forward_inverse_y = nvp.forward(inverse_y, **conditions)
    fldj = nvp.forward_log_det_jacobian(x, event_ndims=1, **conditions)
    # Use identity to invalidate cache.
    ildj = nvp.inverse_log_det_jacobian(
        tf.identity(forward_x), event_ndims=1, **conditions)
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
    self.assertStartsWith(nvp.name, 'real_nvp')
    self.assertAllClose(forward_x_, forward_inverse_y_, rtol=1e-5, atol=1e-5)
    self.assertAllClose(x_, inverse_y_, rtol=1e-5, atol=1e-5)
    self.assertAllClose(ildj_, -fldj_, rtol=1e-5, atol=1e-5)


del RealNVPTestBase

if __name__ == '__main__':
  test_util.main()
