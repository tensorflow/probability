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
"""Autoregressive State Space Model Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import AutoregressiveStateSpaceModel
from tensorflow_probability.python.sts import LocalLevelStateSpaceModel


def ar_explicit_logp(y, coefs, level_scale):
  """Manual log-prob computation for an autoregressive process."""
  num_coefs = len(coefs)
  lp = 0.

  # For the first few steps of y, where previous values
  # are not available, we model them as zero-mean with
  # stddev `prior_scale`.
  for i in range(num_coefs):
    zero_padded_y = np.zeros([num_coefs])
    zero_padded_y[num_coefs - i:num_coefs] = y[:i]
    pred_y = np.dot(zero_padded_y, coefs[::-1])
    lp += tfd.Normal(pred_y, level_scale).log_prob(y[i])

  for i in range(num_coefs, len(y)):
    pred_y = np.dot(y[i - num_coefs:i], coefs[::-1])
    lp += tfd.Normal(pred_y, level_scale).log_prob(y[i])

  return lp


class _AutoregressiveStateSpaceModelTest(test_util.TestCase):

  def testEqualsLocalLevel(self):
    # An AR1 process with coef 1 is just a random walk, equivalent to a local
    # level model. Test that both models define the same distribution
    # (log-prob).
    num_timesteps = 10
    observed_time_series = self._build_placeholder(
        np.random.randn(num_timesteps, 1))

    level_scale = self._build_placeholder(0.1)

    # We'll test an AR1 process, and also (just for kicks) that the trivial
    # embedding as an AR2 process gives the same model.
    coefficients_order1 = np.array([1.]).astype(self.dtype)
    coefficients_order2 = np.array([1., 0.]).astype(self.dtype)

    ar1_ssm = AutoregressiveStateSpaceModel(
        num_timesteps=num_timesteps,
        coefficients=coefficients_order1,
        level_scale=level_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=[level_scale]))
    ar2_ssm = AutoregressiveStateSpaceModel(
        num_timesteps=num_timesteps,
        coefficients=coefficients_order2,
        level_scale=level_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=[level_scale, 1.]))

    local_level_ssm = LocalLevelStateSpaceModel(
        num_timesteps=num_timesteps,
        level_scale=level_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=[level_scale]))

    ar1_lp, ar2_lp, ll_lp = self.evaluate(
        (ar1_ssm.log_prob(observed_time_series),
         ar2_ssm.log_prob(observed_time_series),
         local_level_ssm.log_prob(observed_time_series)))
    self.assertAllClose(ar1_lp, ll_lp)
    self.assertAllClose(ar2_lp, ll_lp)

  def testLogprobCorrectness(self):
    # Compare the state-space model's log-prob to an explicit implementation.
    num_timesteps = 10
    observed_time_series_ = np.random.randn(num_timesteps)
    coefficients_ = np.array([.7, -.1]).astype(self.dtype)
    level_scale_ = 1.0

    observed_time_series = self._build_placeholder(observed_time_series_)
    level_scale = self._build_placeholder(level_scale_)

    expected_logp = ar_explicit_logp(
        observed_time_series_, coefficients_, level_scale_)

    ssm = AutoregressiveStateSpaceModel(
        num_timesteps=num_timesteps,
        coefficients=coefficients_,
        level_scale=level_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=[level_scale, 0.]))

    lp = ssm.log_prob(observed_time_series[..., tf.newaxis])
    self.assertAllClose(self.evaluate(lp), expected_logp)

  def testBatchShape(self):
    # Check that the model builds with batches of parameters.
    order = 3
    batch_shape = [4, 2]

    # No `_build_placeholder`, because coefficients must have static shape.
    coefficients = np.random.randn(*(batch_shape + [order])).astype(self.dtype)

    level_scale = self._build_placeholder(
        np.exp(np.random.randn(*batch_shape)))

    ssm = AutoregressiveStateSpaceModel(
        num_timesteps=10,
        coefficients=coefficients,
        level_scale=level_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder(np.ones([order]))))
    if self.use_static_shape:
      self.assertAllEqual(ssm.batch_shape.as_list(), batch_shape)
    else:
      self.assertAllEqual(self.evaluate(ssm.batch_shape_tensor()), batch_shape)

    y = ssm.sample()
    if self.use_static_shape:
      self.assertAllEqual(y.shape.as_list()[:-2], batch_shape)
    else:
      self.assertAllEqual(self.evaluate(tf.shape(y))[:-2], batch_shape)

  def _build_placeholder(self, ndarray):
    """Convert a numpy array to a TF placeholder.

    Args:
      ndarray: any object convertible to a numpy array via `np.asarray()`.

    Returns:
      placeholder: a TensorFlow `placeholder` with default value given by the
      provided `ndarray`, dtype given by `self.dtype`, and shape specified
      statically only if `self.use_static_shape` is `True`.
    """

    ndarray = np.asarray(ndarray).astype(self.dtype)
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.test_all_tf_execution_regimes
class AutoregressiveStateSpaceModelTestStaticShape32(
    _AutoregressiveStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class AutoregressiveStateSpaceModelTestDynamicShape32(
    _AutoregressiveStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class AutoregressiveStateSpaceModelTestStaticShape64(
    _AutoregressiveStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True

del _AutoregressiveStateSpaceModelTest  # Don't run tests for the base class.

if __name__ == "__main__":
  tf.test.main()
