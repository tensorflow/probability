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
"""Local Linear Trend State Space Model Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.sts import LocalLevelStateSpaceModel

tfd = tfp.distributions
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
tfl = tf.linalg


@test_util.run_all_in_graph_and_eager_modes
class _LocalLevelStateSpaceModelTest(object):

  def test_logprob(self):
    y = self._build_placeholder([1.0, 1.3, 1.9, 2.9, 2.1])

    ssm = LocalLevelStateSpaceModel(
        num_timesteps=5,
        level_scale=0.5,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([1.])))

    lp = ssm.log_prob(y[..., np.newaxis])
    expected_lp = -6.5021
    self.assertAllClose(self.evaluate(lp), expected_lp)

  def test_stats(self):
    # Build a model with expected initial scale 0.
    level_scale = self._build_placeholder(1.0)
    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=self._build_placeholder([0.]),
        scale_diag=self._build_placeholder([1.]))

    ssm = LocalLevelStateSpaceModel(
        num_timesteps=10,
        level_scale=level_scale,
        initial_state_prior=initial_state_prior)

    # In expectation, the process is constant.
    mean = self.evaluate(ssm.mean())
    self.assertAllClose(mean, np.zeros(10)[:, np.newaxis])

    # variance of level[T] is T * level_scale
    expected_variance = np.arange(1, 11)[:, np.newaxis]
    variance = self.evaluate(ssm.variance())
    self.assertAllClose(variance, expected_variance)

  def test_batch_shape(self):
    batch_shape = [4, 2]

    level_scale = self._build_placeholder(
        np.exp(np.random.randn(*batch_shape)))
    initial_state_prior = tfd.MultivariateNormalDiag(
        scale_diag=self._build_placeholder([1.]))

    ssm = LocalLevelStateSpaceModel(
        num_timesteps=10,
        level_scale=level_scale,
        initial_state_prior=initial_state_prior)
    self.assertAllEqual(self.evaluate(ssm.batch_shape_tensor()), batch_shape)

    y = ssm.sample()
    self.assertAllEqual(self.evaluate(tf.shape(input=y))[:-2], batch_shape)

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
    return tf.compat.v1.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.run_all_in_graph_and_eager_modes
class LocalLevelStateSpaceModelTestStaticShape32(
    tf.test.TestCase, _LocalLevelStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class LocalLevelStateSpaceModelTestDynamicShape32(
    tf.test.TestCase, _LocalLevelStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class LocalLevelStateSpaceModelTestStaticShape64(
    tf.test.TestCase, _LocalLevelStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == "__main__":
  tf.test.main()
