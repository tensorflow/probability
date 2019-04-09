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
"""Dynamic Linear Regression model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import dtype_util

from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


class DynamicLinearRegressionStateSpaceModel(tfd.LinearGaussianStateSpaceModel):

  def __init__(self,
               num_timesteps,
               design_matrix,
               weights_scale,
               initial_state_prior,
               observation_noise_scale=0.,
               initial_step=0,
               validate_args=False,
               allow_nan_stats=True,
               name=None):

    with tf.compat.v1.name_scope(
        name, 'DynamicLinearRegressionStateSpaceModel') as name:

      dtype = initial_state_prior.dtype

      design_matrix = tf.convert_to_tensor(
          value=design_matrix, name='design_matrix', dtype=dtype)

      weights_scale = tf.convert_to_tensor(
          value=weights_scale, name='weights_scale', dtype=dtype)

      observation_noise_scale = tf.convert_to_tensor(
          value=observation_noise_scale,
          name='observation_noise_scale',
          dtype=dtype)

      self._weights_scale = weights_scale
      self._observation_noise_scale = observation_noise_scale

      def observation_matrix_fn(t):
        observation_matrix = tf.linalg.LinearOperatorFullMatrix(
            design_matrix[..., t, :], name='observation_matrix')
        return observation_matrix

      super(DynamicLinearRegressionStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=tf.linalg.LinearOperatorIdentity(
              num_rows=tf.shape(design_matrix)[-1],
              dtype=dtype,
              name='transition_matrix'),
          transition_noise=tfd.MultivariateNormalDiag(
              scale_diag=weights_scale[..., tf.newaxis],
              name='transition_noise'),
          observation_matrix=observation_matrix_fn,
          observation_noise=tfd.MultivariateNormalDiag(
              scale_diag=observation_noise_scale[..., tf.newaxis],
              name='observation_noise'),
          initial_state_prior=initial_state_prior,
          initial_step=initial_step,
          allow_nan_stats=allow_nan_stats,
          validate_args=validate_args,
          name=name)

  @property
  def weights_scale(self):
    """Standard deviation of the weights transitions."""
    return self._weights_scale

  @property
  def observation_noise_scale(self):
    """Standard deviation of the observation noise."""
    return self._observation_noise_scale
