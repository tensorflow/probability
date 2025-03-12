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
"""NonIdentifiableQuarticMeasurementModel model."""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from inference_gym.targets import bayesian_model
from inference_gym.targets import model

tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    'NonIdentifiableQuarticMeasurementModel',
]


class NonIdentifiableQuarticMeasurementModel(bayesian_model.BayesianModel):
  """A non-identifiable model with a quartic measurement model.

  This distribution is defined in [1] as the posterior induced by the following
  model:

  ```none
  for i in range(ndims):
    theta[i] ~ Normal(loc=0, scale=1)
  F(theta) = theta[1]**2 + 3 * theta[0]**2 * (theta[0]**2 - 1)
  y ~ Normal(loc=F(theta), scale=noise_scale)
  ```

  with an observed `y = 1`. Note that if `ndims` is > 2, the additional
  dimensions are distributed as a standard normal.

  This distribution is notable for having density on a narrow manifold which
  requires small step sizes when using HMC. It also happens to be multi-modal (4
  modes), but the modes are not well separated and should not pose challenge to
  most inference methods.

  #### References

  1. Au, K. X., Graham, M. M., & Thiery, A. H. (2020). Manifold lifting: scaling
     MCMC to the vanishing noise regime. (2), 1-18. Retrieved from
     http://arxiv.org/abs/2003.03950
  """

  def __init__(
      self,
      ndims=2,
      noise_scale=0.1,
      dtype=tf.float32,
      name='non_identifiable_quartic_measurement_model',
      pretty_name='Non-Identifiable Quartic Measurement Model',
  ):
    """Construct the NonIdentifiableQuarticMeasurementModel model.

    Args:
      ndims: Python integer. Dimensionality of the distribution. Must be at
        least 2.
      noise_scale: Floating point Tensor. Scale of the observation noise.
      dtype: Dtype to use for floating point quantities.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.

    Raises:
      ValueError: If ndims < 2.
    """
    if ndims < 2:
      raise ValueError('ndims must be at least 2, saw: {}'.format(ndims))

    with tf.name_scope(name):

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=lambda params: params,
                  pretty_name='Identity',
                  ground_truth_mean=np.zeros(ndims),  # By symmetry.
                  dtype=dtype,
              )
      }

      self._prior_dist = tfd.Sample(tfd.Normal(tf.zeros([], dtype), 1.0), ndims)
      self._noise_scale = tf.cast(noise_scale, dtype)

    super(NonIdentifiableQuarticMeasurementModel, self).__init__(
        default_event_space_bijector=tfb.Identity(),
        event_shape=self._prior_dist.event_shape,
        dtype=self._prior_dist.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _prior_distribution(self):
    return self._prior_dist

  def _log_likelihood(self, value):
    theta_0_sq = value[..., 0]**2
    theta_1_sq = value[..., 1]**2
    y_hat = theta_1_sq + 3 * theta_0_sq * (theta_0_sq - 1.)
    y = 1.
    observation_dist = tfd.Normal(y_hat, self._noise_scale)
    return observation_dist.log_prob(y)
