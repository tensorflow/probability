# Lint as: python2, python3
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
"""Banana model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp  # Avoid rewriting this to JAX.
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from tensorflow_probability.python.experimental.inference_gym.targets import bayesian_model

__all__ = [
    'Banana',
]


class Banana(bayesian_model.BayesianModel):
  """Creates a banana-shaped distribution.

  This distribution was first described in [1]. The distribution is constructed
  by transforming a 2-D normal distribution with scale [10, 1] by shifting the
  second dimension by `curvature * (x0**2 - 100)` where `x0` is the value of
  the first dimension. If an N > 2 dimensions are requested, the remaining
  dimensions are distributed as a standard normal.

  This distribution is notable for having relatively narrow tails, while being
  derived from a simple, volume-preserving transformation of a normal
  distribution. Despite this simplicity, some inference algorithms have trouble
  sampling from this distribution.

  #### References

  1. Haario, H., Saksman, E., & Tamminen, J. (1999). Adaptive proposal
     distribution for random walk Metropolis algorithm. Computational
     Statistics, 14(3), 375-396.
  """

  def __init__(
      self,
      ndims=2,
      curvature=0.03,
      name='banana',
      pretty_name='Banana',
  ):
    """Construct the banana model.

    Args:
      ndims: Python integer. Dimensionality of the distribution. Must be at
        least 2.
      curvature: Python float. Controls the strength of the curvature of
        the distribution.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.

    Raises:
      ValueError: If ndims < 2.
    """
    if ndims < 2:
      raise ValueError('ndims must be at least 2, saw: {}'.format(ndims))

    with tf.name_scope(name):

      def bijector_fn(x):
        """Banana transform."""
        batch_shape = tf.shape(x)[:-1]
        shift = tf.concat(
            [
                tf.zeros(tf.concat([batch_shape, [1]], axis=0)),
                curvature * (tf.square(x[..., :1]) - 100),
                tf.zeros(tf.concat([batch_shape, [ndims - 2]], axis=0)),
            ],
            axis=-1,
        )
        return tfb.Shift(shift)

      mg = tfd.MultivariateNormalDiag(
          loc=tf.zeros(ndims), scale_diag=[10.] + [1.] * (ndims - 1))
      banana = tfd.TransformedDistribution(
          mg, bijector=tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn))

      sample_transformations = {
          'identity':
              bayesian_model.BayesianModel.SampleTransformation(
                  fn=lambda params: params,
                  pretty_name='Identity',
                  # The second dimension is a sum of scaled Chi2 and normal
                  # distribution.
                  # Mean of Chi2 with one degree of freedom is 1, but since the
                  # first element has variance of 100, it cancels with the shift
                  # (which is why the shift is there).
                  ground_truth_mean=onp.zeros(ndims),
                  # Variance of Chi2 with one degree of freedom is 2.
                  ground_truth_standard_deviation=onp.array(
                      [10.] + [onp.sqrt(1. + 2 * curvature**2 * 10.**4)] +
                      [1.] * (ndims - 2)),
              )
      }

    self._banana = banana

    super(Banana, self).__init__(
        default_event_space_bijector=tfb.Identity(),
        event_shape=banana.event_shape,
        dtype=banana.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _joint_distribution(self):
    return self._banana

  def _evidence(self):
    return

  def _unnormalized_log_prob(self, value):
    return self.joint_distribution().log_prob(value)

  def sample(self, sample_shape=(), seed=None, name='sample'):
    """Generate samples of the specified shape from the target distribution.

    The returned samples are exact (and independent) samples from the target
    distribution of this model.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer or `tfp.util.SeedStream` instance, for seeding PRNG.
      name: Name to give to the prefix the generated ops.

    Returns:
      samples: a `Tensor` with prepended dimensions `sample_shape`.
    """
    return self.joint_distribution().sample(sample_shape, seed=seed, name=name)
