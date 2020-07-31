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
"""Neal's funnel model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.experimental.inference_gym.targets import model
from tensorflow_probability.python.internal import prefer_static as ps


__all__ = [
    'NealsFunnel',
]


class NealsFunnel(model.Model):
  """Creates a funnel-shaped distribution.

  This distribution was first described in [1]. The distribution is constructed
  by transforming a N-D gaussian with scale [3, 1, ...] by scaling all but the
  first dimensions by `exp(x0 / 2)` where  `x0` is the value of the first
  dimension.

  This distribution is notable for having a relatively very narrow "neck" region
  which is challenging for HMC to explore. This distribution resembles the
  posteriors of centrally parameterized hierarchical models.

  #### References

  1. Neal, R. M. (2003). Slice sampling. Annals of Statistics, 31(3), 705-767.
  """

  def __init__(
      self,
      ndims=10,
      name='neals_funnel',
      pretty_name='Neal\'s Funnel',
  ):
    """Construct the Neal's funnel model.

    Args:
      ndims: Python integer. Dimensionality of the distribution. Must be at
        least 2.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.

    Raises:
      ValueError: If ndims < 2.
    """
    if ndims < 2:
      raise ValueError('ndims must be at least 2, saw: {}'.format(ndims))

    with tf.name_scope(name):

      def bijector_fn(x):
        """Funnel transform."""
        batch_shape = ps.shape(x)[:-1]
        scale = tf.concat(
            [
                tf.ones(ps.concat([batch_shape, [1]], axis=0)),
                tf.exp(x[..., :1] / 2) *
                tf.ones(ps.concat([batch_shape, [ndims - 1]], axis=0)),
            ],
            axis=-1,
        )
        return tfb.Scale(scale)

      mg = tfd.MultivariateNormalDiag(
          loc=tf.zeros(ndims), scale_diag=[3.] + [1.] * (ndims - 1))
      funnel = tfd.TransformedDistribution(
          mg, bijector=tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn))

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=lambda params: params,
                  pretty_name='Identity',
                  # The trailing dimensions come from a product distribution of
                  # independent standard normal and a log-normal with a scale of
                  # 3 / 2. See
                  # https://en.wikipedia.org/wiki/Product_distribution for the
                  # formulas. For the mean, the formulas yield zero.
                  ground_truth_mean=np.zeros(ndims),
                  # For the standard deviation, all means are zero and standard
                  # deivations of the normals are 1, so the formula reduces to
                  # `sqrt((sigma_log_normal + mean_log_normal**2))` which
                  # reduces to `exp((sigma_log_normal)**2)`.
                  ground_truth_standard_deviation=np.array(
                      [3.] + [np.exp((3. / 2)**2)] * (ndims - 1)),
              )
      }

    self._funnel = funnel

    super(NealsFunnel, self).__init__(
        default_event_space_bijector=tfb.Identity(),
        event_shape=funnel.event_shape,
        dtype=funnel.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _unnormalized_log_prob(self, value):
    return self._funnel.log_prob(value)

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
    return self._funnel.sample(sample_shape, seed=seed, name=name)
