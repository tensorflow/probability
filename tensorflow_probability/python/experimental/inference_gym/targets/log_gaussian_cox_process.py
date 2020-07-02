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
"""Log-Gaussian Cox process models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.experimental.inference_gym.internal import data
from tensorflow_probability.python.experimental.inference_gym.targets import bayesian_model
from tensorflow_probability.python.experimental.inference_gym.targets import model
from tensorflow_probability.python.math import psd_kernels as tfpk

__all__ = [
    'LogGaussianCoxProcess',
]


class LogGaussianCoxProcess(bayesian_model.BayesianModel):
  """Log-Gaussian Cox Process model."""

  def __init__(
      self,
      train_locations,
      train_extents,
      train_counts,
      dtype=tf.float64,
      name='log_gaussian_cox_process',
      pretty_name='Log-Gaussian Cox Process',
  ):
    """Log-Gaussian Cox Process[1] regression in a D dimensional space.

    This models the observed event counts at a set of locations with associated
    extents. An extent could correspond to an area (in which case location could
    be the centroid of that area), or duration (in which case the location is
    infinitesimal, but the measurements are taken over an extended period of
    time). Counts divided by the extent at a location is termed the intensity at
    that location.

    A Gaussian Process with a Matern 3/2 kernel is used to model log-intensity
    deviations from the mean log-intensity. The regressed intensity is then
    multiplied by the extents to parameterize the rate a Poisson observation
    model. The posterior of the model is over the amplitude and length scale of
    the Matern kernel, as well as the log-intensity deviations themselves. In
    summary:

    ```none
    amplitude ~ LogNormal(-1, 0.5)
    length_scale ~ LogNormal(-1, 1)
    delta_log_intensities ~ GP(Matern32(amplitude, length_scale), locations)
    counts[i] ~ Poisson(extents[i] *
                        exp(delta_log_intensities[i] + mean_log_intensity))
    ```

    The data are encoded into three parallel arrays. I.e.
    `train_counts[i]` and `train_extents[i]` correspond to `train_locations[i]`.

    Args:
      train_locations: Float `Tensor` with shape `[num_train_points, D]`.
        Training set locations where counts were measured.
      train_extents: Float `Tensor` with shape `[num_train_points]`. Training
        set location extents, must be positive.
      train_counts: Integer `Tensor` with shape `[num_train_points]`. Training
        set counts, must be positive.
      dtype: Datatype to use for the model. Gaussian Process regression tends to
        require double precision.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.

    Raises:
      ValueError: If the parallel arrays are not all of the same size.

    #### References

    1. Diggle, P. J., Moraga, P., Rowlingson, B., & Taylor, B. M. (2013).
      Spatial and spatio-temporal log-gaussian cox processes: Extending the
      geostatistical paradigm. Statistical Science, 28(4), 542-563.

    """
    with tf.name_scope(name):
      if not (train_locations.shape[0] == train_counts.shape[0] ==
              train_extents.shape[0]):
        raise ValueError('`train_locations`, `train_counts` and '
                         '`train_extents` must all have the same length. '
                         'Got: {} {} {}'.format(train_locations.shape[0],
                                                train_counts.shape[0],
                                                train_extents.shape[0]))

      train_counts = tf.cast(train_counts, dtype=dtype)
      train_locations = tf.convert_to_tensor(train_locations, dtype=dtype)
      train_extents = tf.convert_to_tensor(train_extents, dtype=dtype)

      mean_log_intensity = tf.reduce_mean(
          tf.math.log(train_counts) - tf.math.log(train_extents),
          axis=-1,
      )

      self._prior_dist = tfd.JointDistributionNamed(
          dict(
              amplitude=tfd.LogNormal(-1, tf.constant(.5, dtype=dtype)),
              length_scale=tfd.LogNormal(-1, tf.constant(1., dtype=dtype)),
              log_intensity=lambda amplitude, length_scale: tfd.GaussianProcess(  # pylint: disable=g-long-lambda
                  mean_fn=lambda x: mean_log_intensity,
                  kernel=tfpk.MaternThreeHalves(
                      amplitude=amplitude + .001,
                      length_scale=length_scale + .001),
                  index_points=train_locations,
                  jitter=1e-6),
          ))

      def observation_noise_fn(log_intensity):
        """Creates the observation noise distribution."""
        return tfd.Poisson(log_rate=tf.math.log(train_extents) + log_intensity)

      self._observation_noise_fn = observation_noise_fn

      def log_likelihood_fn(log_intensity, **_):
        """The log_likelihood function."""
        return tf.reduce_sum(
            observation_noise_fn(log_intensity).log_prob(train_counts), -1)

      self._log_likelihood_fn = log_likelihood_fn

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=lambda params: params,
                  pretty_name='Identity',
                  dtype=self._prior_dist.dtype,
              )
      }

    self._train_locations = train_locations
    self._train_extents = train_extents

    super(LogGaussianCoxProcess, self).__init__(
        default_event_space_bijector=dict(
            amplitude=tfb.Exp(),
            length_scale=tfb.Exp(),
            log_intensity=tfb.Identity(),
        ),
        event_shape=self._prior_dist.event_shape,
        dtype=self._prior_dist.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _sample_dataset(self, seed):
    dataset = dict(
        train_locations=self._train_locations,
        train_extents=self._train_extents,
    )
    prior_samples = self.prior_distribution().sample(seed=seed)
    observation_noise_dist = self._observation_noise_fn(
        prior_samples['log_intensity'])
    counts = observation_noise_dist.sample(seed=seed)
    dataset['train_counts'] = tf.cast(counts, tf.int32)
    return dataset

  def _log_likelihood(self, value):
    return self._log_likelihood_fn(**value)

  def _prior_distribution(self):
    return self._prior_dist


class SyntheticLogGaussianCoxProcess(LogGaussianCoxProcess):
  """Log-Gaussian Cox Process model.

  This dataset was simulated by constructing a 10 by 10 grid of equidistant 2D
  locations with spacing = 1, and then sampling from the prior to determine the
  counts at those locations.
  """

  def __init__(self):
    dataset = data.synthetic_log_gaussian_cox_process()
    super(SyntheticLogGaussianCoxProcess, self).__init__(
        name='synthetic_log_gaussian_cox_process',
        pretty_name='Synthetic Log-Gaussian Cox Process',
        **dataset)
