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
"""Ill-conditioned Gaussian model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp  # Avoid rewriting this to JAX.
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from tensorflow_probability.python.experimental.inference_gym.targets import bayesian_model

__all__ = [
    'IllConditionedGaussian',
]


class IllConditionedGaussian(bayesian_model.BayesianModel):
  """Creates a random ill-conditioned Gaussian.

  The covariance matrix has eigenvalues sampled from the inverse Gamma
  distribution with the specified shape, and then rotated by a random orthogonal
  matrix.

  Note that this function produces reproducible targets, i.e. the constructor
  `seed` argument always needs to be non-`None`.
  """

  def __init__(
      self,
      ndims=100,
      gamma_shape_parameter=0.5,
      max_eigvalue=None,
      seed=10,
      name='ill_conditioned_gaussian',
      pretty_name='Ill-Conditioned Gaussian',
  ):
    """Construct the ill-conditioned Gaussian.

    Args:
      ndims: Python `int`. Dimensionality of the Gaussian.
      gamma_shape_parameter: Python `float`. The shape parameter of the inverse
        Gamma distribution. Anything below 2 is likely to yield poorly
        conditioned covariance matrices.
      max_eigvalue: Python `float`. If set, will normalize the eigenvalues such
        that the maximum is this value.
      seed: Seed to use when generating the eigenvalues and the random
        orthogonal matrix.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.
    """
    rng = onp.random.RandomState(seed=seed & (2**32 - 1))
    eigenvalues = 1. / onp.sort(
        rng.gamma(shape=gamma_shape_parameter, scale=1., size=ndims))
    if max_eigvalue is not None:
      eigenvalues *= max_eigvalue / eigenvalues.max()

    q, r = onp.linalg.qr(rng.randn(ndims, ndims))
    q *= onp.sign(onp.diag(r))

    covariance = (q * eigenvalues).dot(q.T)

    gaussian = tfd.MultivariateNormalTriL(
        loc=tf.zeros(ndims),
        scale_tril=tf.linalg.cholesky(
            tf.convert_to_tensor(covariance, dtype=tf.float32)))
    self._eigenvalues = eigenvalues

    sample_transformations = {
        'identity':
            bayesian_model.BayesianModel.SampleTransformation(
                fn=lambda params: params,
                pretty_name='Identity',
                ground_truth_mean=onp.zeros(ndims),
                ground_truth_standard_deviation=onp.sqrt(onp.diag(covariance)),
            )
    }

    self._gaussian = gaussian

    super(IllConditionedGaussian, self).__init__(
        default_event_space_bijector=tfb.Identity(),
        event_shape=gaussian.event_shape,
        dtype=gaussian.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _joint_distribution(self):
    return self._gaussian

  def _evidence(self):
    return

  def _unnormalized_log_prob(self, value):
    return self.joint_distribution().log_prob(value)

  @property
  def covariance_eigenvalues(self):
    return self._eigenvalues

  def sample(self, sample_shape=(), seed=None, name='sample'):
    """Generate samples of the specified shape from the target distribution.

    The returned samples are exact (and independent) samples from the target
    distribution of this model.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer or `tfp.util.SeedStream` instance, for seeding PRNG.
      name: Name to give to the prefix the generated ops.

    Returns:
      samples: a `Tensor` with prepended dimensions `sample_shape`.
    """
    return self.joint_distribution().sample(sample_shape, seed=seed, name=name)
