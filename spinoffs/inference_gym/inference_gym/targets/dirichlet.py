# Copyright 2025 The TensorFlow Probability Authors.
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
"""Dirichlet model."""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from inference_gym.targets import model
import tensorflow_probability.substrates.numpy as tfp_np


tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    'Dirichlet',
]


class Dirichlet(model.Model):
  """Creates a Dirichlet.

  This function produces a Dirichlet distribution. Low concentration parameters
  (much below 1) produce a distribution that is typically difficult to sample
  from.
  """

  def __init__(
      self,
      concentration_vector=np.ones(100) * 0.1,
      dtype=tf.float32,
      name='dirichlet',
      pretty_name='Dirichlet',
  ):
    """Construct the Dirichlet.

    Args:
      concentration_vector: The concentration parameters of the Dirichlet
        distribution.
      dtype: Dtype to use for floating point quantities.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.
    """

    dirichlet = tfp.distributions.Dirichlet(
        concentration=tf.cast(concentration_vector, dtype),
    )
    dirichlet_np = tfp_np.distributions.Dirichlet(
        concentration=np.asarray(concentration_vector),
    )

    sample_transformations = {
        'identity': model.Model.SampleTransformation(
            fn=lambda params: params,
            pretty_name='Identity',
            ground_truth_mean=dirichlet_np.mean(),
            ground_truth_standard_deviation=dirichlet_np.stddev(),
            dtype=dtype,
        )
    }

    self._dirichlet = dirichlet

    super(Dirichlet, self).__init__(
        default_event_space_bijector=tfb.IteratedSigmoidCentered(
            validate_args=False, name='iterated_sigmoid'
        ),
        event_shape=dirichlet.event_shape,
        dtype=dirichlet.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _unnormalized_log_prob(self, value):
    return self._dirichlet.log_prob(value)

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
    return self._dirichlet.sample(sample_shape, seed=seed, name=name)
