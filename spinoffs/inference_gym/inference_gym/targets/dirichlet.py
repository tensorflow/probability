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
"""Dirichlet model."""

import numpy as np
# import tensorflow.compat.v2 as tf


# import tensorflow_probability as tfp
from inference_gym.targets import model

import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from inference_gym.targets import model


tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    'Dirichlet',
]


class Dirichlet(model.Model):
  """Creates a Dirichlet.

  

  Note that this function produces reproducible targets, i.e. the constructor
  `seed` argument always needs to be non-`None`.
  """

  def __init__(
      self,
      concentration_vector=tf.ones(100)+100,
      dtype=tf.float32,
      name='dirichlet',
      pretty_name='Dirichlet',
  ):
    """Construct the Dirichlet.

    Args:
      dtype: Dtype to use for floating point quantities.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.
    """
    
    dirichlet = tfp.distributions.Dirichlet(
        concentration_vector,
        validate_args=False,
        allow_nan_stats=True,
        force_probs_to_zero_outside_support=False,
        name='Dirichlet'
    )
        
    sample_transformations = {
        'identity':
            model.Model.SampleTransformation(
                fn=lambda params: params,
                pretty_name='Identity',
                ground_truth_mean=dirichlet.mean(),
                ground_truth_standard_deviation=dirichlet.stddev(),
                dtype=dtype,
            )
    }

    self._dirichlet = dirichlet

    self.bij = tfb.IteratedSigmoidCentered(validate_args=False,name='iterated_sigmoid'  )

    super(Dirichlet, self).__init__(
        # default_event_space_bijector=tfb.Identity(),
        default_event_space_bijector=self.bij,
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
    return self.bij.inverse(self._dirichlet.sample(sample_shape, seed=seed, name=name))
  
if __name__ == '__main__':
  # Debug test for Dirichlet model
    import tensorflow_probability.python.internal.test_util as tfp_test_util
    dims = 3
    model = Dirichlet(
        concentration_vector=tf.ones(dims),
    )
    # set seed to 1
    print(model.sample(seed=1))