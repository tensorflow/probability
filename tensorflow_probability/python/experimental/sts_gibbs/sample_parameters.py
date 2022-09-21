# Copyright 2022 The TensorFlow Probability Authors.
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
"""Functions for sampling parameters useful in Gibbs Sampling."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import square
from tensorflow_probability.python.distributions import inverse_gamma
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import prefer_static


def normal_scale_posterior_inverse_gamma_conjugate(variance_prior,
                                                   observations,
                                                   is_missing=None):
  """Returns the conditional posterior of Normal scale given observations.

  We assume the conjugate InverseGamma->Normal model:

  ```
  scale ~ Sqrt(InverseGamma(variance_prior.concentration, variance_prior.scale))
  for i in [1, ..., num_observations]:
    x[i] ~ Normal(0, scale)
  ```

  and return a sample from `p(scale | x)`.

  Args:
    variance_prior: Variance prior distribution as a `tfd.InverseGamma`
      instance. Note that the prior is given on the variance, but the value
      returned is a sample of the scale.
    observations: Float `Tensor` of shape `[..., num_observations]`, specifying
      the centered observations `(x)`.
    is_missing: Optional `bool` `Tensor` of shape `[..., num_observations]`. A
      `True` value indicates that the corresponding observation is missing.

  Returns:
    sampled_scale: A `tfd.Distribution` of the conditional posterior
      of the inverse gamma scale.
  """
  dtype = observations.dtype
  num_observations = prefer_static.shape(observations)[-1]
  if is_missing is not None:
    num_missing = tf.reduce_sum(tf.cast(is_missing, dtype), axis=-1)
    observations = tf.where(is_missing, tf.zeros_like(observations),
                            observations)
    num_observations -= num_missing

  variance_posterior = inverse_gamma.InverseGamma(
      concentration=variance_prior.concentration +
      tf.cast(num_observations / 2, dtype),
      scale=variance_prior.scale +
      tf.reduce_sum(tf.square(observations), axis=-1) / 2.)
  scale_posterior = transformed_distribution.TransformedDistribution(
      bijector=invert.Invert(square.Square()), distribution=variance_posterior)

  if hasattr(variance_prior,
             'upper_bound') and variance_prior.upper_bound is not None:
    variance_posterior.upper_bound = variance_prior.upper_bound
    # TODO(kloveless): This should have sqrt applied, but it is not for
    # temporary backwards compatibility.
    scale_posterior.upper_bound = variance_prior.upper_bound

  return scale_posterior


# TODO(kloveless): This seems like this should be a function on the
# distribution itself.
def sample_with_optional_upper_bound(distribution, sample_shape=(), seed=None):
  """Samples from the given distribution with an optional upper bound."""
  sample = distribution.sample(sample_shape=sample_shape, seed=seed)
  if hasattr(distribution,
             'upper_bound') and distribution.upper_bound is not None:
    sample = tf.minimum(sample, distribution.upper_bound)

  return sample
