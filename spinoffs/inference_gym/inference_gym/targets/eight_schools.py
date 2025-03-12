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
# ============================================================================
"""Eight schools model."""

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import prefer_static as ps

from inference_gym.targets import bayesian_model
from inference_gym.targets import model
from inference_gym.targets.ground_truth import eight_schools

tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    'EightSchools'
]


class EightSchools(bayesian_model.BayesianModel):
  """Eight Schools model.

  The eight schools problem [1, 2] considers the effectiveness of SAT
  coaching programs conducted in parallel at eight schools. It has become a
  classic problem that illustrates the usefulness of hierarchical modeling for
  sharing information between exchangeable groups.

  For each of the eight schools we have an estimated treatment effect
  and a standard error of the effect estimate. The treatment effects in the
  study were obtained by a linear regression on the treatment group using
  PSAT-M and PSAT-V scores as control variables.

  ```python
  num_schools = 8
  treatment_effects = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)
  treatment_stddevs = np.array(
    [15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)
  ```

  This model encodes a partial pooling of these effects towards a global
  estimated `avg_effect`. The strength of the pooling is determined by a
  `log_stddev` variable, which is itself inferred.

  ```none
  avg_effect ~ Normal(loc=0, scale=10)
  log_stddev ~ Normal(loc=5, scale=1)
  for i in range(8):
    school_effects[i] ~ Normal(loc=avg_effect, scale=exp(log_stddev))
    treatment_effects[i] ~ Normal(loc=school_effects[i],
                                  scale=treatment_stddevs[i])
  ```

  #### References

  [1] Donald B. Rubin. Estimation in parallel randomized experiments.
      _Journal of Educational Statistics_, 6(4):377-401, 1981.
  [2] Andrew Gelman, John Carlin, Hal Stern, David Dunson, Aki Vehtari, and
      Donald Rubin. Bayesian Data Analysis, Third Edition.
      Chapman and Hall/CRC, 2013.
  """

  GROUND_TRUTH_MODULE = eight_schools

  def __init__(
      self,
      dtype=tf.float32,
      name='eight_schools',
      pretty_name='Eight Schools',
  ):
    """Construct the Eight Schools model.

    Args:
      dtype: Dtype to use for floating point quantities.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.
    """
    with tf.name_scope(name):

      treatment_effects = tf.constant(
          [28, 8, -3, 7, -1, 1, 18, 12], dtype=dtype)
      treatment_stddevs = tf.constant(
          [15, 10, 16, 11, 9, 11, 10, 18], dtype=dtype)
      num_schools = ps.shape(treatment_effects)[-1]

      self._prior_dist = tfd.JointDistributionNamed({
          'avg_effect': tfd.Normal(
              loc=tf.constant(0.0, dtype), scale=10.0, name='avg_effect'
          ),
          'log_stddev': tfd.Normal(
              loc=tf.constant(5.0, dtype), scale=1.0, name='log_stddev'
          ),
          # Deliberately specify the more challenging 'centered' form of the
          # model (the non-centered form, where all RVs are iid standard normal
          # in the prior, provides an easier inference problem).
          'school_effects': lambda log_stddev, avg_effect: (  # pylint: disable=g-long-lambda
              tfd.Independent(
                  tfd.Normal(
                      loc=avg_effect[..., None]
                      * tf.ones(num_schools, dtype),
                      scale=tf.exp(log_stddev[..., None])
                      * tf.ones(num_schools, dtype),
                      name='school_effects',
                  ),
                  reinterpreted_batch_ndims=1,
              )
          ),
      })

      self._log_likelihood_fn = (
          lambda values: tfd.Independent(  # pylint: disable=g-long-lambda
              tfd.Normal(loc=values['school_effects'],
                         scale=treatment_stddevs),
              reinterpreted_batch_ndims=1).log_prob(treatment_effects))

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=lambda x: x,
                  pretty_name='Identity',
                  # The dtype must match structure returned by `_ext_identity`.
                  dtype=self._prior_dist.dtype)
      }

    super(EightSchools, self).__init__(
        default_event_space_bijector={'avg_effect': tfb.Identity(),
                                      'log_stddev': tfb.Identity(),
                                      'school_effects': tfb.Identity()},
        event_shape=self._prior_dist.event_shape,
        dtype=self._prior_dist.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _prior_distribution(self):
    return self._prior_dist

  def _log_likelihood(self, value):
    return self._log_likelihood_fn(value)
