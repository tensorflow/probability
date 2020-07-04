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
# ============================================================================
"""Implementation of the Bayesian model class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.inference_gym.targets import model

__all__ = [
    'BayesianModel',
]


class BayesianModel(model.Model):
  """Base class for Bayesian models in the Inference Gym.

  Given a Bayesian model described by a prior `P(x)` which we can sample from,
  and a likelihood `P(x | y)` with evidence `y` we construct the posterior by
  multiplying the two terms. The posterior distribution `P(x | y)` is
  represented as a product of the inverse normalization constant and the
  un-normalized density: `1/Z tilde{P}(x | y)`.

  #### Examples

  A simple 2-variable Bayesian model:

  ```python
  class SimpleModel(gym.targets.BayesianModel):

    def __init__(self):
      self._prior_val = tfd.Exponential(0.)
      self._evidence = 1.

      def log_likelihood_fn(x):
        return tfd.Normal(x, 1.).log_prob(self._evidence)

      self._log_likelihood_fn = log_likelihood_fn

      super(SimpleModel, self).__init__(
          default_event_space_bijector=tfb.Exp(),
          event_shape=self._prior_val.event_shape,
          dtype=self._prior_val.dtype,
          name='simple_model',
          pretty_name='SimpleModel',
          sample_transformations=dict(
              identity=gym.targets.Model.SampleTransformation(
                  fn=lambda x: x,
                  pretty_name='Identity',
              ),),
      )

    def _prior_distribution(self):
      return self._prior_val

    def _log_likelihood(self, value):
      return self._log_likelihood_fn(value)
  ```

  Note how we first constructed a prior distribution, and then used its
  properties to specify the Bayesian model. Note that we don't need to define an
  explicit `_unnormalized_log_prob`, as that's automatically constructed from
  the defined `_prior_distribution` and `_log_likelihood` methods.

  We don't specify the ground truth values for the `identity` sample
  transformation as they're not known analytically. See
  `GermanCreditNumericLogisticRegression` Bayesian model for an example of how
  to incorporate Monte-Carlo derived values for ground truth into a sample
  transformation.
  """

  def _prior_distribution(self):
    raise NotImplementedError('_prior_distribution is not implemented.')

  def prior_distribution(self, name='prior_distribution'):
    """The prior distribution over the model parameters."""
    with tf.name_scope(self.name):
      with tf.name_scope(name):
        return self._prior_distribution()

  def _log_likelihood(self, value):
    raise NotImplementedError('_log_likelihood is not implemented.')

  def log_likelihood(self, value, name='log_likelihood'):
    """Evaluates the log_likelihood at `value`."""
    with tf.name_scope(self.name):
      with tf.name_scope(name):
        return self._log_likelihood(value)

  def _unnormalized_log_prob(self, value):
    return (self.prior_distribution().log_prob(value) +
            self.log_likelihood(value))
