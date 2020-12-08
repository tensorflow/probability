# Copyright 2018 The TensorFlow Probability Authors.
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
"""Layers for normalizing flows and masked autoregressive density estimation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import masked_autoregressive as masked_autoregressive_lib
from tensorflow_probability.python.distributions import transformed_distribution as transformed_distribution_lib

from tensorflow_probability.python.layers.distribution_layer import DistributionLambda


__all__ = [
    'AutoregressiveTransform',
]


class AutoregressiveTransform(DistributionLambda):
  """An autoregressive normalizing flow layer.

  Following [Papamakarios et al. (2017)][1], given an autoregressive model p(x)
  with conditional distributions in the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), we can construct a
  normalizing flow for p(x).

  Specifically, suppose `made` is a `tfb.AutoregressiveNetwork` -- a layer
  implementing a Masked Autoencoder for Distribution Estimation (MADE) -- that
  computes location and log-scale parameters `made(x)[i]` for each input `x[i]`.
  Then we can represent the autoregressive model `p(x)` as `x = f(u)` where `u`
  is drawn from from some base distribution and where `f` is an invertible and
  differentiable function (i.e., a `Bijector`) and `f^{-1}(x)` is defined by:
  ```python
  def f_inverse(x):
    shift, log_scale = tf.unstack(made(x), 2, axis=-1)
    return (x - shift) * tf.math.exp(-log_scale)
  ```

  Given a `tfb.AutoregressiveNetwork` layer `made`, an `AutoregressiveTransform`
  layer transforms an input `tfd.Distribution` p(u) to an output
  `tfp.Distribution` p(x) where `x = f(u)`.

  For additional details, see the `tfb.MaskedAutoregressiveFlow` bijector and
  the  `tfb.AutoregressiveNetwork`.

  #### Example

  ```python
  tfd = tfp.distributions
  tfpl = tfp.layers
  tfb = tfp.bijectors
  tfk = tf.keras

  # Generate data -- as in Figure 1 in [Papamakarios et al. (2017)][1]).
  n = 2000
  x2 = np.random.randn(n) * 2
  x1 = np.random.randn(n) + (x2 * x2 / 4)
  data = np.stack([x1, x2], axis=-1)

  # Density estimation with MADE.
  model = tfk.Sequential([
      # NOTE: This model takes no input and outputs a Distribution.  (We use
      # the batch_size and type of the input, but there are no actual input
      # values because the last dimension of the shape is 0.)
      #
      # For conditional density estimation, the model would take the
      # conditioning values as input.)
      tfk.layers.InputLayer(input_shape=(0,), dtype=tf.float32),

      # Given the empty input, return a standard normal distribution with
      # matching batch_shape and event_shape of [2].
      tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
          # pylint: disable=g-long-lambda
          loc=tf.zeros(tf.concat([tf.shape(t)[:-1], [2]], axis=0)),
          scale_diag=[1., 1.])),

      # Transform the standard normal distribution with event_shape of [2] to
      # the target distribution with event_shape of [2].
      tfpl.AutoregressiveTransform(tfb.AutoregressiveNetwork(
          params=2, hidden_units=[10], activation='relu')),
  ])

  model.compile(
      optimizer=tf.compat.v2.optimizers.Adam(),
      loss=lambda y, rv_y: -rv_y.log_prob(y))

  model.fit(x=np.zeros((n, 0)),
            y=data,
            batch_size=25,
            epochs=10,
            steps_per_epoch=n // 25,
            verbose=True)

  # Use the fitted distribution.
  distribution = model(np.zeros((0,)))
  distribution.sample(4)
  distribution.log_prob(np.zeros((5, 3, 2)))
  ```

  #### References

  [1]: George Papamakarios, Theo Pavlakou, Iain Murray, Masked Autoregressive
       Flow for Density Estimation.  In _Neural Information Processing Systems_,
       2017. https://arxiv.org/abs/1705.07057
  """

  def __init__(self, made, **kwargs):
    """Constructs the AutoregressiveTransform layer.

    Args:
      made: A `Made` layer, which must output two parameters for each input.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    super(AutoregressiveTransform, self).__init__(self._transform, **kwargs)

    if made.params != 2:
      raise ValueError('Argument made must output 2 parameters per input, '
                       'found {}.'.format(made.params))

    self._made = made

  def build(self, input_shape):
    tf.keras.Sequential([
        tf.keras.layers.InputLayer(
            input_shape=input_shape[1:], dtype=self.dtype),
        self._made
    ])
    super(AutoregressiveTransform, self).build(input_shape)

  def _transform(self, distribution):
    return transformed_distribution_lib.TransformedDistribution(
        bijector=masked_autoregressive_lib.MaskedAutoregressiveFlow(
            lambda x: tf.unstack(self._made(x), axis=-1)),
        distribution=distribution)
