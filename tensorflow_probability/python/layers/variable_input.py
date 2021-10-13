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
"""VariableInputLayer."""

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf


class VariableLayer(tf.keras.layers.Layer):
  """Simply returns a (trainable) variable, regardless of input.

  This layer implements the mathematical function `f(x) = c` where `c` is a
  constant, i.e., unchanged for all `x`. Like other Keras layers, the constant
  is `trainable`.  This layer can also be interpretted as the special case of
  `tf.keras.layers.Dense` when the `kernel` is forced to be the zero matrix
  (`tf.zeros`).

  #### Examples

  ```python
  trainable_normal = tf.keras.models.Sequential([
      tfp.layers.VariableLayer(
          shape=[3, 4, 2],
          dtype=tf.float64,
          initializer=tfp.layers.BlockwiseInitializer([
              'zeros',
              tf.keras.initializers.Constant(np.log(np.expm1(1.))),
          ], sizes=[1, 1])),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., 0], scale=tf.math.softplus(t[..., 1])),
          reinterpreted_batch_ndims=1)),
  ])
  negloglik = lambda x, rv_x: -rv_x.log_prob(x)
  trainable_normal.compile(optimizer='adam', loss=negloglik)

  # trainable_normal.fit(dataset)

  x = trainable_normal(0.)  # `0.` ignored; like conditioning on emptyset.

  x.dtype
  # ==> tf.float64

  x.batch_shape
  # ==> [3]

  x.event_shape
  # ==> [4]

  x.mean()
  # ==> tf.reduce_mean(dataset)

  x.variance()
  # ==> tfp.stats.variance(dataset)
  ```

  """

  def __init__(self,
               shape,
               dtype=None,
               activation=None,
               initializer='zeros',
               regularizer=None,
               constraint=None,
               **kwargs):
    """Creates the `VariableLayer`.

    Args:
      shape: integer or integer vector specifying the shape of the output of
        this layer.
      dtype: TensorFlow `dtype` of the variable created by this layer.
        Default value: `None` (i.e., `tf.as_dtype(tf.keras.backend.floatx())`).
      activation: Activation function to use.  If you don't specify anything, no
        activation is applied (ie. "linear" activation: `a(x) = x`).
        Default value: `None`.
      initializer: Initializer for the `constant` vector. For example, to
        initialize a trainable (initially standard) `Normal` with
        `tf.math.softplus` transformed scale, one might use:
        ```python
        tfp.layers.BlockwiseInitializer([
            'zeros',
            tf.keras.initializers.Constant(np.log(np.expm1(1.))),  # = 0.541325
        ], sizes=[1, 1])
        ```
        Default value: `'zeros'`.
      regularizer: Regularizer function applied to the `constant` vector.
        Default value: `None`.
      constraint: Constraint function applied to the `constant` vector.
        Default value: `None`.
      **kwargs: Extra arguments forwarded to `tf.keras.layers.Layer`.
    """
    super(VariableLayer, self).__init__(**kwargs)

    self.activation = tf.keras.activations.get(activation)
    self.initializer = tf.keras.initializers.get(initializer)
    self.regularizer = tf.keras.regularizers.get(regularizer)
    self.constraint = tf.keras.constraints.get(constraint)

    shape = tf.get_static_value(shape)
    if shape is None:
      raise ValueError('Shape must be known statically.')
    shape = np.array(shape, dtype=np.int32)
    ndims = len(shape.shape)
    if ndims > 1:
      raise ValueError('Shape must be scalar or vector.')
    shape = shape.reshape(-1)  # Ensures vector shape.

    self._var = self.add_weight(
        'constant',
        shape=shape,
        initializer=self.initializer,
        regularizer=self.regularizer,
        constraint=self.constraint,
        dtype=dtype,
        trainable=kwargs.get('trainable', True))

  def call(self, _):
    x = tf.convert_to_tensor(value=self._var)
    if self.activation is None:
      return x
    return self.activation(x)
