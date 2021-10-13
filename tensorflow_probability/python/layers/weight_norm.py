# Copyright 2019 The TensorFlow Probability Authors.
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
"""Layer wrapper for weight normalization."""

import warnings

import tensorflow.compat.v2 as tf


class WeightNorm(tf.keras.layers.Wrapper):
  """Layer wrapper to decouple magnitude and direction of the layer's weights.

  This wrapper reparameterizes a layer by decoupling the weight's
  magnitude and direction. This speeds up convergence by improving the
  conditioning of the optimization problem. It has an optional data-dependent
  initialization scheme, in which initial values of weights are set as functions
  of the first minibatch of data. Both the weight normalization and data-
  dependent initialization are described in [Salimans and Kingma (2016)][1].

  #### Example

  ```python
    net = WeightNorm(tf.keras.layers.Conv2D(2, 2, activation='relu'),
           input_shape=(32, 32, 3), data_init=True)(x)
    net = WeightNorm(tf.keras.layers.Conv2DTranspose(16, 5, activation='relu'),
                     data_init=True)
    net = WeightNorm(tf.keras.layers.Dense(120, activation='relu'),
                     data_init=True)(net)
    net = WeightNorm(tf.keras.layers.Dense(num_classes),
                     data_init=True)(net)
  ```

  #### References

  [1]: Tim Salimans and Diederik P. Kingma. Weight Normalization: A Simple
       Reparameterization to Accelerate Training of Deep Neural Networks. In
       _30th Conference on Neural Information Processing Systems_, 2016.
       https://arxiv.org/abs/1602.07868
  """

  def __init__(self, layer, data_init=True, **kwargs):
    """Initialize WeightNorm wrapper.

    Args:
      layer: A `tf.keras.layers.Layer` instance. Supported layer types are
        `Dense`, `Conv2D`, and `Conv2DTranspose`. Layers with multiple inputs
        are not supported.
      data_init: `bool`, if `True` use data dependent variable initialization.
      **kwargs: Additional keyword args passed to `tf.keras.layers.Wrapper`.

    Raises:
      ValueError: If `layer` is not a `tf.keras.layers.Layer` instance.

    """
    if not isinstance(layer, tf.keras.layers.Layer):
      raise ValueError(
          'Please initialize `WeightNorm` layer with a `tf.keras.layers.Layer` '
          'instance. You passed: {input}'.format(input=layer))

    layer_type = type(layer).__name__
    if layer_type not in ['Dense', 'Conv2D', 'Conv2DTranspose']:
      warnings.warn('`WeightNorm` is tested only for `Dense`, `Conv2D`, and '
                    '`Conv2DTranspose` layers. You passed a layer of type `{}`'
                    .format(layer_type))

    super(WeightNorm, self).__init__(layer, **kwargs)

    self.data_init = data_init
    self._track_trackable(layer, name='layer')
    self.filter_axis = -2 if layer_type == 'Conv2DTranspose' else -1

  def _compute_weights(self):
    """Generate weights with normalization."""
    # Determine the axis along which to expand `g` so that `g` broadcasts to
    # the shape of `v`.
    new_axis = -self.filter_axis - 3

    # `self.kernel_norm_axes` is determined by `self.filter_axis` and the rank
    # of the layer kernel, and is thus statically known.
    self.layer.kernel = tf.nn.l2_normalize(
        self.v, axis=self.kernel_norm_axes) * tf.expand_dims(self.g, new_axis)

  def _init_norm(self):
    """Set the norm of the weight vector."""
    kernel_norm = tf.sqrt(
        tf.reduce_sum(tf.square(self.v), axis=self.kernel_norm_axes))
    self.g.assign(kernel_norm)

  def _data_dep_init(self, inputs):
    """Data dependent initialization."""
    # Normalize kernel first so that calling the layer calculates
    # `tf.dot(v, x)/tf.norm(v)` as in (5) in ([Salimans and Kingma, 2016][1]).
    self._compute_weights()

    activation = self.layer.activation
    self.layer.activation = None

    use_bias = self.layer.bias is not None
    if use_bias:
      bias = self.layer.bias
      self.layer.bias = tf.zeros_like(bias)

    # Since the bias is initialized as zero, setting the activation to zero and
    # calling the initialized layer (with normalized kernel) yields the correct
    # computation ((5) in Salimans and Kingma (2016))
    x_init = self.layer(inputs)
    norm_axes_out = list(range(x_init.shape.rank - 1))
    m_init, v_init = tf.nn.moments(x_init, norm_axes_out)
    scale_init = 1. / tf.sqrt(v_init + 1e-10)

    self.g.assign(self.g * scale_init)
    if use_bias:
      self.layer.bias = bias
      self.layer.bias.assign(-m_init * scale_init)
    self.layer.activation = activation

  def build(self, input_shape=None):
    """Build `Layer`.

    Args:
      input_shape: The shape of the input to `self.layer`.

    Raises:
      ValueError: If `Layer` does not contain a `kernel` of weights
    """

    input_shape = tf.TensorShape(input_shape).as_list()
    input_shape[0] = None
    self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

    if not self.layer.built:
      self.layer.build(input_shape)

      if not hasattr(self.layer, 'kernel'):
        raise ValueError('`WeightNorm` must wrap a layer that'
                         ' contains a `kernel` for weights')

      kernel_norm_axes = list(range(self.layer.kernel.shape.rank))
      kernel_norm_axes.pop(self.filter_axis)
      # Convert `kernel_norm_axes` from a list to a constant Tensor to allow
      # TF checkpoint saving.
      self.kernel_norm_axes = tf.constant(kernel_norm_axes)

      self.v = self.layer.kernel

      # to avoid a duplicate `kernel` variable after `build` is called
      self.layer.kernel = None
      self.g = self.add_weight(
          name='g',
          shape=(int(self.v.shape[self.filter_axis]),),
          initializer='ones',
          dtype=self.v.dtype,
          trainable=True)
      self.initialized = self.add_weight(
          name='initialized',
          dtype=tf.bool,
          trainable=False)
      self.initialized.assign(False)

    super(WeightNorm, self).build()

  @tf.function
  def call(self, inputs):
    """Call `Layer`."""
    if not self.initialized:
      if self.data_init:
        self._data_dep_init(inputs)
      else:
        # initialize `g` as the norm of the initialized kernel
        self._init_norm()

      self.initialized.assign(True)

    self._compute_weights()
    output = self.layer(inputs)
    return output

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(
        self.layer.compute_output_shape(input_shape).as_list())
