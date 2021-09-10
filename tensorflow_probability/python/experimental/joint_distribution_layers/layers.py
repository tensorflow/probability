# Copyright 2021 The TensorFlow Probability Authors.
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
"""Joint Distribution Layers."""

import collections
import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import joint_distribution_coroutine
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.internal import vectorization_util

__all__ = [
    'Affine',
    'AffineLayer',
    'Conv2D',
    'Lambda',
    'make_conv2d_layer_class',
    'make_lambda_layer_class',
    'Sequential',
    'SequentialLayer',
]

Root = joint_distribution_coroutine.JointDistributionCoroutine.Root


class AffineLayer(collections.namedtuple('AffineLayer', [
    'weights',
    'bias',
])):
  """Affine layer.

  This represents a linear map: `y = weights @ x + bias`.

  Attributes:
    weights: A floating point Tensor with shape `[out_units, in_units]`.
    bias: A floating point Tensor with shape `[out_units]`.
  """
  __slots__ = ()

  def __call__(self, x):
    """Applies the layer to an input.

    Args:
      x: A floating point Tensor with shape [in_units].

    Returns:
      y: A floating point Tensor with shape [out_units].
    """
    x = tf.convert_to_tensor(x, dtype=self.weights.dtype)
    return tf.einsum('...oi,...i->...o', self.weights, x) + self.bias


def _affine_normal_params_model_fn(out_units, in_units, dtype):
  yield Root(
      sample.Sample(
          normal.Normal(tf.zeros([], dtype), 1.), [out_units, in_units],
          name='weights'))
  yield Root(
      sample.Sample(
          normal.Normal(tf.zeros([], dtype), 1.), out_units, name='bias'))


class Affine(joint_distribution_coroutine.JointDistributionCoroutine):
  """Distribution over affine transformations.

  By default, the distribution is effected by placing isotropic Gaussian priors
  over the `weights` and `bias` in this transformation: `y = weights @ x +
  bias`.

  #### Example

  ```python
  layer_dist = Affine(in_units=5, out_units=10)
  layer = layer_dist.sample(seed=tfp.random.sanitize_seed(0))
  y = layer(x)
  ```

  """

  def __init__(self,
               out_units,
               in_units,
               dtype=tf.float32,
               params_model_fn=_affine_normal_params_model_fn,
               name='Affine'):
    """Initialize the `Affine` distribution.

    Args:
      out_units: Integer Tensor. Number of output units.
      in_units: Integer Tensor. Number of input units.
      dtype: Dtype to use for the inner distributions.
      params_model_fn: Callable with signature `(out_units, in_units, dtype) ->
        model_fn`. A model functions (in the `JointDistributionCoroutine` sense)
        used to define the `weights` and `bias` random variables.
      name: Name to use for the ops defined by this distribution.
    """
    parameters = dict(locals())
    model = functools.partial(params_model_fn, out_units, in_units, dtype)
    super().__init__(
        model,
        name=name,
        sample_dtype=AffineLayer(weights=dtype, bias=dtype),
    )
    self._parameters = parameters


def make_conv2d_layer_class(strides, padding):
  """Creates a `Conv2DLayer` class.

  Args:
    strides: A 2-tuple of positive integers. Strides for the spatial dimensions.
    padding: A Python string. Can be either 'SAME' or 'VALID'.

  Returns:
    conv2d_layer_class: A new `Conv2DLayer` class that closes over the args to
      this function.
  """

  # TODO(siege): We do this rather than storing parameters explicitly inside the
  # class because we want to avoid having to use a CompositeTensor-like
  # functionality, as that requires annotating a large porting of TFP with
  # expand_composites=True. This isn't worth doing for this experimental
  # library.

  class Conv2DLayer(collections.namedtuple('Conv2DLayer', [
      'kernel',
  ])):
    """2-dimensional convolution (in the standard deep learning sense) layer.

    See `tf.nn.conv` for the mathematical details of what this does.

    Attributes:
      kernel: A floating point Tensor with shape `[width, height, in_channels,
        out_channels]`.
    """

    __slots__ = ()

    @property
    def strides(self):
      """Strides for the spatial dimensions."""
      return strides

    @property
    def padding(self):
      """Padding."""
      return padding

    def __call__(self, x):
      """Applies the layer to an input.

      Args:
        x: A floating point Tensor with shape `[batch, height, width,
          in_channels]`.

      Returns:
        y: A floating point Tensor with shape `[batch, height', width',
          out_channels]`. The output `width'` and `height'` depend on the value
          of
          `padding`.
      """

      @functools.partial(
          vectorization_util.make_rank_polymorphic, core_ndims=(4, 4))
      # In an ideal world we'd broadcast the kernel shape with the batch shape
      # of the input, but the hardware does not like that.
      def do_conv(kernel, x):
        return tf.nn.conv2d(
            x,
            filters=kernel,
            strides=(1,) + self.strides + (1,),
            padding=self.padding,
        )

      return do_conv(self.kernel, x)

  return Conv2DLayer


def _conv2d_normal_params_model_fn(out_channels, size, in_channels, dtype):
  yield Root(
      sample.Sample(
          normal.Normal(tf.zeros([], dtype), 1.),
          list(size) + [in_channels, out_channels],
          name='kernel'))


class Conv2D(joint_distribution_coroutine.JointDistributionCoroutine):
  """Distribution over convolutions.

  By default, the distribution is effected by placing isotropic Gaussian prior
  over the `kernel`.

  #### Example

  ```python
  layer_dist = Conv2D(in_channels=5, out_channels=10, size=5)
  layer = layer_dist.sample(seed=tfp.random.sanitize_seed(0))
  y = layer(x)
  ```

  """

  def __init__(self,
               out_channels,
               size,
               in_channels,
               dtype=tf.float32,
               params_model_fn=_conv2d_normal_params_model_fn,
               strides=(1, 1),
               padding='SAME',
               name='Conv2D'):
    """Initialize the `Conv2D` distribution.

    Args:
      out_channels: Integer Tensor. Number of output channels.
      size: Integer Tensor or a pair of integer-Tensors. Spatial extent of the
        kernel. If a single value is provided, it is used for both width and
        height.
      in_channels: Integer Tensor. Number of input channels.
      dtype: Dtype to use for the inner distribution.
      params_model_fn: Callable with signature `(out_channels, size,
        in_channels, dtype) -> model_fn`. A model functions (in the
        `JointDistributionCoroutine` sense) used to define the `kernel` random
        variable.
      strides: A 2-tuple of positive integers. Strides for the spatial
        dimensions.
      padding: A Python string. Can be either 'SAME' or 'VALID'.
      name: Name to use for the ops defined by this distribution.
    """
    parameters = dict(locals())
    if not isinstance(size, collections.abc.Sequence):
      size = (size, size)
    elif len(size) != 2:
      raise ValueError(
          '`size` must be a single integer or a 2-tuple of integers. '
          f'Saw: {size}.')

    model = functools.partial(params_model_fn, out_channels, size, in_channels,
                              dtype)

    super().__init__(
        model,
        name=name,
        sample_dtype=make_conv2d_layer_class(
            strides=tuple(strides),
            padding=padding,
        )(kernel=None),
    )
    self._parameters = parameters


def make_lambda_layer_class(fn):
  """Creates a `LambdaLayer` class.

  Args:
    fn: A callable.

  Returns:
    lambda_layer_class: A new `LambdaLayer` class that closes over `fn`.
  """

  # TODO(siege): We do this rather than storing parameters explicitly inside the
  # class because we want to avoid having to use a CompositeTensor-like
  # functionality, as that requires annotating a large porting of TFP with
  # expand_composites=True. This isn't worth doing for this experimental
  # library.

  class LambdaLayer(collections.namedtuple('LambdaLayer', ['dummy_value'])):
    """Lambda layer.

    This layer applies a callable over its input.
    """

    @property
    def fn(self):
      return fn

    def __call__(self, *args, **kwargs):
      return self.fn(*args, **kwargs)

  return LambdaLayer


class Lambda(joint_distribution_coroutine.JointDistributionCoroutine):
  """Distribution over callable atoms.

  This can be thought of as a `tfd.VectorDeterministic` over a point in a
  0-dimensional space, where the singular event happens to be callable.

  #### Example

  ```python
  dist = Lambda(tf.square)
  layer = layer_dist.sample(seed=tfp.random.sanitize_seed(0))
  y = layer(x)  # Same as tf.square(x)
  ```

  """

  def __init__(self, fn, dtype=tf.float32, name='Lambda'):
    """Construct the `Lambda` distribution.

    Args:
      fn: Callable.
      dtype: Dtype of the zero-sized dummy tensor used to convey batch
        information.
      name: Name to use for the ops defined by this distribution.
    """
    parameters = dict(locals())

    def model():
      # The primary utility of assocating a Tensor with this distribution is to
      # propagate the sample shape in situations like `log_prob(sample(shape))`.
      yield self.Root(
          deterministic.VectorDeterministic(
              tf.zeros([0], dtype), name='dummy_value'))

    super().__init__(
        model,
        name=name,
        sample_dtype=make_lambda_layer_class(fn)(dummy_value=None),
    )
    self._parameters = parameters


class SequentialLayer(collections.namedtuple('SequentialLayer', ['layers'])):
  """Sequential application of multiple layers.

  Attributes:
    layers: A sequence of layers to apply in sequence.
  """

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class Sequential(joint_distribution_coroutine.JointDistributionCoroutine):
  """Sequential layer distribution.

  ### Example

  ```python
  network = Sequential(
    Affine(4, 3),
    tf.nn.softplus,
  )
  y = network(x)
  ```
  """

  def __init__(self, *layers, name='Sequential'):
    """Construct the `Sequential`.

    For convenience, callable non-distribution layers are wrapped in the
    `Lambda` layer distribution.

    Args:
      *layers: Layer distributions to apply in sequence.
      name: Name to use for the ops defined by this distribution.
    """
    parameters = dict(locals())

    def model():
      for layer in layers:
        if callable(layer) and not hasattr(layer, 'log_prob'):
          layer = Lambda(layer)
        yield self.Root(layer)

    self._layers = layers

    super().__init__(
        model,
        name=name,
    )
    self._parameters = parameters

  @property
  def layers(self):
    return self._layers

  def _model_flatten(self, value):
    return tuple(value.layers)

  def _model_unflatten(self, value):
    return SequentialLayer(tuple(value))
