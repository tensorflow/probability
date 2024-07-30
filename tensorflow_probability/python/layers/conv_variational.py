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
"""Convolutional variational layers."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import random as tfp_random
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.internal import tf_keras
from tensorflow_probability.python.layers import util as tfp_layers_util
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow.python.ops import nn_ops  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'Convolution1DFlipout',
    'Convolution1DReparameterization',
    'Convolution2DFlipout',
    'Convolution2DReparameterization',
    'Convolution3DFlipout',
    'Convolution3DReparameterization',
]


doc_args = """activation: Activation function. Set it to None to maintain a
      linear activation.
  activity_regularizer: Regularizer function for the output.
  kernel_posterior_fn: Python `callable` which creates
    `tfd.Distribution` instance representing the surrogate
    posterior of the `kernel` parameter. Default value:
    `default_mean_field_normal_fn()`.
  kernel_posterior_tensor_fn: Python `callable` which takes a
    `tfd.Distribution` instance and returns a representative
    value. Default value: `lambda d: d.sample()`.
  kernel_prior_fn: Python `callable` which creates `tfd`
    instance. See `default_mean_field_normal_fn` docstring for required
    parameter signature.
    Default value: `tfd.Normal(loc=0., scale=1.)`.
  kernel_divergence_fn: Python `callable` which takes the surrogate posterior
    distribution, prior distribution and random variate sample(s) from the
    surrogate posterior and computes or approximates the KL divergence. The
    distributions are `tfd.Distribution`-like instances and the
    sample is a `Tensor`.
  bias_posterior_fn: Python `callable` which creates
    `tfd.Distribution` instance representing the surrogate
    posterior of the `bias` parameter. Default value:
    `default_mean_field_normal_fn(is_singular=True)` (which creates an
    instance of `tfd.Deterministic`).
  bias_posterior_tensor_fn: Python `callable` which takes a
    `tfd.Distribution` instance and returns a representative
    value. Default value: `lambda d: d.sample()`.
  bias_prior_fn: Python `callable` which creates `tfd` instance.
    See `default_mean_field_normal_fn` docstring for required parameter
    signature. Default value: `None` (no prior, no variational inference)
  bias_divergence_fn: Python `callable` which takes the surrogate posterior
    distribution, prior distribution and random variate sample(s) from the
    surrogate posterior and computes or approximates the KL divergence. The
    distributions are `tfd.Distribution`-like instances and the
    sample is a `Tensor`."""


class _ConvVariational(tf_keras.layers.Layer):
  """Abstract nD convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      rank,
      filters,
      kernel_size,
      strides=1,
      padding='valid',
      data_format='channels_last',
      dilation_rate=1,
      activation=None,
      activity_regularizer=None,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, ...,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      ${args}
    """
    # pylint: enable=g-doc-args
    super(_ConvVariational, self).__init__(
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = normalize_tuple(strides, rank, 'strides')
    self.padding = normalize_padding(padding)
    self.data_format = normalize_data_format(data_format)
    self.dilation_rate = normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.activation = tf_keras.activations.get(activation)
    self.input_spec = tf_keras.layers.InputSpec(ndim=self.rank + 2)
    self.kernel_posterior_fn = kernel_posterior_fn
    self.kernel_posterior_tensor_fn = kernel_posterior_tensor_fn
    self.kernel_prior_fn = kernel_prior_fn
    self.kernel_divergence_fn = kernel_divergence_fn
    self.bias_posterior_fn = bias_posterior_fn
    self.bias_posterior_tensor_fn = bias_posterior_tensor_fn
    self.bias_prior_fn = bias_prior_fn
    self.bias_divergence_fn = bias_divergence_fn

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    input_dim = tf.compat.dimension_value(input_shape[channel_axis])
    if input_dim is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf_keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, kernel_shape, 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, kernel_shape, 'kernel_prior',
          self.trainable, self.add_variable)

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, (self.filters,), 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, (self.filters,), 'bias_prior',
          self.trainable, self.add_variable)

    self.input_spec = tf_keras.layers.InputSpec(
        ndim=self.rank + 2, axes={channel_axis: input_dim})
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=tf.TensorShape(kernel_shape),
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=convert_data_format(
            self.data_format, self.rank + 2))

    self.built = True

  def call(self, inputs):
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)

    outputs = self._apply_variational_kernel(inputs)
    outputs = self._apply_variational_bias(outputs)
    if self.activation is not None:
      outputs = self.activation(outputs)
    self._apply_divergence(self.kernel_divergence_fn,
                           self.kernel_posterior,
                           self.kernel_prior,
                           self.kernel_posterior_tensor,
                           name='divergence_kernel')
    self._apply_divergence(self.bias_divergence_fn,
                           self.bias_posterior,
                           self.bias_prior,
                           self.bias_posterior_tensor,
                           name='divergence_bias')
    return outputs

  def compute_output_shape(self, input_shape):
    """Computes the output shape of the layer.

    Args:
      input_shape: Shape tuple (tuple of integers) or list of shape tuples
        (one per output tensor of the layer). Shape tuples can include None for
        free dimensions, instead of an integer.

    Returns:
      output_shape: A tuple representing the output shape.
    """
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tf.TensorShape([input_shape[0]] + new_space + [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tf.TensorShape([input_shape[0], self.filters] + new_space)

  def get_config(self):
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable) containing the
    configuration of a layer. The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    Returns:
      config: A Python dictionary of class keyword arguments and their
        serialized values.
    """
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': (tf_keras.activations.serialize(self.activation)
                       if self.activation else None),
        'activity_regularizer':
            tf_keras.initializers.serialize(self.activity_regularizer),
    }
    function_keys = [
        'kernel_posterior_fn',
        'kernel_posterior_tensor_fn',
        'kernel_prior_fn',
        'kernel_divergence_fn',
        'bias_posterior_fn',
        'bias_posterior_tensor_fn',
        'bias_prior_fn',
        'bias_divergence_fn',
    ]
    for function_key in function_keys:
      function = getattr(self, function_key)
      if function is None:
        function_name = None
        function_type = None
      else:
        function_name, function_type = tfp_layers_util.serialize_function(
            function)
      config[function_key] = function_name
      config[function_key + '_type'] = function_type
    base_config = super(_ConvVariational, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    """Creates a layer from its config.

    This method is the reverse of `get_config`, capable of instantiating the
    same layer from the config dictionary.

    Args:
      config: A Python dictionary, typically the output of `get_config`.

    Returns:
      layer: A layer instance.
    """
    config = config.copy()
    function_keys = [
        'kernel_posterior_fn',
        'kernel_posterior_tensor_fn',
        'kernel_prior_fn',
        'kernel_divergence_fn',
        'bias_posterior_fn',
        'bias_posterior_tensor_fn',
        'bias_prior_fn',
        'bias_divergence_fn',
    ]
    for function_key in function_keys:
      serial = config[function_key]
      function_type = config.pop(function_key + '_type')
      if serial is not None:
        config[function_key] = tfp_layers_util.deserialize_function(
            serial,
            function_type=function_type)
    return cls(**config)

  def _apply_variational_bias(self, inputs):
    if self.bias_posterior is None:
      self.bias_posterior_tensor = None
      return inputs
    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)
    outputs = inputs
    if self.data_format == 'channels_first':
      if self.rank == 1:
        # tf.nn.bias_add does not accept a 1D input tensor.
        bias = tf.reshape(self.bias_posterior_tensor,
                          [1, self.filters, 1])
        outputs += bias
      if self.rank == 2:
        outputs = tf.nn.bias_add(outputs,
                                 self.bias_posterior_tensor,
                                 data_format='NCHW')
      if self.rank == 3:
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients. To use bias_add, we collapse Z
        # and Y into a single dimension to obtain a 4D input tensor.
        outputs_shape = tf.shape(outputs)
        outputs_4d = tf.reshape(outputs,
                                [outputs_shape[0], outputs_shape[1],
                                 outputs_shape[2] * outputs_shape[3],
                                 outputs_shape[4]])
        outputs_4d = tf.nn.bias_add(outputs_4d,
                                    self.bias_posterior_tensor,
                                    data_format='NCHW')
        outputs = tf.reshape(outputs_4d, outputs_shape)
    else:
      outputs = tf.nn.bias_add(outputs,
                               self.bias_posterior_tensor,
                               data_format='NHWC')
    return outputs

  def _apply_divergence(self, divergence_fn, posterior, prior,
                        posterior_tensor, name):
    if (divergence_fn is None or
        posterior is None or
        prior is None):
      divergence = None
      return
    divergence = tf.identity(
        divergence_fn(posterior, prior, posterior_tensor),
        name=name)
    self.add_loss(divergence)


class _ConvReparameterization(_ConvVariational):
  """Abstract nD convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [(Kingma and Welling, 2014)][1], which performs a Monte Carlo
  approximation of the distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  #### References

  [1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
       _International Conference on Learning Representations_, 2014.
       https://arxiv.org/abs/1312.6114
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      rank,
      filters,
      kernel_size,
      strides=1,
      padding='valid',
      data_format='channels_last',
      dilation_rate=1,
      activation=None,
      activity_regularizer=None,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, ...,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      ${args}
    """
    # pylint: enable=g-doc-args
    super(_ConvReparameterization, self).__init__(
        rank=rank,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=tf_keras.activations.get(activation),
        activity_regularizer=activity_regularizer,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)

  def _apply_variational_kernel(self, inputs):
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    outputs = self._convolution_op(inputs, self.kernel_posterior_tensor)
    return outputs


class Conv1DReparameterization(_ConvReparameterization):
  """1D convolution layer (e.g. temporal convolution).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [(Kingma and Welling, 2014)][1], which performs a Monte Carlo
  approximation of the distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Upon being built, this layer adds losses (accessible via the `losses`
  property) representing the divergences of `kernel` and/or `bias` surrogate
  posteriors and their respective priors. When doing minibatch stochastic
  optimization, make sure to scale this loss such that it is applied just once
  per epoch (e.g. if `kl` is the sum of `losses` for each element of the batch,
  you should pass `kl / num_examples_per_epoch` to your optimizer).

  You can access the `kernel` and/or `bias` posterior and prior distributions
  after the layer is built via the `kernel_posterior`, `kernel_prior`,
  `bias_posterior` and `bias_prior` properties.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  model = tf_keras.Sequential([
      tf_keras.layers.Reshape([128, 1]),
      tfp.layers.Convolution1DReparameterization(
          64, kernel_size=5, padding='SAME', activation=tf.nn.relu),
      tf_keras.layers.Flatten(),
      tfp.layers.DenseReparameterization(10),
  ])

  logits = model(features)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(model.losses)
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  #### References

  [1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
       _International Conference on Learning Representations_, 2014.
       https://arxiv.org/abs/1312.6114
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      filters,
      kernel_size,
      strides=1,
      padding='valid',
      data_format='channels_last',
      dilation_rate=1,
      activation=None,
      activity_regularizer=None,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer, specifying the
        length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, length,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, channels, length)`.
      dilation_rate: An integer or tuple/list of a single integer, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      ${args}
    """
    # pylint: enable=g-doc-args
    super(Conv1DReparameterization, self).__init__(
        rank=1,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=tf_keras.activations.get(activation),
        activity_regularizer=activity_regularizer,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)


class Conv2DReparameterization(_ConvReparameterization):
  """2D convolution layer (e.g. spatial convolution over images).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [(Kingma and Welling, 2014)][1], which performs a Monte Carlo
  approximation of the distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Upon being built, this layer adds losses (accessible via the `losses`
  property) representing the divergences of `kernel` and/or `bias` surrogate
  posteriors and their respective priors. When doing minibatch stochastic
  optimization, make sure to scale this loss such that it is applied just once
  per epoch (e.g. if `kl` is the sum of `losses` for each element of the batch,
  you should pass `kl / num_examples_per_epoch` to your optimizer).

  You can access the `kernel` and/or `bias` posterior and prior distributions
  after the layer is built via the `kernel_posterior`, `kernel_prior`,
  `bias_posterior` and `bias_prior` properties.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  model = tf_keras.Sequential([
      tf_keras.layers.Reshape([32, 32, 3]),
      tfp.layers.Convolution2DReparameterization(
          64, kernel_size=5, padding='SAME', activation=tf.nn.relu),
      tf_keras.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=[2, 2],
                                   padding='SAME'),
      tf_keras.layers.Flatten(),
      tfp.layers.DenseReparameterization(10),
  ])

  logits = model(features)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(model.losses)
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  #### References

  [1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
       _International Conference on Learning Representations_, 2014.
       https://arxiv.org/abs/1312.6114
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      filters,
      kernel_size,
      strides=(1, 1),
      padding='valid',
      data_format='channels_last',
      dilation_rate=(1, 1),
      activation=None,
      activity_regularizer=None,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, height,
        width, channels)` while `channels_first` corresponds to inputs with
        shape `(batch, channels, height, width)`.
      dilation_rate: An integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      ${args}
    """
    # pylint: enable=g-doc-args
    super(Conv2DReparameterization, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=tf_keras.activations.get(activation),
        activity_regularizer=activity_regularizer,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)


class Conv3DReparameterization(_ConvReparameterization):
  """3D convolution layer (e.g. spatial convolution over volumes).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [(Kingma and Welling, 2014)][1], which performs a Monte Carlo
  approximation of the distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Upon being built, this layer adds losses (accessible via the `losses`
  property) representing the divergences of `kernel` and/or `bias` surrogate
  posteriors and their respective priors. When doing minibatch stochastic
  optimization, make sure to scale this loss such that it is applied just once
  per epoch (e.g. if `kl` is the sum of `losses` for each element of the batch,
  you should pass `kl / num_examples_per_epoch` to your optimizer).

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  model = tf_keras.Sequential([
      tf_keras.layers.Reshape([256, 32, 32, 3]),
      tfp.layers.Convolution3DReparameterization(
          64, kernel_size=5, padding='SAME', activation=tf.nn.relu),
      tf_keras.layers.MaxPooling3D(pool_size=[2, 2, 2],
                                   strides=[2, 2, 2],
                                   padding='SAME'),
      tf_keras.layers.Flatten(),
      tfp.layers.DenseReparameterization(10),
  ])

  logits = model(features)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(model.losses)
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  #### References

  [1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
       _International Conference on Learning Representations_, 2014.
       https://arxiv.org/abs/1312.6114
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      filters,
      kernel_size,
      strides=(1, 1, 1),
      padding='valid',
      data_format='channels_last',
      dilation_rate=(1, 1, 1),
      activation=None,
      activity_regularizer=None,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of 3 integers, specifying the
        depth, height and width of the 3D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 3 integers,
        specifying the strides of the convolution along the depth,
        height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, depth,
        height, width, channels)` while `channels_first` corresponds to inputs
        with shape `(batch, channels, depth, height, width)`.
      dilation_rate: An integer or tuple/list of 3 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      ${args}
    """
    # pylint: enable=g-doc-args
    super(Conv3DReparameterization, self).__init__(
        rank=3,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=tf_keras.activations.get(activation),
        activity_regularizer=activity_regularizer,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)


class _ConvFlipout(_ConvVariational):
  """Abstract nD convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [(Wen et al., 2018)][1], which performs a Monte Carlo approximation
  of the distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the reparameterization
  estimator but has the advantage of significantly lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  #### References

  [1]: Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout:
       Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. In
       _International Conference on Learning Representations_, 2018.
       https://arxiv.org/abs/1803.04386
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      rank,
      filters,
      kernel_size,
      strides=1,
      padding='valid',
      data_format='channels_last',
      dilation_rate=1,
      activation=None,
      activity_regularizer=None,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      seed=None,
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, ...,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      ${args}
      seed: Python scalar `int` which initializes the random number
        generator. Default value: `None` (i.e., use global seed).
    """
    # pylint: enable=g-doc-args
    super(_ConvFlipout, self).__init__(
        rank=rank,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=tf_keras.activations.get(activation),
        activity_regularizer=activity_regularizer,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    # Set additional attributes which do not exist in the parent class.
    self.seed = seed

  def _apply_variational_kernel(self, inputs):
    if (not isinstance(self.kernel_posterior, independent_lib.Independent) or
        not isinstance(self.kernel_posterior.distribution, normal_lib.Normal)):
      raise TypeError(
          '`{}` requires '
          '`kernel_posterior_fn` produce an instance of '
          '`tfd.Independent(tfd.Normal)` '
          '(saw: \"{}\").'.format(
              type(self).__name__, self.kernel_posterior.name))
    self.kernel_posterior_affine = normal_lib.Normal(
        loc=tf.zeros_like(self.kernel_posterior.distribution.loc),
        scale=self.kernel_posterior.distribution.scale)
    self.kernel_posterior_affine_tensor = (
        self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
    self.kernel_posterior_tensor = None

    outputs = self._convolution_op(
        inputs, self.kernel_posterior.distribution.loc)

    input_shape = tf.shape(inputs)
    batch_shape = tf.expand_dims(input_shape[0], 0)
    if self.data_format == 'channels_first':
      channels = input_shape[1]
    else:
      channels = input_shape[-1]

    seed_stream = SeedStream(self.seed, salt='ConvFlipout')

    sign_input = tfp_random.rademacher(
        tf.concat([batch_shape,
                   tf.expand_dims(channels, 0)], 0),
        dtype=inputs.dtype,
        seed=seed_stream())
    sign_output = tfp_random.rademacher(
        tf.concat([batch_shape,
                   tf.expand_dims(self.filters, 0)], 0),
        dtype=inputs.dtype,
        seed=seed_stream())

    if self.data_format == 'channels_first':
      for _ in range(self.rank):
        sign_input = tf.expand_dims(sign_input, -1)  # 2D ex: (B, C, 1, 1)
        sign_output = tf.expand_dims(sign_output, -1)
    else:
      for _ in range(self.rank):
        sign_input = tf.expand_dims(sign_input, 1)  # 2D ex: (B, 1, 1, C)
        sign_output = tf.expand_dims(sign_output, 1)

    perturbed_inputs = self._convolution_op(
        inputs * sign_input, self.kernel_posterior_affine_tensor) * sign_output

    outputs += perturbed_inputs
    return outputs

  def get_config(self):
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable) containing the
    configuration of a layer. The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    Returns:
      config: A Python dictionary of class keyword arguments and their
        serialized values.
    """
    config = {
        'seed': self.seed,
    }
    base_config = super(_ConvFlipout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Conv1DFlipout(_ConvFlipout):
  """1D convolution layer (e.g. temporal convolution) with Flipout.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [(Wen et al., 2018)][1], which performs a Monte Carlo approximation
  of the distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the reparameterization
  estimator but has the advantage of significantly lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Upon being built, this layer adds losses (accessible via the `losses`
  property) representing the divergences of `kernel` and/or `bias` surrogate
  posteriors and their respective priors. When doing minibatch stochastic
  optimization, make sure to scale this loss such that it is applied just once
  per epoch (e.g. if `kl` is the sum of `losses` for each element of the batch,
  you should pass `kl / num_examples_per_epoch` to your optimizer).

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  model = tf_keras.Sequential([
      tf_keras.layers.Reshape([128, 1]),
      tfp.layers.Convolution1DFlipout(
          64, kernel_size=5, padding='SAME', activation=tf.nn.relu),
      tf_keras.layers.Flatten(),
      tfp.layers.DenseFlipout(10),
  ])

  logits = model(features)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(model.losses)
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses the Flipout gradient estimator to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  #### References

  [1]: Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout:
       Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. In
       _International Conference on Learning Representations_, 2018.
       https://arxiv.org/abs/1803.04386
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      filters,
      kernel_size,
      strides=1,
      padding='valid',
      data_format='channels_last',
      dilation_rate=1,
      activation=None,
      activity_regularizer=None,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      seed=None,
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer, specifying the
        length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, length,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, channels, length)`.
      dilation_rate: An integer or tuple/list of a single integer, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      ${args}
      seed: Python scalar `int` which initializes the random number
        generator. Default value: `None` (i.e., use global seed).
    """
    # pylint: enable=g-doc-args
    super(Conv1DFlipout, self).__init__(
        rank=1,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=tf_keras.activations.get(activation),
        activity_regularizer=activity_regularizer,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        seed=seed,
        **kwargs)


class Conv2DFlipout(_ConvFlipout):
  """2D convolution layer (e.g. spatial convolution over images) with Flipout.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [(Wen et al., 2018)][1], which performs a Monte Carlo approximation
  of the distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the reparameterization
  estimator but has the advantage of significantly lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Upon being built, this layer adds losses (accessible via the `losses`
  property) representing the divergences of `kernel` and/or `bias` surrogate
  posteriors and their respective priors. When doing minibatch stochastic
  optimization, make sure to scale this loss such that it is applied just once
  per epoch (e.g. if `kl` is the sum of `losses` for each element of the batch,
  you should pass `kl / num_examples_per_epoch` to your optimizer).

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  model = tf_keras.Sequential([
      tf_keras.layers.Reshape([32, 32, 3]),
      tfp.layers.Convolution2DFlipout(
          64, kernel_size=5, padding='SAME', activation=tf.nn.relu),
      tf_keras.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=[2, 2],
                                   padding='SAME'),
      tf_keras.layers.Flatten(),
      tfp.layers.DenseFlipout(10),
  ])

  logits = model(features)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(model.losses)
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses the Flipout gradient estimator to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  #### References

  [1]: Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout:
       Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. In
       _International Conference on Learning Representations_, 2018.
       https://arxiv.org/abs/1803.04386
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      filters,
      kernel_size,
      strides=(1, 1),
      padding='valid',
      data_format='channels_last',
      dilation_rate=(1, 1),
      activation=None,
      activity_regularizer=None,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      seed=None,
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, height,
        width, channels)` while `channels_first` corresponds to inputs with
        shape `(batch, channels, height, width)`.
      dilation_rate: An integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      ${args}
      seed: Python scalar `int` which initializes the random number
        generator. Default value: `None` (i.e., use global seed).
    """
    # pylint: enable=g-doc-args
    super(Conv2DFlipout, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=tf_keras.activations.get(activation),
        activity_regularizer=activity_regularizer,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        seed=seed,
        **kwargs)


class Conv3DFlipout(_ConvFlipout):
  """3D convolution layer (e.g. spatial convolution over volumes) with Flipout.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [(Wen et al., 2018)][1], which performs a Monte Carlo approximation
  of the distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the reparameterization
  estimator but has the advantage of significantly lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Upon being built, this layer adds losses (accessible via the `losses`
  property) representing the divergences of `kernel` and/or `bias` surrogate
  posteriors and their respective priors. When doing minibatch stochastic
  optimization, make sure to scale this loss such that it is applied just once
  per epoch (e.g. if `kl` is the sum of `losses` for each element of the batch,
  you should pass `kl / num_examples_per_epoch` to your optimizer).

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  model = tf_keras.Sequential([
      tf_keras.layers.Reshape([256, 32, 32, 3]),
      tfp.layers.Convolution3DFlipout(
          64, kernel_size=5, padding='SAME', activation=tf.nn.relu),
      tf_keras.layers.MaxPooling3D(pool_size=[2, 2, 2],
                                   strides=[2, 2, 2],
                                   padding='SAME'),
      tf_keras.layers.Flatten(),
      tfp.layers.DenseFlipout(10),
  ])

  logits = model(features)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(model.losses)
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses the Flipout gradient estimator to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  #### References

  [1]: Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout:
       Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. In
       _International Conference on Learning Representations_, 2018.
       https://arxiv.org/abs/1803.04386
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      filters,
      kernel_size,
      strides=(1, 1, 1),
      padding='valid',
      data_format='channels_last',
      dilation_rate=(1, 1, 1),
      activation=None,
      activity_regularizer=None,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      seed=None,
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of 3 integers, specifying the
        depth, height and width of the 3D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 3 integers,
        specifying the strides of the convolution along the depth,
        height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, depth,
        height, width, channels)` while `channels_first` corresponds to inputs
        with shape `(batch, channels, depth, height, width)`.
      dilation_rate: An integer or tuple/list of 3 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      ${args}
      seed: Python scalar `int` which initializes the random number
        generator. Default value: `None` (i.e., use global seed).
    """
    # pylint: enable=g-doc-args
    super(Conv3DFlipout, self).__init__(
        rank=3,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=tf_keras.activations.get(activation),
        activity_regularizer=activity_regularizer,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        seed=seed,
        **kwargs)


# Aliases

Convolution1DReparameterization = Conv1DReparameterization
Convolution2DReparameterization = Conv2DReparameterization
Convolution3DReparameterization = Conv3DReparameterization
Convolution1DFlipout = Conv1DFlipout
Convolution2DFlipout = Conv2DFlipout
Convolution3DFlipout = Conv3DFlipout


def convert_data_format(data_format, ndim):  # pylint: disable=missing-function-docstring
  if data_format == 'channels_last':
    if ndim == 3:
      return 'NWC'
    elif ndim == 4:
      return 'NHWC'
    elif ndim == 5:
      return 'NDHWC'
    else:
      raise ValueError(f'Input rank: {ndim} not supported. We only support '
                       'input rank 3, 4 or 5.')
  elif data_format == 'channels_first':
    if ndim == 3:
      return 'NCW'
    elif ndim == 4:
      return 'NCHW'
    elif ndim == 5:
      return 'NCDHW'
    else:
      raise ValueError(f'Input rank: {ndim} not supported. We only support '
                       'input rank 3, 4 or 5.')
  else:
    raise ValueError(f'Invalid data_format: {data_format}. We only support '
                     '"channels_first" or "channels_last"')


def normalize_tuple(value, n, name):
  """Transforms a single integer or iterable of integers into an integer tuple.

  Args:
    value: The value to validate and convert. Could an int, or any iterable
      of ints.
    n: The size of the tuple to be returned.
    name: The name of the argument being validated, e.g. "strides" or
      "kernel_size". This is only used to format error messages.

  Returns:
    A tuple of n integers.

  Raises:
    ValueError: If something else than an int/long or iterable thereof was
      passed.
  """
  if isinstance(value, int):
    return (value,) * n
  else:
    try:
      value_tuple = tuple(value)
    except TypeError:
      raise ValueError(f'Argument `{name}` must be a tuple of {str(n)} '
                       f'integers. Received: {str(value)}') from None
    if len(value_tuple) != n:
      raise ValueError(f'Argument `{name}` must be a tuple of {str(n)} '
                       f'integers. Received: {str(value)}')
    for single_value in value_tuple:
      try:
        int(single_value)
      except (ValueError, TypeError):
        raise ValueError(f'Argument `{name}` must be a tuple of {str(n)} '
                         f'integers. Received: {str(value)} including element '
                         f'{str(single_value)} of type '
                         f'{str(type(single_value))}') from None
    return value_tuple


def normalize_data_format(value):
  data_format = value.lower()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('The `data_format` argument must be one of '
                     '"channels_first", "channels_last". Received: '
                     f'{str(value)}.')
  return data_format


def normalize_padding(value):
  padding = value.lower()
  if padding not in {'valid', 'same'}:
    raise ValueError('The `padding` argument must be one of "valid", "same". '
                     f'Received: {str(padding)}.')
  return padding


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """Determines output length of a convolution given input length.

  Args:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding == 'same':
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  else:
    raise ValueError(f'Invalid padding: {padding}')
  return (output_length + stride - 1) // stride
