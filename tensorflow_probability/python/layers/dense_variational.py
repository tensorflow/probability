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
"""Dense variational layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import util as tfp_layers_util
from tensorflow_probability.python.math import random_rademacher
from tensorflow_probability.python.util import docstring as docstring_util


__all__ = [
    'DenseFlipout',
    'DenseLocalReparameterization',
    'DenseReparameterization',
]


doc_args = """units: Integer or Long, dimensionality of the output space.
  activation: Activation function (`callable`). Set it to None to maintain a
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


class _DenseVariational(tf.keras.layers.Layer):
  """Abstract densely-connected class (private, used as implementation base).

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      units,
      activation=None,
      activity_regularizer=None,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(_DenseVariational, self).__init__(
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.units = units
    self.activation = tf.keras.activations.get(activation)
    self.input_spec = tf.layers.InputSpec(min_ndim=2)
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
    in_size = input_shape.with_rank_at_least(2)[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self._input_spec = tf.layers.InputSpec(min_ndim=2, axes={-1: in_size})

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [in_size, self.units], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [in_size, self.units], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)

    outputs = self._apply_variational_kernel(inputs)
    outputs = self._apply_variational_bias(outputs)
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable
    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True
    return outputs

  def compute_output_shape(self, input_shape):
    """Computes the output shape of the layer.

    Args:
      input_shape: Shape tuple (tuple of integers) or list of shape tuples
        (one per output tensor of the layer). Shape tuples can include None for
        free dimensions, instead of an integer.

    Returns:
      output_shape: A tuple representing the output shape.

    Raises:
      ValueError: If innermost dimension of `input_shape` is not defined.
    """
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

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
        'units': self.units,
        'activation': (tf.keras.activations.serialize(self.activation)
                       if self.activation else None),
        'activity_regularizer':
            tf.keras.initializers.serialize(self.activity_regularizer),
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
    base_config = super(_DenseVariational, self).get_config()
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
    return tf.nn.bias_add(inputs, self.bias_posterior_tensor)

  def _apply_divergence(self, divergence_fn, posterior, prior,
                        posterior_tensor, name):
    if (divergence_fn is None or
        posterior is None or
        prior is None):
      divergence = None
      return
    divergence = tf.identity(
        divergence_fn(
            posterior, prior, posterior_tensor),
        name=name)
    self.add_loss(divergence)

  def _matmul(self, inputs, kernel):
    if inputs.shape.ndims <= 2:
      return tf.matmul(inputs, kernel)
    # To handle broadcasting, we must use `tensordot`.
    return tf.tensordot(inputs, kernel, axes=[[-1], [0]])


class DenseReparameterization(_DenseVariational):
  """Densely-connected layer class with reparameterization estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  It uses the reparameterization estimator [(Kingma and Welling, 2014)][1],
  which performs a Monte Carlo approximation of the distribution integrating
  over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  model = tf.keras.Sequential([
      tfp.layers.DenseReparameterization(512, activation=tf.nn.relu),
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
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(DenseReparameterization, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
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
    return self._matmul(inputs, self.kernel_posterior_tensor)


class DenseLocalReparameterization(_DenseVariational):
  """Densely-connected layer class with local reparameterization estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  It uses the local reparameterization estimator [(Kingma et al., 2015)][1],
  which performs a Monte Carlo approximation of the distribution on the hidden
  units induced by the `kernel` and `bias`. The default `kernel_posterior_fn`
  is a normal distribution which factorizes across all elements of the weight
  matrix and bias vector. Unlike [1]'s multiplicative parameterization, this
  distribution has trainable location and scale parameters which is known as
  an additive noise parameterization [(Molchanov et al., 2017)][2].

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  import tensorflow_probability as tfp

  model = tf.keras.Sequential([
      tfp.layers.DenseReparameterization(512, activation=tf.nn.relu),
      tfp.layers.DenseReparameterization(10),
  ])

  logits = model(features)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(model.losses)
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses local reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  #### References

  [1]: Diederik Kingma, Tim Salimans, and Max Welling. Variational Dropout and
       the Local Reparameterization Trick. In _Neural Information Processing
       Systems_, 2015. https://arxiv.org/abs/1506.02557
  [2]: Dmitry Molchanov, Arsenii Ashukha, Dmitry Vetrov. Variational Dropout
       Sparsifies Deep Neural Networks. In _International Conference on Machine
       Learning_, 2017. https://arxiv.org/abs/1701.05369
  """

  @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(DenseLocalReparameterization, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
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
    if (not isinstance(self.kernel_posterior, tfd.Independent) or
        not isinstance(self.kernel_posterior.distribution, tfd.Normal)):
      raise TypeError(
          '`DenseLocalReparameterization` requires '
          '`kernel_posterior_fn` produce an instance of '
          '`tfd.Independent(tfd.Normal)` '
          '(saw: \"{}\").'.format(self.kernel_posterior.name))
    self.kernel_posterior_affine = tfd.Normal(
        loc=self._matmul(inputs, self.kernel_posterior.distribution.loc),
        scale=tf.sqrt(self._matmul(
            tf.square(inputs),
            tf.square(self.kernel_posterior.distribution.scale))))
    self.kernel_posterior_affine_tensor = (
        self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
    self.kernel_posterior_tensor = None
    return self.kernel_posterior_affine_tensor


class DenseFlipout(_DenseVariational):
  """Densely-connected layer class with Flipout estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  It uses the Flipout estimator [(Wen et al., 2018)][1], which performs a Monte
  Carlo approximation of the distribution integrating over the `kernel` and
  `bias`. Flipout uses roughly twice as many floating point operations as the
  reparameterization estimator but has the advantage of significantly lower
  variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  import tensorflow_probability as tfp

  model = tf.keras.Sequential([
      tfp.layers.DenseFlipout(512, activation=tf.nn.relu),
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
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      seed=None,
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.

    Args:
      ${args}
      seed: Python scalar `int` which initializes the random number
        generator. Default value: `None` (i.e., use global seed).
    """
    # pylint: enable=g-doc-args
    super(DenseFlipout, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
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
    if (not isinstance(self.kernel_posterior, tfd.Independent) or
        not isinstance(self.kernel_posterior.distribution, tfd.Normal)):
      raise TypeError(
          '`DenseFlipout` requires '
          '`kernel_posterior_fn` produce an instance of '
          '`tfd.Independent(tfd.Normal)` '
          '(saw: \"{}\").'.format(self.kernel_posterior.name))
    self.kernel_posterior_affine = tfd.Normal(
        loc=tf.zeros_like(self.kernel_posterior.distribution.loc),
        scale=self.kernel_posterior.distribution.scale)
    self.kernel_posterior_affine_tensor = (
        self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
    self.kernel_posterior_tensor = None

    input_shape = tf.shape(inputs)
    batch_shape = input_shape[:-1]

    seed_stream = tfd.SeedStream(self.seed, salt='DenseFlipout')

    sign_input = random_rademacher(
        input_shape,
        dtype=inputs.dtype,
        seed=seed_stream())
    sign_output = random_rademacher(
        tf.concat([batch_shape,
                   tf.expand_dims(self.units, 0)], 0),
        dtype=inputs.dtype,
        seed=seed_stream())
    perturbed_inputs = self._matmul(
        inputs * sign_input, self.kernel_posterior_affine_tensor) * sign_output

    outputs = self._matmul(inputs, self.kernel_posterior.distribution.loc)
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
    base_config = super(DenseFlipout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
