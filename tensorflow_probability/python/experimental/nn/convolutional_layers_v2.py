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
"""Convolution layers for building neural networks."""
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.experimental.nn import layers as layers_lib
from tensorflow_probability.python.experimental.nn import variational_base as vi_lib
from tensorflow_probability.python.experimental.nn.util import convolution_util
from tensorflow_probability.python.experimental.nn.util import kernel_bias as kernel_bias_lib
from tensorflow_probability.python.internal import prefer_static as ps


__all__ = [
    'ConvolutionV2',
    'ConvolutionVariationalFlipoutV2',
    'ConvolutionVariationalReparameterizationV2',
]


# The following aliases ensure docstrings read more succinctly.
tfd = distribution_lib
unpack_kernel_and_bias = vi_lib.unpack_kernel_and_bias


class ConvolutionV2(layers_lib.KernelBiasLayer):
  """Convolution layer.

  This layer creates a Convolution kernel that is convolved (actually
  cross-correlated) with the layer input to produce a tensor of outputs.

  This V2 version supports alternative batch semantics. V1 layers, with batch
  size `B` produced outputs of shape `[N, H, W, B, C]` (where N, H, W, C are
  minibatch size, height, width and number of channels, as usual). V2 layers
  reorder these to `[N, B, H, W, C]`.

  This layer has two learnable parameters, `kernel` and `bias`.
  - The `kernel` (aka `filters` argument of `tf.nn.convolution`) is a
    `tf.Variable` with `rank + 2` `ndims` and shape given by
    `concat([filter_shape, [input_size, output_size]], axis=0)`. Argument
    `filter_shape` is either a  length-`rank` vector or expanded as one, i.e.,
    `filter_size * tf.ones(rank)` when `filter_shape` is an `int` (which we
    denote as `filter_size`).
  - The `bias` is a `tf.Variable` with `1` `ndims` and shape `[output_size]`.

  In summary, the shape of learnable parameters is governed by the following
  arguments: `filter_shape`, `input_size`, `output_size` and possibly `rank` (if
  `filter_shape` needs expansion).

  For more information on convolution layers, we recommend the following:
  - [Deconvolution Checkerboard][https://distill.pub/2016/deconv-checkerboard/]
  - [Convolution Animations][https://github.com/vdumoulin/conv_arithmetic]
  - [What are Deconvolutional Layers?][
    https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers]

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfb = tfp.bijectors
  tfd = tfp.distributions
  tfn = tfp.experimental.nn

  Convolution1DV2 = functools.partial(tfn.ConvolutionV2, rank=1)
  Convolution2DV2 = tfn.ConvolutionV2
  Convolution3DV2 = functools.partial(tfn.ConvolutionV2, rank=3)
  ```

  """

  def __init__(
      self,
      input_size,
      output_size,          # keras::Conv::filters
      # Conv specific.
      filter_shape,         # keras::Conv::kernel_size
      rank=2,               # keras::Conv::rank
      strides=1,            # keras::Conv::strides
      padding='VALID',      # keras::Conv::padding; 'CAUSAL' not implemented.
                            # keras::Conv::data_format is not implemented
      dilations=1,          # keras::Conv::dilation_rate
      # Weights
      kernel_initializer=None,  # tfp.nn.initializers.glorot_uniform()
      bias_initializer=None,    # tf_keras.initializers.zeros()
      make_kernel_bias_fn=kernel_bias_lib.make_kernel_bias,
      dtype=tf.float32,
      index_dtype=tf.int32,
      batch_shape=(),
      # Misc
      activation_fn=None,
      validate_args=False,
      name=None):
    """Constructs layer.

    Note: `data_format` is not supported since all nn layers operate on
    the rightmost column. If your channel dimension is not rightmost, use
    `tf.transpose` before calling this layer. For example, if your channel
    dimension is second from the left, the following code will move it
    rightmost:

    ```python
    inputs = tf.transpose(inputs, tf.concat([
        [0], tf.range(2, tf.rank(inputs)), [1]], axis=0))
    ```

    Args:
      input_size: ...
        In Keras, this argument is inferred from the rightmost input shape,
        i.e., `tf.shape(inputs)[-1]`. This argument specifies the size of the
        second from the rightmost dimension of both `inputs` and `kernel`.
        Default value: `None`.
      output_size: ...
        In Keras, this argument is called `filters`. This argument specifies the
        rightmost dimension size of both `kernel` and `bias`.
      filter_shape: ...
        In Keras, this argument is called `kernel_size`. This argument specifies
        the leftmost `rank` dimensions' sizes of `kernel`.
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution. This argument implies the number of `kernel` dimensions,
        i.e., `kernel.shape.rank == rank + 2`.
        In Keras, this argument has the same name and semantics.
        Default value: `2`.
      strides: An integer or tuple/list of n integers, specifying the stride
        length of the convolution.
        In Keras, this argument has the same name and semantics.
        Default value: `1`.
      padding: One of `"VALID"` or `"SAME"` (case-insensitive).
        In Keras, this argument has the same name and semantics (except we don't
        support `"CAUSAL"`).
        Default value: `'VALID'`.
      dilations: An integer or tuple/list of `rank` integers, specifying the
        dilation rate to use for dilated convolution. Currently, specifying any
        `dilations` value != 1 is incompatible with specifying any `strides`
        value != 1.
        In Keras, this argument is called `dilation_rate`.
        Default value: `1`.
      kernel_initializer: ...
        Default value: `None` (i.e.,
        `tfp.experimental.nn.initializers.glorot_uniform()`).
      bias_initializer: ...
        Default value: `None` (i.e., `tf_keras.initializers.zeros()`).
      make_kernel_bias_fn: ...
        Default value: `tfp.experimental.nn.util.make_kernel_bias`.
      dtype: ...
        Default value: `tf.float32`.
      index_dtype: ...
      batch_shape: ...
        Default value: `()`.
      activation_fn: ...
        Default value: `None`.
      validate_args: ...
      name: ...
        Default value: `None` (i.e., `'ConvolutionV2'`).
    """
    filter_shape = convolution_util.prepare_tuple_argument(
        filter_shape, rank, arg_name='filter_shape',
        validate_args=validate_args)
    batch_shape = (tf.constant([], dtype=tf.int32) if batch_shape is None
                   else ps.cast(ps.reshape(batch_shape, shape=[-1]), tf.int32))
    batch_ndims = ps.size(batch_shape)

    apply_kernel_fn = convolution_util.make_convolution_fn(
        filter_shape, rank=2, strides=strides, padding=padding,
        dilations=dilations, dtype=index_dtype, validate_args=validate_args)

    kernel_shape = ps.concat(
        [batch_shape, [ps.reduce_prod(filter_shape) * input_size, output_size]],
        axis=0)
    bias_shape = ps.concat(
        [batch_shape, tf.ones(rank), [output_size]], axis=0)
    kernel, bias = make_kernel_bias_fn(
        kernel_shape, bias_shape,
        kernel_initializer, bias_initializer,
        batch_ndims, batch_ndims,
        dtype)

    self._make_kernel_bias_fn = make_kernel_bias_fn  # For tracking.
    super(ConvolutionV2, self).__init__(
        kernel=kernel,
        bias=bias,
        apply_kernel_fn=apply_kernel_fn,
        dtype=dtype,
        activation_fn=activation_fn,
        validate_args=validate_args,
        name=name)


class ConvolutionVariationalReparameterizationV2(
    vi_lib.VariationalReparameterizationKernelBiasLayer):
  """Convolution layer class with reparameterization estimator.

  This layer implements the Bayesian variational inference analogue to
  a Convolution layer by assuming the `kernel` and/or the `bias` are
  drawn from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = matmul(inputs, kernel) + bias
  ```

  It uses the reparameterization estimator [(Kingma and Welling, 2014)][1],
  which performs a Monte Carlo approximation of the distribution integrating
  over the `kernel` and `bias`.

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

  This V2 version supports alternative batch semantics. V1 layers, with batch
  size `B` produced outputs of shape `[N, H, W, B, C]` (where N, H, W, C are
  minibatch size, height, width and number of channels, as usual). V2 layers
  reorder these to `[N, B, H, W, C]`.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of images and length-10 one-hot `targets`.

  ```python
  import functools
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  import tensorflow_datasets as tfds
  tfb = tfp.bijectors
  tfd = tfp.distributions
  tfn = tfp.experimental.nn

  # 1  Prepare Dataset

  [train_dataset, eval_dataset], datasets_info = tfds.load(
      name='mnist',
      split=['train', 'test'],
      with_info=True,
      as_supervised=True,
      shuffle_files=True)
  def _preprocess(image, label):
    # image = image < tf.random.uniform(tf.shape(image))   # Randomly binarize.
    image = tf.cast(image, tf.float32) / 255.  # Scale to unit interval.
    lo = 0.001
    image = (1. - 2. * lo) * image + lo  # Rescale to *open* unit interval.
    return image, label
  batch_size = 32
  train_size = datasets_info.splits['train'].num_examples
  train_dataset = tfn.util.tune_dataset(
      train_dataset,
      batch_shape=(batch_size,),
      shuffle_size=int(train_size / 7),
      preprocess_fn=_preprocess)
  train_iter = iter(train_dataset)
  eval_iter = iter(eval_dataset)
  x, y = next(train_iter)
  evidence_shape = x.shape[1:]
  targets_shape = y.shape[1:]

  # 2  Specify Model

  n = tf.cast(train_size, tf.float32)

  BayesConv2D = functools.partial(
      tfn.ConvolutionVariationalReparameterizationV2,
      rank=2,
      padding='same',
      filter_shape=5,
      # Use `he_uniform` because we'll use the `relu` family.
      kernel_initializer=tf_keras.initializers.he_uniform(),
      penalty_weight=1. / n)

  BayesAffine = functools.partial(
      tfn.AffineVariationalReparameterization,
      penalty_weight=1. / n)

  scale = tfp.util.TransformedVariable(1., tfb.Softplus())
  bnn = tfn.Sequential([
      BayesConv2D(evidence_shape[-1], 32, filter_shape=7, strides=2,
                  activation_fn=tf.nn.leaky_relu),           # [b, 14, 14, 32]
      tfn.util.flatten_rightmost(ndims=3),                   # [b, 14 * 14 * 32]
      BayesAffine(14 * 14 * 32, np.prod(target_shape) - 1),  # [b, 9]
      lambda loc: tfb.SoftmaxCentered()(
          tfd.Independent(tfd.Normal(loc, scale),
                          reinterpreted_batch_ndims=1)),  # [b, 10]
  ], name='bayesian_neural_network')

  print(bnn.summary())

  # 3  Train.

  def loss_fn():
    x, y = next(train_iter)
    nll = -tf.reduce_mean(bnn(x).log_prob(y), axis=-1)
    kl = bnn.extra_loss  # Already normalized via `penalty_weight` arg.
    loss = nll + kl
    return loss, (nll, kl)
  opt = tf_keras.optimizers.Adam()
  fit_op = tfn.util.make_fit_op(loss_fn, opt, bnn.trainable_variables)
  for _ in range(200):
    loss, (nll, kl), g = fit_op()
  ```

  This example uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the negative
  Evidence Lower Bound. It consists of the sum of two terms: the expected
  negative log-likelihood, which we approximate via Monte Carlo; and the KL
  divergence, which is added via regularizer terms which are arguments to the
  layer.

  #### References

  [1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
       _International Conference on Learning Representations_, 2014.
       https://arxiv.org/abs/1312.6114
  """

  def __init__(
      self,
      input_size,
      output_size,          # keras::Conv::filters
      # Conv specific.
      filter_shape,         # keras::Conv::kernel_size
      rank=2,               # keras::Conv::rank
      strides=1,            # keras::Conv::strides
      padding='VALID',      # keras::Conv::padding; 'CAUSAL' not implemented.
                            # keras::Conv::data_format is not implemented
      dilations=1,          # keras::Conv::dilation_rate
      # Weights
      kernel_initializer=None,  # tfp.nn.initializers.glorot_uniform()
      bias_initializer=None,    # tf_keras.initializers.zeros()
      make_posterior_fn=kernel_bias_lib.make_kernel_bias_posterior_mvn_diag,
      make_prior_fn=kernel_bias_lib.make_kernel_bias_prior_spike_and_slab,
      posterior_value_fn=tfd.Distribution.sample,
      unpack_weights_fn=unpack_kernel_and_bias,
      dtype=tf.float32,
      index_dtype=tf.int32,
      # Misc
      activation_fn=None,
      seed=None,
      validate_args=False,
      name=None):
    """Constructs layer.

    Note: `data_format` is not supported since all nn layers operate on
    the rightmost column. If your channel dimension is not rightmost, use
    `tf.transpose` before calling this layer. For example, if your channel
    dimension is second from the left, the following code will move it
    rightmost:

    ```python
    inputs = tf.transpose(inputs, tf.concat([
        [0], tf.range(2, tf.rank(inputs)), [1]], axis=0))
    ```

    Args:
      input_size: ...
        In Keras, this argument is inferred from the rightmost input shape,
        i.e., `tf.shape(inputs)[-1]`. This argument specifies the size of the
        second from the rightmost dimension of both `inputs` and `kernel`.
        Default value: `None`.
      output_size: ...
        In Keras, this argument is called `filters`. This argument specifies the
        rightmost dimension size of both `kernel` and `bias`.
      filter_shape: ...
        In Keras, this argument is called `kernel_size`. This argument specifies
        the leftmost `rank` dimensions' sizes of `kernel`.
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution. This argument implies the number of `kernel` dimensions,
        i.e., `kernel.shape.rank == rank + 2`.
        In Keras, this argument has the same name and semantics.
        Default value: `2`.
      strides: An integer or tuple/list of n integers, specifying the stride
        length of the convolution.
        In Keras, this argument has the same name and semantics.
        Default value: `1`.
      padding: One of `"VALID"` or `"SAME"` (case-insensitive).
        In Keras, this argument has the same name and semantics (except we don't
        support `"CAUSAL"`).
        Default value: `'VALID'`.
      dilations: An integer or tuple/list of `rank` integers, specifying the
        dilation rate to use for dilated convolution. Currently, specifying any
        `dilations` value != 1 is incompatible with specifying any `strides`
        value != 1.
        In Keras, this argument is called `dilation_rate`.
        Default value: `1`.
      kernel_initializer: ...
        Default value: `None` (i.e.,
        `tfp.experimental.nn.initializers.glorot_uniform()`).
      bias_initializer: ...
        Default value: `None` (i.e., `tf_keras.initializers.zeros()`).
      make_posterior_fn: ...
        Default value:
          `tfp.experimental.nn.util.make_kernel_bias_posterior_mvn_diag`.
      make_prior_fn: ...
        Default value:
          `tfp.experimental.nn.util.make_kernel_bias_prior_spike_and_slab`.
      posterior_value_fn: ...
        Default valye: `tfd.Distribution.sample`
      unpack_weights_fn:
        Default value: `unpack_kernel_and_bias`
      dtype: ...
        Default value: `tf.float32`.
      index_dtype: ...
      activation_fn: ...
        Default value: `None`.
      seed: ...
        Default value: `None` (i.e., no seed).
      validate_args: ...
      name: ...
        Default value: `None` (i.e.,
        `'ConvolutionVariationalReparameterizationV2'`).
    """
    filter_shape = convolution_util.prepare_tuple_argument(
        filter_shape, rank, arg_name='filter_shape',
        validate_args=validate_args)
    kernel_shape = ps.convert_to_shape_tensor(
        [ps.reduce_prod(filter_shape) * input_size, output_size])
    self._make_posterior_fn = make_posterior_fn  # For variable tracking.
    self._make_prior_fn = make_prior_fn  # For variable tracking.
    batch_ndims = 0

    apply_kernel_fn = convolution_util.make_convolution_fn(
        filter_shape, rank=2, strides=strides, padding=padding,
        dilations=dilations, dtype=index_dtype,
        validate_args=validate_args)
    super(ConvolutionVariationalReparameterizationV2, self).__init__(
        posterior=make_posterior_fn(
            kernel_shape, [output_size],
            kernel_initializer, bias_initializer,
            batch_ndims, batch_ndims,
            dtype),
        prior=make_prior_fn(
            kernel_shape, [output_size],
            kernel_initializer, bias_initializer,
            batch_ndims, batch_ndims,
            dtype),
        apply_kernel_fn=apply_kernel_fn,
        posterior_value_fn=posterior_value_fn,
        unpack_weights_fn=unpack_weights_fn,
        dtype=dtype,
        activation_fn=activation_fn,
        seed=seed,
        validate_args=validate_args,
        name=name)


class ConvolutionVariationalFlipoutV2(vi_lib.VariationalFlipoutKernelBiasLayer):
  """Convolution layer class with Flipout estimator.

  This layer implements the Bayesian variational inference analogue to
  a Convolution layer by assuming the `kernel` and/or the `bias` are
  drawn from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = tf.nn.convolution(inputs, kernel) + bias
  ```

  It uses the Flipout estimator [(Wen et al., 2018)][1], which performs a Monte
  Carlo approximation of the distribution integrating over the `kernel` and
  `bias`. Flipout uses roughly twice as many floating point operations as the
  reparameterization estimator but has the advantage of significantly lower
  variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Upon being built, this layer adds losses (accessible via the `losses`
  property) representing the divergences of `kernel` and/or `bias` surrogate
  posteriors and their respective priors. When doing minibatch stochastic
  optimization, make sure to scale this loss such that it is applied just once
  per epoch (e.g. if `kl` is the sum of `losses` for each element of the batch,
  you should pass `kl / num_examples_per_epoch` to your optimizer).

  ```python
  inputs = tf.transpose(inputs, tf.concat([
      [0], tf.range(2, tf.rank(inputs)), [1]], axis=0))
  ```

  This V2 version supports alternative batch semantics. V1 layers, with batch
  size `B` produced outputs of shape `[N, H, W, B, C]` (where N, H, W, C are
  minibatch size, height, width and number of channels, as usual). V2 layers
  reorder these to `[N, B, H, W, C]`.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of images and length-10 one-hot `targets`.

  ```python
  # Using the following substitution, see:
  tfn = tfp.experimental.nn
  help(tfn.ConvolutionVariationalReparameterizationV2)
  BayesConv2D = functools.partial(
      tfn.ConvolutionVariationalFlipoutV2,
      penalty_weight=1. / n)
  ```

  This example uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the negative
  Evidence Lower Bound. It consists of the sum of two terms: the expected
  negative log-likelihood, which we approximate via Monte Carlo; and the KL
  divergence, which is added via regularizer terms which are arguments to the
  layer.

  #### References

  [1]: Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout:
       Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. In
       _International Conference on Learning Representations_, 2018.
       https://arxiv.org/abs/1803.04386
  """

  def __init__(
      self,
      input_size,
      output_size,          # keras::Conv::filters
      # Conv specific.
      filter_shape,         # keras::Conv::kernel_size
      rank=2,               # keras::Conv::rank
      strides=1,            # keras::Conv::strides
      padding='VALID',      # keras::Conv::padding; 'CAUSAL' not implemented.
                            # keras::Conv::data_format is not implemented
      dilations=1,          # keras::Conv::dilation_rate
      # Weights
      kernel_initializer=None,  # tfp.nn.initializers.glorot_uniform()
      bias_initializer=None,    # tf_keras.initializers.zeros()
      make_posterior_fn=kernel_bias_lib.make_kernel_bias_posterior_mvn_diag,
      make_prior_fn=kernel_bias_lib.make_kernel_bias_prior_spike_and_slab,
      posterior_value_fn=tfd.Distribution.sample,
      unpack_weights_fn=unpack_kernel_and_bias,
      dtype=tf.float32,
      index_dtype=tf.int32,
      # Misc
      activation_fn=None,
      seed=None,
      validate_args=False,
      name=None):
    """Constructs layer.

    Note: `data_format` is not supported since all nn layers operate on
    the rightmost column. If your channel dimension is not rightmost, use
    `tf.transpose` before calling this layer. For example, if your channel
    dimension is second from the left, the following code will move it
    rightmost:

    ```python
    inputs = tf.transpose(inputs, tf.concat([
        [0], tf.range(2, tf.rank(inputs)), [1]], axis=0))
    ```

    Args:
      input_size: ...
        In Keras, this argument is inferred from the rightmost input shape,
        i.e., `tf.shape(inputs)[-1]`. This argument specifies the size of the
        second from the rightmost dimension of both `inputs` and `kernel`.
        Default value: `None`.
      output_size: ...
        In Keras, this argument is called `filters`. This argument specifies the
        rightmost dimension size of both `kernel` and `bias`.
      filter_shape: ...
        In Keras, this argument is called `kernel_size`. This argument specifies
        the leftmost `rank` dimensions' sizes of `kernel`.
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution. This argument implies the number of `kernel` dimensions,
        i.e., `kernel.shape.rank == rank + 2`.
        In Keras, this argument has the same name and semantics.
        Default value: `2`.
      strides: An integer or tuple/list of n integers, specifying the stride
        length of the convolution.
        In Keras, this argument has the same name and semantics.
        Default value: `1`.
      padding: One of `"VALID"` or `"SAME"` (case-insensitive).
        In Keras, this argument has the same name and semantics (except we don't
        support `"CAUSAL"`).
        Default value: `'VALID'`.
      dilations: An integer or tuple/list of `rank` integers, specifying the
        dilation rate to use for dilated convolution. Currently, specifying any
        `dilations` value != 1 is incompatible with specifying any `strides`
        value != 1.
        In Keras, this argument is called `dilation_rate`.
        Default value: `1`.
      kernel_initializer: ...
        Default value: `None` (i.e.,
        `tfp.experimental.nn.initializers.glorot_uniform()`).
      bias_initializer: ...
        Default value: `None` (i.e., `tf_keras.initializers.zeros()`).
      make_posterior_fn: ...
        Default value:
          `tfp.experimental.nn.util.make_kernel_bias_posterior_mvn_diag`.
      make_prior_fn: ...
        Default value:
          `tfp.experimental.nn.util.make_kernel_bias_prior_spike_and_slab`.
      posterior_value_fn: ...
        Default valye: `tfd.Distribution.sample`
      unpack_weights_fn:
        Default value: `unpack_kernel_and_bias`
      dtype: ...
        Default value: `tf.float32`.
      index_dtype: ...
      activation_fn: ...
        Default value: `None`.
      seed: ...
        Default value: `None` (i.e., no seed).
      validate_args: ...
      name: ...
        Default value: `None` (i.e.,
        `'ConvolutionVariationalFlipoutV2'`).
    """

    filter_shape = convolution_util.prepare_tuple_argument(
        filter_shape, rank, arg_name='filter_shape',
        validate_args=validate_args)

    kernel_shape = ps.convert_to_shape_tensor(
        [ps.reduce_prod(filter_shape) * input_size, output_size])
    self._make_posterior_fn = make_posterior_fn  # For variable tracking.
    self._make_prior_fn = make_prior_fn  # For variable tracking.
    batch_ndims = 0

    apply_kernel_fn = convolution_util.make_convolution_fn(
        filter_shape, rank=2, strides=strides, padding=padding,
        dilations=dilations, dtype=index_dtype,
        validate_args=validate_args)
    super(ConvolutionVariationalFlipoutV2, self).__init__(
        posterior=make_posterior_fn(
            kernel_shape, [output_size],
            kernel_initializer, bias_initializer,
            batch_ndims, batch_ndims,
            dtype),
        prior=make_prior_fn(
            kernel_shape, [output_size],
            kernel_initializer, bias_initializer,
            batch_ndims, batch_ndims,
            dtype),
        apply_kernel_fn=apply_kernel_fn,
        posterior_value_fn=posterior_value_fn,
        unpack_weights_fn=unpack_weights_fn,
        dtype=dtype,
        activation_fn=activation_fn,
        seed=seed,
        validate_args=validate_args,
        name=name)
