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
"""ConvolutionTranspose layers for building neural networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.experimental.nn import layers as layers_lib
from tensorflow_probability.python.experimental.nn import util as nn_util_lib
from tensorflow_probability.python.experimental.nn import variational_base as vi_lib
from tensorflow_probability.python.internal import prefer_static


__all__ = [
    'ConvolutionTranspose',
    'ConvolutionTransposeVariationalFlipout',
    'ConvolutionTransposeVariationalReparameterization',
]


# The following aliases ensure docstrings read more succinctly.
tfd = distribution_lib
kl_divergence_monte_carlo = vi_lib.kl_divergence_monte_carlo
unpack_kernel_and_bias = vi_lib.unpack_kernel_and_bias


class ConvolutionTranspose(layers_lib.KernelBiasLayer):
  """ConvolutionTranspose layer.

  This layer creates a ConvolutionTranspose kernel that is convolved (actually
  cross-correlated) with the layer input to produce a tensor of outputs.

  This layer has two learnable parameters, `kernel` and `bias`.
  - The `kernel` (aka `filters` argument of `tf.nn.conv_transpose`) is a
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

  ConvolutionTranspose1D = functools.partial(tfn.ConvolutionTranspose, rank=1)
  ConvolutionTranspose2D = tfn.ConvolutionTranspose
  ConvolutionTranspose3D = functools.partial(tfn.ConvolutionTranspose, rank=3)
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
      output_padding=None,  # keras::ConvTranspose::output_padding
      # Weights
      init_kernel_fn=None,  # tfp.experimental.nn.initializers.glorot_uniform()
      init_bias_fn=None,    # tf.initializers.zeros()
      make_kernel_bias_fn=nn_util_lib.make_kernel_bias,
      dtype=tf.float32,
      # Misc
      activation_fn=None,
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
      output_padding: An `int` or length-`rank` tuple/list representing the
        amount of padding along the input spatial dimensions (e.g., depth,
        height, width). A single `int` indicates the same value for all spatial
        dimensions. The amount of output padding along a given dimension must be
        lower than the stride along that same dimension.  If set to `None`
        (default), the output shape is inferred.
        In Keras, this argument has the same name and semantics.
        Default value: `None` (i.e., inferred).
      init_kernel_fn: ...
        Default value: `None` (i.e.,
        `tfp.experimental.nn.initializers.glorot_uniform()`).
      init_bias_fn: ...
        Default value: `None` (i.e., `tf.initializers.zeros()`).
      make_kernel_bias_fn: ...
        Default value: `tfp.experimental.nn.util.make_kernel_bias`.
      dtype: ...
        Default value: `tf.float32`.
      activation_fn: ...
        Default value: `None`.
      name: ...
        Default value: `None` (i.e., `'ConvolutionTranspose'`).
    """
    filter_shape = nn_util_lib.prepare_tuple_argument(
        filter_shape, rank, 'filter_shape')
    kernel_shape = filter_shape + (output_size, input_size)  # Note transpose.
    batch_ndims = 0
    kernel, bias = make_kernel_bias_fn(
        kernel_shape, [output_size],
        init_kernel_fn, init_bias_fn,
        batch_ndims, batch_ndims,
        dtype)
    super(ConvolutionTranspose, self).__init__(
        kernel=kernel,
        bias=bias,
        apply_kernel_fn=_make_convolution_transpose_fn(
            rank, strides, padding, dilations,
            filter_shape, output_size, output_padding),
        dtype=dtype,
        activation_fn=activation_fn,
        name=name)


class ConvolutionTransposeVariationalReparameterization(
    vi_lib.VariationalReparameterizationKernelBiasLayer):
  """ConvolutionTranspose layer class with reparameterization estimator.

  This layer implements the Bayesian variational inference analogue to
  a ConvolutionTranspose layer by assuming the `kernel` and/or the `bias` are
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

  #### Examples

  We illustrate a Bayesian autoencoder network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods), assuming a
  dataset of images. Note that this examples is *not* a variational autoencoder,
  rather it is a Bayesian Autoencoder which also uses variational inference.

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
  x, _ = next(train_iter)  # Ignore labels.
  evidence_shape = x.shape[1:]

  # 2  Specify Model

  bottleneck_size = 2

  BayesConv2D = functools.partial(
      tfn.ConvolutionVariationalReparameterization,
      rank=2,
      padding='same',
      filter_shape=5,
      # Use `he_uniform` because we'll use the `relu` family.
      init_kernel_fn=tf.initializers.he_uniform())

  BayesDeconv2D = functools.partial(
      tfn.ConvolutionTransposeVariationalReparameterization,
      rank=2,
      padding='same',
      filter_shape=5,
      # Use `he_uniform` because we'll use the `relu` family.
      init_kernel_fn=tf.initializers.he_uniform())

  scale = tfp.util.TransformedVariable(1., tfb.Softplus())
  bnn = tfn.Sequential([
      BayesConv2D(evidence_shape[-1], 32, filter_shape=5, strides=2,
                  activation_fn=tf.nn.leaky_relu),        # [b, 14, 14, 32]
      tfn.util.flatten_rightmost(ndims=3),                # [b, 14 * 14 * 32]
      tfn.AffineVariationalReparameterization(
          14 * 14 * 32, bottleneck_size),                 # [b, 2]
      lambda x: x[..., tf.newaxis, tf.newaxis, :],        # [b, 1, 1, 2]
      BayesDeconv2D(2, 64, filter_shape=7, strides=1, padding='valid',
                    activation_fn=tf.nn.leaky_relu),      # [b, 7, 7, 64]
      BayesDeconv2D(64, 32, filter_shape=4, strides=4,
                    activation_fn=tf.nn.leaky_relu),      # [2, 28, 28, 32]
      BayesConv2D(32, 1, filter_shape=2, strides=1),      # [2, 28, 28, 1]
      tfn.Lambda(
          eval_fn=lambda loc: (
              tfd.Independent(tfb.Sigmoid()(tfd.Normal(loc, scale)),
                              reinterpreted_batch_ndims=3)),
          also_track=scale),                              # [b, 28, 28, 1]
  ], name='bayesian_autoencoder')

  print(bnn.summary())

  # 3  Train.

  def loss_fn():
    x, _ = next(train_iter)  # Ignore the label.
    nll = -tf.reduce_mean(bnn(x).log_prob(x), axis=-1)
    kl = bnn.extra_loss / tf.cast(train_size, tf.float32)
    loss = nll + kl
    return loss, (nll, kl)
  opt = tf.optimizers.Adam()
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
      output_padding=None,  # keras::ConvTranspose::output_padding
      # Weights
      init_kernel_fn=None,  # tfp.experimental.nn.initializers.glorot_uniform()
      init_bias_fn=None,    # tf.initializers.zeros()
      make_posterior_fn=nn_util_lib.make_kernel_bias_posterior_mvn_diag,
      make_prior_fn=nn_util_lib.make_kernel_bias_prior_spike_and_slab,
      posterior_value_fn=tfd.Distribution.sample,
      unpack_weights_fn=unpack_kernel_and_bias,
      dtype=tf.float32,
      # Penalty.
      penalty_weight=None,
      posterior_penalty_fn=kl_divergence_monte_carlo,
      # Misc
      activation_fn=None,
      seed=None,
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
      output_padding: An `int` or length-`rank` tuple/list representing the
        amount of padding along the input spatial dimensions (e.g., depth,
        height, width). A single `int` indicates the same value for all spatial
        dimensions. The amount of output padding along a given dimension must be
        lower than the stride along that same dimension.  If set to `None`
        (default), the output shape is inferred.
        In Keras, this argument has the same name and semantics.
        Default value: `None` (i.e., inferred).
      init_kernel_fn: ...
        Default value: `None` (i.e.,
        `tfp.experimental.nn.initializers.glorot_uniform()`).
      init_bias_fn: ...
        Default value: `None` (i.e., `tf.initializers.zeros()`).
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
      penalty_weight: ...
        Default value: `None` (i.e., weight is `1`).
      posterior_penalty_fn: ...
        Default value: `kl_divergence_monte_carlo`.
      activation_fn: ...
        Default value: `None`.
      seed: ...
        Default value: `None` (i.e., no seed).
      name: ...
        Default value: `None` (i.e.,
        `'ConvolutionTransposeVariationalReparameterization'`).
    """
    filter_shape = nn_util_lib.prepare_tuple_argument(
        filter_shape, rank, 'filter_shape')
    kernel_shape = filter_shape + (output_size, input_size)  # Note transpose.
    batch_ndims = 0
    super(ConvolutionTransposeVariationalReparameterization, self).__init__(
        posterior=make_posterior_fn(
            kernel_shape, [output_size],
            init_kernel_fn, init_bias_fn,
            batch_ndims, batch_ndims,
            dtype),
        prior=make_prior_fn(
            kernel_shape, [output_size],
            init_kernel_fn, init_bias_fn,
            batch_ndims, batch_ndims,
            dtype),
        apply_kernel_fn=_make_convolution_transpose_fn(
            rank, strides, padding, dilations,
            filter_shape, output_size, output_padding),
        posterior_value_fn=posterior_value_fn,
        unpack_weights_fn=unpack_weights_fn,
        dtype=dtype,
        penalty_weight=penalty_weight,
        posterior_penalty_fn=posterior_penalty_fn,
        activation_fn=activation_fn,
        seed=seed,
        name=name)


class ConvolutionTransposeVariationalFlipout(
    vi_lib.VariationalFlipoutKernelBiasLayer):
  """ConvolutionTranspose layer class with Flipout estimator.

  This layer implements the Bayesian variational inference analogue to
  a ConvolutionTranspose layer by assuming the `kernel` and/or the `bias` are
  drawn from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = tf.nn.conv_transpose(inputs, kernel) + bias
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

  #### Examples

  We illustrate a Bayesian autoencoder network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods), assuming a
  dataset of images. Note that this examples is *not* a variational autoencoder,
  rather it is a Bayesian Autoencoder which also uses variational inference.

  ```python
  # Using the following substitution, see:
  tfn = tfp.experimental.nn
  help(tfn.ConvolutionTransposeVariationalReparameterization)
  BayesConv2D = functools.partial(
      tfn.ConvolutionVariationalFlipout,
      rank=2,
      padding='same',
      filter_shape=5,
      # Use `he_uniform` because we'll use the `relu` family.
      init_kernel_fn=tf.initializers.he_uniform())
  BayesDeconv2D = functools.partial(
      tfn.ConvolutionTransposeVariationalFlipout,
      rank=2,
      padding='same',
      filter_shape=5,
      # Use `he_uniform` because we'll use the `relu` family.
      init_kernel_fn=tf.initializers.he_uniform())
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
      # ConvTranspose specific.
      filter_shape,         # keras::Conv::kernel_size
      rank=2,               # keras::Conv::rank
      strides=1,            # keras::Conv::strides
      padding='VALID',      # keras::Conv::padding; 'CAUSAL' not implemented.
                            # keras::Conv::data_format is not implemented
      dilations=1,          # keras::Conv::dilation_rate
      output_padding=None,  # keras::ConvTranspose::output_padding
      # Weights
      init_kernel_fn=None,  # tfp.experimental.nn.initializers.glorot_uniform()
      init_bias_fn=None,    # tf.initializers.zeros()
      make_posterior_fn=nn_util_lib.make_kernel_bias_posterior_mvn_diag,
      make_prior_fn=nn_util_lib.make_kernel_bias_prior_spike_and_slab,
      posterior_value_fn=tfd.Distribution.sample,
      unpack_weights_fn=unpack_kernel_and_bias,
      dtype=tf.float32,
      # Penalty.
      penalty_weight=None,
      posterior_penalty_fn=kl_divergence_monte_carlo,
      # Misc
      activation_fn=None,
      seed=None,
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
      output_padding: An `int` or length-`rank` tuple/list representing the
        amount of padding along the input spatial dimensions (e.g., depth,
        height, width). A single `int` indicates the same value for all spatial
        dimensions. The amount of output padding along a given dimension must be
        lower than the stride along that same dimension.  If set to `None`
        (default), the output shape is inferred.
        In Keras, this argument has the same name and semantics.
        Default value: `None` (i.e., inferred).
      init_kernel_fn: ...
        Default value: `None` (i.e.,
        `tfp.experimental.nn.initializers.glorot_uniform()`).
      init_bias_fn: ...
        Default value: `None` (i.e., `tf.initializers.zeros()`).
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
      penalty_weight: ...
        Default value: `None` (i.e., weight is `1`).
      posterior_penalty_fn: ...
        Default value: `kl_divergence_monte_carlo`.
      activation_fn: ...
        Default value: `None`.
      seed: ...
        Default value: `None` (i.e., no seed).
      name: ...
        Default value: `None` (i.e.,
        `'ConvolutionTransposeVariationalFlipout'`).
    """
    filter_shape = nn_util_lib.prepare_tuple_argument(
        filter_shape, rank, 'filter_shape')
    kernel_shape = filter_shape + (output_size, input_size)  # Note transpose.
    batch_ndims = 0
    super(ConvolutionTransposeVariationalFlipout, self).__init__(
        posterior=make_posterior_fn(
            kernel_shape, [output_size],
            init_kernel_fn, init_bias_fn,
            batch_ndims, batch_ndims,
            dtype),
        prior=make_prior_fn(
            kernel_shape, [output_size],
            init_kernel_fn, init_bias_fn,
            batch_ndims, batch_ndims,
            dtype),
        apply_kernel_fn=_make_convolution_transpose_fn(
            rank, strides, padding, dilations,
            filter_shape, output_size, output_padding),
        posterior_value_fn=posterior_value_fn,
        unpack_weights_fn=unpack_weights_fn,
        dtype=dtype,
        penalty_weight=penalty_weight,
        posterior_penalty_fn=posterior_penalty_fn,
        activation_fn=activation_fn,
        seed=seed,
        name=name)


def _make_convolution_transpose_fn(rank, strides, padding, dilations,
                                   filter_shape, output_size, output_padding):
  """Helper to create tf convolution op."""
  [
      rank,
      strides,
      padding,
      dilations,
      data_format,
  ] = nn_util_lib.prepare_conv_args(rank, strides, padding, dilations)
  def op(x, kernel):
    output_shape, strides_ = _get_output_shape(
        rank, strides, padding, dilations,
        prefer_static.shape(x), output_size, filter_shape, output_padding)
    return tf.nn.conv_transpose(
        x, kernel,
        output_shape=output_shape,
        strides=strides_,
        padding=padding,
        data_format=data_format,
        dilations=dilations)
  return lambda x, kernel: nn_util_lib.batchify_op(op, rank + 1, x, kernel)


def _get_output_shape(rank, strides, padding, dilations,
                      input_shape, output_size, filter_shape, output_padding):
  """Compute the `output_shape` and `strides` arg used by `conv_transpose`."""
  if output_padding is None:
    output_padding = (None,) * rank
  else:
    output_padding = nn_util_lib.prepare_tuple_argument(
        output_padding, rank, 'output_padding')
    for stride, out_pad in zip(strides, output_padding):
      if out_pad >= stride:
        raise ValueError('Stride {} must be greater than output '
                         'padding {}.'.format(strides, output_padding))
  assert len(filter_shape) == rank
  assert len(strides) == rank
  assert len(output_padding) == rank
  event_shape = []
  for i in range(-rank, 0):
    event_shape.append(_deconv_output_length(
        input_shape[i - 1],
        filter_shape[i],
        padding=padding,
        output_padding=output_padding[i],
        stride=strides[i],
        dilation=dilations[i]))
  event_shape.append(output_size)
  batch_shape = input_shape[:-rank-1]
  output_shape = prefer_static.concat([batch_shape, event_shape], axis=0)
  strides = (1,) + strides + (1,)
  return output_shape, strides


def _deconv_output_length(input_size, filter_size, padding, output_padding,
                          stride, dilation):
  """Determines output length of a transposed convolution given input length.

  Args:
    input_size: `int`.
    filter_size: `int`.
    padding: one of `"SAME"`, `"VALID"`, `"FULL"`.
    output_padding: `int`, amount of padding along the output dimension. Can
      be set to `None` in which case the output length is inferred.
    stride: `int`.
    dilation: `int`.

  Returns:
    output_length: The output length (`int`).
  """
  assert padding in {'SAME', 'VALID', 'FULL'}
  if input_size is None:
    return None
  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == 'VALID':
      return input_size * stride + max(filter_size - stride, 0)
    elif padding == 'FULL':
      return input_size * stride - (stride + filter_size - 2)
    elif padding == 'SAME':
      return input_size * stride
  if padding == 'SAME':
    pad = filter_size // 2
  elif padding == 'VALID':
    pad = 0
  elif padding == 'FULL':
    pad = filter_size - 1
  return (input_size - 1) * stride + filter_size - 2 * pad + output_padding
