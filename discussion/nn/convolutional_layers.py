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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from discussion.nn import layers as layers_lib
from discussion.nn import util as nn_util_lib
from discussion.nn import variational_base as vi_lib
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'Convolution',
    'ConvolutionVariationalFlipout',
    'ConvolutionVariationalReparameterization',
]


# The following aliases ensure docstrings read more succinctly.
tfd = distribution_lib
kl_divergence_monte_carlo = vi_lib.kl_divergence_monte_carlo
unpack_kernel_and_bias = vi_lib.unpack_kernel_and_bias


class Convolution(layers_lib.KernelBiasLayer):
  """Convolution layer.

  This layer creates a Convolution kernel that is convolved (actually
  cross-correlated) with the layer input to produce a tensor of outputs.

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
  from discussion import nn
  tfb = tfp.bijectors
  tfd = tfp.distributions

  Convolution1D = functools.partial(nn.Convolution, rank=1)
  Convolution2D = nn.Convolution
  Convolution3D = functools.partial(nn.Convolution, rank=3)
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
      make_kernel_bias_fn=nn_util_lib.make_kernel_bias,
      init_kernel_fn=None,  # glorot_uniform
      init_bias_fn=tf.zeros,  # Same as Keras.
      # Misc
      dtype=tf.float32,
      batch_size=0,
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
        i.e.`, `kernel.shape.rank == rank + 2`.
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
      make_kernel_bias_fn: ...
        Default value: `nn.util.make_kernel_bias`.
      init_kernel_fn: ...
        Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
      init_bias_fn: ...
        Default value: `tf.zeros`.
      dtype: ...
        Default value: `tf.float32`.
      batch_size: ...
        Default value: `0`.
      name: ...
        Default value: `None` (i.e., `'Convolution'`).
    """
    filter_shape = prepare_tuple_argument(
        filter_shape, rank, arg_name='filter_shape')
    if batch_size == 0:
      kernel_shape = filter_shape + (input_size, output_size)
      bias_shape = (output_size,)
      apply_kernel_fn = _make_convolution_fn(
          rank, strides, padding, dilations)
    else:
      kernel_shape = (batch_size,) + filter_shape + (input_size, output_size)
      bias_shape = (batch_size, output_size)
      apply_kernel_fn = lambda x, k: convolution_batch(  # pylint: disable=g-long-lambda
          x, k,
          rank=rank,
          strides=strides,
          padding=padding,
          data_format='NHWBC',
          dilations=dilations)
    kernel, bias = make_kernel_bias_fn(
        kernel_shape, bias_shape, dtype, init_kernel_fn, init_bias_fn)
    self._make_kernel_bias_fn = make_kernel_bias_fn  # For tracking.
    super(Convolution, self).__init__(
        kernel=kernel,
        bias=bias,
        apply_kernel_fn=apply_kernel_fn,
        dtype=dtype,
        name=name)


class ConvolutionVariationalReparameterization(
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

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of images and length-10 one-hot `targets`.

  ```python
  import functools
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  import tensorflow_datasets as tfds
  from discussion import nn
  tfb = tfp.bijectors
  tfd = tfp.distributions

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
  train_dataset = nn.util.tune_dataset(
      train_dataset,
      batch_size=batch_size,
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
      nn.ConvolutionVariationalReparameterization,
      rank=2,
      padding='same',
      filter_shape=5,
      init_kernel_fn=tf.initializers.he_normal(),
      penalty_weight=1. / n)

  BayesAffine = functools.partial(
      nn.AffineVariationalReparameterization,
      init_kernel_fn=tf.initializers.he_normal(),
      init_bias_fn=tf.initializers.he_normal(),
      penalty_weight=1. / n)

  scale = tfp.util.TransformedVariable(1., tfb.Softplus())
  bnn = nn.Sequential([
      BayesConv2D(evidence_shape[-1], 32, filter_shape=7, strides=2),
      tf.nn.elu,
      nn.util.trace('conv1'),  # [b, 14, 14, 32]

      nn.util.flatten_rightmost(ndims=3),
      nn.util.trace('flat1'),  # [b, 14 * 14 * 32]

      BayesAffine(14 * 14 * 32, np.prod(target_shape) - 1),
      nn.util.trace('affine1'),  # [b, 9]

      nn.Lambda(
          eval_fn=lambda loc: tfb.SoftmaxCentered()(
              tfd.Independent(tfd.Normal(loc, scale),
                              reinterpreted_batch_ndims=1)),
          also_track=scale),
      nn.util.trace('head'),  # [b, 10]
  ], name='bayesian_neural_network')

  print(bnn.summary())

  # 3  Train.

  def loss_fn():
    x, y = next(train_iter)
    nll = -tf.reduce_mean(bnn(x).log_prob(y), axis=-1)
    kl = bnn.extra_loss  # Already normalized via `penalty_weight` arg.
    loss = nll + kl
    return loss, (nll, kl)
  opt = tf.optimizers.Adam()
  fit_op = nn.util.make_fit_op(loss_fn, opt, bnn.trainable_variables)
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
      make_posterior_fn=nn_util_lib.make_kernel_bias_posterior_mvn_diag,
      make_prior_fn=nn_util_lib.make_kernel_bias_prior_spike_and_slab,
      init_kernel_fn=None,  # glorot_uniform
      init_bias_fn=tf.zeros,
      posterior_value_fn=tfd.Distribution.sample,
      unpack_weights_fn=unpack_kernel_and_bias,
      # Penalty.
      penalty_weight=None,
      posterior_penalty_fn=kl_divergence_monte_carlo,
      # Misc
      seed=None,
      dtype=tf.float32,
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
        i.e.`, `kernel.shape.rank == rank + 2`.
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
      make_posterior_fn: ...
        Default value: `nn.util.make_kernel_bias_posterior_mvn_diag`.
      make_prior_fn: ...
        Default value: `nn.util.make_kernel_bias_prior_spike_and_slab`.
      init_kernel_fn: ...
        Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
      init_bias_fn: ...
        Default value: `tf.zeros`.
      posterior_value_fn: ...
        Default valye: `tfd.Distribution.sample`
      unpack_weights_fn:
        Default value: `unpack_kernel_and_bias`
      penalty_weight: ...
        Default value: `None` (i.e., weight is `1`).
      posterior_penalty_fn: ...
        Default value: `kl_divergence_monte_carlo`.
      seed: ...
        Default value: `None` (i.e., no seed).
      dtype: ...
        Default value: `tf.float32`.
      name: ...
        Default value: `None` (i.e.,
        `'ConvolutionVariationalReparameterization'`).
    """
    filter_shape = prepare_tuple_argument(
        filter_shape, rank, arg_name='filter_shape')
    kernel_shape = filter_shape + (input_size, output_size)
    self._make_posterior_fn = make_posterior_fn  # For variable tracking.
    self._make_prior_fn = make_prior_fn  # For variable tracking.
    super(ConvolutionVariationalReparameterization, self).__init__(
        posterior=make_posterior_fn(
            kernel_shape, [output_size], dtype, init_kernel_fn, init_bias_fn),
        prior=make_prior_fn(
            kernel_shape, [output_size], dtype, init_kernel_fn, init_bias_fn),
        apply_kernel_fn=_make_convolution_fn(
            rank, strides, padding, dilations),
        posterior_value_fn=posterior_value_fn,
        unpack_weights_fn=unpack_weights_fn,
        penalty_weight=penalty_weight,
        posterior_penalty_fn=posterior_penalty_fn,
        seed=seed,
        dtype=dtype,
        name=name)


class ConvolutionVariationalFlipout(
    vi_lib.VariationalFlipoutKernelBiasLayer):
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

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of images and length-10 one-hot `targets`.

  ```python
  # Using the following substitution, see:
  help(nn.ConvolutionVariationalReparameterization)

  BayesConv2D = functools.partial(
      nn.ConvolutionVariationalFlipout,
      init_kernel_fn=tf.initializers.he_normal(),
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
      make_posterior_fn=nn_util_lib.make_kernel_bias_posterior_mvn_diag,
      make_prior_fn=nn_util_lib.make_kernel_bias_prior_spike_and_slab,
      init_kernel_fn=None,  # glorot_uniform
      init_bias_fn=tf.zeros,
      posterior_value_fn=tfd.Distribution.sample,
      unpack_weights_fn=unpack_kernel_and_bias,
      # Penalty.
      penalty_weight=None,
      posterior_penalty_fn=kl_divergence_monte_carlo,
      # Misc
      seed=None,
      dtype=tf.float32,
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
        i.e.`, `kernel.shape.rank == rank + 2`.
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
      make_posterior_fn: ...
        Default value: `nn.util.make_kernel_bias_posterior_mvn_diag`.
      make_prior_fn: ...
        Default value: `nn.util.make_kernel_bias_prior_spike_and_slab`.
      init_kernel_fn: ...
        Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
      init_bias_fn: ...
        Default value: `tf.zeros`.
      posterior_value_fn: ...
        Default valye: `tfd.Distribution.sample`
      unpack_weights_fn:
        Default value: `unpack_kernel_and_bias`
      penalty_weight: ...
        Default value: `None` (i.e., weight is `1`).
      posterior_penalty_fn: ...
        Default value: `kl_divergence_monte_carlo`.
      seed: ...
        Default value: `None` (i.e., no seed).
      dtype: ...
        Default value: `tf.float32`.
      name: ...
        Default value: `None` (i.e.,
        `'ConvolutionVariationalFlipout'`).
    """
    filter_shape = prepare_tuple_argument(
        filter_shape, rank, arg_name='filter_shape')
    kernel_shape = filter_shape + (input_size, output_size)
    self._make_posterior_fn = make_posterior_fn  # For variable tracking.
    self._make_prior_fn = make_prior_fn  # For variable tracking.
    super(ConvolutionVariationalFlipout, self).__init__(
        posterior=make_posterior_fn(
            kernel_shape, [output_size], dtype, init_kernel_fn, init_bias_fn),
        prior=make_prior_fn(
            kernel_shape, [output_size], dtype, init_kernel_fn, init_bias_fn),
        apply_kernel_fn=_make_convolution_fn(
            rank, strides, padding, dilations),
        posterior_value_fn=posterior_value_fn,
        unpack_weights_fn=unpack_weights_fn,
        penalty_weight=penalty_weight,
        posterior_penalty_fn=posterior_penalty_fn,
        seed=seed,
        dtype=dtype,
        name=name)


def _make_convolution_fn(rank, strides, padding, dilations):
  """Helper to create tf convolution op."""
  [
      rank,
      strides,
      padding,
      dilations,
      data_format,
  ] = prepare_conv_args(rank, strides, padding, dilations)
  def op(x, kernel):
    dtype = dtype_util.common_dtype([x, kernel], dtype_hint=tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype, name='x')
    kernel = tf.convert_to_tensor(kernel, dtype=dtype, name='kernel')
    return tf.nn.convolution(
        x, kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations)
  return lambda x, kernel: batchify_op(op, rank + 1, x, kernel)


def batchify_op(op, op_min_input_ndims, x, *other_op_args):
  """Reshape `op` input `x` to be a vec of `op_min_input_ndims`-rank tensors."""
  if x.shape.rank == op_min_input_ndims + 1:
    # Input is already a vector of `op_min_input_ndims`-rank tensors.
    return op(x, *other_op_args)
  batch_shape, op_shape = prefer_static.split(
      prefer_static.shape(x),
      num_or_size_splits=[-1, op_min_input_ndims])
  flat_shape = prefer_static.pad(
      op_shape,
      paddings=[[1, 0]],
      constant_values=-1)
  y = tf.reshape(x, flat_shape)
  y = op(y, *other_op_args)
  unflat_shape = prefer_static.concat([
      batch_shape,
      prefer_static.shape(y)[1:],
  ], axis=0)
  y = tf.reshape(y, unflat_shape)
  return y


def prepare_conv_args(rank, strides, padding, dilations):
  """Sanitizes use provided input."""
  try:
    rank = int(tf.get_static_value(rank))
  except TypeError:
    raise TypeError('Argument `rank` must be statically known `int`.')
  valid_rank = {1, 2, 3}
  if rank not in valid_rank:
    raise ValueError('Argument `rank` must be in {}.'.format(valid_rank))
  strides = prepare_tuple_argument(strides, rank, arg_name='strides')
  padding = _prepare_padding_argument(padding)
  dilations = prepare_tuple_argument(dilations, rank, arg_name='dilations')
  data_format = {1: 'NWC', 2: 'NHWC', 3: 'NDHWC'}.get(rank)
  return rank, strides, padding, dilations, data_format


def prepare_tuple_argument(x, n, arg_name):
  """Helper which puts tuples in standard form."""
  if isinstance(x, int):
    return (x,) * n
  try:
    x = tuple(x)
  except TypeError:
    raise ValueError('Argument {} must be convertible to tuple.'.format(
        arg_name))
  if n != len(x):
    raise ValueError('Argument {} has invalid length; expected:{}, '
                     'saw:{}.'.format(arg_name, n, len(x)))
  for x_ in x:
    try:
      int(x_)
    except (ValueError, TypeError):
      raise ValueError('Argument {} contains non-integer input; '
                       'saw: {}.'.format(arg_name, x_))
  return x


def _prepare_padding_argument(x):
  """Helper which processes the padding argument."""
  if not hasattr(x, 'upper'):
    return tuple(x)
  padding = x.upper()
  if padding in {'CAUSAL', 'FULL'}:
    raise NotImplementedError(
        'Argument `padding` value "{}" currently not supported. If you '
        'require this feature, please create an issue on '
        '`https://github.com/tensorflow/probability` or email '
        '`tfprobability@tensorflow.org`.'.format(padding))
  valid_values = {'VALID', 'SAME'}
  if padding not in valid_values:
    raise ValueError('Argument `padding` must be convertible to a tuple '
                     'or one of {}; saw: "{}".'.format(valid_values, padding))
  return padding


def convolution_batch(x, kernel, rank, strides, padding, data_format=None,
                      dilations=None, name=None):
  """Like `tf.nn.conv2d` except applies batch of kernels to batch of `x`."""
  if rank != 2:
    raise NotImplementedError('Argument `rank` currently only supports `2`; '
                              'saw "{}".'.format(rank))
  if data_format is not None and data_format.upper() != 'NHWBC':
    raise ValueError('Argument `data_format` currently only supports "NHWBC"; '
                     'saw "{}".'.format(data_format))
  with tf.name_scope(name or 'conv2d_nhwbc'):
    # Prepare arguments.
    [
        rank,
        _,  # strides
        padding,
        dilations,
        data_format,
    ] = prepare_conv_args(rank, strides, padding, dilations)
    strides = prepare_tuple_argument(
        strides, rank + 2, arg_name='strides')

    dtype = dtype_util.common_dtype([x, kernel], dtype_hint=tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype, name='x')
    kernel = tf.convert_to_tensor(kernel, dtype=dtype, name='kernel')

    x_shape = prefer_static.shape(x)
    x_shape_ = x.shape
    x = tf.reshape(
        x,  # [n, h, w, b, c]
        shape=prefer_static.pad(x_shape[:-2],
                                paddings=[[0, 1]],
                                constant_values=-1))  # [n, h, w, bc]

    kernel_shape = prefer_static.shape(kernel)  # [b, fh, fw, c, c']
    kernel_shape_ = kernel.shape
    kernel = tf.transpose(kernel, [1, 2, 0, 3, 4])
    kernel = tf.reshape(
        kernel,  # [fh, fw, b, c, c']
        shape=prefer_static.concat([
            kernel_shape[1:-2],
            [-1, kernel_shape[-1]],
        ], axis=0))  # [fh, fw, bc, c']

    y = tf.nn.depthwise_conv2d(
        x, kernel,
        strides=strides,
        padding=padding,
        data_format='NHWC',
        dilations=dilations)
    #  SAME: y.shape = [n, h,      w,      bcc']
    # VALID: y.shape = [n, h-fh+1, w-fw+1, bcc']
    y = tf.reshape(
        y,
        shape=prefer_static.concat([
            prefer_static.shape(y)[:-1],
            kernel_shape[:1],
            kernel_shape[-2:],
        ], axis=0))  # [n, h, w, b, c, c']
    y = tf.reduce_sum(y, axis=-2)  # [n, h, w, b, c']
    tensorshape_util.set_shape(
        y.shape,
        tensorshape_util.concatenate(x_shape_[:-1], kernel_shape_[-1]))
    return y
