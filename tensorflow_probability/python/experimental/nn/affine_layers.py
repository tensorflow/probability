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
"""Affine layers for building neural networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.experimental.nn import layers as layers_lib
from tensorflow_probability.python.experimental.nn import util as nn_util_lib
from tensorflow_probability.python.experimental.nn import variational_base as vi_lib


__all__ = [
    'Affine',
    'AffineVariationalFlipout',
    'AffineVariationalReparameterization',
    'AffineVariationalReparameterizationLocal',
]


# The following aliases ensure docstrings read more succinctly.
tfd = distribution_lib
kl_divergence_monte_carlo = vi_lib.kl_divergence_monte_carlo
unpack_kernel_and_bias = vi_lib.unpack_kernel_and_bias


class Affine(layers_lib.KernelBiasLayer):
  """Basic affine layer."""

  def __init__(
      self,
      input_size,
      output_size,
      make_kernel_bias_fn=nn_util_lib.make_kernel_bias,
      init_kernel_fn=None,  # tf.initializers.glorot_uniform()
      init_bias_fn=None,    # tf.zeros
      dtype=tf.float32,
      batch_shape=(),
      name=None):
    """Constructs layer.

    Args:
      input_size: ...
      output_size: ...
      make_kernel_bias_fn: ...
        Default value: `tfp.experimental.nn.util.make_kernel_bias`.
      init_kernel_fn: ...
        Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
      init_bias_fn: ...
        Default value: `None` (i.e., `tf.zeros`).
      dtype: ...
        Default value: `tf.float32`.
      batch_shape: ...
        Default value: `()`.
      name: ...
        Default value: `None` (i.e., `'Affine'`).
    """
    if not batch_shape:
      kernel_shape = (input_size, output_size)
      bias_shape = (output_size,)
      apply_kernel_fn = tf.matmul
    else:
      batch_shape = tuple(batch_shape)
      kernel_shape = batch_shape + (input_size, output_size)
      bias_shape = batch_shape + (output_size,)
      # apply_kernel_fn = lambda x, k: tf.matmul(
      #     x[..., tf.newaxis, :], k)[..., 0, :]
      apply_kernel_fn = lambda x, k: tf.linalg.matvec(k, x, adjoint_a=True)
    kernel, bias = make_kernel_bias_fn(
        kernel_shape, bias_shape, dtype, init_kernel_fn, init_bias_fn)
    self._make_kernel_bias_fn = make_kernel_bias_fn  # For tracking.
    super(Affine, self).__init__(
        kernel=kernel,
        bias=bias,
        apply_kernel_fn=apply_kernel_fn,
        dtype=dtype,
        name=name)


class AffineVariationalReparameterization(
    vi_lib.VariationalReparameterizationKernelBiasLayer):
  """Densely-connected layer class with reparameterization estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
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

  BayesConv2D = functools.partial(
      tfn.ConvolutionVariationalReparameterization,
      rank=2,
      padding='same',
      filter_shape=5,
      init_kernel_fn=tf.initializers.he_uniform())  # Because we'll use `elu`.

  BayesAffine = tfn.AffineVariationalReparameterization

  scale = tfp.util.TransformedVariable(1., tfb.Softplus())
  bnn = tfn.Sequential([
      BayesConv2D(evidence_shape[-1], 32, filter_shape=7, strides=2),
      tf.nn.elu,
      tfn.util.trace('conv1'),  # [b, 14, 14, 32]

      tfn.util.flatten_rightmost(ndims=3),
      tfn.util.trace('flat1'),  # [b, 14 * 14 * 32]

      BayesAffine(14 * 14 * 32, np.prod(target_shape) - 1),
      tfn.util.trace('affine1'),  # [b, 9]

      tfn.Lambda(
          eval_fn=lambda loc: tfb.SoftmaxCentered()(
              tfd.Independent(tfd.Normal(loc, scale),
                              reinterpreted_batch_ndims=1)),
          also_track=scale),
      tfn.util.trace('head'),  # [b, 10]
  ], name='bayesian_neural_network')

  print(bnn.summary())

  # 3  Train.

  def loss_fn():
    x, y = next(train_iter)
    nll = -tf.reduce_mean(bnn(x).log_prob(y), axis=-1)
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
      output_size,
      # Weights
      make_posterior_fn=nn_util_lib.make_kernel_bias_posterior_mvn_diag,
      make_prior_fn=nn_util_lib.make_kernel_bias_prior_spike_and_slab,
      init_kernel_fn=None,  # tf.initializers.glorot_uniform()
      init_bias_fn=None,    # tf.zeros
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

    Args:
      input_size: ...
      output_size: ...
      make_posterior_fn: ...
        Default value:
          `tfp.experimental.nn.util.make_kernel_bias_posterior_mvn_diag`.
      make_prior_fn: ...
        Default value:
          `tfp.experimental.nn.util.make_kernel_bias_prior_spike_and_slab`.
      init_kernel_fn: ...
        Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
      init_bias_fn: ...
        Default value: `None` (i.e., `tf.zeros`).
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
        Default value: `None` (i.e., `'AffineVariationalReparameterization'`).
    """
    self._make_posterior_fn = make_posterior_fn  # For variable tracking.
    self._make_prior_fn = make_prior_fn  # For variable tracking.
    super(AffineVariationalReparameterization, self).__init__(
        posterior=make_posterior_fn(
            [input_size, output_size], [output_size], dtype,
            init_kernel_fn, init_bias_fn),
        prior=make_prior_fn(
            [input_size, output_size], [output_size], dtype,
            init_kernel_fn, init_bias_fn),
        apply_kernel_fn=tf.matmul,
        posterior_value_fn=posterior_value_fn,
        unpack_weights_fn=unpack_weights_fn,
        penalty_weight=penalty_weight,
        posterior_penalty_fn=posterior_penalty_fn,
        seed=seed,
        dtype=dtype,
        name=name)


class AffineVariationalFlipout(vi_lib.VariationalFlipoutKernelBiasLayer):
  """Densely-connected layer class with Flipout estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = matmul(inputs, kernel) + bias
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

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of images and length-10 one-hot `targets`.

  ```python
  # Using the following substitution, see:
  tfn = tfp.experimental.nn
  help(tfn.AffineVariationalReparameterization)
  BayesAffine = tfn.AffineVariationalFlipout
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
      output_size,
      # Weights
      make_posterior_fn=nn_util_lib.make_kernel_bias_posterior_mvn_diag,
      make_prior_fn=nn_util_lib.make_kernel_bias_prior_spike_and_slab,
      init_kernel_fn=None,  # tf.initializers.glorot_uniform()
      init_bias_fn=None,    # tf.zeros,
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

    Args:
      input_size: ...
      output_size: ...
      make_posterior_fn: ...
        Default value:
          `tfp.experimental.nn.util.make_kernel_bias_posterior_mvn_diag`.
      make_prior_fn: ...
        Default value:
          `tfp.experimental.nn.util.make_kernel_bias_prior_spike_and_slab`.
      init_kernel_fn: ...
        Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
      init_bias_fn: ...
        Default value: `None` (i.e., `tf.zeros`).
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
        Default value: `None` (i.e., `'AffineVariationalFlipout'`).
    """
    self._make_posterior_fn = make_posterior_fn  # For variable tracking.
    self._make_prior_fn = make_prior_fn  # For variable tracking.
    super(AffineVariationalFlipout, self).__init__(
        posterior=make_posterior_fn(
            [input_size, output_size], [output_size], dtype,
            init_kernel_fn, init_bias_fn),
        prior=make_prior_fn(
            [input_size, output_size], [output_size], dtype,
            init_kernel_fn, init_bias_fn),
        apply_kernel_fn=tf.matmul,
        posterior_value_fn=posterior_value_fn,
        unpack_weights_fn=unpack_weights_fn,
        penalty_weight=penalty_weight,
        posterior_penalty_fn=posterior_penalty_fn,
        seed=seed,
        dtype=dtype,
        name=name)


class AffineVariationalReparameterizationLocal(vi_lib.VariationalLayer):
  """Densely-connected layer class with local reparameterization estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = matmul(inputs, kernel) + bias
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
  # Using the following substitution, see:
  tfn = tfp.experimental.nn
  help(tfn.AffineVariationalReparameterization)
  BayesAffine =  tfn.AffineVariationalReparameterizationLocal
  ```

  This example uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the negative
  Evidence Lower Bound. It consists of the sum of two terms: the expected
  negative log-likelihood, which we approximate via Monte Carlo; and the KL
  divergence, which is added via regularizer terms which are arguments to the
  layer.

  #### References

  [1]: Diederik Kingma, Tim Salimans, and Max Welling. Variational Dropout and
       the Local Reparameterization Trick. In _Neural Information Processing
       Systems_, 2015. https://arxiv.org/abs/1506.02557
  [2]: Dmitry Molchanov, Arsenii Ashukha, Dmitry Vetrov. Variational Dropout
       Sparsifies Deep Neural Networks. In _International Conference on Machine
       Learning_, 2017. https://arxiv.org/abs/1701.05369
  """

  def __init__(
      self,
      input_size,
      output_size,
      # Weights
      make_posterior_fn=nn_util_lib.make_kernel_bias_posterior_mvn_diag,
      make_prior_fn=nn_util_lib.make_kernel_bias_prior_spike_and_slab,
      init_kernel_fn=None,  # tf.initializers.glorot_uniform()
      init_bias_fn=None,    # tf.zeros
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

    Args:
      input_size: ...
      output_size: ...
      make_posterior_fn: ...
        Default value:
          `tfp.experimental.nn.util.make_kernel_bias_posterior_mvn_diag`.
      make_prior_fn: ...
        Default value:
          `tfp.experimental.nn.util.make_kernel_bias_prior_spike_and_slab`.
      init_kernel_fn: ...
        Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
      init_bias_fn: ...
        Default value: `None` (i.e., `tf.zeros`).
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
        Default value: `None` (i.e., `'AffineVariationalFlipout'`).
    """
    self._make_posterior_fn = make_posterior_fn  # For variable tracking.
    self._make_prior_fn = make_prior_fn  # For variable tracking.
    super(AffineVariationalReparameterizationLocal, self).__init__(
        posterior=make_posterior_fn(
            [input_size, output_size], [output_size], dtype,
            init_kernel_fn, init_bias_fn),
        prior=make_prior_fn(
            [input_size, output_size], [output_size], dtype,
            init_kernel_fn, init_bias_fn),
        penalty_weight=penalty_weight,
        posterior_penalty_fn=posterior_penalty_fn,
        posterior_value_fn=posterior_value_fn,
        seed=seed,
        dtype=dtype,
        name=name)
    self._unpack_weights_fn = unpack_weights_fn

  @property
  def unpack_weights_fn(self):
    return self._unpack_weights_fn

  def _eval(self, x, weights):
    kernel_dist, bias_dist = self.unpack_weights_fn(  # pylint: disable=not-callable
        self.posterior.sample_distributions(value=weights)[0])
    kernel_loc, kernel_scale = vi_lib.get_spherical_normal_loc_scale(
        kernel_dist)
    loc = tf.matmul(x, kernel_loc)
    scale = tf.sqrt(tf.matmul(tf.square(x), tf.square(kernel_scale)))
    _, sampled_bias = self.unpack_weights_fn(weights)  # pylint: disable=not-callable
    if sampled_bias is not None:
      try:
        bias_loc, bias_scale = vi_lib.get_spherical_normal_loc_scale(
            bias_dist)
        is_bias_spherical_normal = True
      except TypeError:
        is_bias_spherical_normal = False
      if is_bias_spherical_normal:
        loc = loc + bias_loc
        scale = tf.sqrt(tf.square(scale) + tf.square(bias_scale))
      else:
        loc = loc + sampled_bias
    y = normal_lib.Normal(loc=loc, scale=scale).sample(seed=self._seed())
    return y
