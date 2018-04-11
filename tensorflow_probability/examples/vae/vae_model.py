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
"""Model definitions for an example VAE."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def make_encoder_net(images,
                     num_outputs,
                     hidden_layer_sizes,
                     activation=tf.nn.elu,
                     name=None):
  """Creates a deep, dense network to parameterize the encoder distribution.

  Args:
    images: A `float`-like `Tensor` representing the inputs to be encoded.
      The first dimension (axis 0) indexes batch elements; all other
      dimensions index event elements.
    num_outputs: Python scalar `int` representing the size of the
      network output. If the input images have shape `[b, E1, .., Ek]`
      then the output shape is `[b, num_outputs]`.
    hidden_layer_sizes: A Python enumerable of Python scalar integers each
      indicating the number of units in each hidden layer. E.g., `[128, 32]`
      implies the network has two hidden layers with `128` and `32` outputs,
      respectively.
    activation: The activation function as a TensorFlow op (e.g., tf.nn.elu).
    name: An optional name scope for the network parameters.

  Returns:
    net: An `images.dtype` `Tensor` with shape `[images.shape[0], num_outputs]`
      which represents the network evaluated at `images`.
  """
  with tf.name_scope(name, "make_encoder_net", [images]):
    net = 2 * tf.layers.flatten(images) - 1
    for size in hidden_layer_sizes:
      net = tf.layers.dense(net, size, activation=activation)
    net = tf.layers.dense(net, num_outputs, activation=None)
    return net


def make_encoder_mvndiag(net, latent_size, name=None):
  """Creates a spherical multivariate Normal encoder distribution.

  Args:
    net: A `float`-like `Tensor` having shape [batch_size,
      latent_size*2], containing logits for the Gaussian
      `loc` and `scale_diag` parameters of the encoder.
      This is typically the output of `make_encoder_net`.
    latent_size: The number of elements in the latent random
      variable.
    name: An optional name scope for the encoder.

  Returns:
    encoder: A Distribution object representing the encoder.

  """
  with tf.name_scope(name, "make_encoder_mvndiag", values=[net]):
    mu = net[..., :latent_size]
    sigma = tf.nn.softplus(net[..., latent_size:] + 0.5)
    return tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma,
                                      name="encoder_distribution")


def make_decoder_net(encoding_draw,
                     num_outputs,
                     hidden_layer_sizes,
                     activation=tf.nn.elu,
                     name=None):
  """Creates a deep, dense network to parameterize the decoder distribution.

  Args:
    encoding_draw: A `float`-like `Tensor` containing the latent
      vectors to be decoded. These are assumed to be rank-1, so
      the encoding `Tensor` is rank-2 with shape `[batch_size, latent_size]`.
    num_outputs: Python scalar integer representing the size of
      the network output. If the input encodings have shape `[batch_size,
      latent_size]` then the output shape is `[batch_size, num_outputs]`.
    hidden_layer_sizes: A Python enumerable of Python scalar integers each
      indicating the number of units in each hidden layer. E.g., `[128, 32]`
      implies the network has two hidden layers with `128` and `32` outputs,
      respectively.
    activation: The activation function as a TensorFlow op (e.g., tf.nn.elu).
    name: An optional name scope for network parameters.

  Returns:
    net: An `encoding_draw.dtype` `Tensor` with shape
      `encoding_draw.shape[:-1] + [num_outputs,]` which represents the
       network evaluated at `encoding_draw`.
  """
  with tf.name_scope(name, "make_decoder_net", [encoding_draw]):
    net = encoding_draw
    for size in hidden_layer_sizes:
      net = tf.layers.dense(net, size, activation=activation)
    net = tf.layers.dense(net, num_outputs, activation=None)
    return net


def make_decoder_bernoulli(net, event_shape, name=None):
  """Creates a decoder using a factored Bernoulli likelihood.

  Args:
    net: A `float`-like `Tensor` of shape `[batch_size, prod(event_shape)]`
      containing (flattened) logits for the Bernoulli decoder. This is
      typically the output of `make_decoder_net`.
    event_shape: A Python enumerable of Python integers indicating the
      desired shape for decoded outputs.
    name: An optional name scope.

  Returns:
    decoder: A Distribution object representing the decoder.
  """

  with tf.name_scope(name, "make_decoder_bernoulli", values=[net]):
    new_shape = tf.concat([tf.shape(net)[:-1], event_shape],
                          axis=0)
    logits = tf.reshape(net, shape=new_shape)
    decoder = tfd.Independent(tfd.Bernoulli(logits=logits),
                              reinterpreted_batch_ndims=len(event_shape),
                              name="decoder_distribution")
    return decoder


def make_prior_mvndiag(latent_size, dtype=tf.float32, name=None):
  """Creates a factored (diagonal) Gaussian prior distribution.

  Args:
    latent_size: The number of elements in the latent random variable.
    dtype: The TensorFlow datatype of the latent code.
    name: An optional name scope.

  Returns:
    prior: A Distribution object representing the prior.
  """
  with tf.name_scope(name, "make_prior_mvndiag", values=[]):
    sigma = tf.fill([latent_size],
                    np.array(1., dtype=dtype.as_numpy_dtype))
    prior = tfd.MultivariateNormalDiag(scale_diag=sigma,
                                       name="prior_distribution")
    return prior


def make_vae(images,
             make_encoder,
             make_decoder,
             make_prior,
             return_full=False):
  """Defines the model and loss for a variational autoencoder (VAE).

  The VAE defines a generative model in which a latent encoding `Z` is
  sampled from a prior `p(Z)`, then used to generate an observation `X`
  by way of a decoder distribution `p(X|Z)`. To fit the model, we assume
  an approximate representation of the posterior, in the form of an encoder
  `q(Z|X)`, and then minimize the KL divergence between `q(Z|X)` and the
  true posterior `p(Z|X)`, which is intractable. This is equivalent to
  maximizing the evidence lower bound (ELBO)
  ```
   L =  E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z)]
     <= log p(X)
  ```
  which also provides a lower bound on the marginal likelihood `p(X)`. See
  [Kingma and Welling (2014)][1] for further information.

  ### Example Usage

  This method takes callables to build the model components. Typically these
  will be lambdas defined to encode additional modeling choices, e.g.,

  ```
  def my_encoder(images):
    net = vae_model.make_encoder_net(images,
           num_outputs=latent_size * 2,
           hidden_layer_sizes=encoder_layers)
    return vae_model.make_encoder_mvndiag(net,
        latent_size)
  def my_decoder(encoding):
    net = vae_model.make_decoder_net(encoding,
            num_outputs=np.prod(image_shape),
            hidden_layer_sizes=decoder_layers)
    return vae_model.make_decoder_bernoulli(net, image_shape)
  def my_prior():
    return vae_model.make_prior_mvndiag(latent_size)

  elbo = vae_model.make_vae(images_placeholder, my_encoder,
                            my_decoder, my_prior)
  ```

  Args:
    images: A `float`-like `Tensor` containing observed inputs X. The first
      dimension (axis 0) indexes batch elements; all other dimensions index
      event elements.
    make_encoder: A callable to generate the encoder distribution `q(Z|X)`. This
      takes a single argument, a `float`-like `Tensor` representing a batch of
      inputs `X`, and returns a Distribution over the batch of latent codes `Z`.
    make_decoder: A callable to generate the decoder distribution `p(X|Z)`. This
      takes a single argument, a `float`-like `Tensor` representing a batch of
      latent codes `Z`, and returns a Distribution over the batch of
      observations `X`.
    make_prior: A callable to generate the prior distribution `p(Z)`. This takes
      no arguments, and returns a Distribution over a single latent code (
    return_full: If True, also return the model components and the
      sampled encoding.

  Returns:
    elbo_loss: A scalar `Tensor` computing the negation of the variational
      evidence bound (i.e., `elbo_loss >= -log p(X)`).

  #### References

  [1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
       _International Conference on Learning Representations_, 2014.
       https://arxiv.org/abs/1312.6114
  """

  # Create the three components of a VAE: encoder, prior, and decoder
  with tf.variable_scope("encoder"):
    encoder = make_encoder(images)

  with tf.variable_scope("prior"):
    prior = make_prior()

  def joint_log_prob(z):
    with tf.variable_scope("decoder"):
      decoder = make_decoder(z)
    return decoder.log_prob(images) + prior.log_prob(z)

  elbo_loss = tf.reduce_sum(
      tfp.vi.csiszar_divergence.monte_carlo_csiszar_f_divergence(
          f=tfp.vi.csiszar_divergence.kl_reverse,
          p_log_prob=joint_log_prob,
          q=encoder,
          num_draws=1))
  tf.summary.scalar("elbo", elbo_loss)

  if return_full:
    # Rebuild (and reuse!) the decoder so we can compute stats from it.
    encoding_draw = encoder.sample()
    with tf.variable_scope("decoder", reuse=True):
      decoder = make_decoder(encoding_draw)
    return elbo_loss, encoder, decoder, prior, encoding_draw

  return elbo_loss
