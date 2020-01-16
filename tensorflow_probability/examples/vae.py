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
"""Trains a variational auto-encoder (VAE) on binarized MNIST.

The VAE defines a generative model in which a latent code `Z` is sampled from a
prior `p(Z)`, then used to generate an observation `X` by way of a decoder
`p(X|Z)`. The full reconstruction follows

```none
   X ~ p(X)              # A random image from some dataset.
   Z ~ q(Z | X)          # A random encoding of the original image ("encoder").
Xhat ~ p(Xhat | Z)       # A random reconstruction of the original image
                         #   ("decoder").
```

To fit the VAE, we assume an approximate representation of the posterior in the
form of an encoder `q(Z|X)`. We minimize the KL divergence between `q(Z|X)` and
the true posterior `p(Z|X)`: this is equivalent to maximizing the evidence lower
bound (ELBO),

```none
-log p(x)
= -log int dz p(x|z) p(z)
= -log int dz q(z|x) p(x|z) p(z) / q(z|x)
<= int dz q(z|x) (-log[ p(x|z) p(z) / q(z|x) ])   # Jensen's Inequality
=: KL[q(Z|x) || p(x|Z)p(Z)]
= -E_{Z~q(Z|x)}[log p(x|Z)] + KL[q(Z|x) || p(Z)]
```

-or-

```none
-log p(x)
= KL[q(Z|x) || p(x|Z)p(Z)] - KL[q(Z|x) || p(Z|x)]
<= KL[q(Z|x) || p(x|Z)p(Z)                        # Positivity of KL
= -E_{Z~q(Z|x)}[log p(x|Z)] + KL[q(Z|x) || p(Z)]
```

The `-E_{Z~q(Z|x)}[log p(x|Z)]` term is an expected reconstruction loss and
`KL[q(Z|x) || p(Z)]` is a kind of distributional regularizer. See
[Kingma and Welling (2014)][1] for more details.

This script supports both a (learned) mixture of Gaussians prior as well as a
fixed standard normal prior. You can enable the fixed standard normal prior by
setting `mixture_components` to 1. Note that fixing the parameters of the prior
(as opposed to fitting them with the rest of the model) incurs no loss in
generality when using only a single Gaussian. The reasoning for this is
two-fold:

  * On the generative side, the parameters from the prior can simply be absorbed
    into the first linear layer of the generative net. If `z ~ N(mu, Sigma)` and
    the first layer of the generative net is given by `x = Wz + b`, this can be
    rewritten,

      s ~ N(0, I)
      x = Wz + b
        = W (As + mu) + b
        = (WA) s + (W mu + b)

    where Sigma has been decomposed into A A^T = Sigma. In other words, the log
    likelihood of the model (E_{Z~q(Z|x)}[log p(x|Z)]) is independent of whether
    or not we learn mu and Sigma.

  * On the inference side, we can adjust any posterior approximation
    q(z | x) ~ N(mu[q], Sigma[q]), with

    new_mu[p] := 0
    new_Sigma[p] := eye(d)
    new_mu[q] := inv(chol(Sigma[p])) @ (mu[p] - mu[q])
    new_Sigma[q] := inv(Sigma[q]) @ Sigma[p]

    A bit of algebra on the KL divergence term `KL[q(Z|x) || p(Z)]` reveals that
    it is also invariant to the prior parameters as long as Sigma[p] and
    Sigma[q] are invertible.

This script also supports using the analytic KL (KL[q(Z|x) || p(Z)]) with the
`analytic_kl` flag. Using the analytic KL is only supported when
`mixture_components` is set to 1 since otherwise no analytic form is known.

Here we also compute tighter bounds, the IWAE [Burda et. al. (2015)][2].

These as well as image summaries can be seen in Tensorboard. For help using
Tensorboard see
https://www.tensorflow.org/guide/summaries_and_tensorboard
which can be run with
  `python -m tensorboard.main --logdir=MODEL_DIR`

#### References

[1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
     _International Conference on Learning Representations_, 2014.
     https://arxiv.org/abs/1312.6114
[2]: Yuri Burda, Roger Grosse, Ruslan Salakhutdinov. Importance Weighted
     Autoencoders. In _International Conference on Learning Representations_,
     2015.
     https://arxiv.org/abs/1509.00519
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports
from absl import flags
import numpy as np
from six.moves import urllib
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

IMAGE_SHAPE = [28, 28, 1]

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_integer(
    "max_steps", default=5001, help="Number of training steps to run.")
flags.DEFINE_integer(
    "latent_size",
    default=16,
    help="Number of dimensions in the latent code (z).")
flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_integer(
    "n_samples", default=16, help="Number of samples to use in encoding.")
flags.DEFINE_integer(
    "mixture_components",
    default=100,
    help="Number of mixture components to use in the prior. Each component is "
         "a diagonal normal distribution. The parameters of the components are "
         "intialized randomly, and then learned along with the rest of the "
         "parameters. If `analytic_kl` is True, `mixture_components` must be "
         "set to `1`.")
flags.DEFINE_bool(
    "analytic_kl",
    default=False,
    help="Whether or not to use the analytic version of the KL. When set to "
         "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
         "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
         "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
         "then you must also specify `mixture_components=1`.")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=500, help="Frequency at which to save visualizations.")
flags.DEFINE_bool(
    "fake_data",
    default=False,
    help="If true, uses fake data instead of MNIST.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")

FLAGS = flags.FLAGS


def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.math.log(tf.math.expm1(x))


def make_encoder(activation, latent_size, base_depth):
  """Creates the encoder function.

  Args:
    activation: Activation function in hidden layers.
    latent_size: The dimensionality of the encoding.
    base_depth: The lowest depth for a layer.

  Returns:
    encoder: A `callable` mapping a `Tensor` of images to a
      `tfd.Distribution` instance over encodings.
  """
  conv = functools.partial(
      tf.keras.layers.Conv2D, padding="SAME", activation=activation)

  encoder_net = tf.keras.Sequential([
      conv(base_depth, 5, 1),
      conv(base_depth, 5, 2),
      conv(2 * base_depth, 5, 1),
      conv(2 * base_depth, 5, 2),
      conv(4 * latent_size, 7, padding="VALID"),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(2 * latent_size, activation=None),
  ])

  def encoder(images):
    images = 2 * tf.cast(images, dtype=tf.float32) - 1
    net = encoder_net(images)
    return tfd.MultivariateNormalDiag(
        loc=net[..., :latent_size],
        scale_diag=tf.nn.softplus(net[..., latent_size:] +
                                  _softplus_inverse(1.0)),
        name="code")

  return encoder


def make_decoder(activation, latent_size, output_shape, base_depth):
  """Creates the decoder function.

  Args:
    activation: Activation function in hidden layers.
    latent_size: Dimensionality of the encoding.
    output_shape: The output image shape.
    base_depth: Smallest depth for a layer.

  Returns:
    decoder: A `callable` mapping a `Tensor` of encodings to a
      `tfd.Distribution` instance over images.
  """
  deconv = functools.partial(
      tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
  conv = functools.partial(
      tf.keras.layers.Conv2D, padding="SAME", activation=activation)

  decoder_net = tf.keras.Sequential([
      deconv(2 * base_depth, 7, padding="VALID"),
      deconv(2 * base_depth, 5),
      deconv(2 * base_depth, 5, 2),
      deconv(base_depth, 5),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5),
      conv(output_shape[-1], 5, activation=None),
  ])

  def decoder(codes):
    original_shape = tf.shape(input=codes)
    # Collapse the sample and batch dimension and convert to rank-4 tensor for
    # use with a convolutional decoder network.
    codes = tf.reshape(codes, (-1, 1, 1, latent_size))
    logits = decoder_net(codes)
    logits = tf.reshape(
        logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
    return tfd.Independent(tfd.Bernoulli(logits=logits),
                           reinterpreted_batch_ndims=len(output_shape),
                           name="image")

  return decoder


def make_mixture_prior(latent_size, mixture_components):
  """Creates the mixture of Gaussians prior distribution.

  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.

  Returns:
    random_prior: A `tfd.Distribution` instance representing the distribution
      over encodings in the absence of any evidence.
  """
  if mixture_components == 1:
    # See the module docstring for why we don't learn the parameters here.
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([latent_size]),
        scale_identity_multiplier=1.0)

  loc = tf.compat.v1.get_variable(
      name="loc", shape=[mixture_components, latent_size])
  raw_scale_diag = tf.compat.v1.get_variable(
      name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.compat.v1.get_variable(
      name="mixture_logits", shape=[mixture_components])

  return tfd.MixtureSameFamily(
      components_distribution=tfd.MultivariateNormalDiag(
          loc=loc,
          scale_diag=tf.nn.softplus(raw_scale_diag)),
      mixture_distribution=tfd.Categorical(logits=mixture_logits),
      name="prior")


def pack_images(images, rows, cols):
  """Helper utility to make a field of images."""
  shape = tf.shape(input=images)
  width = shape[-3]
  height = shape[-2]
  depth = shape[-1]
  images = tf.reshape(images, (-1, width, height, depth))
  batch = tf.shape(input=images)[0]
  rows = tf.minimum(rows, batch)
  cols = tf.minimum(batch // rows, cols)
  images = images[:rows * cols]
  images = tf.reshape(images, (rows, cols, width, height, depth))
  images = tf.transpose(a=images, perm=[0, 2, 1, 3, 4])
  images = tf.reshape(images, [1, rows * width, cols * height, depth])
  return images


def image_tile_summary(name, tensor, rows=8, cols=8):
  tf.compat.v1.summary.image(
      name, pack_images(tensor, rows, cols), max_outputs=1)


def model_fn(features, labels, mode, params, config):
  """Builds the model function for use in an estimator.

  Arguments:
    features: The input features for the estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.

  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """
  del labels, config

  if params["analytic_kl"] and params["mixture_components"] != 1:
    raise NotImplementedError(
        "Using `analytic_kl` is only supported when `mixture_components = 1` "
        "since there's no closed form otherwise.")

  encoder = make_encoder(params["activation"],
                         params["latent_size"],
                         params["base_depth"])
  decoder = make_decoder(params["activation"],
                         params["latent_size"],
                         IMAGE_SHAPE,
                         params["base_depth"])
  latent_prior = make_mixture_prior(params["latent_size"],
                                    params["mixture_components"])

  image_tile_summary(
      "input", tf.cast(features, dtype=tf.float32), rows=1, cols=16)

  approx_posterior = encoder(features)
  approx_posterior_sample = approx_posterior.sample(params["n_samples"])
  decoder_likelihood = decoder(approx_posterior_sample)
  image_tile_summary(
      "recon/sample",
      tf.cast(decoder_likelihood.sample()[:3, :16], dtype=tf.float32),
      rows=3,
      cols=16)
  image_tile_summary(
      "recon/mean",
      decoder_likelihood.mean()[:3, :16],
      rows=3,
      cols=16)

  # `distortion` is just the negative log likelihood.
  distortion = -decoder_likelihood.log_prob(features)
  avg_distortion = tf.reduce_mean(input_tensor=distortion)
  tf.compat.v1.summary.scalar("distortion", avg_distortion)

  if params["analytic_kl"]:
    rate = tfd.kl_divergence(approx_posterior, latent_prior)
  else:
    rate = (approx_posterior.log_prob(approx_posterior_sample)
            - latent_prior.log_prob(approx_posterior_sample))
  avg_rate = tf.reduce_mean(input_tensor=rate)
  tf.compat.v1.summary.scalar("rate", avg_rate)

  elbo_local = -(rate + distortion)

  elbo = tf.reduce_mean(input_tensor=elbo_local)
  loss = -elbo
  tf.compat.v1.summary.scalar("elbo", elbo)

  importance_weighted_elbo = tf.reduce_mean(
      input_tensor=tf.reduce_logsumexp(input_tensor=elbo_local, axis=0) -
      tf.math.log(tf.cast(params["n_samples"], dtype=tf.float32)))
  tf.compat.v1.summary.scalar("elbo/importance_weighted",
                              importance_weighted_elbo)

  # Decode samples from the prior for visualization.
  random_image = decoder(latent_prior.sample(16))
  image_tile_summary(
      "random/sample",
      tf.cast(random_image.sample(), dtype=tf.float32),
      rows=4,
      cols=4)
  image_tile_summary("random/mean", random_image.mean(), rows=4, cols=4)

  # Perform variational inference by minimizing the -ELBO.
  global_step = tf.compat.v1.train.get_or_create_global_step()
  learning_rate = tf.compat.v1.train.cosine_decay(
      params["learning_rate"], global_step, params["max_steps"])
  tf.compat.v1.summary.scalar("learning_rate", learning_rate)
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={
          "elbo":
              tf.compat.v1.metrics.mean(elbo),
          "elbo/importance_weighted":
              tf.compat.v1.metrics.mean(importance_weighted_elbo),
          "rate":
              tf.compat.v1.metrics.mean(avg_rate),
          "distortion":
              tf.compat.v1.metrics.mean(avg_distortion),
      },
  )


ROOT_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"


def download(directory, filename):
  """Downloads a file."""
  filepath = os.path.join(directory, filename)
  if tf.io.gfile.exists(filepath):
    return filepath
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)
  url = os.path.join(ROOT_PATH, filename)
  print("Downloading %s to %s" % (url, filepath))
  urllib.request.urlretrieve(url, filepath)
  return filepath


def static_mnist_dataset(directory, split_name):
  """Returns binary static MNIST tf.data.Dataset."""
  amat_file = download(directory, FILE_TEMPLATE.format(split=split_name))
  dataset = tf.data.TextLineDataset(amat_file)
  str_to_arr = lambda string: np.array([c == b"1" for c in string.split()])

  def _parser(s):
    booltensor = tf.compat.v1.py_func(str_to_arr, [s], tf.bool)
    reshaped = tf.reshape(booltensor, [28, 28, 1])
    return tf.cast(reshaped, dtype=tf.float32), tf.constant(0, tf.int32)

  return dataset.map(_parser)


def build_fake_input_fns(batch_size):
  """Builds fake MNIST-style data for unit testing."""
  random_sample = np.random.rand(batch_size, *IMAGE_SHAPE).astype("float32")

  def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        random_sample).map(lambda row: (row, 0)).batch(batch_size).repeat()
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        random_sample).map(lambda row: (row, 0)).batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn


def build_input_fns(data_dir, batch_size):
  """Builds an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  def train_input_fn():
    dataset = static_mnist_dataset(data_dir, "train")
    dataset = dataset.shuffle(50000).repeat().batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  # Build an iterator over the heldout set.
  def eval_input_fn():
    eval_dataset = static_mnist_dataset(data_dir, "valid")
    eval_dataset = eval_dataset.batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(eval_dataset).get_next()

  return train_input_fn, eval_input_fn


def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  params["activation"] = getattr(tf.nn, params["activation"])
  if FLAGS.delete_existing and tf.io.gfile.exists(FLAGS.model_dir):
    tf.compat.v1.logging.warn("Deleting old log directory at {}".format(
        FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)

  if FLAGS.fake_data:
    train_input_fn, eval_input_fn = build_fake_input_fns(FLAGS.batch_size)
  else:
    train_input_fn, eval_input_fn = build_input_fns(FLAGS.data_dir,
                                                    FLAGS.batch_size)

  estimator = tf.estimator.Estimator(
      model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.viz_steps,
      ),
  )

  for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
    estimator.train(train_input_fn, steps=FLAGS.viz_steps)
    eval_results = estimator.evaluate(eval_input_fn)
    print("Evaluation_results:\n\t%s\n" % eval_results)


if __name__ == "__main__":
  tf.compat.v1.app.run()
