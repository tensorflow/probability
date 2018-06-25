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

The VAE defines a generative model in which a latent code `Z` is
sampled from a prior `p(Z)`, then used to generate an observation `X`
by way of a decoder `p(X|Z)`. To fit the model, we assume an approximate
representation of the posterior in the form of an encoder
`q(Z|X)`. We minimize the KL divergence between `q(Z|X)` and the
true posterior `p(Z|X)`: this is equivalent to maximizing the evidence
lower bound (ELBO)

```none
 L =  E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z)]
   <= log p(X)
```

which also provides a lower bound on the marginal likelihood `p(X)`. See
[Kingma and Welling (2014)][1] for more details.

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
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

IMAGE_SHAPE = (28, 28, 1)

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
    help="Batch size. Must divide evenly into dataset sizes.")
flags.DEFINE_integer(
    "n_samples", default=16, help="Number of samples to use in encoding.")
flags.DEFINE_integer(
    "mixture_components", default=100, help="Number of mixture components.")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=500, help="Frequency at which save visualizations.")
flags.DEFINE_bool("fake_data", default=False, help="If true, uses fake data.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing directory.")

FLAGS = flags.FLAGS


def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.log(tf.expm1(x))


def make_encoder(activation, latent_size, base_depth):
  """Create the encoder function.

  Args:
    activation: Activation function to use.
    latent_size: The dimensionality of the encoding.
    base_depth: The lowest depth for a layer.

  Returns:
    encoder: A `callable` mapping a `Tensor` of images to a
      `tf.distributions.Distribution` instance over encodings.
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
  """Create the decoder function.

  Args:
    activation: Activation function to use.
    latent_size: Dimensionality of the encoding.
    output_shape: The output image shape.
    base_depth: Smallest depth for a layer.

  Returns:
    decoder: A `callable` mapping a `Tensor` of encodings to a
      `tf.distributions.Distribution` instance over images.
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
    original_shape = tf.shape(codes)
    # Collapse the sample and batch dimension
    # and convert to rank-4 tensor for use with
    # a convolutional decoder network.
    codes = tf.reshape(codes, (-1, 1, 1, latent_size))
    logits = decoder_net(codes)
    logits = tf.reshape(
        logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
    return tfd.Independent(
        tfd.Bernoulli(logits=logits),
        reinterpreted_batch_ndims=len(output_shape),
        name="image")

  return decoder


def make_random_prior(latent_size, mixture_components):
  """Create the prior distribution.

  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.

  Returns:
    random_prior: A `tf.distributions.Distribution` instance
      representing the distribution over encodings in the absence of any
      evidence.
  """
  loc = tf.get_variable(name="loc", shape=[mixture_components, latent_size])
  raw_scale_diag = tf.get_variable(
      name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.get_variable(
      name="mixture_logits", shape=[mixture_components])

  return tfd.MixtureSameFamily(
      components_distribution=tfd.MultivariateNormalDiag(
          loc=loc,
          scale_diag=tf.nn.softplus(raw_scale_diag + _softplus_inverse(1.))),
      mixture_distribution=tfd.Categorical(logits=mixture_logits),
      name="prior")


def pack_images(images, rows, cols):
  """Helper utility to make a field of images."""
  shape = tf.shape(images)
  width = shape[-3]
  height = shape[-2]
  depth = shape[-1]
  images = tf.reshape(images, (-1, width, height, depth))
  batch = tf.shape(images)[0]
  rows = tf.minimum(rows, batch)
  cols = tf.minimum(batch // rows, cols)
  images = images[:rows * cols]
  images = tf.reshape(images, (rows, cols, width, height, depth))
  images = tf.transpose(images, [0, 2, 1, 3, 4])
  images = tf.reshape(images, [1, rows * width, cols * height, depth])
  return images


def image_tile_summary(name, tensor, rows=8, cols=8):
  tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)


def model_fn(features, labels, mode, params, config):
  """Build the model function for use in an estimator.

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
  encoder = make_encoder(params["activation"],
                         params["latent_size"],
                         params["base_depth"])
  decoder = make_decoder(params["activation"],
                         params["latent_size"],
                         IMAGE_SHAPE,
                         params["base_depth"])
  random_prior = make_random_prior(params["latent_size"],
                                   params["mixture_components"])

  image_tile_summary("input", tf.to_float(features), rows=1, cols=16)

  random_encoding = encoder(features)
  encoding = random_encoding.sample(params["n_samples"])
  random_reconstruction = decoder(encoding)
  image_tile_summary(
      "recon/sample",
      tf.to_float(random_reconstruction.sample()[:3, :16]),
      rows=3,
      cols=16)
  image_tile_summary(
      "recon/mean",
      random_reconstruction.mean()[:3, :16],
      rows=3,
      cols=16)

  distortion = -random_reconstruction.log_prob(features)
  avg_distortion = tf.reduce_mean(distortion)
  tf.summary.scalar("distortion", avg_distortion)
  rate = random_encoding.log_prob(encoding) - random_prior.log_prob(encoding)
  avg_rate = tf.reduce_mean(rate)
  tf.summary.scalar("rate", avg_rate)

  elbo_local = -(rate + distortion)

  elbo = tf.reduce_mean(elbo_local)
  loss = -elbo
  tf.summary.scalar("elbo", elbo)
  importance_weighted_elbo = tf.reduce_mean(
      tf.reduce_logsumexp(elbo_local, axis=0) -
      tf.log(tf.to_float(params["n_samples"])))
  tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)

  # Decode samples from the prior for visualization.
  prior = random_prior.sample(16)
  random_image = decoder(prior)
  image_tile_summary(
      "random/sample", tf.to_float(random_image.sample()), rows=4, cols=4)
  image_tile_summary("random/mean", random_image.mean(), rows=4, cols=4)

  # Perform variational inference by minimizing the -ELBO.
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                        params["max_steps"])
  tf.summary.scalar("learning_rate", learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={
          "elbo": tf.metrics.mean(elbo),
          "elbo/importance_weighted": tf.metrics.mean(importance_weighted_elbo),
          "rate": tf.metrics.mean(avg_rate),
          "distortion": tf.metrics.mean(avg_distortion),
      },
  )


ROOT_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"


def download(directory, filename):
  """Download a file."""
  filepath = os.path.join(directory, filename)
  if tf.gfile.Exists(filepath):
    return filepath
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  url = os.path.join(ROOT_PATH, filename)
  print("Downloading %s to %s" % (url, filepath))
  urllib.request.urlretrieve(url, filepath)
  return filepath


def static_mnist_dataset(directory, split_name):
  """Return binary static MNIST tf.data.Dataset."""
  amat_file = download(directory, FILE_TEMPLATE.format(split=split_name))
  dataset = tf.data.TextLineDataset(amat_file)
  str_to_arr = lambda string: np.array([char == "1" for char in string.split()])

  def _parser(s):
    booltensor = tf.py_func(str_to_arr, [s], tf.bool)
    reshaped = tf.reshape(booltensor, [28, 28, 1])
    return tf.to_float(reshaped), tf.constant(0, tf.int32)

  return dataset.map(_parser)


def build_fake_input_fns(batch_size):
  """Build fake MNIST-style data for unit testing."""
  dataset = tf.data.Dataset.from_tensor_slices(
      np.random.rand(batch_size, *IMAGE_SHAPE).astype("float32")).map(
          lambda row: (row, 0)).batch(batch_size)

  train_input_fn = lambda: dataset.repeat().make_one_shot_iterator().get_next()
  eval_input_fn = lambda: dataset.make_one_shot_iterator().get_next()
  return train_input_fn, eval_input_fn


def build_input_fns(data_dir, batch_size):
  """Build an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  training_dataset = static_mnist_dataset(data_dir, "train")
  training_dataset = training_dataset.shuffle(50000).repeat().batch(batch_size)
  train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

  # Build an iterator over the heldout set.
  eval_dataset = static_mnist_dataset(data_dir, "valid")
  eval_dataset = eval_dataset.batch(batch_size)
  eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()

  return train_input_fn, eval_input_fn


def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  params["activation"] = getattr(tf.nn, params["activation"])
  if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

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
  tf.app.run()
