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
"""Trains a variational auto-encoder (VAE) on dynamically binarized MNIST.

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

#### References

[1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
     _International Conference on Learning Representations_, 2014.
     https://arxiv.org/abs/1312.6114
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

# Dependency imports
from absl import flags
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.contrib.learn.python.learn.datasets import mnist

tfd = tf.contrib.distributions

IMAGE_SHAPE = [28, 28]

flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=10000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("latent_size",
                     default=16,
                     help="Number of dimensions in the latent code (z).")
flags.DEFINE_string("encoder_layers",
                    default="256,128",
                    help="Comma-separated list of layer sizes for the encoder.")
flags.DEFINE_string("decoder_layers",
                    default="128,256",
                    help="Comma-separated list of layer sizes for the decoder.")
flags.DEFINE_string("activation",
                    default="elu",
                    help="Activation function for all hidden layers.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size. Must divide evenly into dataset sizes.")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "vae/data"),
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=500,
                     help="Frequency at which save visualizations.")
flags.DEFINE_bool("fake_data",
                  default=False,
                  help="If true, uses fake data.")

FLAGS = flags.FLAGS


def make_encoder(images):
  """Build encoder which takes a batch of images and returns a latent code.

  Args:
    images: A `int`-like `Tensor` representing the inputs to be encoded.
      The first dimension (axis 0) indexes batch elements; all other
      dimensions index event elements.

  Returns:
    encoder: A multivariate `Normal` distribution.
  """
  encoder_net = tf.keras.Sequential()
  encoder_net.add(tf.keras.layers.Flatten())
  for units in FLAGS.encoder_layers:
    encoder_net.add(tf.keras.layers.Dense(units,
                                          activation=FLAGS.activation))
  encoder_net.add(tf.keras.layers.Dense(FLAGS.latent_size * 2,
                                        activation=None))
  images = tf.cast(images, dtype=tf.float32)
  net = encoder_net(images)
  loc = net[..., :FLAGS.latent_size]
  scale_diag = tf.nn.softplus(net[..., FLAGS.latent_size:] + 0.5)
  return tfd.MultivariateNormalDiag(loc=loc,
                                    scale_diag=scale_diag,
                                    name="encoder_distribution")


def make_decoder(codes):
  """Build decoder which takes a batch of codes and returns generated images.

  Args:
    codes: A `float`-like `Tensor` containing the latent
      vectors to be decoded. These are assumed to be rank-1, so
      the encoding `Tensor` is rank-2 with shape `[batch_size, latent_size]`.

  Returns:
    decoder: A multivariate `Bernoulli` distribution.
  """
  decoder_net = tf.keras.Sequential()
  for units in FLAGS.decoder_layers:
    decoder_net.add(tf.keras.layers.Dense(units,
                                          activation=FLAGS.activation))
  decoder_net.add(tf.keras.layers.Dense(np.prod(IMAGE_SHAPE),
                                        activation=None))
  net = decoder_net(codes)
  new_shape = tf.concat([tf.shape(net)[:-1], IMAGE_SHAPE], axis=0)
  logits = tf.reshape(net, shape=new_shape)
  return tfd.Independent(tfd.Bernoulli(logits=logits),
                         reinterpreted_batch_ndims=len(IMAGE_SHAPE),
                         name="decoder_distribution")


def make_prior():
  """Build prior distribution over latent codes.

  Returns:
    prior: A multivariate standard `Normal` distribution.
  """
  return tfd.MultivariateNormalDiag(scale_diag=tf.ones(FLAGS.latent_size),
                                    name="prior_distribution")


def make_vae(images, encoder_fn, decoder_fn, prior_fn, return_full=False):
  """Builds the variational auto-encoder and its loss function.

  Args:
    images: A `int`-like `Tensor` containing observed inputs X. The first
      dimension (axis 0) indexes batch elements; all other dimensions index
      event elements.
    encoder_fn: A callable to build the encoder `q(Z|X)`. This takes a single
      argument, a `int`-like `Tensor` representing a batch of inputs `X`, and
      returns a Distribution over the batch of latent codes `Z`.
    decoder_fn: A callable to build the decoder `p(X|Z)`. This takes a single
      argument, a `float`-like `Tensor` representing a batch of latent codes
      `Z`, and returns a Distribution over the batch of observations `X`.
    prior_fn: A callable to build the prior `p(Z)`. This takes no arguments and
      returns a Distribution over a single latent code (
    return_full: If True, also return the model components and the encoding.

  Returns:
    elbo_loss: A scalar `Tensor` computing the negation of the variational
      evidence bound (i.e., `elbo_loss >= -log p(X)`).
  """
  with tf.variable_scope("encoder"):
    encoder = encoder_fn(images)

  with tf.variable_scope("prior"):
    prior = prior_fn()

  def joint_log_prob(z):
    with tf.variable_scope("decoder"):
      decoder = decoder_fn(z)
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
      decoder = decoder_fn(encoding_draw)
    return elbo_loss, encoder, decoder, prior, encoding_draw

  return elbo_loss


def save_imgs(x, fname):
  """Helper method to save a grid of images to a PNG file.

  Args:
    x: A numpy array of shape [n_images, height, width].
    fname: The filename to write to (including extension).
  """
  n = x.shape[0]
  fig = figure.Figure(figsize=(n, 1), frameon=False)
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(1, n, i+1)
    ax.imshow(x[i].squeeze(),
              interpolation="none",
              cmap=cm.get_cmap("binary"))
    ax.axis("off")
  canvas.print_figure(fname, format="png")
  print("saved %s" % fname)


def visualize_training(images_val,
                       reconstructed_images_val,
                       random_images_val,
                       log_dir, prefix, viz_n=10):
  """Helper method to save images visualizing model reconstructions.

  Args:
    images_val: Numpy array containing a batch of input images.
    reconstructed_images_val: Numpy array giving the expected output
      (mean) of the decoder.
    random_images_val: Optionally, a Numpy array giving the expected output
      (mean) of decoding samples from the prior, or `None`.
    log_dir: The directory to write images (Python `str`).
    prefix: A specific label for the saved visualizations, which
      determines their filenames (Python `str`).
    viz_n: The number of images from each batch to visualize (Python `int`).
  """
  save_imgs(images_val[:viz_n],
            os.path.join(log_dir, "{}_inputs.png".format(prefix)))
  save_imgs(reconstructed_images_val[:viz_n],
            os.path.join(log_dir,
                         "{}_reconstructions.png".format(prefix)))

  if random_images_val is not None:
    save_imgs(random_images_val[:viz_n],
              os.path.join(log_dir,
                           "{}_prior_samples.png".format(prefix)))


def build_fake_data(num_examples=10):
  """Build fake MNIST-style data for unit testing."""

  class Dummy(object):
    pass

  num_examples = 10
  mnist_data = Dummy()
  mnist_data.train = Dummy()
  mnist_data.train.images = np.float32(np.random.randn(
      num_examples, np.prod(IMAGE_SHAPE)))
  mnist_data.train.labels = np.int32(np.random.permutation(
      np.arange(num_examples)))
  mnist_data.train.num_examples = num_examples
  mnist_data.validation = Dummy()
  mnist_data.validation.images = np.float32(np.random.randn(
      num_examples, np.prod(IMAGE_SHAPE)))
  mnist_data.validation.labels = np.int32(np.random.permutation(
      np.arange(num_examples)))
  mnist_data.validation.num_examples = num_examples
  return mnist_data


def build_input_pipeline(mnist_data, batch_size, heldout_size):
  """Build an Iterator switching between train and heldout data."""
  # Build an iterator over training batches.
  training_dataset = tf.data.Dataset.from_tensor_slices(
      (mnist_data.train.images, np.int32(mnist_data.train.labels)))
  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = training_batches.make_one_shot_iterator()

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  heldout_dataset = tf.data.Dataset.from_tensor_slices(
      (mnist_data.validation.images,
       np.int32(mnist_data.validation.labels)))
  heldout_frozen = (heldout_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
  heldout_iterator = heldout_frozen.make_one_shot_iterator()

  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.placeholder(tf.string, shape=[])
  feedable_iterator = tf.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()

  return images, labels, handle, training_iterator, heldout_iterator


def main(argv):
  del argv  # unused
  FLAGS.encoder_layers = [int(units) for units
                          in FLAGS.encoder_layers.split(",")]
  FLAGS.decoder_layers = [int(units) for units
                          in FLAGS.decoder_layers.split(",")]
  FLAGS.activation = getattr(tf.nn, FLAGS.activation)
  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  if FLAGS.fake_data:
    mnist_data = build_fake_data()
  else:
    mnist_data = mnist.read_data_sets(FLAGS.data_dir)

  with tf.Graph().as_default():
    (images, _, handle,
     training_iterator, heldout_iterator) = build_input_pipeline(
         mnist_data, FLAGS.batch_size, mnist_data.validation.num_examples)

    # Reshape as a pixel image and dynamically binarize pixels.
    images = tf.reshape(images, shape=[-1] + IMAGE_SHAPE)
    images = tf.cast(images > 0.5, dtype=tf.int32)

    # Build the model and ELBO loss function.
    elbo_loss, _, decoder, prior, _ = make_vae(images,
                                               make_encoder,
                                               make_decoder,
                                               make_prior,
                                               return_full=True)
    reconstructed_images = decoder.mean()

    # Decode samples from the prior for visualization.
    prior_samples = prior.sample(10)
    with tf.variable_scope("decoder", reuse=True):
      decoded = make_decoder(prior_samples)
      random_images = decoded.mean()

    # Perform variational inference by minimizing the -ELBO.
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_op = optimizer.minimize(elbo_loss)

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
      sess.run(init)

      # Run the training loop.
      train_handle = sess.run(training_iterator.string_handle())
      heldout_handle = sess.run(heldout_iterator.string_handle())
      for step in range(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, elbo_loss],
                                 feed_dict={handle: train_handle})
        duration = time.time() - start_time
        if step % 100 == 0:
          print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(
              step, loss_value, duration))

          # Update the events file.
          summary_str = sess.run(summary, feed_dict={handle: train_handle})
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush()

        # Periodically save a checkpoint and visualize model progress.
        if (step + 1) % FLAGS.viz_steps == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_file = os.path.join(FLAGS.model_dir, "model.ckpt")
          saver.save(sess, checkpoint_file, global_step=step)

          # Visualize inputs and model reconstructions from the training set.
          images_val, reconstructions_val, random_images_val = sess.run(
              (images, reconstructed_images, random_images),
              feed_dict={handle: train_handle})
          visualize_training(images_val,
                             reconstructions_val,
                             random_images_val,
                             log_dir=FLAGS.model_dir,
                             prefix="step{:05d}_train".format(step))

          # Visualize inputs and model reconstructions from the validation set.
          heldout_images_val, heldout_reconstructions_val = sess.run(
              (images, reconstructed_images),
              feed_dict={handle: heldout_handle})
          visualize_training(heldout_images_val,
                             heldout_reconstructions_val,
                             None,
                             log_dir=FLAGS.model_dir,
                             prefix="step{:05d}_validation".format(step))

if __name__ == "__main__":
  tf.app.run()
