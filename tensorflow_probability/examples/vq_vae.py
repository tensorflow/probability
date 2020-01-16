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
"""Trains a vector quantized-variational autoencoder (VQ-VAE) on MNIST.

The VQ-VAE is similar to a variational autoencoder (VAE), but the latent
code Z goes through a discrete bottleneck before being passed to the encoder.
The bottleneck uses vector quantization to match the latent code to its nearest
neighbor in a codebook. To train, we minimize the weighted sum of the
reconstruction loss and a commitment loss that ensures the encoder commits to
entries in the codebook. In addition, we use exponential moving averaging (EMA)
to update the codebook for each minibatch.

#### References

[1]: Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. Neural Discrete
     Representation Learning. In _Conference on Neural Information Processing
     Systems_, 2017. https://arxiv.org/abs/1711.00937
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

# Dependency imports
from absl import flags
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
from six.moves import urllib
import tensorflow.compat.v1 as tf

from tensorflow_probability import distributions as tfd
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.python.training import moving_averages

IMAGE_SHAPE = [28, 28, 1]

flags.DEFINE_float("learning_rate",
                   default=0.001,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=10000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("latent_size",
                     default=1,
                     help="Number of latent variables.")
flags.DEFINE_integer("num_codes",
                     default=64,
                     help="Number of discrete codes in codebook.")
flags.DEFINE_integer("code_size",
                     default=16,
                     help="Dimension of each entry in codebook.")
flags.DEFINE_integer("base_depth",
                     default=32,
                     help="Base depth for encoder and decoder CNNs.")
flags.DEFINE_string("activation",
                    default="elu",
                    help="Activation function for all hidden layers.")
flags.DEFINE_float("beta",
                   default=0.25,
                   help="Scaling for commitment loss.")
flags.DEFINE_float("decay",
                   default=0.99,
                   help="Decay for exponential moving average.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size.")
flags.DEFINE_string("mnist_type",
                    default="threshold",
                    help="""Type of MNIST used. Choices include 'fake_data',
                    'bernoulli' for Hugo Larochelle's randomly binarized MNIST,
                     and 'threshold' for binarized MNIST at 0.5 threshold.""")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "vq_vae/data"),
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vq_vae/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=500,
                     help="Frequency at which to save visualizations.")

FLAGS = flags.FLAGS
BERNOULLI_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"


class MnistType(object):
  """MNIST types for input data."""
  FAKE_DATA = "fake_data"
  THRESHOLD = "threshold"
  BERNOULLI = "bernoulli"


class VectorQuantizer(object):
  """Creates a vector-quantizer.

  It quantizes a continuous vector under a codebook. The codebook is also known
  as "embeddings" or "memory", and it is learned using an exponential moving
  average.
  """

  def __init__(self, num_codes, code_size):
    self.num_codes = num_codes
    self.code_size = code_size
    self.codebook = tf.compat.v1.get_variable(
        "codebook",
        [num_codes, code_size],
        dtype=tf.float32,
    )
    self.ema_count = tf.compat.v1.get_variable(
        name="ema_count",
        shape=[num_codes],
        initializer=tf.compat.v1.initializers.constant(0),
        trainable=False)
    self.ema_means = tf.compat.v1.get_variable(
        name="ema_means",
        initializer=self.codebook.initialized_value(),
        trainable=False)

  def __call__(self, codes):
    """Uses codebook to find nearest neighbor for each code.

    Args:
      codes: A `float`-like `Tensor` containing the latent
        vectors to be compared to the codebook. These are rank-3 with shape
        `[batch_size, latent_size, code_size]`.

    Returns:
      nearest_codebook_entries: The 1-nearest neighbor in Euclidean distance for
        each code in the batch.
      one_hot_assignments: The one-hot vectors corresponding to the matched
        codebook entry for each code in the batch.
    """
    distances = tf.norm(
        tensor=tf.expand_dims(codes, 2) -
        tf.reshape(self.codebook, [1, 1, self.num_codes, self.code_size]),
        axis=3)
    assignments = tf.argmin(input=distances, axis=2)
    one_hot_assignments = tf.one_hot(assignments, depth=self.num_codes)
    nearest_codebook_entries = tf.reduce_sum(
        input_tensor=tf.expand_dims(one_hot_assignments, -1) *
        tf.reshape(self.codebook, [1, 1, self.num_codes, self.code_size]),
        axis=2)
    return nearest_codebook_entries, one_hot_assignments


def make_encoder(base_depth, activation, latent_size, code_size):
  """Creates the encoder function.

  Args:
    base_depth: Layer base depth in encoder net.
    activation: Activation function in hidden layers.
    latent_size: The number of latent variables in the code.
    code_size: The dimensionality of each latent variable.

  Returns:
    encoder: A `callable` mapping a `Tensor` of images to a `Tensor` of shape
      `[..., latent_size, code_size]`.
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
      tf.keras.layers.Dense(latent_size * code_size, activation=None),
      tf.keras.layers.Reshape([latent_size, code_size])
  ])

  def encoder(images):
    """Encodes a batch of images.

    Args:
      images: A `Tensor` representing the inputs to be encoded, of shape `[...,
        channels]`.

    Returns:
      codes: A `float`-like `Tensor` of shape `[..., latent_size, code_size]`.
        It represents latent vectors to be matched with the codebook.
    """
    images = 2 * tf.cast(images, dtype=tf.float32) - 1
    codes = encoder_net(images)
    return codes

  return encoder


def make_decoder(base_depth, activation, input_size, output_shape):
  """Creates the decoder function.

  Args:
    base_depth: Layer base depth in decoder net.
    activation: Activation function in hidden layers.
    input_size: The flattened latent input shape as an int.
    output_shape: The output image shape as a list.

  Returns:
    decoder: A `callable` mapping a `Tensor` of encodings to a
      `tfd.Distribution` instance over images.
  """
  deconv = functools.partial(
      tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
  conv = functools.partial(
      tf.keras.layers.Conv2D, padding="SAME", activation=activation)
  decoder_net = tf.keras.Sequential([
      tf.keras.layers.Reshape((1, 1, input_size)),
      deconv(2 * base_depth, 7, padding="VALID"),
      deconv(2 * base_depth, 5),
      deconv(2 * base_depth, 5, 2),
      deconv(base_depth, 5),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5),
      conv(output_shape[-1], 5, activation=None),
      tf.keras.layers.Reshape(output_shape),
  ])

  def decoder(codes):
    """Builds a distribution over images given codes.

    Args:
      codes: A `Tensor` representing the inputs to be decoded, of shape `[...,
        code_size]`.

    Returns:
      decoder_distribution: A multivariate `Bernoulli` distribution.
    """
    logits = decoder_net(codes)
    return tfd.Independent(tfd.Bernoulli(logits=logits),
                           reinterpreted_batch_ndims=len(output_shape),
                           name="decoder_distribution")

  return decoder


def add_ema_control_dependencies(vector_quantizer,
                                 one_hot_assignments,
                                 codes,
                                 commitment_loss,
                                 decay):
  """Add control dependencies to the commmitment loss to update the codebook.

  Args:
    vector_quantizer: An instance of the VectorQuantizer class.
    one_hot_assignments: The one-hot vectors corresponding to the matched
      codebook entry for each code in the batch.
    codes: A `float`-like `Tensor` containing the latent vectors to be compared
      to the codebook.
    commitment_loss: The commitment loss from comparing the encoder outputs to
      their neighboring codebook entries.
    decay: Decay factor for exponential moving average.

  Returns:
    commitment_loss: Commitment loss with control dependencies.
  """
  # Use an exponential moving average to update the codebook.
  updated_ema_count = moving_averages.assign_moving_average(
      vector_quantizer.ema_count,
      tf.reduce_sum(input_tensor=one_hot_assignments, axis=[0, 1]),
      decay,
      zero_debias=False)
  updated_ema_means = moving_averages.assign_moving_average(
      vector_quantizer.ema_means,
      tf.reduce_sum(
          input_tensor=tf.expand_dims(codes, 2) *
          tf.expand_dims(one_hot_assignments, 3),
          axis=[0, 1]),
      decay,
      zero_debias=False)

  # Add small value to avoid dividing by zero.
  perturbed_ema_count = updated_ema_count + 1e-5
  with tf.control_dependencies([commitment_loss]):
    update_means = tf.compat.v1.assign(
        vector_quantizer.codebook,
        updated_ema_means / perturbed_ema_count[..., tf.newaxis])
    with tf.control_dependencies([update_means]):
      return tf.identity(commitment_loss)


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
  """Builds fake MNIST-style data for unit testing."""

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


def download(directory, filename):
  """Downloads a file."""
  filepath = os.path.join(directory, filename)
  if tf.io.gfile.exists(filepath):
    return filepath
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)
  url = os.path.join(BERNOULLI_PATH, filename)
  print("Downloading %s to %s" % (url, filepath))
  urllib.request.urlretrieve(url, filepath)
  return filepath


def load_bernoulli_mnist_dataset(directory, split_name):
  """Returns Hugo Larochelle's binary static MNIST tf.data.Dataset."""
  amat_file = download(directory, FILE_TEMPLATE.format(split=split_name))
  dataset = tf.data.TextLineDataset(amat_file)
  str_to_arr = lambda string: np.array([c == b"1" for c in string.split()])

  def _parser(s):
    booltensor = tf.compat.v1.py_func(str_to_arr, [s], tf.bool)
    reshaped = tf.reshape(booltensor, [28, 28, 1])
    return tf.cast(reshaped, dtype=tf.float32), tf.constant(0, tf.int32)

  return dataset.map(_parser)


def build_input_pipeline(data_dir, batch_size, heldout_size, mnist_type):
  """Builds an Iterator switching between train and heldout data."""
  # Build an iterator over training batches.
  if mnist_type in [MnistType.FAKE_DATA, MnistType.THRESHOLD]:
    if mnist_type == MnistType.FAKE_DATA:
      mnist_data = build_fake_data()
    else:
      mnist_data = mnist.read_data_sets(data_dir)
    training_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.train.images, np.int32(mnist_data.train.labels)))
    heldout_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.validation.images,
         np.int32(mnist_data.validation.labels)))
  elif mnist_type == MnistType.BERNOULLI:
    training_dataset = load_bernoulli_mnist_dataset(data_dir, "train")
    heldout_dataset = load_bernoulli_mnist_dataset(data_dir, "valid")
  else:
    raise ValueError("Unknown MNIST type.")

  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  heldout_frozen = (heldout_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
  heldout_iterator = tf.compat.v1.data.make_one_shot_iterator(heldout_frozen)

  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.compat.v1.placeholder(tf.string, shape=[])
  feedable_iterator = tf.compat.v1.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()
  # Reshape as a pixel image and binarize pixels.
  images = tf.reshape(images, shape=[-1] + IMAGE_SHAPE)
  if mnist_type in [MnistType.FAKE_DATA, MnistType.THRESHOLD]:
    images = tf.cast(images > 0.5, dtype=tf.int32)

  return images, labels, handle, training_iterator, heldout_iterator


def main(argv):
  del argv  # unused
  FLAGS.activation = getattr(tf.nn, FLAGS.activation)
  if tf.io.gfile.exists(FLAGS.model_dir):
    tf.compat.v1.logging.warn("Deleting old log directory at {}".format(
        FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)

  with tf.Graph().as_default():
    # TODO(b/113163167): Speed up and tune hyperparameters for Bernoulli MNIST.
    (images, _, handle,
     training_iterator, heldout_iterator) = build_input_pipeline(
         FLAGS.data_dir, FLAGS.batch_size, heldout_size=10000,
         mnist_type=FLAGS.mnist_type)

    encoder = make_encoder(FLAGS.base_depth,
                           FLAGS.activation,
                           FLAGS.latent_size,
                           FLAGS.code_size)
    decoder = make_decoder(FLAGS.base_depth,
                           FLAGS.activation,
                           FLAGS.latent_size * FLAGS.code_size,
                           IMAGE_SHAPE)
    vector_quantizer = VectorQuantizer(FLAGS.num_codes, FLAGS.code_size)

    codes = encoder(images)
    nearest_codebook_entries, one_hot_assignments = vector_quantizer(codes)
    codes_straight_through = codes + tf.stop_gradient(
        nearest_codebook_entries - codes)
    decoder_distribution = decoder(codes_straight_through)
    reconstructed_images = decoder_distribution.mean()

    reconstruction_loss = -tf.reduce_mean(
        input_tensor=decoder_distribution.log_prob(images))
    commitment_loss = tf.reduce_mean(
        input_tensor=tf.square(codes -
                               tf.stop_gradient(nearest_codebook_entries)))
    commitment_loss = add_ema_control_dependencies(
        vector_quantizer,
        one_hot_assignments,
        codes,
        commitment_loss,
        FLAGS.decay)
    prior_dist = tfd.Multinomial(
        total_count=1.0, logits=tf.zeros([FLAGS.latent_size, FLAGS.num_codes]))
    prior_loss = -tf.reduce_mean(
        input_tensor=tf.reduce_sum(
            input_tensor=prior_dist.log_prob(one_hot_assignments), axis=1))

    loss = reconstruction_loss + FLAGS.beta * commitment_loss + prior_loss
    # Upper bound marginal negative log-likelihood as prior loss +
    # reconstruction loss.
    marginal_nll = prior_loss + reconstruction_loss

    tf.compat.v1.summary.scalar("losses/total_loss", loss)
    tf.compat.v1.summary.scalar("losses/reconstruction_loss",
                                reconstruction_loss)
    tf.compat.v1.summary.scalar("losses/prior_loss", prior_loss)
    tf.compat.v1.summary.scalar("losses/commitment_loss",
                                FLAGS.beta * commitment_loss)

    # Decode samples from a uniform prior for visualization.
    prior_samples = tf.reduce_sum(
        input_tensor=tf.expand_dims(prior_dist.sample(10), -1) *
        tf.reshape(vector_quantizer.codebook,
                   [1, 1, FLAGS.num_codes, FLAGS.code_size]),
        axis=2)
    decoded_distribution_given_random_prior = decoder(prior_samples)
    random_images = decoded_distribution_given_random_prior.mean()

    # Perform inference by minimizing the loss function.
    optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate)
    train_op = optimizer.minimize(loss)

    summary = tf.compat.v1.summary.merge_all()
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
      summary_writer = tf.compat.v1.summary.FileWriter(FLAGS.model_dir,
                                                       sess.graph)
      sess.run(init)

      # Run the training loop.
      train_handle = sess.run(training_iterator.string_handle())
      heldout_handle = sess.run(heldout_iterator.string_handle())
      for step in range(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict={handle: train_handle})
        duration = time.time() - start_time
        if step % 100 == 0:
          marginal_nll_val = sess.run(marginal_nll,
                                      feed_dict={handle: heldout_handle})
          print("Step: {:>3d} Training Loss: {:.3f} Heldout NLL: {:.3f} "
                "({:.3f} sec)".format(step, loss_value, marginal_nll_val,
                                      duration))

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
  tf.compat.v1.app.run()
