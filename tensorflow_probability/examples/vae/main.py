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
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Trains a variational autoencoder (VAE) on MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

# Dependency imports
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf

from tensorflow_probability.examples.vae import vae_model
from tensorflow.contrib.learn.python.learn.datasets import mnist

# Hardcode the image shape. Change this if using non-MNIST datasets.
IMAGE_SHAPE = [28, 28]


def save_imgs(x, fname):
  """Helper method to save a grid of images to a PNG file.

  Args:
    x: A numpy array of shape [n_images, height, width].
    fname: The filename to write to (including extension).

  Raises:
    ImportError: if matplotlib is not available.
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


def run_training():
  """Constructs the model and runs the training loop."""

  if FLAGS.fake_data:
    # Generate dummy images and labels for fast unit testing
    mnist_data = build_fake_data()
  else:
    # Get MNIST images and labels for training, validation, and test sets.
    mnist_data = mnist.read_data_sets(FLAGS.input_data_dir)

  with tf.Graph().as_default():

    (images, _, handle,
     training_iterator, heldout_iterator) = build_input_pipeline(
         mnist_data, FLAGS.batch_size, mnist_data.validation.num_examples)

    images = tf.reshape(images, [-1] + IMAGE_SHAPE)

    # Build the model and ELBO loss function
    def my_encoder(images):
      net = vae_model.make_encoder_net(
          images,
          num_outputs=FLAGS.latent_size * 2,
          hidden_layer_sizes=FLAGS.encoder_layers,
          activation=FLAGS.activation)
      return vae_model.make_encoder_mvndiag(net, FLAGS.latent_size)

    def my_decoder(encoding):
      net = vae_model.make_decoder_net(
          encoding,
          num_outputs=np.prod(IMAGE_SHAPE),
          hidden_layer_sizes=FLAGS.decoder_layers,
          activation=FLAGS.activation)
      return vae_model.make_decoder_bernoulli(net, IMAGE_SHAPE)

    def my_prior():
      return vae_model.make_prior_mvndiag(FLAGS.latent_size)

    elbo_loss, _, decoder, prior, _ = vae_model.make_vae(images,
                                                         my_encoder,
                                                         my_decoder,
                                                         my_prior,
                                                         return_full=True)
    reconstructed_images = decoder.mean()

    # Also decode some samples from the prior, just for visualization
    prior_samples = prior.sample(10)
    with tf.variable_scope("decoder", reuse=True):
      decoded = my_decoder(prior_samples)
      random_images = decoded.mean()

    # Perform variational inference by maximizing the ELBO
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_op = optimizer.minimize(elbo_loss)

    # General bookkeeping: construct the session and summary writer.
    summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
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
          print("Step {}: loss = {:.2f} ({:.3f} sec)".format
                (step, loss_value, duration))

          # Update the events file.
          summary_str = sess.run(summary, feed_dict={handle: train_handle})
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush()

        # Periodically save a checkpoint and visualize model progress.
        if (step+1) % 500 == 0 or (step + 1) == FLAGS.max_steps:

          checkpoint_file = os.path.join(FLAGS.log_dir, "model.ckpt")
          saver.save(sess, checkpoint_file, global_step=step)

          # Visualize inputs and model reconstructions from the
          # training set...
          images_val, reconstructions_val, random_images_val = sess.run(
              (images, reconstructed_images, random_images),
              feed_dict={handle: train_handle})
          visualize_training(images_val,
                             reconstructions_val,
                             random_images_val,
                             log_dir=FLAGS.log_dir,
                             prefix="step{:05d}_train".format(step))

          # ... and from the validation set.
          heldout_images_val, heldout_reconstructions_val = sess.run(
              (images, reconstructed_images),
              feed_dict={handle: heldout_handle})
          visualize_training(heldout_images_val,
                             heldout_reconstructions_val,
                             None,
                             log_dir=FLAGS.log_dir,
                             prefix="step{:05d}_validation".format(step))


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.logging.warn("Deleting old log directory at {}".format(FLAGS.log_dir))
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  run_training()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.01,
      help="Initial learning rate."
  )
  parser.add_argument(
      "--max_steps",
      type=int,
      default=2800,
      help="Number of training steps to run."
  )
  parser.add_argument(
      "--latent_size",
      type=int,
      default=8,
      help="Number of dimensions in the latent code (z)."
  )
  parser.add_argument(
      "--encoder_layers",
      type=str,
      default="128,32",
      help="Comma-separated list of layer sizes for the encoder."
  )
  parser.add_argument(
      "--decoder_layers",
      type=str,
      default="32,128",
      help="Comma-separated list of layer sizes for the decoder."
  )
  parser.add_argument(
      "--activation",
      type=str,
      default="elu",
      help="Activation function for the encoder and decoder networks."
      )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=128,
      help="Batch size.  Must divide evenly into the dataset sizes."
  )
  parser.add_argument(
      "--input_data_dir",
      type=str,
      default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                           "vae/input_data"),
      help="Directory to put the input data."
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                           "vae/logs/"),
      help="Directory to put the log data."
  )
  parser.add_argument(
      "--viz_steps",
      type=int,
      default=400,
      help="Frequency at which save visualizations."
  )
  parser.add_argument(
      "--fake_data",
      default=False,
      help="If true, uses fake data for unit testing.",
      action="store_true"
  )

  FLAGS, unparsed = parser.parse_known_args()

  FLAGS.encoder_layers = [int(units) for units
                          in FLAGS.encoder_layers.split(",")]
  FLAGS.decoder_layers = [int(units) for units
                          in FLAGS.decoder_layers.split(",")]

  FLAGS.activation = tf.nn.__getattribute__(FLAGS.activation)

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
