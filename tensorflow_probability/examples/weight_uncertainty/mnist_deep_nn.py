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
"""Trains a deep Bayesian neural net to classify MNIST digits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse  # TODO(b/74538173): Use absl.flags rather than argparse.
import os
import sys

# Dependency imports
import matplotlib
matplotlib.use("Agg")
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow_probability.examples.weight_uncertainty import weight_uncertainty
from tensorflow.contrib.learn.python.learn.datasets import mnist

tfd = tf.contrib.distributions

IMAGE_SHAPE = (28, 28)


def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
  """Save a PNG plot with histograms of weight means and stddevs.

  Args:
    names: A Python `iterable` of `str` variable names.
    qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.

  Raises:
    ImportError: if matplotlib is not available.
  """

  fig = figure.Figure(figsize=(6, 3))
  canvas = backend_agg.FigureCanvasAgg(fig)

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names, qm_vals):
    sns.distplot(qm.flatten(), ax=ax, label=n)
  ax.set_title("weight means")
  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([0, 4.])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names, qs_vals):
    sns.distplot(qs.flatten(), ax=ax)
  ax.set_title("weight stddevs")
  ax.set_xlim([0, 1.])
  ax.set_ylim([0, 25.])

  fig.tight_layout()
  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))


def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=""):
  """Save a PNG plot visualizing posterior uncertainty on heldout data.

  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `(n_heldout,) + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `(n_monte_carlo,
      n_heldout, n_classes)` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.

  Raises:
    ImportError: if matplotlib is not available.
  """

  fig = figure.Figure(figsize=(9, 3*n))
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(n, 3, 3*i + 1)
    ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE), interpolation="None")

    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title("posterior samples")

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(10), np.mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title("predictive probs")
  fig.suptitle(title)
  fig.tight_layout()

  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))


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


def run_training():
  """Run the main training loop."""

  if FLAGS.fake_data:
    # Generate dummy images and labels for fast unit testing
    mnist_data = build_fake_data()
  else:
    mnist_data = mnist.read_data_sets(FLAGS.input_data_dir)

  def build_deep_classifier(images, name=None):
    with tf.name_scope(name, "build_deep_classifier", [images]):
      net = images
      for layer_size in FLAGS.encoder_layers:
        net = tf.layers.dense(net, layer_size, activation=FLAGS.activation)
      logits = tf.layers.dense(net, 10, activation=None)
      model = tfd.Categorical(logits=logits)
      return model

  with tf.Graph().as_default():

    (images, labels, handle,
     training_iterator, heldout_iterator) = build_input_pipeline(
         mnist_data, FLAGS.batch_size, mnist_data.validation.num_examples)

    # Build a Bayesian neural net.
    elbo_loss, model = weight_uncertainty.bayesianify(
        build_deep_classifier, images, labels, mnist_data.train.num_examples)

    # Extract weight posterior statistics for later visualization.
    qs = tf.get_collection(weight_uncertainty.VI_QDISTS)
    names, qmeans, qstds = zip(*[
        (q.name[6:-10], q.mean(), q.stddev()) for q in qs])

    with tf.name_scope("train"):
      opt = tf.train.AdamOptimizer(
          learning_rate=FLAGS.learning_rate)

      train_op = opt.minimize(elbo_loss)
      init = tf.global_variables_initializer()
      sess = tf.Session()
      sess.run(init)

      # Run the training loop
      train_handle = sess.run(training_iterator.string_handle())
      heldout_handle = sess.run(heldout_iterator.string_handle())
      for step in range(FLAGS.max_steps):
        _, loss_value = sess.run([train_op, elbo_loss],
                                 feed_dict={handle: train_handle})

        if step % 10 == 0:
          print("step {:d} loss {:.2f}".format(step, loss_value))

        if (step+1) % FLAGS.viz_steps == 0:

          # Compute log prob of heldout set by averaging draws from the model:
          # p(heldout | train) = int_model p(heldout|model) p(model|train)
          #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
          # where model_i is a draw from the posterior p(model|train)
          probs = np.asarray([sess.run((model.probs),
                                       feed_dict={handle: heldout_handle})
                              for _ in range(FLAGS.n_monte_carlo)])
          mean_probs = np.mean(probs, axis=0)

          image_vals, label_vals = sess.run((images, labels),
                                            feed_dict={handle: heldout_handle})
          heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
                                                 label_vals.flatten()]))
          print(" ... heldout lp {:.2f}".format(heldout_lp))

          qm_vals, qs_vals = sess.run((qmeans, qstds))
          plot_weight_posteriors(names, qm_vals, qs_vals,
                                 fname=os.path.join(
                                     FLAGS.log_dir,
                                     "step{:05d}_weights.png".format(step)))

          plot_heldout_prediction(image_vals, probs,
                                  fname=os.path.join(
                                      FLAGS.log_dir,
                                      "step{:05d}_pred.png".format(step)),
                                  title="mean heldout logprob {:.2f}"
                                  .format(heldout_lp))


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    print("Warning: deleting old log directory at {}".format(FLAGS.log_dir))
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
      default=6000,
      help="Number of training steps to run."
  )
  parser.add_argument(
      "--encoder_layers",
      type=str,
      default="128,32",
      help="Comma-separated list of layer sizes for the encoder."
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
                           "mnist_deep_nn/input_data"),
      help="Directory to put the input data."
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                           "mnist_deep_nn/logs/"),
      help="Directory to put the log data."
  )
  parser.add_argument(
      "--viz_steps",
      type=int,
      default=400,
      help="Frequency at which save visualizations."
  )
  parser.add_argument(
      "--n_monte_carlo",
      type=int,
      default=50,
      help="Network draws used to compute predictive probabilities"
  )
  parser.add_argument(
      "--fake_data",
      default=False,
      action="store_true",
      help="If true, uses fake data for unit testing.",
  )

  FLAGS, unparsed = parser.parse_known_args()

  FLAGS.encoder_layers = [int(units) for units
                          in FLAGS.encoder_layers.split(",")]
  FLAGS.activation = tf.nn.__getattribute__(FLAGS.activation)

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
