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
"""Trains a Bayesian logistic regression model on synthetic data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse  # TODO(b/74538173): Use absl.flags rather than argparse.
import os
import sys

# Dependency imports
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf

from tensorflow_probability.examples.weight_uncertainty import weight_uncertainty


tfd = tf.contrib.distributions


def toy_logistic_data(num_examples, input_size=2, weights_prior_stddev=5.0):
  """Generates synthetic data for binary classification.

  Args:
    num_examples: The number of samples to generate (scalar Python `int`).
    input_size: The input space dimension (scalar Python `int`).
    weights_prior_stddev: The prior standard deviation of the weight
      vector. (scalar Python `float`).

  Returns:
    random_weights: Sampled weights as a Numpy `array` of shape
      `[input_size]`.
    random_bias: Sampled bias as a scalar Python `float`.
    design_matrix: Points sampled uniformly from the cube `[-1,
       1]^{input_size}`, as a Numpy `array` of shape `(num_examples,
       input_size)`.
    labels: Labels sampled from the logistic model `p(label=1) =
      logistic(dot(inputs, random_weights) + random_bias)`, as a Numpy
      `int32` `array` of shape `(num_examples, 1)`.
  """
  random_weights = weights_prior_stddev * np.random.randn(input_size)
  random_bias = np.random.randn()
  design_matrix = np.random.rand(num_examples, input_size) * 2 - 1
  logits = np.reshape(
      np.dot(design_matrix, random_weights) + random_bias,
      (-1, 1))
  p_labels = 1. / (1 + np.exp(-logits))
  labels = np.int32(p_labels > np.random.rand(num_examples, 1))
  return random_weights, random_bias, np.float32(design_matrix), labels


def visualize_decision(inputs, labels, true_w_b, candidate_w_bs, fname):
  """Utility method to visualize decision boundaries in R^2.

  Args:
    inputs: Input points, as a Numpy `array` of shape `[num_examples, 2]`.
    labels: Numpy `float`-like array of shape `[num_examples, 1]` giving a
      label for each point.
    true_w_b: A `tuple` `(w, b)` where `w` is a Numpy array of
       shape `[2]` and `b` is a scalar `float`, interpreted as a
       decision rule of the form `dot(inputs, w) + b > 0`.
    candidate_w_bs: Python `iterable` containing tuples of the same form as
       true_w_b.
    fname: The filename to save the plot as a PNG image (Python `str`).

  Raises:
    ImportError: if matplotlib is not available.
  """

  fig = figure.Figure(figsize=(6, 6))
  canvas = backend_agg.FigureCanvasAgg(fig)
  ax = fig.add_subplot(1, 1, 1)
  ax.scatter(inputs[:, 0], inputs[:, 1],
             c=np.float32(labels[:, 0]),
             cmap=cm.get_cmap("binary"))

  def plot_weights(w, b, **kwargs):
    w1, w2 = w
    x1s = np.linspace(-1, 1, 100)
    x2s = -(w1  * x1s + b) / w2
    ax.plot(x1s, x2s, **kwargs)

  for w, b in candidate_w_bs:
    plot_weights(w, b,
                 alpha=1./np.sqrt(len(candidate_w_bs)),
                 lw=1, color="blue")

  if true_w_b is not None:
    plot_weights(*true_w_b, lw=4,
                 color="green", label="true separator")

  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([-1.5, 1.5])
  ax.legend()

  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))


def build_input_pipeline(x, y, batch_size):
  """Build a Dataset iterator for supervised classification.

  Args:
    x: Numpy `array` of inputs, indexed by the first dimension.
    y: Numpy `array` of labels, with the same first dimension as `x`.
    batch_size: Number of elements in each training batch.

  Returns:
    batch_data: `Tensor` feed  inputs, of shape
      `[batch_size] + x.shape[1:]`.
    batch_labels: `Tensor` feed of labels, of shape
      `[batch_size] + y.shape[1:]`.
  """

  training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = training_batches.make_one_shot_iterator()
  batch_data, batch_labels = training_iterator.get_next()
  return batch_data, batch_labels


def run_training():
  """Generate data and run the training loop."""

  # Generate (and visualize) a toy classification dataset.
  w_true, b_true, x, y = toy_logistic_data(FLAGS.num_examples, 2)

  # Define a logistic regression model as a Bernoulli distribution
  # parameterized by logits from a single linear layer.
  def build_logistic_model(inputs):
    logits = tf.layers.dense(inputs, 1, activation=None)
    model = tfd.Bernoulli(logits=logits)
    return model

  with tf.Graph().as_default():
    inputs, labels = build_input_pipeline(x, y, FLAGS.batch_size)

    # Build a variational Bayesian logistic regression model
    elbo_loss, _ = weight_uncertainty.bayesianify(
        build_logistic_model,
        inputs,
        labels,
        FLAGS.num_examples)

    with tf.name_scope("train"):
      opt = tf.train.AdamOptimizer(
          learning_rate=FLAGS.learning_rate)

      train_op = opt.minimize(elbo_loss)
      init = tf.global_variables_initializer()
      with tf.Session() as sess:
        sess.run(init)

        # Fit the model to data.
        for step in range(FLAGS.max_steps):
          _, loss_value = sess.run([train_op, elbo_loss])
          if step % 400 == 0:
            print("step {:d}: loss {:.2f}".format(step, loss_value))

        # Visualize some draws from the weights posterior.
        qw, qb = tf.get_collection(weight_uncertainty.VI_QDISTS)
        w_draw, b_draw = qw.sample(), qb.sample()
        candidate_w_bs = []
        for _ in range(FLAGS.n_monte_carlo):
          w, b = sess.run((w_draw, b_draw))
          candidate_w_bs.append((w, b))
        visualize_decision(x, y, (w_true, b_true),
                           candidate_w_bs,
                           fname=os.path.join(FLAGS.log_dir,
                                              "weights_inferred.png"))


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.logging.warn(
        "Warning: deleting old log directory at {}".format(
            FLAGS.log_dir))
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
      default=1500,
      help="Number of training steps to run."
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=32,
      help="Batch size.  Must divide evenly into the dataset sizes."
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                           "logistic_regression/logs/"),
      help="Directory to put the log data."
  )
  parser.add_argument(
      "--num_examples",
      type=int,
      default=256,
      help="Number of datapoints to generate."
  )
  parser.add_argument(
      "--n_monte_carlo",
      type=int,
      default=25,
      help="Monte Carlo samples used to visualize the weight posterior"
  )

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
