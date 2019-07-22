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

import os

# Dependency imports
from absl import flags
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=1500,
                     help="Number of training steps to run.")
flags.DEFINE_integer("batch_size",
                     default=32,
                     help="Batch size.")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                         "logistic_regression/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("num_examples",
                     default=256,
                     help="Number of datapoints to generate.")
flags.DEFINE_integer("num_monte_carlo",
                     default=50,
                     help="Monte Carlo samples to visualize weight posterior.")

FLAGS = flags.FLAGS


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
      logistic(dot(features, random_weights) + random_bias)`, as a Numpy
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


def visualize_decision(features, labels, true_w_b, candidate_w_bs, fname):
  """Utility method to visualize decision boundaries in R^2.

  Args:
    features: Input points, as a Numpy `array` of shape `[num_examples, 2]`.
    labels: Numpy `float`-like array of shape `[num_examples, 1]` giving a
      label for each point.
    true_w_b: A `tuple` `(w, b)` where `w` is a Numpy array of
       shape `[2]` and `b` is a scalar `float`, interpreted as a
       decision rule of the form `dot(features, w) + b > 0`.
    candidate_w_bs: Python `iterable` containing tuples of the same form as
       true_w_b.
    fname: The filename to save the plot as a PNG image (Python `str`).
  """
  fig = figure.Figure(figsize=(6, 6))
  canvas = backend_agg.FigureCanvasAgg(fig)
  ax = fig.add_subplot(1, 1, 1)
  ax.scatter(features[:, 0], features[:, 1],
             c=np.float32(labels[:, 0]),
             cmap=cm.get_cmap("binary"),
             edgecolors="k")

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
    x: Numpy `array` of features, indexed by the first dimension.
    y: Numpy `array` of labels, with the same first dimension as `x`.
    batch_size: Number of elements in each training batch.

  Returns:
    batch_features: `Tensor` feed  features, of shape
      `[batch_size] + x.shape[1:]`.
    batch_labels: `Tensor` feed of labels, of shape
      `[batch_size] + y.shape[1:]`.
  """
  training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)
  batch_features, batch_labels = training_iterator.get_next()
  return batch_features, batch_labels


def main(argv):
  del argv  # unused
  if tf.io.gfile.exists(FLAGS.model_dir):
    tf.compat.v1.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)

  # Generate (and visualize) a toy classification dataset.
  w_true, b_true, x, y = toy_logistic_data(FLAGS.num_examples, 2)
  features, labels = build_input_pipeline(x, y, FLAGS.batch_size)

  # Define a logistic regression model as a Bernoulli distribution
  # parameterized by logits from a single linear layer. We use the Flipout
  # Monte Carlo estimator for the layer: this enables lower variance
  # stochastic gradients than naive reparameterization.
  with tf.compat.v1.name_scope("logistic_regression", values=[features]):
    layer = tfp.layers.DenseFlipout(
        units=1,
        activation=None,
        kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
        bias_posterior_fn=tfp.layers.default_mean_field_normal_fn())
    logits = layer(features)
    labels_distribution = tfd.Bernoulli(logits=logits)

  # Compute the -ELBO as the loss, averaged over the batch size.
  neg_log_likelihood = -tf.reduce_mean(
      input_tensor=labels_distribution.log_prob(labels))
  kl = sum(layer.losses) / FLAGS.num_examples
  elbo_loss = neg_log_likelihood + kl

  # Build metrics for evaluation. Predictions are formed from a single forward
  # pass of the probabilistic layers. They are cheap but noisy predictions.
  predictions = tf.cast(logits > 0, dtype=tf.int32)
  accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(
      labels=labels, predictions=predictions)

  with tf.compat.v1.name_scope("train"):
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(elbo_loss)

  init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                     tf.compat.v1.local_variables_initializer())

  with tf.compat.v1.Session() as sess:
    sess.run(init_op)

    # Fit the model to data.
    for step in range(FLAGS.max_steps):
      _ = sess.run([train_op, accuracy_update_op])
      if step % 100 == 0:
        loss_value, accuracy_value = sess.run([elbo_loss, accuracy])
        print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
            step, loss_value, accuracy_value))

    # Visualize some draws from the weights posterior.
    w_draw = layer.kernel_posterior.sample()
    b_draw = layer.bias_posterior.sample()
    candidate_w_bs = []
    for _ in range(FLAGS.num_monte_carlo):
      w, b = sess.run((w_draw, b_draw))
      candidate_w_bs.append((w, b))
    visualize_decision(x, y, (w_true, b_true),
                       candidate_w_bs,
                       fname=os.path.join(FLAGS.model_dir,
                                          "weights_inferred.png"))

if __name__ == "__main__":
  tf.compat.v1.app.run()
