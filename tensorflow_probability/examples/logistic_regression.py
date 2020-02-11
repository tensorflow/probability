# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License');
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
from absl import app
from absl import flags
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tf.enable_v2_behavior()

tfd = tfp.distributions

flags.DEFINE_float('learning_rate',
                   default=0.01,
                   help='Initial learning rate.')
flags.DEFINE_integer('num_epochs',
                     default=50,
                     help='Number of epochs to run.')
flags.DEFINE_integer('batch_size',
                     default=32,
                     help='Batch size.')
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                         'logistic_regression/'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer('num_examples',
                     default=256,
                     help='Number of datapoints to generate.')
flags.DEFINE_integer('num_monte_carlo',
                     default=50,
                     help='Monte Carlo samples to visualize weight posterior.')

FLAGS = flags.FLAGS
# The dimensions of the example data, ie shape=(256, 2)
NUM_DIMENSIONS = 2


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
             cmap=cm.get_cmap('binary'),
             edgecolors='k')

  def plot_weights(w, b, **kwargs):
    w1, w2 = w
    x1s = np.linspace(-1, 1, 100)
    x2s = -(w1  * x1s + b) / w2
    ax.plot(x1s, x2s, **kwargs)

  for w, b in candidate_w_bs:
    plot_weights(w, b,
                 alpha=1./np.sqrt(len(candidate_w_bs)),
                 lw=1, color='blue')

  if true_w_b is not None:
    plot_weights(*true_w_b, lw=4,
                 color='green', label='true separator')

  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([-1.5, 1.5])
  ax.legend()

  canvas.print_figure(fname, format='png')
  print('saved {}'.format(fname))


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


class ToyDataSequence(tf.keras.utils.Sequence):
  """Creates a sequence of labeled points from provided numpy arrays."""

  def __init__(self, features, labels, batch_size):
    """Initializes the sequence.

    Args:
      features: Numpy `array` of features, indexed by the first dimension.
      labels: Numpy `array` of features, with the same first dimension as
              `features`.
      batch_size: Integer, number of elements in each training batch.
    """
    self.features = features
    self.labels = labels
    self.batch_size = batch_size

  def __len__(self):
    return int(np.ceil(len(self.features) / self.batch_size))

  def __getitem__(self, idx):
    batch_x = self.features[self.batch_size * idx : self.batch_size * (idx + 1)]
    batch_y = self.labels[self.batch_size * idx: self.batch_size * (idx + 1)]
    return batch_x, batch_y


def create_model(num_samples, num_dimensions):
  """Creates a Keras model for logistic regression.

  Args:
   num_samples: Integer for number of training samples.
   num_dimensions: Integer for number of features in dataset.

  Returns:
    model: Compiled Keras model.
  """
  # KL divergence weighted by the number of training samples, using
  # lambda function to pass as input to the kernel_divergence_fn on
  # flipout layers.
  kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(num_samples, dtype=tf.float32))

  # Define a logistic regression model as a Bernoulli distribution
  # parameterized by logits from a single linear layer. We use the Flipout
  # Monte Carlo estimator for the layer: this enables lower variance
  # stochastic gradients than naive reparameterization.
  input_layer = tf.keras.layers.Input(shape=num_dimensions)
  dense_layer = tfp.layers.DenseFlipout(
      units=1,
      activation='sigmoid',
      kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
      bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
      kernel_divergence_fn=kl_divergence_function)(input_layer)

  # Model compilation.
  model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)
  optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
  # We use the binary_crossentropy loss since this toy example contains
  # two labels. The Keras API will then automatically add the
  # Kullback-Leibler divergence (contained on the individual layers of
  # the model), to the cross entropy loss, effectively
  # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
  model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
  return model


def main(argv):
  del argv
  if tf.io.gfile.exists(FLAGS.model_dir):
    print('Warning: deleting old log directory at {}'.format(FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)

  # Generate a toy classification dataset.
  w_true, b_true, features, labels = toy_logistic_data(FLAGS.num_examples)
  toy_logistic_sequence = ToyDataSequence(features, labels, FLAGS.batch_size)

  # Define and train a bayesian logistic regression model.
  model = create_model(FLAGS.num_examples, NUM_DIMENSIONS)
  model.fit(toy_logistic_sequence, epochs=FLAGS.num_epochs,
            shuffle=True)
  # Visualize some draws from the weights posterior.
  candidate_w_bs = [(model.layers[-1].kernel_posterior.sample().numpy(),
                     model.layers[-1].bias_posterior.sample().numpy())
                    for _ in range(FLAGS.num_monte_carlo)]
  visualize_decision(features, labels, (w_true, b_true),
                     candidate_w_bs,
                     fname=os.path.join(FLAGS.model_dir,
                                        'weights_inferred.png'))


if __name__ == '__main__':
  app.run(main)
