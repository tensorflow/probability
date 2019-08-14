# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for tensorflow_probability.python.stats.calibration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


class CalibrationTest(parameterized.TestCase, tf.test.TestCase):

  _TEMPERATURES = [0.01, 1.0, 5.0]
  _NLABELS = [2, 4]
  _NSAMPLES = [8192, 16384]

  @parameterized.parameters(
      itertools.product(_TEMPERATURES, _NLABELS, _NSAMPLES)
  )
  def test_brier_decomposition(self, temperature, nlabels, nsamples):
    """Test the accuracy of the estimated Brier decomposition."""
    tf.random.set_seed(1)
    logits = tf.random.normal((nsamples, nlabels)) / temperature
    labels = tf.random.uniform((nsamples,), maxval=nlabels, dtype=tf.int32)

    uncertainty, resolution, reliability = tfp.stats.brier_decomposition(
        labels=labels, logits=logits)
    uncertainty = float(uncertainty)
    resolution = float(resolution)
    reliability = float(reliability)

    # Recover an estimate of the Brier score from the decomposition
    brier = uncertainty - resolution + reliability

    # Estimate Brier score directly
    brier_direct = tfp.stats.brier_score(labels=labels, logits=logits)
    brier_direct = float(tf.reduce_mean(brier_direct, axis=0))

    logging.info("Brier, n=%d k=%d T=%.2f, Unc %.4f - Res %.4f + Rel %.4f = "
                 "Brier %.4f,  Brier-direct %.4f",
                 nsamples, nlabels, temperature,
                 uncertainty, resolution, reliability,
                 brier, brier_direct)

    self.assertGreaterEqual(resolution, 0.0, msg="Brier resolution negative")
    self.assertGreaterEqual(reliability, 0.0, msg="Brier reliability negative")
    self.assertAlmostEqual(
        brier, brier_direct, delta=1.0e-2,
        msg="Brier from decomposition (%.4f) and Brier direct (%.4f) disagree "
        "beyond estimation error." % (brier, brier_direct))

  def test_brier_decomposition_batching(self):
    """Test batching support in Brier decomposition."""
    tf.random.set_seed(1)
    nlabels = 4
    nsamples = 1024
    batch_shape = [3, 5]
    batch_size = np.prod(batch_shape, dtype=np.int32)

    logits = tf.random.normal(batch_shape + [nsamples, nlabels])
    labels = tf.random.uniform(batch_shape + [nsamples],
                               maxval=nlabels, dtype=tf.int32)
    flat_logits = tf.reshape(logits, [batch_size, nsamples, nlabels])
    flat_labels = tf.reshape(labels, [batch_size, nsamples])

    decomps = tf.stack(tfp.stats.brier_decomposition(labels, logits))
    decomps = tf.reshape(decomps, [3, batch_size])

    for i in range(batch_size):
      decomp_i = tfp.stats.brier_decomposition(flat_labels[i], flat_logits[i])
      decomp_i = tf.stack(decomp_i)
      self.assertAllClose(decomps[:, i], decomp_i)

  def _compute_perturbed_reliability(self, data, labels,
                                     weights, bias, perturbation):
    """Compute reliability of data set under perturbed hypothesis."""
    weights_perturbed = weights + perturbation*tf.random.normal(weights.shape)
    logits_perturbed = tf.matmul(data, weights_perturbed)
    logits_perturbed += tf.expand_dims(bias, 0)

    _, _, reliability = tfp.stats.brier_decomposition(
        labels=labels, logits=logits_perturbed)

    return float(reliability)

  def _generate_linear_dataset(self, nfeatures, nlabels, nsamples):
    tf.random.set_seed(1)
    data = tf.random.normal((nsamples, nfeatures))
    weights = tf.random.normal((nfeatures, nlabels))
    bias = tf.random.normal((nlabels,))

    logits_true = tf.matmul(data, weights) + tf.expand_dims(bias, 0)
    prob_true = tfp.distributions.Categorical(logits=logits_true)
    labels = prob_true.sample(1)
    labels = tf.reshape(labels, (tf.size(input=labels),))

    return data, labels, weights, bias

  @parameterized.parameters(
      (5, 2, 20000), (5, 4, 20000),
  )
  def test_reliability_experiment(self, nfeatures, nlabels, nsamples,
                                  tolerance=0.05):
    data, labels, weights, bias = self._generate_linear_dataset(
        nfeatures, nlabels, nsamples)

    nreplicates = 40
    perturbations = np.linspace(0.0, 3.0, 10)
    reliability = np.zeros_like(perturbations)

    for i, perturbation in enumerate(perturbations):
      reliability_replicates = np.array(
          [self._compute_perturbed_reliability(data, labels, weights,
                                               bias, perturbation)
           for _ in range(nreplicates)])
      reliability[i] = np.mean(reliability_replicates)
      logging.info("Reliability at perturbation %.3f: %.4f", perturbation,
                   reliability[i])

    for i in range(1, len(reliability)):
      self.assertLessEqual(reliability[i-1], reliability[i] + tolerance,
                           msg="Reliability decreases (%.4f to %.4f + %.3f) "
                           "with perturbation size increasing from %.4f "
                           "to %.4f" % (reliability[i-1], reliability[i],
                                        tolerance,
                                        perturbations[i-1], perturbations[i]))

  def test_expected_calibration_error(self):
    probs = tf.convert_to_tensor([[0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
    logits = tf.math.log(probs)
    labels = tf.convert_to_tensor([1, 0, 0], dtype=tf.int32)

    ece = tfp.stats.expected_calibration_error(
        3, logits=logits, labels_true=labels)
    ece = float(ece)
    ece_true = 0.33333333

    self.assertAlmostEqual(ece, ece_true, places=6,
                           msg="Computed ECE (%.5f) does not match "
                           "true ECE (%.5f)." % (ece, ece_true))

  def _generate_perfect_calibration_logits(self, nsamples, nclasses):
    """Generate well distributed and well calibrated probabilities.

    Args:
      nsamples: int, >= 1, number of samples to generate.
      nclasses: int, >= 2, number of classes.

    Returns:
      logits: Tensor, shape (nsamples, nclasses), tf.float32, unnormalized log
        probabilities (logits) of the probabilistic predictions.
      labels: Tensor, shape (nsamples,), tf.int32, the true class labels.  Each
        element is in the range 0,..,nclasses-1.
    """
    tf.random.set_seed(1)

    logits = 2.0*tf.random.normal((nsamples, nclasses))
    py = tfp.distributions.Categorical(logits=logits)
    labels = py.sample()

    return logits, labels

  def _generate_random_calibration_logits(self, nsamples, nclasses):
    """Generate well distributed and poorly calibrated probabilities.

    Args:
      nsamples: int, >= 1, number of samples to generate.
      nclasses: int, >= 2, number of classes.

    Returns:
      logits: Tensor, shape (nsamples, nclasses), tf.float32, unnormalized log
        probabilities (logits) of the probabilistic predictions.
      labels: Tensor, shape (nsamples,), tf.int32, the true class labels.  Each
        element is in the range 0,..,nclasses-1.
    """
    tf.random.set_seed(1)

    logits = 2.0*tf.random.normal((nsamples, nclasses))
    py = tfp.distributions.Categorical(logits=logits)
    labels = py.sample()
    logits_other = 2.0*tf.random.normal((nsamples, nclasses))

    return logits_other, labels

  @parameterized.parameters(
      (10, 2, 50000), (10, 5, 50000), (10, 50, 50000),
  )
  def test_expected_calibration_error_wellcalibrated(self, num_bins, nclasses,
                                                     nsamples):
    logits, labels = self._generate_perfect_calibration_logits(
        nsamples, nclasses)
    ece = tfp.stats.expected_calibration_error(
        num_bins, logits=logits, labels_true=labels)
    ece = float(ece)

    logging.info("ECE (well-calibrated), num_bins=%d  nclasses=%d  "
                 "nsamples=%d  ECE %.4f", num_bins, nclasses, nsamples, ece)

    ece_tolerance = 0.01
    self.assertLess(ece, ece_tolerance,
                    msg="Expected calibration error (ECE) computed at %.4f "
                    "for well-calibrated predictions exceeds "
                    "tolerance of %.4f." % (ece, ece_tolerance))

  @parameterized.parameters(
      (10, 2, 50000), (10, 5, 50000), (10, 50, 50000),
  )
  def test_expected_calibration_error_uncalibrated(self, num_bins, nclasses,
                                                   nsamples):
    logits, labels = self._generate_random_calibration_logits(
        nsamples, nclasses)
    ece = tfp.stats.expected_calibration_error(
        num_bins, logits=logits, labels_true=labels)
    ece = float(ece)

    logging.info("ECE (uncalibrated), num_bins=%d  nclasses=%d  "
                 "nsamples=%d  ECE %.4f", num_bins, nclasses, nsamples, ece)

    ece_lower = 0.2
    self.assertGreater(ece, ece_lower,
                       msg="Expected calibration error (ECE) computed at %.4f "
                       "for well-calibrated predictions is smaller than "
                       "minimum ECE of %.4f." % (ece, ece_lower))

if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
