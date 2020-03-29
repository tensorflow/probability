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
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class CalibrationTest(test_util.TestCase):

  _TEMPERATURES = [0.01, 1.0, 5.0]
  _NLABELS = [2, 4]
  _NSAMPLES = [8192, 16384]

  @parameterized.parameters(
      itertools.product(_TEMPERATURES, _NLABELS, _NSAMPLES)
  )
  def test_brier_decomposition(self, temperature, nlabels, nsamples):
    """Test the accuracy of the estimated Brier decomposition."""
    tf.random.set_seed(1)
    seed_stream = test_util.test_seed_stream()
    logits = tf.random.normal(
        (nsamples, nlabels), seed=seed_stream()) / temperature
    labels = tf.random.uniform(
        (nsamples,), maxval=nlabels, seed=seed_stream(), dtype=tf.int32)

    uncertainty, resolution, reliability = tfp.stats.brier_decomposition(
        labels=labels, logits=logits)

    # Recover an estimate of the Brier score from the decomposition
    brier = uncertainty - resolution + reliability

    # Estimate Brier score directly
    brier_direct = tfp.stats.brier_score(labels=labels, logits=logits)
    uncertainty, resolution, reliability, brier, brier_direct = self.evaluate([
        uncertainty,
        resolution,
        reliability,
        brier,
        tf.reduce_mean(brier_direct, axis=0),
    ])

    logging.info('Brier, n=%d k=%d T=%.2f, Unc %.4f - Res %.4f + Rel %.4f = '
                 'Brier %.4f,  Brier-direct %.4f',
                 nsamples, nlabels, temperature,
                 uncertainty, resolution, reliability,
                 brier, brier_direct)

    self.assertGreaterEqual(resolution, 0.0, msg='Brier resolution negative')
    self.assertGreaterEqual(reliability, 0.0, msg='Brier reliability negative')
    self.assertAlmostEqual(
        brier, brier_direct, delta=1.e-1,
        msg='Brier from decomposition (%.4f) and Brier direct (%.4f) disagree '
        'beyond estimation error.' % (brier, brier_direct))

  def test_brier_decomposition_batching(self):
    """Test batching support in Brier decomposition."""
    tf.random.set_seed(1)
    nlabels = 4
    nsamples = 1024
    batch_shape = [3, 5]
    batch_size = np.prod(batch_shape, dtype=np.int32)
    seed_stream = test_util.test_seed_stream()

    logits = tf.random.normal(
        batch_shape + [nsamples, nlabels], seed=seed_stream())
    labels = tf.random.uniform(batch_shape + [nsamples],
                               maxval=nlabels, seed=seed_stream(),
                               dtype=tf.int32)
    flat_logits = tf.reshape(logits, [batch_size, nsamples, nlabels])
    flat_labels = tf.reshape(labels, [batch_size, nsamples])

    decomps = tf.stack(tfp.stats.brier_decomposition(labels, logits))
    decomps = tf.reshape(decomps, [3, batch_size])

    decomps_i = []
    for i in range(batch_size):
      d = tfp.stats.brier_decomposition(flat_labels[i], flat_logits[i])
      decomps_i.append(tf.stack(d))
    decomps, decomps_i = self.evaluate([decomps, tf.stack(decomps_i, axis=-1)])
    for i in range(batch_size):
      self.assertAllClose(decomps[:, i], decomps_i[:, i], rtol=0.20, atol=0.02)

  def _compute_perturbed_reliability(self, data, labels,
                                     weights, bias, perturbation):
    """Compute reliability of data set under perturbed hypothesis."""
    weights_perturbed = weights + perturbation * tf.random.normal(
        weights.shape, seed=test_util.test_seed())
    logits_perturbed = tf.matmul(data, weights_perturbed)
    logits_perturbed += tf.expand_dims(bias, 0)

    _, _, reliability = tfp.stats.brier_decomposition(
        labels=labels, logits=logits_perturbed)

    return reliability

  def _generate_linear_dataset(self, nfeatures, nlabels, nsamples):
    tf.random.set_seed(1)
    seed_stream = test_util.test_seed_stream()
    data = tf.random.normal((nsamples, nfeatures), seed=seed_stream())
    weights = tf.random.normal((nfeatures, nlabels), seed=seed_stream())
    bias = tf.random.normal((nlabels,), seed=seed_stream())

    logits_true = tf.matmul(data, weights) + tf.expand_dims(bias, 0)
    prob_true = tfp.distributions.Categorical(logits=logits_true)
    labels = prob_true.sample(1, seed=seed_stream())
    labels = tf.reshape(labels, (tf.size(labels),))

    return data, labels, weights, bias

  @parameterized.parameters(
      (5, 2, 2000), (5, 4, 2000),
  )
  def test_reliability_experiment(self, nfeatures, nlabels, nsamples,
                                  tolerance=0.05):
    data, labels, weights, bias = self._generate_linear_dataset(
        nfeatures, nlabels, nsamples)

    nreplicates = 5
    perturbations = np.linspace(0.0, 3.0, 10)
    reliability = []

    for i, perturbation in enumerate(perturbations):
      reliability_replicates = tf.stack(
          [self._compute_perturbed_reliability(data, labels, weights,
                                               bias, perturbation)
           for _ in range(nreplicates)], axis=0)
      reliability.append(tf.math.reduce_mean(reliability_replicates, axis=0))
    reliability = self.evaluate(tf.stack(reliability, axis=0))

    for i in range(1, len(reliability)):
      logging.info('Reliability at perturbation %.3f: %.4f',
                   perturbations[i], reliability[i])
      self.assertLessEqual(reliability[i-1], reliability[i] + tolerance,
                           msg='Reliability decreases (%.4f to %.4f + %.3f) '
                           'with perturbation size increasing from %.4f '
                           'to %.4f' % (reliability[i-1], reliability[i],
                                        tolerance,
                                        perturbations[i-1], perturbations[i]))

  def test_expected_calibration_error(self):
    probs = tf.convert_to_tensor([[0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
    logits = tf.math.log(probs)
    labels = tf.convert_to_tensor([1, 0, 0], dtype=tf.int32)

    ece = self.evaluate(tfp.stats.expected_calibration_error(
        3, logits=logits, labels_true=labels))

    self.assertAlmostEqual(0.33333333, ece, places=6,
                           msg='Computed ECE (%.5f) does not match '
                           'true ECE (1/3).' % (ece,))

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
    seed_stream = test_util.test_seed_stream()
    logits = 2. * tf.random.normal(
        (nsamples, nclasses), seed=seed_stream())
    py = tfp.distributions.Categorical(logits=logits)
    labels = py.sample(seed=seed_stream())

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
    seed_stream = test_util.test_seed_stream()

    logits = 2.* tf.random.normal(
        (nsamples, nclasses), seed=seed_stream())
    py = tfp.distributions.Categorical(logits=logits)
    labels = py.sample(seed=seed_stream())
    logits_other = 2. * tf.random.normal(
        (nsamples, nclasses), seed=seed_stream())

    return logits_other, labels

  @parameterized.parameters(
      (10, 2, 50000), (10, 5, 50000), (10, 50, 50000),
  )
  def test_expected_calibration_error_wellcalibrated(self, num_bins, nclasses,
                                                     nsamples):
    logits, labels = self._generate_perfect_calibration_logits(
        nsamples, nclasses)
    ece = self.evaluate(tfp.stats.expected_calibration_error(
        num_bins, logits=logits, labels_true=labels))

    logging.info('ECE (well-calibrated), num_bins=%d  nclasses=%d  '
                 'nsamples=%d  ECE %.4f', num_bins, nclasses, nsamples, ece)

    ece_tolerance = 0.01
    self.assertLess(ece, ece_tolerance,
                    msg='Expected calibration error (ECE) computed at %.4f '
                    'for well-calibrated predictions exceeds '
                    'tolerance of %.4f.' % (ece, ece_tolerance))

  @parameterized.parameters(
      (10, 2, 50000), (10, 5, 50000), (10, 50, 50000),
  )
  def test_expected_calibration_error_uncalibrated(self, num_bins, nclasses,
                                                   nsamples):
    logits, labels = self._generate_random_calibration_logits(
        nsamples, nclasses)
    ece = self.evaluate(tfp.stats.expected_calibration_error(
        num_bins, logits=logits, labels_true=labels))

    logging.info('ECE (uncalibrated), num_bins=%d  nclasses=%d  '
                 'nsamples=%d  ECE %.4f', num_bins, nclasses, nsamples, ece)

    ece_lower = 0.2
    self.assertGreater(ece, ece_lower,
                       msg='Expected calibration error (ECE) computed at %.4f '
                       'for well-calibrated predictions is smaller than '
                       'minimum ECE of %.4f.' % (ece, ece_lower))


@test_util.test_all_tf_execution_regimes
class ExpectedCalibrationErrorQuantiles(test_util.TestCase):

  def test_docstring_simple(self):
    label = tf.cast([0, 0, 1, 0, 1, 1], dtype=tf.bool)
    log_pred = tf.math.log([0.1, 0.05, 0.5, 0.2, 0.99, 0.99])
    ece, acc, conf, cnt, edges, buckets = self.evaluate(
        tfp.stats.expected_calibration_error_quantiles(
            label, log_pred, num_buckets=3))
    self.assertNear(0.145, ece, err=1e-6)
    self.assertAllClose([0., 0., 1.], acc, rtol=1e-6, atol=1e-6)
    self.assertAllClose([0.075, 0.2, 0.826667], conf, rtol=1e-6, atol=0.)
    self.assertAllClose([2., 1., 3.], cnt, rtol=1e-6, atol=0.)
    self.assertAllClose(np.log([0.05, 0.2, 0.5, 0.99]), edges,
                        rtol=1e-6, atol=0.)
    self.assertAllEqual([0, 0, 2, 1, 2, 2], buckets)

  def test_docstring_categorical(self):
    y = tf.constant([[0, 1], [2, 1], [1, 1], [2, 2]], dtype=tf.int64)
    logits = [
        [[0., -1, -2],
         [-1., 0, -1]],
        [[2., -1, 0],
         [-1., 0, -1]],
        [[-1., 0, -1],
         [0., -2, -1]],
        [[-1., 0, -2],
         [0, -1, -2]],
    ]
    d = tfp.distributions.Categorical(logits=logits)
    def all_categories(d):
      num_classes = tf.shape(d.logits_parameter())[-1]
      batch_ndims = tf.size(d.batch_shape_tensor())
      expand_shape = tf.pad(
          [num_classes], paddings=[[0, batch_ndims]], constant_values=1)
      return tf.reshape(tf.range(num_classes, dtype=d.dtype), expand_shape)
    all_pred_log_prob = d.log_prob(all_categories(d))
    yhat = tf.argmax(all_pred_log_prob, axis=0)
    def rollaxis(x, shift):
      return tf.transpose(x, tf.roll(tf.range(tf.rank(x)), shift=shift, axis=0))
    pred_log_prob = tf.gather(rollaxis(all_pred_log_prob, shift=-1),
                              yhat,
                              batch_dims=len(d.batch_shape))
    hit = tf.equal(y, yhat)
    ece, acc, conf, cnt, edges, buckets = self.evaluate(
        tfp.stats.expected_calibration_error_quantiles(
            hit, pred_log_prob, num_buckets=10))
    self.assertEqual((2,), ece.shape)
    self.assertEqual((10, 2), acc.shape)
    self.assertEqual((10, 2), conf.shape)
    self.assertEqual((10, 2), cnt.shape)
    self.assertEqual((11, 2), edges.shape)
    self.assertEqual((4, 2), buckets.shape)

  def test_multidim_and_axis(self):
    label = tf.cast([[0, 0, 1, 0, 1, 1],
                     [0, 0, 1, 0, 1, 1]], dtype=tf.bool)
    log_pred = tf.math.log([[0.1, 0.05, 0.5, 0.2, 0.99, 0.99],
                            [0.1, 0.05, 0.5, 0.2, 0.99, 0.99]])
    ece, acc, conf, cnt, edges, buckets = self.evaluate(
        tfp.stats.expected_calibration_error_quantiles(
            label, log_pred, num_buckets=3, axis=1))
    self.assertAllClose([0.145, 0.145], ece, rtol=1e-6, atol=1e-6)
    self.assertAllClose(np.transpose([[0., 0., 1.],
                                      [0., 0., 1.]]),
                        acc,
                        rtol=1e-6, atol=1e-6)
    self.assertAllClose(np.transpose([[0.075, 0.2, 0.826667],
                                      [0.075, 0.2, 0.826667]]),
                        conf,
                        rtol=1e-6, atol=0.)
    self.assertAllClose(np.transpose([[2., 1., 3.],
                                      [2., 1., 3.]]),
                        cnt,
                        rtol=1e-6, atol=0.)
    self.assertAllClose(np.log(np.transpose([[0.05, 0.2, 0.5, 0.99],
                                             [0.05, 0.2, 0.5, 0.99]])),
                        edges,
                        rtol=1e-6, atol=0.)
    self.assertAllEqual([[0, 0, 2, 1, 2, 2],
                         [0, 0, 2, 1, 2, 2]],
                        buckets)


if __name__ == '__main__':
  tf.test.main()
