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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
class NormalTest(tf.test.TestCase):

  def testNormalConjugateKnownSigmaPosterior(self):
    with tf.compat.v1.Session():
      mu0 = tf.constant([3.0])
      sigma0 = tf.constant([math.sqrt(10.0)])
      sigma = tf.constant([math.sqrt(2.0)])
      x = tf.constant([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0])
      s = tf.reduce_sum(input_tensor=x)
      n = tf.size(input=x)
      prior = tfd.Normal(loc=mu0, scale=sigma0)
      posterior = tfd.normal_conjugates_known_scale_posterior(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertTrue(isinstance(posterior, tfd.Normal))
      posterior_log_pdf = self.evaluate(posterior.log_prob(x))
      self.assertEqual(posterior_log_pdf.shape, (6,))

  def testNormalConjugateKnownSigmaPosteriorND(self):
    with tf.compat.v1.Session():
      batch_size = 6
      mu0 = tf.constant([[3.0, -3.0]] * batch_size)
      sigma0 = tf.constant([[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
      sigma = tf.constant([[math.sqrt(2.0)]] * batch_size)
      x = tf.transpose(
          a=tf.constant([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=tf.float32))
      s = tf.reduce_sum(input_tensor=x)
      n = tf.size(input=x)
      prior = tfd.Normal(loc=mu0, scale=sigma0)
      posterior = tfd.normal_conjugates_known_scale_posterior(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertTrue(isinstance(posterior, tfd.Normal))
      posterior_log_pdf = self.evaluate(posterior.log_prob(x))
      self.assertEqual(posterior_log_pdf.shape, (6, 2))

  def testNormalConjugateKnownSigmaNDPosteriorND(self):
    with tf.compat.v1.Session():
      batch_size = 6
      mu0 = tf.constant([[3.0, -3.0]] * batch_size)
      sigma0 = tf.constant([[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
      sigma = tf.constant([[math.sqrt(2.0), math.sqrt(4.0)]] * batch_size)
      x = tf.constant(
          [[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], [2.5, -2.5, -4.0, 0.0, 1.0, -2.0]],
          dtype=tf.float32)
      s = tf.reduce_sum(input_tensor=x, axis=[1])
      x = tf.transpose(a=x)  # Reshape to shape (6, 2)
      n = tf.constant([6] * 2)
      prior = tfd.Normal(loc=mu0, scale=sigma0)
      posterior = tfd.normal_conjugates_known_scale_posterior(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertTrue(isinstance(posterior, tfd.Normal))

      # Calculate log_pdf under the 2 models
      posterior_log_pdf = posterior.log_prob(x)
      self.assertEqual(posterior_log_pdf.shape, (6, 2))
      self.assertEqual(self.evaluate(posterior_log_pdf).shape, (6, 2))

  def testNormalConjugateKnownSigmaPredictive(self):
    with tf.compat.v1.Session():
      batch_size = 6
      mu0 = tf.constant([3.0] * batch_size)
      sigma0 = tf.constant([math.sqrt(10.0)] * batch_size)
      sigma = tf.constant([math.sqrt(2.0)] * batch_size)
      x = tf.constant([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0])
      s = tf.reduce_sum(input_tensor=x)
      n = tf.size(input=x)
      prior = tfd.Normal(loc=mu0, scale=sigma0)
      predictive = tfd.normal_conjugates_known_scale_predictive(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertTrue(isinstance(predictive, tfd.Normal))
      predictive_log_pdf = self.evaluate(predictive.log_prob(x))
      self.assertEqual(predictive_log_pdf.shape, (6,))


if __name__ == "__main__":
  tf.test.main()
