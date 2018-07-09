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
"""Tests for Wishart."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import linalg
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import errors_impl

tfd = tfp.distributions


def make_pd(start, n):
  """Deterministically create a positive definite matrix."""
  x = np.tril(linalg.circulant(np.arange(start, start + n)))
  return np.dot(x, x.T)


def chol(x):
  """Compute Cholesky factorization."""
  return linalg.cholesky(x).T


def wishart_var(df, x):
  """Compute Wishart variance for numpy scale matrix."""
  x = np.sqrt(df) * np.asarray(x)
  d = np.expand_dims(np.diag(x), -1)
  return x**2 + np.dot(d, d.T)


class WishartTest(tf.test.TestCase):

  def testEntropy(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4
      w = tfd.Wishart(df, scale_tril=chol(scale))
      # sp.stats.wishart(df=4, scale=make_pd(1., 2)).entropy()
      self.assertAllClose(6.301387092430769, w.entropy().eval())

      w = tfd.Wishart(df=1, scale_tril=[[1.]])
      # sp.stats.wishart(df=1,scale=1).entropy()
      self.assertAllClose(0.78375711047393404, w.entropy().eval())

  def testMeanLogDetAndLogNormalizingConstant(self):
    with self.test_session():

      def entropy_alt(w):
        return (
            w.log_normalization()
            - 0.5 * (w.df - w.dimension - 1.) * w.mean_log_det()
            + 0.5 * w.df * w.dimension).eval()

      w = tfd.Wishart(df=4, scale_tril=chol(make_pd(1., 2)))
      self.assertAllClose(w.entropy().eval(), entropy_alt(w))

      w = tfd.Wishart(df=5, scale_tril=[[1.]])
      self.assertAllClose(w.entropy().eval(), entropy_alt(w))

  def testMean(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4
      w = tfd.Wishart(df, scale_tril=chol(scale))
      self.assertAllEqual(df * scale, w.mean().eval())

  def testMode(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4
      w = tfd.Wishart(df, scale_tril=chol(scale))
      self.assertAllEqual((df - 2. - 1.) * scale, w.mode().eval())

  def testStd(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4
      w = tfd.Wishart(df, scale_tril=chol(scale))
      self.assertAllEqual(tf.sqrt(wishart_var(df, scale)), w.stddev().eval())

  def testVariance(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4
      w = tfd.Wishart(df, scale_tril=chol(scale))
      self.assertAllEqual(wishart_var(df, scale), w.variance().eval())

  def testSample(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4

      chol_w = tfd.Wishart(
          df, scale_tril=chol(scale), cholesky_input_output_matrices=False)

      x = chol_w.sample(1, seed=42).eval()
      chol_x = [chol(x[0])]

      full_w = tfd.Wishart(df, scale, cholesky_input_output_matrices=False)
      self.assertAllClose(x, full_w.sample(1, seed=42).eval())

      chol_w_chol = tfd.Wishart(
          df, scale_tril=chol(scale), cholesky_input_output_matrices=True)
      self.assertAllClose(chol_x, chol_w_chol.sample(1, seed=42).eval())
      eigen_values = tf.matrix_diag_part(chol_w_chol.sample(1000, seed=42))
      np.testing.assert_array_less(0., eigen_values.eval())

      full_w_chol = tfd.Wishart(
          df, scale=scale, cholesky_input_output_matrices=True)
      self.assertAllClose(chol_x, full_w_chol.sample(1, seed=42).eval())
      eigen_values = tf.matrix_diag_part(full_w_chol.sample(1000, seed=42))
      np.testing.assert_array_less(0., eigen_values.eval())

      # Check first and second moments.
      df = 4.
      chol_w = tfd.Wishart(
          df=df,
          scale_tril=chol(make_pd(1., 3)),
          cholesky_input_output_matrices=False)
      x = chol_w.sample(10000, seed=42)
      self.assertAllEqual((10000, 3, 3), x.get_shape())

      moment1_estimate = tf.reduce_mean(x, reduction_indices=[0]).eval()
      self.assertAllClose(chol_w.mean().eval(), moment1_estimate, rtol=0.05)

      # The Variance estimate uses the squares rather than outer-products
      # because Wishart.Variance is the diagonal of the Wishart covariance
      # matrix.
      variance_estimate = (tf.reduce_mean(tf.square(x), reduction_indices=[0]) -
                           tf.square(moment1_estimate)).eval()
      self.assertAllClose(
          chol_w.variance().eval(), variance_estimate, rtol=0.05)

  # Test that sampling with the same seed twice gives the same results.
  def testSampleMultipleTimes(self):
    with self.test_session():
      df = 4.
      n_val = 100

      tf.set_random_seed(654321)
      chol_w1 = tfd.Wishart(
          df=df,
          scale_tril=chol(make_pd(1., 3)),
          cholesky_input_output_matrices=False,
          name="wishart1")
      samples1 = chol_w1.sample(n_val, seed=123456).eval()

      tf.set_random_seed(654321)
      chol_w2 = tfd.Wishart(
          df=df,
          scale_tril=chol(make_pd(1., 3)),
          cholesky_input_output_matrices=False,
          name="wishart2")
      samples2 = chol_w2.sample(n_val, seed=123456).eval()

      self.assertAllClose(samples1, samples2)

  def testProb(self):
    with self.test_session():
      # Generate some positive definite (pd) matrices and their Cholesky
      # factorizations.
      x = np.array(
          [make_pd(1., 2), make_pd(2., 2), make_pd(3., 2), make_pd(4., 2)])
      chol_x = np.array([chol(x[0]), chol(x[1]), chol(x[2]), chol(x[3])])

      # Since Wishart wasn"t added to SciPy until 0.16, we'll spot check some
      # pdfs with hard-coded results from upstream SciPy.

      log_prob_df_seq = np.array([
          # math.log(stats.wishart.pdf(x[0], df=2+0, scale=x[0]))
          -3.5310242469692907,
          # math.log(stats.wishart.pdf(x[1], df=2+1, scale=x[1]))
          -7.689907330328961,
          # math.log(stats.wishart.pdf(x[2], df=2+2, scale=x[2]))
          -10.815845159537895,
          # math.log(stats.wishart.pdf(x[3], df=2+3, scale=x[3]))
          -13.640549882916691,
      ])

      # This test checks that batches don't interfere with correctness.
      w = tfd.Wishart(
          df=[2, 3, 4, 5],
          scale_tril=chol_x,
          cholesky_input_output_matrices=True)
      self.assertAllClose(log_prob_df_seq, w.log_prob(chol_x).eval())

      # Now we test various constructions of Wishart with different sample
      # shape.

      log_prob = np.array([
          # math.log(stats.wishart.pdf(x[0], df=4, scale=x[0]))
          -4.224171427529236,
          # math.log(stats.wishart.pdf(x[1], df=4, scale=x[0]))
          -6.3378770664093453,
          # math.log(stats.wishart.pdf(x[2], df=4, scale=x[0]))
          -12.026946850193017,
          # math.log(stats.wishart.pdf(x[3], df=4, scale=x[0]))
          -20.951582705289454,
      ])

      for w in (
          tfd.Wishart(
              df=4,
              scale_tril=chol_x[0],
              cholesky_input_output_matrices=False),
          tfd.Wishart(
              df=4, scale=x[0], cholesky_input_output_matrices=False)):
        self.assertAllEqual((2, 2), w.event_shape_tensor().eval())
        self.assertEqual(2, w.dimension.eval())
        self.assertAllClose(log_prob[0], w.log_prob(x[0]).eval())
        self.assertAllClose(log_prob[0:2], w.log_prob(x[0:2]).eval())
        self.assertAllClose(
            np.reshape(log_prob, (2, 2)),
            w.log_prob(np.reshape(x, (2, 2, 2, 2))).eval())
        self.assertAllClose(
            np.reshape(np.exp(log_prob), (2, 2)),
            w.prob(np.reshape(x, (2, 2, 2, 2))).eval())
        self.assertAllEqual((2, 2),
                            w.log_prob(np.reshape(x, (2, 2, 2, 2))).get_shape())

      for w in (
          tfd.Wishart(df=4,
                      scale_tril=chol_x[0],
                      cholesky_input_output_matrices=True),
          tfd.Wishart(df=4, scale=x[0], cholesky_input_output_matrices=True)):
        self.assertAllEqual((2, 2), w.event_shape_tensor().eval())
        self.assertEqual(2, w.dimension.eval())
        self.assertAllClose(log_prob[0], w.log_prob(chol_x[0]).eval())
        self.assertAllClose(log_prob[0:2], w.log_prob(chol_x[0:2]).eval())
        self.assertAllClose(
            np.reshape(log_prob, (2, 2)),
            w.log_prob(np.reshape(chol_x, (2, 2, 2, 2))).eval())
        self.assertAllClose(
            np.reshape(np.exp(log_prob), (2, 2)),
            w.prob(np.reshape(chol_x, (2, 2, 2, 2))).eval())
        self.assertAllEqual((2, 2),
                            w.log_prob(np.reshape(x, (2, 2, 2, 2))).get_shape())

  def testBatchShape(self):
    with self.test_session() as sess:
      scale = make_pd(1., 2)
      chol_scale = chol(scale)

      w = tfd.Wishart(df=4, scale_tril=chol_scale)
      self.assertAllEqual([], w.batch_shape)
      self.assertAllEqual([], w.batch_shape_tensor().eval())

      w = tfd.Wishart(
          df=[4., 4], scale_tril=np.array([chol_scale, chol_scale]))
      self.assertAllEqual([2], w.batch_shape)
      self.assertAllEqual([2], w.batch_shape_tensor().eval())

      scale_deferred = tf.placeholder(tf.float32)
      w = tfd.Wishart(df=4, scale_tril=scale_deferred)
      self.assertAllEqual(
          [], sess.run(w.batch_shape_tensor(),
                       feed_dict={scale_deferred: chol_scale}))
      self.assertAllEqual(
          [2],
          sess.run(w.batch_shape_tensor(),
                   feed_dict={scale_deferred: [chol_scale, chol_scale]}))

  def testEventShape(self):
    with self.test_session() as sess:
      scale = make_pd(1., 2)
      chol_scale = chol(scale)

      w = tfd.Wishart(df=4, scale_tril=chol_scale)
      self.assertAllEqual([2, 2], w.event_shape)
      self.assertAllEqual([2, 2], w.event_shape_tensor().eval())

      w = tfd.Wishart(
          df=[4., 4], scale_tril=np.array([chol_scale, chol_scale]))
      self.assertAllEqual([2, 2], w.event_shape)
      self.assertAllEqual([2, 2], w.event_shape_tensor().eval())

      scale_deferred = tf.placeholder(tf.float32)
      w = tfd.Wishart(df=4, scale_tril=scale_deferred)
      self.assertAllEqual(
          [2, 2],
          sess.run(w.event_shape_tensor(),
                   feed_dict={scale_deferred: chol_scale}))
      self.assertAllEqual(
          [2, 2],
          sess.run(w.event_shape_tensor(),
                   feed_dict={scale_deferred: [chol_scale, chol_scale]}))

  def testValidateArgs(self):
    with self.test_session() as sess:
      df_deferred = tf.placeholder(tf.float32)
      chol_scale_deferred = tf.placeholder(tf.float32)
      x = make_pd(1., 3)
      chol_scale = chol(x)

      # Check expensive, deferred assertions.
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "cannot be less than"):
        chol_w = tfd.Wishart(
            df=df_deferred,
            scale_tril=chol_scale_deferred,
            validate_args=True)
        sess.run(chol_w.log_prob(np.asarray(
            x, dtype=np.float32)),
                 feed_dict={df_deferred: 2.,
                            chol_scale_deferred: chol_scale})

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Cholesky decomposition was not successful"):
        chol_w = tfd.Wishart(df=df_deferred, scale=chol_scale_deferred)
        # np.ones((3, 3)) is not positive, definite.
        sess.run(chol_w.log_prob(np.asarray(
            x, dtype=np.float32)),
                 feed_dict={
                     df_deferred: 4.,
                     chol_scale_deferred: np.ones(
                         (3, 3), dtype=np.float32)
                 })

      with self.assertRaisesOpError("scale_tril must be square"):
        chol_w = tfd.Wishart(
            df=4.,
            scale_tril=np.array([[2., 3., 4.], [1., 2., 3.]],
                                dtype=np.float32),
            validate_args=True)
        sess.run(chol_w.scale().eval())

      # Ensure no assertions.
      chol_w = tfd.Wishart(
          df=df_deferred, scale_tril=chol_scale_deferred,
          validate_args=False)
      sess.run(chol_w.log_prob(np.asarray(
          x, dtype=np.float32)),
               feed_dict={df_deferred: 4,
                          chol_scale_deferred: chol_scale})
      # Bogus log_prob, but since we have no checks running... c"est la vie.
      sess.run(chol_w.log_prob(np.asarray(
          x, dtype=np.float32)),
               feed_dict={df_deferred: 4,
                          chol_scale_deferred: np.ones((3, 3))})

  def testStaticAsserts(self):
    with self.test_session():
      x = make_pd(1., 3)
      chol_scale = chol(x)

      # Still has these assertions because they're resolveable at graph
      # construction
      with self.assertRaisesRegexp(ValueError, "cannot be less than"):
        tfd.Wishart(df=2, scale_tril=chol_scale, validate_args=False)
      with self.assertRaisesRegexp(TypeError, "Argument tril must have dtype"):
        tfd.Wishart(
            df=4.,
            scale_tril=np.asarray(chol_scale, dtype=np.int32),
            validate_args=False)

  def testSampleBroadcasts(self):
    dims = 2
    batch_shape = [2, 3]
    sample_shape = [2, 1]
    scale = np.float32([
        [[1., 0.5],
         [0.5, 1.]],
        [[0.5, 0.25],
         [0.25, 0.75]],
    ])
    scale = np.reshape(np.concatenate([scale, scale, scale], axis=0),
                       batch_shape + [dims, dims])
    wishart = tfd.Wishart(df=5, scale=scale)
    x = wishart.sample(sample_shape, seed=42)
    with self.test_session() as sess:
      x_ = sess.run(x)
    expected_shape = sample_shape + batch_shape + [dims, dims]
    self.assertAllEqual(expected_shape, x.shape)
    self.assertAllEqual(expected_shape, x_.shape)


if __name__ == "__main__":
  tf.test.main()
