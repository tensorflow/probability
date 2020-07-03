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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


def make_pd(start, n):
  """Deterministically create a positive definite matrix."""
  x = np.tril(linalg.circulant(np.arange(start, start + n)))
  return np.dot(x, x.T)


def chol(x):
  """Compute Cholesky factorization."""
  return linalg.cholesky(x).T


def wishart_var(df, x):
  """Compute Wishart variance for numpy scale matrix."""
  df = np.asarray(df)[..., np.newaxis, np.newaxis]
  x = np.asarray(x)
  d = np.diagonal(x, axis1=-2, axis2=-1)
  return df * (x**2 + np.matmul(
      d[..., np.newaxis], d[..., np.newaxis, :]))


@test_util.test_all_tf_execution_regimes
class WishartTest(test_util.TestCase):

  def testEntropy(self):
    scale = make_pd(1., 2)
    df = 4
    w = tfd.WishartTriL(df, scale_tril=chol(scale), validate_args=True)
    # sp.stats.wishart(df=4, scale=make_pd(1., 2)).entropy()
    self.assertAllClose(6.301387092430769, self.evaluate(w.entropy()))

    w = tfd.WishartTriL(df=1, scale_tril=[[1.]], validate_args=True)
    # sp.stats.wishart(df=1,scale=1).entropy()
    self.assertAllClose(0.78375711047393404, self.evaluate(w.entropy()))

  def testMeanLogDetAndLogNormalizingConstant(self):

    def entropy_alt(w):
      dimension = tf.cast(w.scale.domain_dimension_tensor(), w.dtype)
      return self.evaluate(w.log_normalization() -
                           0.5 * (w.df - dimension - 1.) * w.mean_log_det() +
                           0.5 * w.df * dimension)

    w = tfd.WishartTriL(
        df=4, scale_tril=chol(make_pd(1., 2)), validate_args=True)
    self.assertAllClose(self.evaluate(w.entropy()), entropy_alt(w))

    w = tfd.WishartTriL(df=5, scale_tril=[[1.]], validate_args=True)
    self.assertAllClose(self.evaluate(w.entropy()), entropy_alt(w))

  def testParamBroadcasting(self):
    # A 1-batch of 2x2 scale operators
    scale_tril = np.expand_dims(chol([[1., .5], [.5, 1.]]), 0)

    df = [5, 6, 7]  # A 3-batch of degrees of freedom
    wish = tfp.distributions.WishartTriL(df=df, scale_tril=scale_tril)
    self.assertAllEqual([2, 2], tensorshape_util.as_list(wish.event_shape))
    self.assertAllEqual([2, 2], self.evaluate(wish.event_shape_tensor()))
    self.assertAllEqual([3], tensorshape_util.as_list(wish.batch_shape))
    self.assertAllEqual([3], self.evaluate(wish.batch_shape_tensor()))
    self.assertAllEqual([4, 3, 2, 2], wish.sample(
        sample_shape=(4,), seed=test_util.test_seed()).shape)
    self.assertAllEqual([4, 3, 2, 2],
                        self.evaluate(tf.shape(wish.sample(
                            sample_shape=(4,), seed=test_util.test_seed()))))

  def testMean(self):
    scale = make_pd(1., 2)
    df = 4
    w = tfd.WishartTriL(df, scale_tril=chol(scale), validate_args=True)
    self.assertAllEqual(df * scale, self.evaluate(w.mean()))

  def testMeanBroadcast(self):
    scale = [make_pd(1., 2), make_pd(1., 2)]
    chol_scale = np.float32([chol(s) for s in scale])
    scale = np.float32(scale)
    df = np.array([4., 3.], dtype=np.float32)
    w = tfd.WishartTriL(df, scale_tril=chol_scale, validate_args=True)
    self.assertAllEqual(
        df[..., np.newaxis, np.newaxis] * scale, self.evaluate(w.mean()))

  def testMode(self):
    scale = make_pd(1., 2)
    df = 4
    w = tfd.WishartTriL(df, scale_tril=chol(scale), validate_args=True)
    self.assertAllEqual((df - 2. - 1.) * scale, self.evaluate(w.mode()))

  def testStd(self):
    scale = make_pd(1., 2)
    df = 4
    w = tfd.WishartTriL(df, scale_tril=chol(scale), validate_args=True)
    self.assertAllEqual(
        np.sqrt(wishart_var(df, scale)), self.evaluate(w.stddev()))

  def testVariance(self):
    scale = make_pd(1., 2)
    df = 4
    w = tfd.WishartTriL(df, scale_tril=chol(scale), validate_args=True)
    self.assertAllEqual(wishart_var(df, scale), self.evaluate(w.variance()))

  def testVarianceBroadcast(self):
    scale = [make_pd(1., 2), make_pd(1., 2)]
    chol_scale = np.float32([chol(s) for s in scale])
    scale = np.float32(scale)
    df = np.array([4., 3.], dtype=np.float32)
    w = tfd.WishartTriL(df, scale_tril=chol_scale, validate_args=True)
    self.assertAllEqual(wishart_var(df, scale), self.evaluate(w.variance()))

  def testSamplingEmptyDist(self):
    w = tfd.WishartTriL(df=[1], scale_tril=[[1.]], validate_args=True)
    self.evaluate(w[:0].sample(seed=test_util.test_seed()))

  def testLogProbEmptyDist(self):
    w = tfd.WishartTriL(df=[1], scale_tril=[[1.]], validate_args=True)
    self.evaluate(w[:0].log_prob([[1.]]))

  def testSampleWithSameSeed(self):
    if tf.executing_eagerly():
      return
    scale = make_pd(1., 2)
    df = 4
    seed = test_util.test_seed()

    chol_w = tfd.WishartTriL(
        df,
        scale_tril=chol(scale),
        input_output_cholesky=False,
        validate_args=True)

    x = self.evaluate(chol_w.sample(1, seed=seed))
    chol_x = [chol(x[0])]

    chol_w_chol = tfd.WishartTriL(
        df,
        scale_tril=chol(scale),
        input_output_cholesky=True,
        validate_args=True)
    self.assertAllClose(chol_x, self.evaluate(chol_w_chol.sample(1, seed=seed)))
    eigen_values = tf.linalg.diag_part(chol_w_chol.sample(1000, seed=seed))
    np.testing.assert_array_less(0., self.evaluate(eigen_values))

  def testSample(self):
    # Check first and second moments.
    df = 4.
    chol_w = tfd.WishartTriL(
        df=df,
        scale_tril=chol(make_pd(1., 3)),
        input_output_cholesky=False,
        validate_args=True)
    x = chol_w.sample(10000, seed=test_util.test_seed(hardcoded_seed=42))
    self.assertAllEqual((10000, 3, 3), x.shape)

    moment1_estimate = self.evaluate(tf.reduce_mean(x, axis=[0]))
    self.assertAllClose(
        self.evaluate(chol_w.mean()), moment1_estimate, rtol=0.05)

    # The Variance estimate uses the squares rather than outer-products
    # because Wishart.Variance is the diagonal of the Wishart covariance
    # matrix.
    variance_estimate = self.evaluate(
        tf.reduce_mean(tf.square(x), axis=[0]) - tf.square(moment1_estimate))
    self.assertAllClose(
        self.evaluate(chol_w.variance()), variance_estimate, rtol=0.15)

  # Test that sampling with the same seed twice gives the same results.
  def testSampleMultipleTimes(self):
    df = 4.
    n_val = 100
    seed = test_util.test_seed()

    tf.random.set_seed(seed)
    chol_w1 = tfd.WishartTriL(
        df=df,
        scale_tril=chol(make_pd(1., 3)),
        input_output_cholesky=False,
        validate_args=True,
        name='wishart1')

    samples1 = self.evaluate(chol_w1.sample(n_val, seed=seed))

    tf.random.set_seed(seed)
    chol_w2 = tfd.WishartTriL(
        df=df,
        scale_tril=chol(make_pd(1., 3)),
        input_output_cholesky=False,
        validate_args=True,
        name='wishart2')
    samples2 = self.evaluate(chol_w2.sample(n_val, seed=seed))

    self.assertAllClose(samples1, samples2)

  def testProb(self):
    # Generate some positive definite (pd) matrices and their Cholesky
    # factorizations.
    x = np.array(
        [make_pd(1., 2),
         make_pd(2., 2),
         make_pd(3., 2),
         make_pd(4., 2)])
    chol_x = np.array([chol(x[0]), chol(x[1]), chol(x[2]), chol(x[3])])

    # Since Wishart wasn't added to SciPy until 0.16, we'll spot check some
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
    w = tfd.WishartTriL(
        df=[2, 3, 4, 5],
        scale_tril=chol_x,
        input_output_cholesky=True,
        validate_args=True)
    self.assertAllClose(log_prob_df_seq, self.evaluate(w.log_prob(chol_x)))

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

    w = tfd.WishartTriL(
        df=4,
        scale_tril=chol_x[0],
        input_output_cholesky=False,
        validate_args=True)
    dimension = w.scale.domain_dimension_tensor()
    self.assertAllEqual((2, 2), self.evaluate(w.event_shape_tensor()))
    self.assertEqual(2, self.evaluate(dimension))
    self.assertAllClose(log_prob[0], self.evaluate(w.log_prob(x[0])))
    self.assertAllClose(log_prob[0:2], self.evaluate(w.log_prob(x[0:2])))
    self.assertAllClose(
        np.reshape(log_prob, (2, 2)),
        self.evaluate(w.log_prob(np.reshape(x, (2, 2, 2, 2)))))
    self.assertAllClose(
        np.reshape(np.exp(log_prob), (2, 2)),
        self.evaluate(w.prob(np.reshape(x, (2, 2, 2, 2)))))
    self.assertAllEqual((2, 2),
                        w.log_prob(np.reshape(x, (2, 2, 2, 2))).shape)

    w = tfd.WishartTriL(
        df=4,
        scale_tril=chol_x[0],
        input_output_cholesky=True,
        validate_args=True)
    dimension = w.scale.domain_dimension_tensor()
    self.assertAllEqual((2, 2), self.evaluate(w.event_shape_tensor()))
    self.assertEqual(2, self.evaluate(dimension))
    self.assertAllClose(log_prob[0], self.evaluate(w.log_prob(chol_x[0])))
    self.assertAllClose(log_prob[0:2], self.evaluate(w.log_prob(chol_x[0:2])))
    self.assertAllClose(
        np.reshape(log_prob, (2, 2)),
        self.evaluate(w.log_prob(np.reshape(chol_x, (2, 2, 2, 2)))))
    self.assertAllClose(
        np.reshape(np.exp(log_prob), (2, 2)),
        self.evaluate(w.prob(np.reshape(chol_x, (2, 2, 2, 2)))))
    self.assertAllEqual((2, 2),
                        w.log_prob(np.reshape(x, (2, 2, 2, 2))).shape)

  def testBatchShape(self):
    scale = make_pd(1., 2)
    chol_scale = chol(scale)

    w = tfd.WishartTriL(df=4, scale_tril=chol_scale, validate_args=True)
    self.assertAllEqual([], w.batch_shape)
    self.assertAllEqual([], self.evaluate(w.batch_shape_tensor()))

    w = tfd.WishartTriL(
        df=[4., 4],
        scale_tril=np.array([chol_scale, chol_scale]),
        validate_args=True)
    self.assertAllEqual([2], w.batch_shape)
    self.assertAllEqual([2], self.evaluate(w.batch_shape_tensor()))

    scale_deferred = tf1.placeholder_with_default(
        chol_scale, shape=chol_scale.shape)
    w = tfd.WishartTriL(df=4, scale_tril=scale_deferred, validate_args=True)
    self.assertAllEqual([], self.evaluate(w.batch_shape_tensor()))

    scale_deferred = tf1.placeholder_with_default(
        np.array([chol_scale, chol_scale]), shape=None)
    w = tfd.WishartTriL(df=4, scale_tril=scale_deferred, validate_args=True)
    self.assertAllEqual([2], self.evaluate(w.batch_shape_tensor()))

  def testEventShape(self):
    scale = make_pd(1., 2)
    chol_scale = chol(scale)

    w = tfd.WishartTriL(df=4, scale_tril=chol_scale, validate_args=True)
    self.assertAllEqual([2, 2], w.event_shape)
    self.assertAllEqual([2, 2], self.evaluate(w.event_shape_tensor()))

    w = tfd.WishartTriL(
        df=[4., 4],
        scale_tril=np.array([chol_scale, chol_scale]),
        validate_args=True)
    self.assertAllEqual([2, 2], w.event_shape)
    self.assertAllEqual([2, 2], self.evaluate(w.event_shape_tensor()))

    scale_deferred = tf1.placeholder_with_default(
        chol_scale, shape=chol_scale.shape)
    w = tfd.WishartTriL(df=4, scale_tril=scale_deferred, validate_args=True)
    self.assertAllEqual([2, 2], self.evaluate(w.event_shape_tensor()))

    scale_deferred = tf1.placeholder_with_default(
        np.array([chol_scale, chol_scale]), shape=None)
    w = tfd.WishartTriL(df=4, scale_tril=scale_deferred, validate_args=True)
    self.assertAllEqual([2, 2], self.evaluate(w.event_shape_tensor()))

  def testValidateArgs(self):
    x = make_pd(1., 3)
    chol_scale = chol(x)
    df_deferred = tf1.placeholder_with_default(2., shape=None)
    chol_scale_deferred = tf1.placeholder_with_default(
        np.float32(chol_scale), shape=chol_scale.shape)

    # In eager mode, these checks are done statically and hence
    # ValueError is returned on object construction.
    error_type = tf.errors.InvalidArgumentError
    if tf.executing_eagerly():
      error_type = ValueError

    # Check expensive, deferred assertions.
    with self.assertRaisesRegexp(error_type, 'cannot be less than'):
      chol_w = tfd.WishartTriL(
          df=df_deferred, scale_tril=chol_scale_deferred, validate_args=True)
      self.evaluate(chol_w.log_prob(np.asarray(x, dtype=np.float32)))

    with self.assertRaisesOpError('`scale_tril` must be square.'):
      chol_w = tfd.WishartTriL(
          df=4.,
          scale_tril=np.array([[2., 3., 4.], [1., 2., 3.]], dtype=np.float32),
          validate_args=True)
      self.evaluate(chol_w.scale())

    # Ensure no assertions.
    df_deferred = tf1.placeholder_with_default(4., shape=None)
    chol_scale_deferred = tf1.placeholder_with_default(
        np.float32(chol_scale), shape=chol_scale.shape)
    chol_w = tfd.WishartTriL(
        df=df_deferred, scale_tril=chol_scale_deferred, validate_args=False)
    self.evaluate(chol_w.log_prob(np.asarray(x, dtype=np.float32)))

    chol_scale_deferred = tf1.placeholder_with_default(
        np.ones([3, 3], dtype=np.float32), shape=[3, 3])
    chol_w = tfd.WishartTriL(
        df=df_deferred, scale_tril=chol_scale_deferred, validate_args=False)
    # Bogus log_prob, but since we have no checks running... c'est la vie.
    self.evaluate(chol_w.log_prob(np.asarray(x, dtype=np.float32)))

  def testStaticAsserts(self):
    x = make_pd(1., 3)
    chol_scale = chol(x)

    # Still has these assertions because they're resolveable at graph
    # construction:
    # df < rank
    with self.assertRaisesRegexp(ValueError, 'cannot be less than'):
      tfd.WishartTriL(df=2, scale_tril=chol_scale, validate_args=False)
    # non-float dtype
    with self.assertRaisesRegexp(TypeError, '.'):
      tfd.WishartTriL(
          df=4,
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
    scale_tril = np.stack([chol(s) for s in scale])
    scale_tril = np.reshape(np.concatenate([scale_tril]*3, axis=0),
                            batch_shape + [dims, dims])
    wishart = tfd.WishartTriL(df=5, scale_tril=scale_tril, validate_args=True)
    x = wishart.sample(sample_shape, seed=test_util.test_seed())
    x_ = self.evaluate(x)
    expected_shape = sample_shape + batch_shape + [dims, dims]
    self.assertAllEqual(expected_shape, x.shape)
    self.assertAllEqual(expected_shape, x_.shape)

  def testLogProbBroadcastsX(self):
    dims = 2
    batch_shape = [2, 3]
    scale = np.float32([
        [[1., 0.5],
         [0.5, 1.]],
        [[0.5, 0.25],
         [0.25, 0.75]],
    ])
    scale_tril = np.stack([chol(s) for s in scale])
    scale_tril = np.reshape(np.concatenate([scale_tril]*3, axis=0),
                            batch_shape + [dims, dims])
    wishart = tfd.WishartTriL(df=5, scale_tril=scale_tril, validate_args=True)
    x = np.random.randn(dims, dims)
    x = np.matmul(x, x.T)
    lp = wishart.log_prob(x)
    lp_bc = wishart.log_prob(x * np.ones([2, 3, 1, 1]))
    lp_, lp_bc_ = self.evaluate([lp, lp_bc])
    self.assertAllEqual(batch_shape, lp.shape)
    self.assertAllEqual(batch_shape, lp_.shape)
    self.assertAllEqual(batch_shape, lp_bc.shape)
    self.assertAllEqual(batch_shape, lp_bc_.shape)
    self.assertAllClose(lp_bc_, lp_)

  def testLogProbBroadcastOverDfInsideMixture(self):
    dims = 2
    scale = np.float32([[0.5, 0.25],  #
                        [0.25, 0.75]])
    df = np.arange(3., 8., dtype=np.float32)
    dist = tfd.MixtureSameFamily(
        components_distribution=tfd.WishartTriL(
            df=df, scale_tril=chol(scale), validate_args=True),
        mixture_distribution=tfd.Categorical(logits=tf.zeros(df.shape)))
    x = np.random.randn(dims, dims)
    x = np.matmul(x, x.T)
    lp = dist.log_prob(x)
    lp_ = self.evaluate(lp)
    self.assertAllEqual([], dist.batch_shape)
    self.assertAllEqual([], lp.shape)
    self.assertAllEqual([], lp_.shape)

  def testStaticAssertNonFlatDfDoesntRaise(self):
    # Check we don't get ValueError: The truth value of an array with more than
    # one element is ambiguous. Use a.any() or a.all()
    tfd.WishartTriL(
        df=[[2., 2]], scale_tril=chol(make_pd(1., 2)), validate_args=True)

  def testAssertsVariableDf(self):
    df = tf.Variable(2.)
    scale_tril = chol(make_pd(1., 4)).astype(np.float32)
    self.evaluate(df.initializer)
    msg = ('cannot be less than dimension of scale matrix.')
    with self.assertRaisesOpError(msg):
      d = tfd.WishartTriL(df=df, scale_tril=scale_tril, validate_args=True)
      self.evaluate(d.entropy())

  def testAssertsVariableDfAfterMutation(self):
    df = tf.Variable(3.)
    scale_tril = chol(make_pd(2., 2)).astype(np.float32)
    self.evaluate(df.initializer)
    d = tfd.WishartTriL(df=df, scale_tril=scale_tril, validate_args=True)
    msg = ('cannot be less than dimension of scale matrix.')
    with self.assertRaisesOpError(msg):
      with tf.control_dependencies([df.assign(1.)]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsVariableScale(self):
    scale_tril = tf.Variable(chol(make_pd(3., 4.)).astype(np.float32),
                             shape=tf.TensorShape(None))
    df = 3
    self.evaluate(scale_tril.initializer)
    with self.assertRaisesOpError('cannot be less than'):
      d = tfd.WishartTriL(df=df, scale_tril=scale_tril, validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsVariableScaleAfterMutation(self):
    df = tf.Variable(3.)
    scale_tril = tf.Variable(chol(make_pd(2., 2.)), dtype=np.float32,
                             shape=tf.TensorShape(None))
    self.evaluate([df.initializer, scale_tril.initializer])
    d = tfd.WishartTriL(df=df, scale_tril=scale_tril, validate_args=True)
    msg = 'cannot be less than dimension of scale matrix'
    with self.assertRaisesOpError(msg):
      with tf.control_dependencies([df.assign(-2.)]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

    invalid_scale_tril = chol(make_pd(2., 4.)).astype(np.float32)
    with self.assertRaisesOpError(msg):
      with tf.control_dependencies([scale_tril.assign(invalid_scale_tril)]):
        self.evaluate(d.mean())

  def testAssertsVariableScaleTril(self):
    df = 4
    scale_tril = tf.Variable([[-1., 0], [1., 0]])
    self.evaluate(scale_tril.initializer)
    with self.assertRaisesOpError('`scale_tril` must be positive definite.'):
      d = tfd.WishartTriL(df=df, scale_tril=scale_tril, validate_args=True)
      self.evaluate(d.entropy())

    scale_tril = tf.Variable([[1., 0., 0.], [1., 2., 0]])
    self.evaluate(scale_tril.initializer)
    with self.assertRaisesOpError('`scale_tril` must be square.'):
      d = tfd.WishartTriL(df=df, scale_tril=scale_tril, validate_args=True)
      self.evaluate(d.event_shape_tensor())

  def testAssertsVariableScaleTrilAfterMutation(self):
    df = 3
    scale_tril = tf.Variable(
        chol(make_pd(2., 2)), shape=[None, 2], validate_shape=False)
    self.evaluate(scale_tril.initializer)
    d = tfd.WishartTriL(df=df, scale_tril=scale_tril, validate_args=True)
    with self.assertRaisesOpError('`scale_tril` must be positive definite.'):
      non_pd_scale = [[-2., 0.], [1., 3.]]
      with tf.control_dependencies([scale_tril.assign(non_pd_scale)]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

    with self.assertRaisesOpError('`scale_tril` must be square.'):
      non_square_scale = [[1., 0], [-2., 1.], [1., 1.]]
      with tf.control_dependencies([scale_tril.assign(non_square_scale)]):
        self.evaluate(d.event_shape_tensor())


if __name__ == '__main__':
  tf.test.main()
