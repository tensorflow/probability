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
"""Tests for the JointDistributionSequential."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
from absl.testing import parameterized

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


tfd = tfp.distributions


class Dummy(object):
  """Dummy object to ensure `tf_inspect.getfullargspec` works for `__init__`."""

  # To ensure no code is keying on the unspecial name "self", we use "me".
  def __init__(me, arg1, arg2, arg3=None, **named):  # pylint: disable=no-self-argument
    pass


@test_util.run_all_in_graph_and_eager_modes
class JointDistributionSequentialTest(
    test_case.TestCase, parameterized.TestCase):

  def test_sample_log_prob(self):
    d = tfd.JointDistributionSequential(
        [
            tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
            lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
            tfd.Normal(loc=0, scale=2.),
            tfd.Normal,  # Or, `lambda loc, scale: tfd.Normal(loc, scale)`.
            lambda m: tfd.Sample(tfd.Bernoulli(logits=m), 12),
        ],
        validate_args=True)

    self.assertEqual(
        (
            ('e', ()),
            ('scale', ('e',)),
            ('loc', ()),
            ('m', ('loc', 'scale')),
            ('x', ('m',)),
        ),
        d._resolve_graph())

    xs = d.sample(seed=tfp_test_util.test_seed())
    self.assertLen(xs, 5)
    # We'll verify the shapes work as intended when we plumb these back into the
    # respective log_probs.

    ds, _ = d.sample_distributions(value=xs)
    self.assertLen(ds, 5)
    self.assertIsInstance(ds[0], tfd.Independent)
    self.assertIsInstance(ds[1], tfd.Gamma)
    self.assertIsInstance(ds[2], tfd.Normal)
    self.assertIsInstance(ds[3], tfd.Normal)
    self.assertIsInstance(ds[4], tfd.Sample)

    # Static properties.
    self.assertAllEqual(
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32],
        d.dtype)
    for expected, actual_tensorshape, actual_shapetensor in zip(
        [[2], [], [], [], [12]],
        d.event_shape,
        self.evaluate(d.event_shape_tensor())):
      self.assertAllEqual(expected, actual_tensorshape)
      self.assertAllEqual(expected, actual_shapetensor)

    for expected, actual_tensorshape, actual_shapetensor in zip(
        [[], [], [], []],
        d.batch_shape,
        self.evaluate(d.batch_shape_tensor())):
      self.assertAllEqual(expected, actual_tensorshape)
      self.assertAllEqual(expected, actual_shapetensor)

    expected_jlp = sum(d_.log_prob(x) for d_, x in zip(ds, xs))
    actual_jlp = d.log_prob(xs)
    self.assertAllEqual(*self.evaluate([expected_jlp, actual_jlp]))

  def test_kl_divergence(self):
    d0 = tfd.JointDistributionSequential(
        [
            tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
            tfd.Normal(loc=0, scale=2.),
        ],
        validate_args=True)
    d1 = tfd.JointDistributionSequential(
        [
            tfd.Independent(tfd.Exponential(rate=[10, 12]), 1),
            tfd.Normal(loc=1, scale=1.),
        ],
        validate_args=True)
    expected_kl = sum(tfd.kl_divergence(d0_, d1_) for d0_, d1_
                      in zip(d0.model, d1.model))
    actual_kl = tfd.kl_divergence(d0, d1)
    other_actual_kl = d0.kl_divergence(d1)
    expected_kl_, actual_kl_, other_actual_kl_ = self.evaluate([
        expected_kl, actual_kl, other_actual_kl])
    self.assertNear(expected_kl_, actual_kl_, err=1e-5)
    self.assertNear(expected_kl_, other_actual_kl_, err=1e-5)

  def test_cross_entropy(self):
    d0 = tfd.JointDistributionSequential(
        [
            tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
            tfd.Normal(loc=0, scale=2.),
        ],
        validate_args=True)
    d1 = tfd.JointDistributionSequential(
        [
            tfd.Independent(tfd.Exponential(rate=[10, 12]), 1),
            tfd.Normal(loc=1, scale=1.),
        ],
        validate_args=True)
    expected_xent = sum(
        d0_.cross_entropy(d1_) for d0_, d1_
        in zip(d0.model, d1.model))
    actual_xent = d0.cross_entropy(d1)
    expected_xent_, actual_xent_ = self.evaluate([expected_xent, actual_xent])
    self.assertNear(actual_xent_, expected_xent_, err=1e-5)

  def test_norequired_args_maker(self):
    """Test that only non-default args are passed through."""
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Must pass probs or logits, but not both.'):
      tfd.JointDistributionSequential([tfd.Normal(0., 1.), tfd.Bernoulli])

  def test_graph_resolution(self):
    d = tfd.JointDistributionSequential(
        [
            tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
            lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
            tfd.HalfNormal(2.5),
            lambda s: tfd.Normal(loc=0, scale=s),
            tfd.Exponential(2),
            lambda df, loc, _, scale: tfd.StudentT(df, loc, scale),
        ],
        validate_args=True)
    self.assertEqual(
        (('e', ()),
         ('scale', ('e',)),
         ('s', ()),
         ('loc', ('s',)),
         ('df', ()),
         ('x', ('df', 'loc', '_', 'scale'))),
        d._resolve_graph())

  @parameterized.parameters('mean', 'mode', 'stddev', 'variance')
  def test_summary_statistic(self, attr):
    d = tfd.JointDistributionSequential(
        [tfd.Normal(0., 1.), tfd.Bernoulli(logits=0.)],
        validate_args=True)
    expected = tuple(getattr(d_, attr)() for d_ in d.model)
    actual = getattr(d, attr)()
    self.assertAllEqual(*self.evaluate([expected, actual]))

  @parameterized.parameters(('covariance',))
  def test_notimplemented_summary_statistic(self, attr):
    d = tfd.JointDistributionSequential([tfd.Normal(0., 1.),
                                         tfd.Bernoulli(probs=0.5)],
                                        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionSequential'):
      getattr(d, attr)()

  @parameterized.parameters(
      'quantile', 'log_cdf', 'cdf',
      'log_survival_function', 'survival_function',
  )
  def test_notimplemented_evaluative_statistic(self, attr):
    d = tfd.JointDistributionSequential([tfd.Normal(0., 1.),
                                         tfd.Bernoulli(probs=0.5)],
                                        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionSequential'):
      getattr(d, attr)([0.]*len(d.model))

  def test_copy(self):
    pgm = [tfd.Normal(0., 1.), tfd.Bernoulli(probs=0.5)]
    d = tfd.JointDistributionSequential(pgm, validate_args=True)
    d_copy = d.copy()
    self.assertAllEqual(
        {'model': pgm,
         'validate_args': True,
         'name': None},
        d_copy.parameters)

  def test_batch_slicing(self):
    d = tfd.JointDistributionSequential(
        [
            tfd.Exponential(rate=[10, 12, 14]),
            lambda s: tfd.Normal(loc=0, scale=s),
            lambda: tfd.Beta(concentration0=[3, 2, 1], concentration1=1),
        ],
        validate_args=True)

    d0, d1 = d[:1], d[1:]
    x0 = d0.sample(seed=tfp_test_util.test_seed())
    x1 = d1.sample(seed=tfp_test_util.test_seed())

    self.assertLen(x0, 3)
    self.assertEqual([1], x0[0].shape)
    self.assertEqual([1], x0[1].shape)
    self.assertEqual([1], x0[2].shape)

    self.assertLen(x1, 3)
    self.assertEqual([2], x1[0].shape)
    self.assertEqual([2], x1[1].shape)
    self.assertEqual([2], x1[2].shape)

  def test_sample_shape_propagation_default_behavior(self):
    d = tfd.JointDistributionSequential(
        [
            tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
            lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
            tfd.HalfNormal(2.5),
            lambda s: tfd.Normal(loc=0, scale=s),
            tfd.Exponential(2),
            lambda df, loc, _, scale: tfd.StudentT(df, loc, scale),
        ],
        validate_args=True)
    x = d.sample([2, 3], seed=tfp_test_util.test_seed())
    self.assertLen(x, 6)
    self.assertEqual((2, 3, 2), x[0].shape)
    self.assertEqual((2, 3), x[1].shape)
    self.assertEqual((2, 3), x[2].shape)
    self.assertEqual((2, 3), x[3].shape)
    self.assertEqual((2, 3), x[4].shape)
    self.assertEqual((2, 3), x[5].shape)
    lp = d.log_prob(x)
    self.assertEqual((2, 3), lp.shape)

  def test_sample_shape_propagation_nondefault_behavior(self):
    d = tfd.JointDistributionSequential(
        [
            tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),          # 0
            lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),  # 1
            tfd.HalfNormal(2.5),                                           # 2
            lambda s: tfd.Normal(loc=0, scale=s),                          # 3
            tfd.Exponential(2),                                            # 4
            lambda df, loc, _, scale: tfd.StudentT(df, loc, scale),        # 5
        ],
        validate_args=False)  # So log_prob doesn't complain.
    # The following enables the nondefault sample shape behavior.
    d._always_use_specified_sample_shape = True
    sample_shape = (2, 3)
    x = d.sample(sample_shape, seed=tfp_test_util.test_seed())
    self.assertLen(x, 6)
    self.assertEqual(sample_shape + (2,), x[0].shape)
    self.assertEqual(sample_shape * 2, x[1].shape)  # Has 1 arg.
    self.assertEqual(sample_shape * 1, x[2].shape)  # Has 0 args.
    self.assertEqual(sample_shape * 2, x[3].shape)  # Has 1 arg.
    self.assertEqual(sample_shape * 1, x[4].shape)  # Has 0 args.
    # Has 3 args, one being scalar.
    self.assertEqual(sample_shape * 3, x[5].shape)
    lp = d.log_prob(x)
    self.assertEqual(sample_shape * 3, lp.shape)

  def test_only_memoize_non_user_input(self):
    d = tfd.JointDistributionSequential(
        [
            lambda: tfd.Bernoulli(logits=0.),
            lambda c: tfd.Normal(loc=tf.cast(c, tf.float32), scale=1.),
        ],
        validate_args=True)
    self.assertEqual([tf.int32, None], d.dtype)
    d.sample(value=[1., None])  # For the *potential* side-effect
    self.assertEqual([tf.int32, None], d.dtype)
    d.sample()  # For the *actual* side-effect
    self.assertEqual([tf.int32, tf.float32], d.dtype)

  def test_argspec(self):
    argspec = tf_inspect.getfullargspec(Dummy)
    self.assertAllEqual(['me', 'arg1', 'arg2', 'arg3'], argspec.args)
    self.assertIs(None, argspec.varargs)
    self.assertIs('named', argspec.varkw)
    self.assertAllEqual((None,), argspec.defaults)

  def test_invalid_structure_raises_error(self):
    with self.assertRaisesWithPredicateMatch(
        TypeError, 'Unable to unflatten like `model` with type "model".'):
      tfd.JointDistributionSequential(collections.namedtuple('model', 'a b')(
          a=tfd.Normal(0, 1),
          b=tfd.Normal(1, 2)))

  def test_latent_dirichlet_allocation(self):
    """Tests Latent Dirichlet Allocation joint model.

    The LDA generative process can be written as:

    ```none
    N[i] ~ Poisson(xi)
    theta[i] ~ Dirichlet(alpha)
    Z[i] ~ Multinomial(N[i], theta[i])
    for k in 1...K:
      X[i,k] ~ Multinomial(Z[i, k], beta[j])
    ```

    Typically `xi` is specified and `alpha`, `beta` are fit using type-II
    maximum likelihood estimators.

    Reference: http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
    """

    # Hyperparameters.
    num_topics = 3
    num_words = 10
    avg_doc_length = 5
    u = tfd.Uniform(low=-1., high=1.)
    alpha = tfp.util.DeferredTensor(
        tf.nn.softplus,
        tf.Variable(u.sample([num_topics]), name='raw_alpha'))
    beta = tf.Variable(u.sample([num_topics, num_words]), name='beta')

    # LDA Model.
    # Note near 1:1 with mathematical specification. The main distinction is the
    # use of Independent--this lets us easily aggregate multinomials across
    # topics (and in any "shape" of documents).
    lda = tfd.JointDistributionSequential([
        tfd.Poisson(rate=avg_doc_length),                              # n
        tfd.Dirichlet(concentration=alpha),                            # theta
        lambda theta, n: tfd.Multinomial(total_count=n, probs=theta),  # z
        lambda z: tfd.Independent(                                     # x  pylint: disable=g-long-lambda
            tfd.Multinomial(total_count=z, logits=beta),
            reinterpreted_batch_ndims=1),
    ])

    # Now, let's sample some "documents" and compute the log-prob of each.
    docs_shape = [2, 4]  # That is, 8 docs in the shape of [2, 4].
    [n, theta, z, x] = lda.sample(docs_shape)
    log_probs = lda.log_prob([n, theta, z, x])
    self.assertEqual(docs_shape, log_probs.shape)

    # Verify we correctly track trainable variables.
    self.assertAllEqual((alpha.pretransformed_input, beta),
                        lda.trainable_variables)

    # Ensure we can compute gradients.
    with tf.GradientTape() as tape:
      # Note: The samples are not taped, hence implicitly "stop_gradient."
      negloglik = -lda.log_prob([n, theta, z, x])
    grads = tape.gradient(negloglik, lda.trainable_variables)

    self.assertLen(grads, 2)
    self.assertAllEqual((alpha.pretransformed_input.shape, beta.shape),
                        (grads[0].shape, grads[1].shape))
    self.assertAllNotNone(grads)


if __name__ == '__main__':
  tf.test.main()
