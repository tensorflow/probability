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
"""Tests for the JointDistributionAutoBatched."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections

# Dependency imports
from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


Root = tfd.JointDistributionCoroutineAutoBatched.Root


@test_util.test_all_tf_execution_regimes
class JointDistributionAutoBatchedTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'coroutine',
       'jd_class': tfd.JointDistributionCoroutineAutoBatched},
      {'testcase_name': 'sequential',
       'jd_class': tfd.JointDistributionSequentialAutoBatched},
      {'testcase_name': 'named',
       'jd_class': tfd.JointDistributionNamedAutoBatched})
  def test_batch_and_event_shape_with_plate(self, jd_class):

    models = {}

    def coroutine_model():
      g = yield Root(tfd.Gamma(2, 2))
      df = yield Root(tfd.Exponential(1.))
      loc = yield tfd.Sample(tfd.Normal(0, g), 20)
      yield tfd.StudentT(tf.expand_dims(df, -1), loc, 1)
    models[tfd.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[tfd.JointDistributionSequentialAutoBatched] = [
        tfd.Gamma(2, 2),
        tfd.Exponential(1.),
        lambda _, g: tfd.Sample(tfd.Normal(0, g), 20),
        lambda loc, df: tfd.StudentT(tf.expand_dims(df, -1), loc, 1)
    ]

    models[tfd.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('g', tfd.Gamma(2, 2)),
        ('df', tfd.Exponential(1.)),
        ('loc', lambda g: tfd.Sample(tfd.Normal(0, g), 20)),
        ('x', lambda loc, df: tfd.StudentT(tf.expand_dims(df, -1), loc, 1))))

    joint = jd_class(models[jd_class], validate_args=True)

    # Properties `event_shape` and `batch_shape` should be defined
    # even before any sampling calls have occurred.
    self.assertAllEqual(joint._model_flatten(joint.event_shape),
                        [[], [], [20], [20]])
    self.assertAllEqual(joint.batch_shape, [])

    is_scalar = joint._model_flatten(joint.is_scalar_event())
    self.assertAllEqual(is_scalar[0], True)
    self.assertAllEqual(is_scalar[1], True)
    self.assertAllEqual(is_scalar[2], False)
    self.assertAllEqual(is_scalar[3], False)

    event_shape = joint._model_flatten(joint.event_shape_tensor())
    self.assertAllEqual(event_shape[0], [])
    self.assertAllEqual(event_shape[1], [])
    self.assertAllEqual(event_shape[2], [20])
    self.assertAllEqual(event_shape[3], [20])

    self.assertEqual(joint.is_scalar_batch(), True)

    batch_shape = joint.batch_shape_tensor()
    self.assertAllEqual(batch_shape, [])

  @parameterized.named_parameters(
      {'testcase_name': 'coroutine',
       'jd_class': tfd.JointDistributionCoroutineAutoBatched},
      {'testcase_name': 'sequential',
       'jd_class': tfd.JointDistributionSequentialAutoBatched},
      {'testcase_name': 'named',
       'jd_class': tfd.JointDistributionNamedAutoBatched})
  def test_model_with_nontrivial_batch_shape(self, jd_class):

    models = {}
    def coroutine_model():
      g = yield Root(tfd.Gamma(2, [2, 3.]))
      df = yield Root(tfd.Exponential([1., 2.]))
      loc = yield tfd.Sample(tfd.Normal(0, g), 20)
      yield tfd.StudentT(tf.expand_dims(df, -1), loc, 1)
    models[tfd.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[tfd.JointDistributionSequentialAutoBatched] = [
        tfd.Gamma(2, [2, 3.]),
        tfd.Exponential([1., 2.]),
        lambda _, g: tfd.Sample(tfd.Normal(0, g), 20),
        lambda loc, df: tfd.StudentT(tf.expand_dims(df, -1), loc, 1)
    ]

    models[tfd.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('g', tfd.Gamma(2, [2, 3.])),
        ('df', tfd.Exponential([1., 2.])),
        ('loc', lambda g: tfd.Sample(tfd.Normal(0, g), 20)),
        ('x', lambda loc, df: tfd.StudentT(tf.expand_dims(df, -1), loc, 1))))

    joint = jd_class(models[jd_class], batch_ndims=1, validate_args=True)

    self.assertAllEqual(joint._model_flatten(joint.event_shape),
                        [[], [], [20], [20]])
    self.assertAllEqual(joint.batch_shape, [2])

    is_scalar = joint._model_flatten(joint.is_scalar_event())
    self.assertAllEqual(is_scalar[0], True)
    self.assertAllEqual(is_scalar[1], True)
    self.assertAllEqual(is_scalar[2], False)
    self.assertAllEqual(is_scalar[3], False)

    self.assertAllEqual(joint.is_scalar_batch(), False)

    batch_shape = self.evaluate(joint.batch_shape_tensor())
    self.assertAllEqual(batch_shape, [2])

    x = joint.sample([5], seed=test_util.test_seed())
    lp = self.evaluate(joint.log_prob(x))
    self.assertAllEqual(lp.shape, [5, 2])

  @parameterized.named_parameters(
      {'testcase_name': 'coroutine',
       'base_jd_class': tfd.JointDistributionCoroutine,
       'jda_class': tfd.JointDistributionCoroutineAutoBatched},
      {'testcase_name': 'sequential',
       'base_jd_class': tfd.JointDistributionSequential,
       'jda_class': tfd.JointDistributionSequentialAutoBatched},
      {'testcase_name': 'named',
       'base_jd_class': tfd.JointDistributionNamed,
       'jda_class': tfd.JointDistributionNamedAutoBatched})
  def test_broadcast_ragged_batch_shape(self, base_jd_class, jda_class):

    base_jd_models = {}

    # Writing a JDC with ragged batch shape will broadcast the first
    # distribution over the second.
    # (though note, this model breaks `log_prob` with nontrivial sample shape).
    def coroutine():
      x = yield Root(tfd.Normal(0., scale=1.))
      yield tfd.Normal(x[..., tf.newaxis], [1., 2., 3., 4., 5.])
    base_jd_models[tfd.JointDistributionCoroutine] = coroutine
    base_jd_models[tfd.JointDistributionSequential] = [
        tfd.Normal(0., scale=1.),
        lambda x: tfd.Normal(x[..., tf.newaxis], [1., 2., 3., 4., 5.])
    ]
    base_jd_models[tfd.JointDistributionNamed] = {
        'x': tfd.Normal(0., scale=1.),
        'y': lambda x: tfd.Normal(x[..., tf.newaxis], [1., 2., 3., 4., 5.])
    }

    # But we can get equivalent behavior in a JDCA by expanding dims so that
    # the batch dimensions line up.
    jd_auto_models = {}
    def coroutine_auto():
      x = yield Root(tfd.Normal(0., scale=[1.]))
      yield tfd.Normal(x, [1., 2., 3., 4., 5.])
    jd_auto_models[tfd.JointDistributionCoroutineAutoBatched] = coroutine_auto
    jd_auto_models[tfd.JointDistributionSequentialAutoBatched] = [
        tfd.Normal(0., scale=[1.]),
        lambda x: tfd.Normal(x, [1., 2., 3., 4., 5.])
    ]
    jd_auto_models[tfd.JointDistributionNamedAutoBatched] = (
        collections.OrderedDict((
            ('x', tfd.Normal(0., scale=[1.])),
            ('y', lambda x: tfd.Normal(x, [1., 2., 3., 4., 5.])))))

    # Writing a JD with ragged batch shape will broadcast the first
    # distribution over the second.
    # (though note, this model breaks `log_prob` with nontrivial sample shape).
    jd_broadcasting = base_jd_class(base_jd_models[base_jd_class])

    # This model's broadcasting behavior is a footgun (it can break inference
    # routines and cause silently incorrect optimization); it should be
    # disallowed by `validate_args`.
    with self.assertRaisesRegexp(Exception,
                                 'Component batch shapes are inconsistent'):
      jda_invalid = jda_class(jd_auto_models[jda_class],
                              batch_ndims=1, validate_args=True)
      _ = self.evaluate(jda_invalid.log_prob(
          jda_invalid.sample(seed=test_util.test_seed())))

    # But, if the user wants to run with no guardrails, one can eke out
    # performance wins when evaluating a shared value over multiple models.
    jda_broadcasting = jda_class(jd_auto_models[jda_class], batch_ndims=1)

    self.assertAllEqual(
        jda_broadcasting._model_flatten(jda_broadcasting.event_shape),
        [[], []])
    self.assertAllEqual(jda_broadcasting.batch_shape, [5])

    joint_sample = jda_broadcasting.sample(seed=test_util.test_seed())
    x_sample, y_sample = self.evaluate(
        list(joint_sample.values()) if hasattr(joint_sample, 'values')
        else joint_sample)
    # The model samples only a single value for x, shared across the batch.
    self.assertAllEqual(x_sample.shape, [1])
    self.assertAllEqual(y_sample.shape, [5])

    lp_jd_broadcast = self.evaluate(jd_broadcasting.log_prob(
        jd_broadcasting._model_unflatten([x_sample[..., 0], y_sample])))
    lp_jda_broadcast = self.evaluate(jda_broadcasting.log_prob(
        jda_broadcasting._model_unflatten([x_sample, y_sample])))
    self.assertAllEqual(lp_jda_broadcast.shape, [5])
    self.assertAllEqual(lp_jd_broadcast, lp_jda_broadcast)

    # Try drawing multiple samples and computing log-prob.
    joint_sample = self.evaluate(jda_broadcasting.sample(
        [2, 3], seed=test_util.test_seed()))
    lp_jda_broadcast = self.evaluate(jda_broadcasting.log_prob(joint_sample))
    self.assertAllEqual(lp_jda_broadcast.shape, [2, 3, 5])

  @parameterized.named_parameters(
      {'testcase_name': 'coroutine',
       'jd_class': tfd.JointDistributionCoroutineAutoBatched},
      {'testcase_name': 'sequential',
       'jd_class': tfd.JointDistributionSequentialAutoBatched},
      {'testcase_name': 'named',
       'jd_class': tfd.JointDistributionNamedAutoBatched})
  def test_log_prob_and_prob_with_plate(self, jd_class):

    models = {}
    def coroutine_model():
      a = yield Root(tfd.Bernoulli(probs=0.5, dtype=tf.float32))
      b = yield tfd.Sample(tfd.Bernoulli(probs=0.25 + 0.5*a,
                                         dtype=tf.float32), 2)
      yield tfd.Normal(loc=a, scale=1. + b)
    models[tfd.JointDistributionCoroutineAutoBatched] = coroutine_model
    models[tfd.JointDistributionSequentialAutoBatched] = [
        tfd.Bernoulli(probs=0.5, dtype=tf.float32),
        lambda a: tfd.Sample(tfd.Bernoulli(  # pylint: disable=g-long-lambda
            probs=0.25 + 0.5*a, dtype=tf.float32), 2),
        lambda b, a: tfd.Normal(loc=a, scale=1. + b)
    ]
    models[tfd.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('a', tfd.Bernoulli(probs=0.5, dtype=tf.float32)),
        ('b', lambda a: tfd.Sample(tfd.Bernoulli(  # pylint: disable=g-long-lambda
            probs=0.25 + 0.5*a, dtype=tf.float32), 2)),
        ('c', lambda b, a: tfd.Normal(loc=a, scale=1. + b))))

    joint = jd_class(models[jd_class], validate_args=True)

    z = self.evaluate(joint.sample(seed=test_util.test_seed()))
    a, b, c = z.values() if hasattr(z, 'values') else z

    log_prob = self.evaluate(joint.log_prob(z))
    prob = self.evaluate(joint.prob(z))

    expected_log_prob = self.evaluate(
        np.log(0.5) +
        tf.reduce_sum(tf.math.log(b * (0.25 + 0.5 * a) +
                                  (1 - b) * (0.75 - 0.5 * a))) +
        tf.reduce_sum(-0.5 * ((c - a) / (1. + b))**2 -
                      0.5 * np.log(2. * np.pi) -
                      tf.math.log((1. + b))))
    self.assertAllClose(log_prob, expected_log_prob)
    self.assertAllClose(prob, np.exp(expected_log_prob))

  @parameterized.named_parameters(
      {'testcase_name': 'coroutine',
       'jd_class': tfd.JointDistributionCoroutineAutoBatched},
      {'testcase_name': 'sequential',
       'jd_class': tfd.JointDistributionSequentialAutoBatched},
      {'testcase_name': 'named',
       'jd_class': tfd.JointDistributionNamedAutoBatched})
  def test_log_prob_multiple_samples(self, jd_class):

    models = {}
    def coroutine_model():
      a = yield Root(tfd.Bernoulli(probs=0.5, dtype=tf.float32))
      b = yield tfd.Bernoulli(probs=0.25 + 0.5*a,
                              dtype=tf.float32)
      yield tfd.Normal(loc=a, scale=1. + b)
    models[tfd.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[tfd.JointDistributionSequentialAutoBatched] = [
        tfd.Bernoulli(probs=0.5, dtype=tf.float32),
        lambda a: tfd.Bernoulli(probs=0.25 + 0.5*a, dtype=tf.float32),
        lambda b, a: tfd.Normal(loc=a, scale=1. + b)
    ]

    models[tfd.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('a', tfd.Bernoulli(probs=0.5, dtype=tf.float32)),
        ('b', lambda a: tfd.Bernoulli(probs=0.25 + 0.5*a, dtype=tf.float32)),
        ('c', lambda b, a: tfd.Normal(loc=a, scale=1. + b))))

    joint = jd_class(models[jd_class], validate_args=True)

    z = joint.sample(4, seed=test_util.test_seed())

    log_prob = joint.log_prob(z)

    a, b, c = z.values() if hasattr(z, 'values') else z  # pylint: disable=unbalanced-tuple-unpacking

    expected_log_prob = (
        np.log(0.5) +
        tf.math.log(b * (0.25 + 0.5 * a) +
                    (1 - b) * (0.75 -0.5 * a)) +
        -0.5 * ((c - a) / (1. + b)) ** 2 -
        0.5 * np.log(2. * np.pi) -
        tf.math.log((1. + b)))

    self.assertAllClose(*self.evaluate([log_prob, expected_log_prob]))

  def test_sample_dtype_structures_output(self):

    num_features = 4

    def dist():
      scale_variance = yield Root(
          tfd.InverseGamma(0.5, 0.5))
      scale_noncentered = yield Root(
          tfd.Sample(tfd.HalfNormal(1.), num_features))
      scale = scale_noncentered * scale_variance[..., None]**0.5
      weights_noncentered = yield Root(
          tfd.Sample(tfd.Normal(0., 1.), num_features))
      yield tfd.Deterministic(weights_noncentered * scale)

    # Currently sample_dtype is only used for `tf.nest.pack_structure_as`. In
    # the future we may use it for error checking and/or casting.
    sample_dtype = collections.namedtuple('Model', [
        'scale_variance',
        'scale_noncentered',
        'weights_noncentered',
        'weights',
    ])(*([None]*4))
    joint = tfd.JointDistributionCoroutineAutoBatched(
        dist, sample_dtype=sample_dtype, validate_args=True)
    self.assertAllEqual(sorted(sample_dtype._fields),
                        sorted(joint.sample(
                            seed=test_util.test_seed())._fields))
    ds, xs = joint.sample_distributions(seed=test_util.test_seed())
    tf.nest.assert_same_structure(sample_dtype, ds)
    tf.nest.assert_same_structure(sample_dtype, xs)
    self.assertEqual([3, 4], joint.log_prob(joint.sample(
        [3, 4], seed=test_util.test_seed())).shape)

  def test_repr_with_custom_sample_dtype(self):
    sd = collections.namedtuple('Model', ['s', 'w'])(None, None)

    def dist():
      s = yield Root(tfd.Sample(tfd.InverseGamma(2, 2), 100))
      yield tfd.Normal(0, s)

    m = tfd.JointDistributionCoroutineAutoBatched(dist, sample_dtype=sd)
    self.assertEqual(
        ('<tfp.distributions.JointDistributionCoroutineAutoBatched'
         ' \'JointDistributionCoroutineAutoBatched\''
         ' batch_shape=[]'
         ' event_shape=Model(s=[100], w=[100])'
         ' dtype=Model(s=float32, w=float32)>'),
        repr(m))

  @parameterized.named_parameters(
      {'testcase_name': 'coroutine',
       'jd_class': tfd.JointDistributionCoroutineAutoBatched},
      {'testcase_name': 'sequential',
       'jd_class': tfd.JointDistributionSequentialAutoBatched},
      {'testcase_name': 'named',
       'jd_class': tfd.JointDistributionNamedAutoBatched})
  @test_util.jax_disable_variable_test
  def test_latent_dirichlet_allocation(self, jd_class):  # pylint: disable=g-doc-args
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
    alpha = tfp.util.TransformedVariable(
        u.sample([num_topics], seed=test_util.test_seed()),
        tfb.Softplus(), name='alpha')
    beta = tf.Variable(u.sample([num_topics, num_words]), name='beta')

    # Note near 1:1 with mathematical specification. The main distinction is the
    # use of Independent--this lets us easily aggregate multinomials across
    # topics (and in any "shape" of documents).
    def lda_coroutine_model():
      n = yield Root(tfd.Poisson(rate=avg_doc_length))
      theta = yield Root(tfd.Dirichlet(concentration=alpha))
      z = yield tfd.Multinomial(total_count=n, probs=theta)
      yield tfd.Multinomial(total_count=z, logits=beta)
    if jd_class is tfd.JointDistributionCoroutineAutoBatched:
      model = lda_coroutine_model
    elif jd_class is tfd.JointDistributionSequentialAutoBatched:
      model = [
          tfd.Poisson(rate=avg_doc_length),  # n
          tfd.Dirichlet(concentration=alpha),  # theta
          lambda theta, n: tfd.Multinomial(total_count=n, probs=theta),  # z
          lambda z: tfd.Multinomial(total_count=z, logits=beta)
      ]
    elif jd_class is tfd.JointDistributionNamedAutoBatched:
      model = collections.OrderedDict((
          ('n', tfd.Poisson(rate=avg_doc_length)),
          ('theta', tfd.Dirichlet(concentration=alpha)),
          ('z', lambda theta, n: tfd.Multinomial(total_count=n, probs=theta)),
          ('X', lambda z: tfd.Multinomial(total_count=z, logits=beta))))

    lda = jd_class(model, validate_args=True)

    # Now, let's sample some "documents" and compute the log-prob of each.
    docs_shape = [2, 4]  # That is, 8 docs in the shape of [2, 4].
    sample = lda.sample(docs_shape, seed=test_util.test_seed())
    log_probs = lda.log_prob(sample)
    self.assertEqual(docs_shape, log_probs.shape)

    # Verify we correctly track trainable variables.
    self.assertLen(lda.trainable_variables, 2)
    self.assertIs(alpha.pretransformed_input, lda.trainable_variables[0])
    self.assertIs(beta, lda.trainable_variables[1])

    # Ensure we can compute gradients.
    with tf.GradientTape() as tape:
      # Note: The samples are not taped, hence implicitly "stop_gradient."
      negloglik = -lda.log_prob(sample)
    grads = tape.gradient(negloglik, lda.trainable_variables)

    self.assertLen(grads, 2)
    self.assertAllEqual((alpha.pretransformed_input.shape, beta.shape),
                        (grads[0].shape, grads[1].shape))
    self.assertAllNotNone(grads)

  @parameterized.named_parameters(
      {'testcase_name': 'coroutine',
       'jd_class': tfd.JointDistributionCoroutineAutoBatched},
      {'testcase_name': 'sequential',
       'jd_class': tfd.JointDistributionSequentialAutoBatched},
      {'testcase_name': 'named',
       'jd_class': tfd.JointDistributionNamedAutoBatched})
  def test_default_event_space_bijector(self, jd_class):

    models = {}
    def coroutine_model():
      g = yield Root(tfd.Gamma(2, [2, 3.]))
      df = yield Root(tfd.Exponential([1., 2.]))
      loc = yield tfd.Sample(tfd.Normal(0, g), 20)
      yield tfd.StudentT(tf.expand_dims(df, -1), loc, 1)
    models[tfd.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[tfd.JointDistributionSequentialAutoBatched] = [
        tfd.Gamma(2, [2, 3.]),
        tfd.Exponential([1., 2.]),
        lambda _, g: tfd.Sample(tfd.Normal(0, g), 20),
        lambda loc, df: tfd.StudentT(tf.expand_dims(df, -1), loc, 1)
    ]

    models[tfd.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('g', tfd.Gamma(2, [2, 3.])),
        ('df', tfd.Exponential([1., 2.])),
        ('loc', lambda g: tfd.Sample(tfd.Normal(0, g), 20)),
        ('x', lambda loc, df: tfd.StudentT(tf.expand_dims(df, -1), loc, 1))))

    joint = jd_class(models[jd_class], batch_ndims=1, validate_args=True)
    joint_bijector = joint._experimental_default_event_space_bijector()
    x = self.evaluate(joint.sample(seed=test_util.test_seed()))
    self.assertAllClose(
        x,
        self.evaluate(joint_bijector.forward(joint_bijector.inverse(x))))


if __name__ == '__main__':
  tf.test.main()
