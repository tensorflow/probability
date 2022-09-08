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

import collections
import os

# Dependency imports
from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import dirichlet
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import inverse_gamma
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import multinomial
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.distributions import sample as sample_lib
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.util import deferred_tensor


JAX_MODE = False
Root = jdab.JointDistributionCoroutineAutoBatched.Root


@test_util.test_all_tf_execution_regimes
class JointDistributionAutoBatchedTest(test_util.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'coroutine',
          'jd_class': jdab.JointDistributionCoroutineAutoBatched
      }, {
          'testcase_name': 'sequential',
          'jd_class': jdab.JointDistributionSequentialAutoBatched
      }, {
          'testcase_name': 'named',
          'jd_class': jdab.JointDistributionNamedAutoBatched
      })
  def test_batch_and_event_shape_with_plate(self, jd_class):

    models = {}

    def coroutine_model():
      g = yield lognormal.LogNormal(0., 1.)
      df = yield exponential.Exponential(1.)
      loc = yield sample_lib.Sample(normal.Normal(0, g), 20)
      yield student_t.StudentT(tf.expand_dims(df, -1), loc, 1)

    models[jdab.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[jdab.JointDistributionSequentialAutoBatched] = [
        lognormal.LogNormal(0., 1.),
        exponential.Exponential(1.),
        lambda _, g: sample_lib.Sample(normal.Normal(0, g), 20),
        lambda loc, df: student_t.StudentT(tf.expand_dims(df, -1), loc, 1)
    ]

    models[jdab.JointDistributionNamedAutoBatched] = collections.OrderedDict(
        (('g', lognormal.LogNormal(0.,
                                   1.)), ('df', exponential.Exponential(1.)),
         ('loc', lambda g: sample_lib.Sample(normal.Normal(0, g), 20)),
         ('x',
          lambda loc, df: student_t.StudentT(tf.expand_dims(df, -1), loc, 1))))

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

    self.assertIsInstance(joint, tf.__internal__.CompositeTensor)
    flat = tf.nest.flatten(joint, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(
        joint, flat, expand_composites=True)
    self.assertIsInstance(unflat, jd_class)
    self.assertIs(type(joint.model), type(unflat.model))

  @parameterized.named_parameters(
      {
          'testcase_name': 'CoroutineStateless',
          'sampler_type': 'stateless',
          'jd_class': jdab.JointDistributionCoroutineAutoBatched
      }, {
          'testcase_name': 'CoroutineStateful',
          'sampler_type': 'stateful',
          'jd_class': jdab.JointDistributionCoroutineAutoBatched
      }, {
          'testcase_name': 'SequentialStateless',
          'sampler_type': 'stateless',
          'jd_class': jdab.JointDistributionSequentialAutoBatched
      }, {
          'testcase_name': 'SequentialStateful',
          'sampler_type': 'stateful',
          'jd_class': jdab.JointDistributionSequentialAutoBatched
      }, {
          'testcase_name': 'NamedStateful',
          'sampler_type': 'stateful',
          'jd_class': jdab.JointDistributionNamedAutoBatched
      }, {
          'testcase_name': 'NamedStateless',
          'sampler_type': 'stateless',
          'jd_class': jdab.JointDistributionNamedAutoBatched
      })
  def test_model_with_nontrivial_batch_shape(self, jd_class, sampler_type):
    models = {}
    def coroutine_model():
      g = yield lognormal.LogNormal(0., [1., 2.])
      df = yield exponential.Exponential([1., 2.])
      loc = yield sample_lib.Sample(normal.Normal(0, g), 20)
      yield student_t.StudentT(tf.expand_dims(df, -1), loc, 1)

    models[jdab.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[jdab.JointDistributionSequentialAutoBatched] = [
        lognormal.LogNormal(0., [1., 2.]),
        exponential.Exponential([1., 2.]),
        lambda _, g: sample_lib.Sample(normal.Normal(0, g), 20),
        lambda loc, df: student_t.StudentT(tf.expand_dims(df, -1), loc, 1)
    ]

    models[jdab.JointDistributionNamedAutoBatched] = collections.OrderedDict(
        (('g', lognormal.LogNormal(0., [1., 2.])),
         ('df', exponential.Exponential([1., 2.])),
         ('loc', lambda g: sample_lib.Sample(normal.Normal(0, g), 20)),
         ('x',
          lambda loc, df: student_t.StudentT(tf.expand_dims(df, -1), loc, 1))))

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

    x = joint.sample([5], seed=test_util.test_seed(sampler_type=sampler_type))
    lp = self.evaluate(joint.log_prob(x))
    self.assertAllEqual(lp.shape, [5, 2])

  def test_model_with_dynamic_batch_ndims(self):
    if tf.executing_eagerly():
      self.skipTest('Dynamic shape.')

    def coroutine_model():
      g = yield lognormal.LogNormal(0., [1., 2.])
      df = yield exponential.Exponential([1., 2.])
      loc = yield sample_lib.Sample(normal.Normal(0, g), 20)
      yield student_t.StudentT(tf.expand_dims(df, -1), loc, 1)

    joint = jdab.JointDistributionCoroutineAutoBatched(
        coroutine_model,
        batch_ndims=tf1.placeholder_with_default(1, shape=[]),
        validate_args=True)

    batch_shape_tensor = self.evaluate(joint.batch_shape_tensor())
    self.assertAllEqual(batch_shape_tensor, [2])
    event_shape_tensor = self.evaluate(joint.event_shape_tensor())
    self.assertAllEqual(event_shape_tensor[0], [])
    self.assertAllEqual(event_shape_tensor[1], [])
    self.assertAllEqual(event_shape_tensor[2], [20])
    self.assertAllEqual(event_shape_tensor[3], [20])

    self.assertAllEqual(joint.batch_shape, tf.TensorShape(None))
    self.assertAllEqual(joint._model_flatten(joint.event_shape),
                        [tf.TensorShape(None)] * 4)

    x = joint.sample([5], seed=test_util.test_seed(sampler_type='stateless'))
    lp = self.evaluate(joint.log_prob(x))
    self.assertAllEqual(lp.shape, [5, 2])

  @parameterized.named_parameters(
      {
          'testcase_name': 'coroutine',
          'base_jd_class': jdc.JointDistributionCoroutine,
          'jda_class': jdab.JointDistributionCoroutineAutoBatched
      }, {
          'testcase_name': 'sequential',
          'base_jd_class': jds.JointDistributionSequential,
          'jda_class': jdab.JointDistributionSequentialAutoBatched
      }, {
          'testcase_name': 'named',
          'base_jd_class': jdn.JointDistributionNamed,
          'jda_class': jdab.JointDistributionNamedAutoBatched
      })
  def test_broadcast_ragged_batch_shape(self, base_jd_class, jda_class):

    base_jd_models = {}

    # Writing a JDC with ragged batch shape will broadcast the first
    # distribution over the second.
    # (though note, this model breaks `log_prob` with nontrivial sample shape).
    def coroutine():
      x = yield Root(normal.Normal(0., scale=1.))
      yield normal.Normal(x[..., tf.newaxis], [1., 2., 3., 4., 5.])

    base_jd_models[jdc.JointDistributionCoroutine] = coroutine
    base_jd_models[jds.JointDistributionSequential] = [
        normal.Normal(0., scale=1.),
        lambda x: normal.Normal(x[..., tf.newaxis], [1., 2., 3., 4., 5.])
    ]
    base_jd_models[jdn.JointDistributionNamed] = {
        'x': normal.Normal(0., scale=1.),
        'y': lambda x: normal.Normal(x[..., tf.newaxis], [1., 2., 3., 4., 5.])
    }

    # But we can get equivalent behavior in a JDCA by expanding dims so that
    # the batch dimensions line up.
    jd_auto_models = {}
    def coroutine_auto():
      x = yield normal.Normal(0., scale=[1.])
      yield normal.Normal(x, [1., 2., 3., 4., 5.])

    jd_auto_models[jdab.JointDistributionCoroutineAutoBatched] = coroutine_auto
    jd_auto_models[jdab.JointDistributionSequentialAutoBatched] = [
        normal.Normal(0., scale=[1.]),
        lambda x: normal.Normal(x, [1., 2., 3., 4., 5.])
    ]
    jd_auto_models[jdab.JointDistributionNamedAutoBatched] = (
        collections.OrderedDict(
            (('x', normal.Normal(0., scale=[1.])),
             ('y', lambda x: normal.Normal(x, [1., 2., 3., 4., 5.])))))

    # Writing a JD with ragged batch shape will broadcast the first
    # distribution over the second.
    # (though note, this model breaks `log_prob` with nontrivial sample shape).
    jd_broadcasting = base_jd_class(base_jd_models[base_jd_class])

    # This model's broadcasting behavior is a footgun (it can break inference
    # routines and cause silently incorrect optimization); it should be
    # disallowed by `validate_args`.
    with self.assertRaisesRegexp(
        Exception,
        ('Component batch shapes are inconsistent|'
         'Broadcasting probably indicates an error in model specification')):
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
      {
          'testcase_name': 'coroutine',
          'jd_class': jdab.JointDistributionCoroutineAutoBatched
      }, {
          'testcase_name': 'sequential',
          'jd_class': jdab.JointDistributionSequentialAutoBatched
      }, {
          'testcase_name': 'named',
          'jd_class': jdab.JointDistributionNamedAutoBatched
      })
  def test_log_prob_and_prob_with_plate(self, jd_class):

    models = {}
    def coroutine_model():
      a = yield bernoulli.Bernoulli(probs=0.5, dtype=tf.float32)
      b = yield sample_lib.Sample(  # pylint: disable=g-long-lambda
          bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32), 2)
      yield normal.Normal(loc=a, scale=1. + b)

    models[jdab.JointDistributionCoroutineAutoBatched] = coroutine_model
    models[jdab.JointDistributionSequentialAutoBatched] = [
        bernoulli.Bernoulli(probs=0.5, dtype=tf.float32),
        lambda a: sample_lib.Sample(  # pylint: disable=g-long-lambda
            bernoulli.Bernoulli(
                probs=0.25 + 0.5 * a,
                dtype=tf.float32),
            2),
        lambda b, a: normal.Normal(loc=a, scale=1. + b)
    ]
    models[jdab.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('a', bernoulli.Bernoulli(probs=0.5, dtype=tf.float32)),
        (
            'b',
            lambda a: sample_lib.Sample(  # pylint: disable=g-long-lambda
                bernoulli.Bernoulli(
                    probs=0.25 + 0.5 * a,
                    dtype=tf.float32),
                2)),
        ('c', lambda b, a: normal.Normal(loc=a, scale=1. + b))))

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
      {
          'testcase_name': 'coroutine',
          'jd_class': jdab.JointDistributionCoroutineAutoBatched
      }, {
          'testcase_name': 'sequential',
          'jd_class': jdab.JointDistributionSequentialAutoBatched
      }, {
          'testcase_name': 'named',
          'jd_class': jdab.JointDistributionNamedAutoBatched
      })
  def test_log_prob_multiple_samples(self, jd_class):

    models = {}
    def coroutine_model():
      a = yield bernoulli.Bernoulli(probs=0.5, dtype=tf.float32)
      b = yield bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32)
      yield normal.Normal(loc=a, scale=1. + b)

    models[jdab.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[jdab.JointDistributionSequentialAutoBatched] = [
        bernoulli.Bernoulli(probs=0.5, dtype=tf.float32),
        lambda a: bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32),
        lambda b, a: normal.Normal(loc=a, scale=1. + b)
    ]

    models[jdab.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('a', bernoulli.Bernoulli(probs=0.5, dtype=tf.float32)),
        ('b',
         lambda a: bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32)),
        ('c', lambda b, a: normal.Normal(loc=a, scale=1. + b))))

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

  @parameterized.named_parameters(
      {
          'testcase_name': 'coroutine',
          'jd_class': jdab.JointDistributionCoroutineAutoBatched
      }, {
          'testcase_name': 'sequential',
          'jd_class': jdab.JointDistributionSequentialAutoBatched
      }, {
          'testcase_name': 'named',
          'jd_class': jdab.JointDistributionNamedAutoBatched
      })
  def test_sample_and_log_prob(self, jd_class):

    # Define a bijector to detect if/when `inverse` is called.
    inverted_values = []

    class InverseTracingExp(exp.Exp):

      def _inverse(self, y):
        inverted_values.append(y)
        return tf.math.log(y)

    models = {}

    def coroutine_model():
      g = yield InverseTracingExp()(normal.Normal(0., 1.), name='g')
      df = yield exponential.Exponential(1., name='df')
      loc = yield sample_lib.Sample(normal.Normal(0, g), 20, name='loc')
      yield student_t.StudentT(df, loc, 1, name='x')

    models[jdab.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[jdab.JointDistributionSequentialAutoBatched] = [
        InverseTracingExp()(normal.Normal(0., 1.), name='g'),
        exponential.Exponential(1., name='df'),
        lambda _, g: sample_lib.Sample(normal.Normal(0, g), 20, name='loc'),
        lambda loc, df: student_t.StudentT(df, loc, 1, name='x')
    ]

    models[jdab.JointDistributionNamedAutoBatched] = collections.OrderedDict(
        (('g', InverseTracingExp()(normal.Normal(0., 1.))),
         ('df', exponential.Exponential(1.)),
         ('loc', lambda g: sample_lib.Sample(normal.Normal(0, g), 20)),
         ('x', lambda loc, df: student_t.StudentT(df, loc, 1))))

    joint = jd_class(models[jd_class], validate_args=True)

    seed = test_util.test_seed(sampler_type='stateless')
    for sample_shape in ([], [5]):
      inverted_values.clear()
      x1, lp1 = self.evaluate(
          joint.experimental_sample_and_log_prob(
              sample_shape,
              seed=seed,
              df=2.7))  # Check that kwargs are supported.
      x2 = self.evaluate(
          joint.sample(sample_shape, seed=seed, df=2.7))
      self.assertAllCloseNested(x1, x2)

      self.assertLen(inverted_values, 0)
      lp2 = joint.log_prob(x1)
      self.assertLen(inverted_values, 1)
      self.assertAllClose(lp1, lp2)

  @test_util.jax_disable_test_missing_functionality('b/157594634')
  def test_sample_distributions(self):
    def coroutine_model():
      g = yield normal.Normal(0., 1., name='g')
      df = yield exponential.Exponential(1., name='df')
      loc = yield normal.Normal(tf.zeros([20]), g, name='loc')
      yield student_t.StudentT(df, loc, 1, name='x')

    joint = jdab.JointDistributionCoroutineAutoBatched(coroutine_model)

    ds, xs = joint.sample_distributions([4, 3], seed=test_util.test_seed())
    for d, x in zip(ds, xs):
      self.assertGreaterEqual(len(d.batch_shape), 2)
      lp = d.log_prob(x)
      self.assertAllEqual(lp.shape[:2], [4, 3])

  @test_util.jax_disable_test_missing_functionality('b/201586404')
  def test_sample_distributions_not_composite_tensor_raises_error(self):
    def coroutine_model():
      yield transformed_distribution.TransformedDistribution(
          normal.Normal(0., 1.), test_util.NonCompositeTensorExp(), name='td')

    joint = jdab.JointDistributionCoroutineAutoBatched(coroutine_model)

    # Sampling with trivial sample shape avoids the vmap codepath.
    ds, _ = joint.sample_distributions([], seed=test_util.test_seed())
    self.assertIsInstance(
        ds[0],
        # Non-CompositeTensor version of TransformedDistribution.
        transformed_distribution._TransformedDistribution)

    with self.assertRaisesRegex(
        TypeError, r'Some component distribution\(s\) cannot be returned'):
      joint.sample_distributions([4, 3], seed=test_util.test_seed())

  def test_sample_with_batch_value(self):

    @jdab.JointDistributionCoroutineAutoBatched
    def dist():
      a = yield sample_lib.Sample(normal.Normal(0, 1.), 2)
      b = yield sample_lib.Sample(normal.Normal(0, 1.), 3)
      # The following line fails if not autovectorized.
      yield normal.Normal(a[tf.newaxis, ...] * b[..., tf.newaxis], 1.)
    x = self.evaluate(dist.sample(123, seed=test_util.test_seed()))
    x2 = self.evaluate(dist.sample(value=x, seed=test_util.test_seed()))
    self.assertAllCloseNested(x, x2)

    # Also test a dict-type value (JDNamed).
    dist = jdab.JointDistributionNamedAutoBatched({
        'a':
            sample_lib.Sample(normal.Normal(0, 1.), 2),
        'b':
            sample_lib.Sample(normal.Normal(0, 1.), 3),
        'c':
            lambda a, b: normal.Normal(  # pylint: disable=g-long-lambda
                a[tf.newaxis, ...] * b[..., tf.newaxis], 1.)
    })
    x = self.evaluate(dist.sample(123, seed=test_util.test_seed()))
    x2 = self.evaluate(dist.sample(value=x, seed=test_util.test_seed()))
    self.assertAllCloseNested(x, x2)

  def test_sample_with_value_as_kwarg(self):

    @jdab.JointDistributionCoroutineAutoBatched
    def dist():
      a = yield sample_lib.Sample(normal.Normal(0, 1.), 2, name='a')
      b = yield sample_lib.Sample(normal.Normal(0, 1.), 3, name='b')
      # The following line fails if not autovectorized.
      yield normal.Normal(a[tf.newaxis, ...] * b[..., tf.newaxis], 1., name='c')

    x = self.evaluate(dist.sample(4, seed=test_util.test_seed()))
    x2 = self.evaluate(dist.sample(seed=test_util.test_seed(), a=x.a))
    self.assertAllClose(x.a, x2.a)
    self.assertAllEqual(x2.b.shape, [4, 3])
    self.assertAllEqual(x2.c.shape, [4, 3, 2])

  @parameterized.named_parameters(
      dict(testcase_name='stateful', sampler_type='stateful'),
      dict(testcase_name='stateless', sampler_type='stateless'))
  def test_sample_with_partially_specified_value(self, sampler_type):

    num_features = 5

    def dist():
      scale_variance = yield inverse_gamma.InverseGamma(0.5, 0.5)
      scale_noncentered = yield sample_lib.Sample(
          half_normal.HalfNormal(1.), num_features)
      scale = scale_noncentered * scale_variance[..., None]**0.5
      weights_noncentered = yield sample_lib.Sample(
          normal.Normal(0., 1.), num_features)
      yield deterministic.Deterministic(weights_noncentered * scale)

    joint = jdab.JointDistributionCoroutineAutoBatched(dist, validate_args=True)

    value_partial_batch_dim = 4
    value_ = (3.,
              None,
              None,
              np.ones([value_partial_batch_dim, num_features]))
    value = [None if v is None else tf.cast(v, tf.float32) for v in value_]

    # The sample should keep the specified values.
    xs = self.evaluate(
        joint.sample(
            value=value, seed=test_util.test_seed(sampler_type=sampler_type)))
    self.assertAllEqual(xs[0], tf.fill([value_partial_batch_dim], value[0]))
    self.assertAllEqual(xs[1].shape, [value_partial_batch_dim, num_features])
    self.assertAllEqual(xs[2].shape, [value_partial_batch_dim, num_features])
    self.assertAllEqual(xs[3], value[3])

    # With sample shape.
    sample_shape = [6, 2]
    samples = joint.sample(sample_shape, value=value,
                           seed=test_util.test_seed(sampler_type=sampler_type))
    xs = self.evaluate(samples)
    expect_shp = sample_shape + [value_partial_batch_dim, num_features]
    self.assertAllEqual(
        xs[0], tf.fill(sample_shape + [value_partial_batch_dim], value[0]))
    self.assertAllEqual(xs[1].shape, expect_shp)
    self.assertAllEqual(xs[2].shape, expect_shp)
    self.assertAllEqual(xs[3], value[3] * tf.ones(expect_shp))

    sample_shape_dynamic = tf1.placeholder_with_default(
        sample_shape, shape=None)
    samples = joint.sample(sample_shape_dynamic, value=value,
                           seed=test_util.test_seed(sampler_type=sampler_type))
    xs = self.evaluate(samples)
    self.assertAllEqual(
        xs[0], tf.fill(sample_shape + [value_partial_batch_dim], value[0]))
    self.assertAllEqual(xs[1].shape, expect_shp)
    self.assertAllEqual(xs[2].shape, expect_shp)
    self.assertAllEqual(xs[3], value[3] * tf.ones(expect_shp))

  @parameterized.named_parameters(
      dict(testcase_name='stateful', sampler_type='stateful'),
      dict(testcase_name='stateless', sampler_type='stateless'))
  def test_sample_with_prefix_of_values(self, sampler_type):
    num_rows = 4
    num_columns = 5
    def dist():
      a = yield sample_lib.Sample(normal.Normal(0., 1.), num_rows, name='a')
      b = yield sample_lib.Sample(normal.Normal(0., 1.), num_columns, name='b')
      yield normal.Normal(a[..., None] * b[None, ...], 1., name='c')

    tuple_joint = jdab.JointDistributionCoroutineAutoBatched(
        dist, validate_args=True)
    namedtuple_joint = jdab.JointDistributionCoroutineAutoBatched(
        dist,
        sample_dtype=collections.namedtuple('ModelSpec', ['a', 'b', 'c'])(
            a=tf.float32, b=tf.float32, c=tf.float32),
        validate_args=True)

    value_partial_batch_dim = 3
    v0 = 3. * np.ones([value_partial_batch_dim, num_rows]).astype(np.float32)

    # Tuple (or namedtuple) value contains only the first variable.
    tuple_value = (v0,)
    namedtuple_value = collections.namedtuple('ValueSpec', ['a'])(a=v0)

    for joint in (tuple_joint, namedtuple_joint):
      for value in (tuple_value, namedtuple_value):
        xs = self.evaluate(
            joint.sample(value=value,
                         seed=test_util.test_seed(sampler_type=sampler_type)))
        self.assertAllEqual(xs[0], v0)
        self.assertAllEqual(xs[1].shape,
                            [value_partial_batch_dim, num_columns])
        self.assertAllEqual(xs[2].shape,
                            [value_partial_batch_dim, num_rows, num_columns])

  def test_unit_sample_shape_avoids_vectorization(self):
    xs = []  # Collect (possibly symbolic) Tensors sampled inside the model.

    @jdab.JointDistributionCoroutineAutoBatched
    def dist():
      x = yield normal.Normal(0., 1., name='x')
      xs.append(x)

    # Try sampling with a variety of unit sample shapes.
    self.assertEqual(
        [1],
        dist.sample(
            1, seed=test_util.test_seed(sampler_type='seedless')).x.shape)
    self.assertEqual(
        [1],
        dist.sample([1],
                    seed=test_util.test_seed(sampler_type='seedless')).x.shape)
    self.assertEqual(
        [1, 1],
        dist.sample([1, 1],
                    seed=test_util.test_seed(sampler_type='seedless')).x.shape)

    # Check that the model only ever saw the trivial sample shape.
    for x in xs:
      self.assertEqual(x.shape, [])

  def test_unit_sample_shape(self):

    @jdab.JointDistributionCoroutineAutoBatched
    def dist():
      x = yield normal.Normal(loc=tf.zeros([3]), scale=1., name='x')
      yield bernoulli.Bernoulli(logits=tf.einsum('n->', x), name='y')

    for sample_shape in [(), 1, [1], [1, 1], [2]]:
      self.assertAllEqual(
          dist.log_prob(
              dist.sample(sample_shape,
                          seed=test_util.test_seed())).shape,
          np.reshape(sample_shape, [-1]))

  def test_sample_dtype_structures_output(self):

    num_features = 4

    def dist():
      scale_variance = yield Root(inverse_gamma.InverseGamma(0.5, 0.5))
      scale_noncentered = yield Root(
          sample_lib.Sample(half_normal.HalfNormal(1.), num_features))
      scale = scale_noncentered * scale_variance[..., None]**0.5
      weights_noncentered = yield Root(
          sample_lib.Sample(normal.Normal(0., 1.), num_features))
      yield deterministic.Deterministic(weights_noncentered * scale)

    # Currently sample_dtype is only used for `tf.nest.pack_structure_as`. In
    # the future we may use it for error checking and/or casting.
    sample_dtype = collections.namedtuple('Model', [
        'scale_variance',
        'scale_noncentered',
        'weights_noncentered',
        'weights',
    ])(*([None]*4))
    joint = jdab.JointDistributionCoroutineAutoBatched(
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
      s = yield sample_lib.Sample(inverse_gamma.InverseGamma(2, 2), 100)
      yield normal.Normal(0, s)

    m = jdab.JointDistributionCoroutineAutoBatched(dist, sample_dtype=sd)
    self.assertEqual(
        ('<tfp.distributions.JointDistributionCoroutineAutoBatched'
         ' \'JointDistributionCoroutineAutoBatched\''
         ' batch_shape=[]'
         ' event_shape=Model(s=[100], w=[100])'
         ' dtype=Model(s=float32, w=float32)>'),
        repr(m))

  @parameterized.named_parameters(
      {
          'testcase_name': 'coroutine',
          'jd_class': jdab.JointDistributionCoroutineAutoBatched
      }, {
          'testcase_name': 'sequential',
          'jd_class': jdab.JointDistributionSequentialAutoBatched
      }, {
          'testcase_name': 'named',
          'jd_class': jdab.JointDistributionNamedAutoBatched
      })
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
    seed = test_util.test_seed_stream()
    # Hyperparameters.
    num_topics = 3
    num_words = 10
    avg_doc_length = 5
    u = uniform.Uniform(low=-1., high=1.)
    alpha = deferred_tensor.TransformedVariable(
        u.sample([num_topics], seed=seed()), softplus.Softplus(), name='alpha')
    beta = tf.Variable(u.sample([num_topics, num_words],
                                seed=seed()), name='beta')

    # Note near 1:1 with mathematical specification. The main distinction is the
    # use of Independent--this lets us easily aggregate multinomials across
    # topics (and in any "shape" of documents).
    def lda_coroutine_model():
      n = yield Root(poisson.Poisson(rate=avg_doc_length))
      theta = yield Root(dirichlet.Dirichlet(concentration=alpha))
      z = yield multinomial.Multinomial(total_count=n, probs=theta)
      yield multinomial.Multinomial(total_count=z, logits=beta)

    if jd_class is jdab.JointDistributionCoroutineAutoBatched:
      model = lda_coroutine_model
    elif jd_class is jdab.JointDistributionSequentialAutoBatched:
      model = [
          poisson.Poisson(rate=avg_doc_length),  # n
          dirichlet.Dirichlet(concentration=alpha),  # theta
          # z
          lambda theta, n: multinomial.Multinomial(total_count=n, probs=theta),
          lambda z: multinomial.Multinomial(total_count=z, logits=beta)
      ]
    elif jd_class is jdab.JointDistributionNamedAutoBatched:
      model = collections.OrderedDict(
          (('n', poisson.Poisson(rate=avg_doc_length)),
           ('theta', dirichlet.Dirichlet(concentration=alpha)),
           ('z', lambda theta, n: multinomial.Multinomial(  # pylint: disable=g-long-lambda
               total_count=n, probs=theta)),
           ('X',
            lambda z: multinomial.Multinomial(total_count=z, logits=beta))))

    # TODO(b/159842104): Enable autovectorization for Multinomial sampling.
    lda = jd_class(model, validate_args=True, use_vectorized_map=False)

    # Now, let's sample some "documents" and compute the log-prob of each.
    docs_shape = [2, 4]  # That is, 8 docs in the shape of [2, 4].
    sample = lda.sample(docs_shape, seed=seed())
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
      {
          'testcase_name': 'coroutine',
          'jd_class': jdab.JointDistributionCoroutineAutoBatched
      }, {
          'testcase_name': 'sequential',
          'jd_class': jdab.JointDistributionSequentialAutoBatched
      }, {
          'testcase_name': 'named',
          'jd_class': jdab.JointDistributionNamedAutoBatched
      })
  def test_default_event_space_bijector(self, jd_class):

    models = {}
    def coroutine_model():
      high = yield lognormal.LogNormal(0., [1.])
      yield uniform.Uniform(low=[[-1., -2.]], high=high[..., tf.newaxis])
      yield deterministic.Deterministic([[0., 1., 2.]])

    models[jdab.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[jdab.JointDistributionSequentialAutoBatched] = [
        lognormal.LogNormal(0., [1.]), lambda high: uniform.Uniform(  # pylint: disable=g-long-lambda
            low=[[-1., -2.]], high=high[..., tf.newaxis]),
        deterministic.Deterministic([[0., 1., 2.]])
    ]

    models[jdab.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('high', lognormal.LogNormal(0., [1.])),
        (
            'x',
            lambda high: uniform.Uniform(  # pylint: disable=g-long-lambda
                low=[[-1., -2.]],
                high=high[..., tf.newaxis])),
        ('y', deterministic.Deterministic([[0., 1., 2.]]))))

    joint = jd_class(models[jd_class], batch_ndims=1, validate_args=True)
    self.assertAllEqual(joint.batch_shape, [1])
    self.assertAllEqualNested(tf.nest.flatten(joint.event_shape),
                              [[], [2], [3]])
    joint_bijector = joint.experimental_default_event_space_bijector()

    y = self.evaluate(joint.sample([2, 3], seed=test_util.test_seed()))
    x = joint_bijector.inverse(y)
    self.assertAllCloseNested(y, joint_bijector.forward(x))

    fldj = joint_bijector.forward_log_det_jacobian(
        x, event_ndims=tf.nest.pack_sequence_as(joint.dtype, [0, 1, 2]))
    ildj = joint_bijector.inverse_log_det_jacobian(
        y, event_ndims=tf.nest.pack_sequence_as(joint.dtype, [0, 1, 1]))
    self.assertAllEqual(fldj.shape, joint.log_prob(y).shape)
    self.assertAllClose(fldj, -ildj)

    # Passing inputs *without* batch shape should return sane outputs.
    y = self.evaluate(joint.sample([], seed=test_util.test_seed()))
    # Strip the sample to represent just a single event.
    unbatched_y = tf.nest.map_structure(lambda t: t[0, ...], y)
    self.assertAllEqualNested(tf.nest.map_structure(tf.shape, unbatched_y),
                              joint.event_shape_tensor())
    ildj = joint_bijector.inverse_log_det_jacobian(
        unbatched_y,
        event_ndims=tf.nest.pack_sequence_as(joint.dtype, [0, 1, 1]))
    self.assertAllEqual(ildj.shape, joint.log_prob(unbatched_y).shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'coroutine',
          'jd_class': jdab.JointDistributionCoroutineAutoBatched
      }, {
          'testcase_name': 'sequential',
          'jd_class': jdab.JointDistributionSequentialAutoBatched
      }, {
          'testcase_name': 'named',
          'jd_class': jdab.JointDistributionNamedAutoBatched
      })
  def test_default_event_space_bijector_constant_jacobian(self, jd_class):

    models = {}
    def coroutine_model():
      yield normal.Normal(0., [1., 2.], name='x')

    models[jdab.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[jdab.JointDistributionSequentialAutoBatched] = [
        normal.Normal(0., [1., 2.], name='x')
    ]

    models[jdab.JointDistributionNamedAutoBatched] = {
        'x': normal.Normal(0., [1., 2.], name='x')
    }

    joint = jd_class(models[jd_class], batch_ndims=1, validate_args=True)
    self.assertAllEqual(joint.batch_shape, [2])
    joint_bijector = joint.experimental_default_event_space_bijector()

    y = self.evaluate(joint.sample([3], seed=test_util.test_seed()))
    x = joint_bijector.inverse(y)
    self.assertAllCloseNested(y, joint_bijector.forward(x))

    fldj = joint_bijector.forward_log_det_jacobian(x)
    ildj = joint_bijector.inverse_log_det_jacobian(y)
    self.assertAllEqual(fldj.shape, joint.log_prob(y).shape)
    self.assertAllClose(fldj, -ildj)

  def test_nested_joint_distributions(self):
    batch_shape = [2, 3]

    def inner_fn():
      xy = yield jdab.JointDistributionNamedAutoBatched(
          {
              'x':
                  normal.Normal(
                      loc=tf.zeros(batch_shape),
                      scale=tf.ones(batch_shape),
                      name='x'),
              'y':
                  lambda x: poisson.Poisson(log_rate=x, name='y')
          },
          batch_ndims=2,
          name='xy')
      _ = yield normal.Normal(loc=0., scale=xy['y'], name='z')

    joint = jdab.JointDistributionSequentialAutoBatched([
        jdab.JointDistributionCoroutineAutoBatched(
            inner_fn, batch_ndims=1, name='a')
    ])
    z = joint.sample(seed=test_util.test_seed())

    # Batch and event shape.
    self.assertAllEqual(joint.batch_shape, [])
    self.assertAllEqualNested(
        tf.nest.map_structure(lambda x: tf.TensorShape(x.shape), z),
        joint.event_shape)

    # Sample shape.
    z2 = self.evaluate(
        joint.sample(5, seed=test_util.test_seed()))
    lp2 = joint.log_prob(z2)
    self.assertAllEqual(lp2.shape, [5])

    z3 = joint.sample(value=z2, seed=test_util.test_seed())
    self.assertAllCloseNested(z2, z3)

  # pylint: disable=g-complex-comprehension

  @parameterized.named_parameters(*[
      dict(
          testcase_name='_{}{}'.format(
              jd_class.__name__,
              '_jit' if jit else ''),
          jd_class=jd_class,
          jit=jit)
      for jd_class in (jdab.JointDistributionCoroutineAutoBatched,
                       jdab.JointDistributionSequentialAutoBatched,
                       jdab.JointDistributionNamedAutoBatched)
      for jit in (False, True)])
  def test_kahan_precision(self, jd_class, jit):
    maybe_jit = lambda f: f
    if jit:
      self.skip_if_no_xla()
      if not JAX_MODE and not tf.test.is_gpu_available():
        self.skipTest('b/179303849')
      maybe_jit = tf.function(jit_compile=True)

    def make_models(dtype):
      models = {}
      def mk_20k_poisson(log_rate):
        return poisson.Poisson(
            log_rate=tf.broadcast_to(log_rate[..., tf.newaxis], log_rate.shape +
                                     (20_000,)))
      def coroutine_model():
        log_rate = yield normal.Normal(0., dtype(.2), name='log_rate')
        yield mk_20k_poisson(log_rate).copy(name='x')

      models[jdab.JointDistributionCoroutineAutoBatched] = coroutine_model

      models[jdab.JointDistributionSequentialAutoBatched] = [
          normal.Normal(0., dtype(.2)), mk_20k_poisson
      ]

      models[jdab.JointDistributionNamedAutoBatched] = collections.OrderedDict(
          (('log_rate', normal.Normal(0., dtype(.2))), ('x', mk_20k_poisson)))
      return models

    joint = jd_class(make_models(np.float32)[jd_class], validate_args=True,
                     experimental_use_kahan_sum=True)
    joint64 = jd_class(make_models(np.float64)[jd_class], validate_args=True)
    stream = test_util.test_seed_stream()
    nsamp = 7
    xs = self.evaluate(
        joint.sample(log_rate=tf.zeros([nsamp]), seed=stream()))
    if isinstance(xs, dict):
      xs['log_rate'] = normal.Normal(0, .2).sample(nsamp, seed=stream())
    else:
      xs = (normal.Normal(0, .2).sample(nsamp, seed=stream()), xs[1])
    xs64 = tf.nest.map_structure(lambda x: tf.cast(x, tf.float64), xs)
    lp = maybe_jit(joint.copy(validate_args=not jit).log_prob)(xs)
    lp64 = joint64.log_prob(xs64)
    lp, lp64 = self.evaluate((tf.cast(lp, tf.float64), lp64))
    # Without Kahan, example max-abs-diff: ~0.06
    self.assertAllClose(lp64, lp, rtol=0., atol=.01)

  def test_kahan_broadcasting_check(self):
    def model():
      _ = yield normal.Normal(0., 1.)  # Batch shape ()
      _ = yield normal.Normal([0., 1., 2.], 1.)  # Batch shape [3]

    dist = jdab.JointDistributionCoroutineAutoBatched(
        model,
        validate_args=True,
        experimental_use_kahan_sum=True,
        batch_ndims=1)
    sample = self.evaluate(dist.sample(seed=test_util.test_seed(
        sampler_type='stateless')))
    with self.assertRaises(ValueError):
      self.evaluate(dist.log_prob(sample))


if __name__ == '__main__':
  # TODO(b/173158845): XLA:CPU reassociates away the Kahan correction term.
  os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'
  test_util.main()
