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
import os

# Dependency imports
from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


JAX_MODE = False
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
      g = yield tfd.LogNormal(0., 1.)
      df = yield tfd.Exponential(1.)
      loc = yield tfd.Sample(tfd.Normal(0, g), 20)
      yield tfd.StudentT(tf.expand_dims(df, -1), loc, 1)
    models[tfd.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[tfd.JointDistributionSequentialAutoBatched] = [
        tfd.LogNormal(0., 1.),
        tfd.Exponential(1.),
        lambda _, g: tfd.Sample(tfd.Normal(0, g), 20),
        lambda loc, df: tfd.StudentT(tf.expand_dims(df, -1), loc, 1)
    ]

    models[tfd.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('g', tfd.LogNormal(0., 1.)),
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
      *(dict(  # pylint: disable=g-complex-comprehension
          testcase_name=jd_type + '_' + sampler_type,
          jd_class=getattr(tfd, 'JointDistribution' + jd_type + 'AutoBatched'),
          sampler_type=sampler_type)
        for jd_type in ('Coroutine', 'Sequential', 'Named')
        for sampler_type in ('stateful', 'stateless')))
  def test_model_with_nontrivial_batch_shape(self, jd_class, sampler_type):
    models = {}
    def coroutine_model():
      g = yield tfd.LogNormal(0., [1., 2.])
      df = yield tfd.Exponential([1., 2.])
      loc = yield tfd.Sample(tfd.Normal(0, g), 20)
      yield tfd.StudentT(tf.expand_dims(df, -1), loc, 1)
    models[tfd.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[tfd.JointDistributionSequentialAutoBatched] = [
        tfd.LogNormal(0., [1., 2.]),
        tfd.Exponential([1., 2.]),
        lambda _, g: tfd.Sample(tfd.Normal(0, g), 20),
        lambda loc, df: tfd.StudentT(tf.expand_dims(df, -1), loc, 1)
    ]

    models[tfd.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('g', tfd.LogNormal(0., [1., 2.])),
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

    x = joint.sample([5], seed=test_util.test_seed(sampler_type=sampler_type))
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
      x = yield tfd.Normal(0., scale=[1.])
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
      {'testcase_name': 'coroutine',
       'jd_class': tfd.JointDistributionCoroutineAutoBatched},
      {'testcase_name': 'sequential',
       'jd_class': tfd.JointDistributionSequentialAutoBatched},
      {'testcase_name': 'named',
       'jd_class': tfd.JointDistributionNamedAutoBatched})
  def test_log_prob_and_prob_with_plate(self, jd_class):

    models = {}
    def coroutine_model():
      a = yield tfd.Bernoulli(probs=0.5, dtype=tf.float32)
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
      a = yield tfd.Bernoulli(probs=0.5, dtype=tf.float32)
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

  def test_sample_with_batch_value(self):
    @tfd.JointDistributionCoroutineAutoBatched
    def dist():
      a = yield tfd.Sample(tfd.Normal(0, 1.), 2)
      b = yield tfd.Sample(tfd.Normal(0, 1.), 3)
      # The following line fails if not autovectorized.
      yield tfd.Normal(a[tf.newaxis, ...] * b[..., tf.newaxis], 1.)
    x = self.evaluate(dist.sample(123, seed=test_util.test_seed()))
    x2 = self.evaluate(dist.sample(value=x, seed=test_util.test_seed()))
    self.assertAllCloseNested(x, x2)

    # Also test a dict-type value (JDNamed).
    dist = tfd.JointDistributionNamedAutoBatched({
        'a': tfd.Sample(tfd.Normal(0, 1.), 2),
        'b': tfd.Sample(tfd.Normal(0, 1.), 3),
        'c': lambda a, b: tfd.Normal(  # pylint: disable=g-long-lambda
            a[tf.newaxis, ...] * b[..., tf.newaxis], 1.)})
    x = self.evaluate(dist.sample(123, seed=test_util.test_seed()))
    x2 = self.evaluate(dist.sample(value=x, seed=test_util.test_seed()))
    self.assertAllCloseNested(x, x2)

  def test_sample_with_value_as_kwarg(self):
    @tfd.JointDistributionCoroutineAutoBatched
    def dist():
      a = yield tfd.Sample(tfd.Normal(0, 1.), 2, name='a')
      b = yield tfd.Sample(tfd.Normal(0, 1.), 3, name='b')
      # The following line fails if not autovectorized.
      yield tfd.Normal(a[tf.newaxis, ...] * b[..., tf.newaxis], 1., name='c')

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
      scale_variance = yield tfd.InverseGamma(0.5, 0.5)
      scale_noncentered = yield tfd.Sample(tfd.HalfNormal(1.), num_features)
      scale = scale_noncentered * scale_variance[..., None]**0.5
      weights_noncentered = yield tfd.Sample(tfd.Normal(0., 1.), num_features)
      yield tfd.Deterministic(weights_noncentered * scale)

    joint = tfd.JointDistributionCoroutineAutoBatched(dist, validate_args=True)

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
      a = yield tfd.Sample(tfd.Normal(0., 1.), num_rows, name='a')
      b = yield tfd.Sample(tfd.Normal(0., 1.), num_columns, name='b')
      yield tfd.Normal(a[..., None] * b[None, ...], 1., name='c')

    tuple_joint = tfd.JointDistributionCoroutineAutoBatched(
        dist, validate_args=True)
    namedtuple_joint = tfd.JointDistributionCoroutineAutoBatched(
        dist,
        sample_dtype=collections.namedtuple(
            'ModelSpec', ['a', 'b', 'c'])(
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
    if not tf.executing_eagerly():
      self.skipTest('Test relies on eager execution.')

    @tfd.JointDistributionCoroutineAutoBatched
    def dist():
      # Because `pfor` operates by tracing its loop body, to ensure we're
      # not inside of a `pfor` loop body it's sufficient to check that we're
      # not inside of a tf.function.
      if not tf.executing_eagerly():
        raise ValueError('Model is running inside tf.function. This may '
                         'indicate that auto-vectorization is being '
                         'triggered unnecessarily.')
      yield tfd.Normal(0., 1., name='x')
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

  def test_unit_sample_shape(self):
    @tfd.JointDistributionCoroutineAutoBatched
    def dist():
      x = yield tfd.Normal(loc=tf.zeros([3]), scale=1., name='x')
      yield tfd.Bernoulli(logits=tf.einsum('n->', x), name='y')

    for sample_shape in [(), 1, [1], [1, 1], [2]]:
      self.assertAllEqual(
          dist.log_prob(
              dist.sample(sample_shape,
                          seed=test_util.test_seed())).shape,
          np.reshape(sample_shape, [-1]))

  def test_sample_dtype_structures_output(self):

    num_features = 4

    def dist():
      scale_variance = yield Root(tfd.InverseGamma(0.5, 0.5))
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
    # TODO(b/139710644): Enable `use_vectorized_map` here once
    # `sample_distributions` is supported.
    joint = tfd.JointDistributionCoroutineAutoBatched(
        dist, sample_dtype=sample_dtype, validate_args=True,
        use_vectorized_map=False)
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
      s = yield tfd.Sample(tfd.InverseGamma(2, 2), 100)
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
    seed = test_util.test_seed_stream()
    # Hyperparameters.
    num_topics = 3
    num_words = 10
    avg_doc_length = 5
    u = tfd.Uniform(low=-1., high=1.)
    alpha = tfp.util.TransformedVariable(
        u.sample([num_topics], seed=seed()),
        tfb.Softplus(), name='alpha')
    beta = tf.Variable(u.sample([num_topics, num_words],
                                seed=seed()), name='beta')

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
      {'testcase_name': 'coroutine',
       'jd_class': tfd.JointDistributionCoroutineAutoBatched},
      {'testcase_name': 'sequential',
       'jd_class': tfd.JointDistributionSequentialAutoBatched},
      {'testcase_name': 'named',
       'jd_class': tfd.JointDistributionNamedAutoBatched})
  def test_default_event_space_bijector(self, jd_class):

    models = {}
    def coroutine_model():
      high = yield tfd.LogNormal(0., 1)
      yield tfd.Uniform(low=[-1., -2.], high=high)
    models[tfd.JointDistributionCoroutineAutoBatched] = coroutine_model

    models[tfd.JointDistributionSequentialAutoBatched] = [
        tfd.LogNormal(0., 1.),
        lambda high: tfd.Uniform(low=[-1., -2.], high=high)
    ]

    models[tfd.JointDistributionNamedAutoBatched] = collections.OrderedDict((
        ('high', tfd.LogNormal(0., 1.)),
        ('x', lambda high: tfd.Uniform(low=[-1., -2.], high=high))))

    joint = jd_class(models[jd_class], validate_args=True)
    joint_bijector = joint.experimental_default_event_space_bijector()

    y = self.evaluate(joint.sample([2, 3], seed=test_util.test_seed()))
    x = joint_bijector.inverse(y)
    self.assertAllClose(y, joint_bijector.forward(x))

    event_ndims = tf.nest.pack_sequence_as(joint.dtype, [0, 1])
    fldj = joint_bijector.forward_log_det_jacobian(x, event_ndims=event_ndims)
    ildj = joint_bijector.inverse_log_det_jacobian(y, event_ndims=event_ndims)
    self.assertAllEqual(fldj.shape, [2, 3])
    self.assertAllClose(fldj, -ildj)

  def test_nested_joint_distributions(self):
    batch_shape = [2, 3]

    def inner_fn():
      xy = yield tfd.JointDistributionNamedAutoBatched(
          {'x': tfd.Normal(loc=tf.zeros(batch_shape),
                           scale=tf.ones(batch_shape),
                           name='x'),
           'y': lambda x: tfd.Poisson(log_rate=x, name='y')},
          batch_ndims=2,
          name='xy')
      _ = yield tfd.Normal(loc=0., scale=xy['y'], name='z')

    joint = tfd.JointDistributionSequentialAutoBatched([
        tfd.JointDistributionCoroutineAutoBatched(inner_fn,
                                                  batch_ndims=1,
                                                  name='a')])
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

  @parameterized.named_parameters(*[
      dict(testcase_name='_{}{}'.format(jd_class.__name__,  # pylint: disable=g-complex-comprehension
                                        '_jit' if jit else ''),
           jd_class=jd_class, jit=jit)
      for jd_class in (tfd.JointDistributionCoroutineAutoBatched,
                       tfd.JointDistributionSequentialAutoBatched,
                       tfd.JointDistributionNamedAutoBatched)
      for jit in (False, True)
  ])
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
        return tfd.Poisson(log_rate=tf.broadcast_to(log_rate[..., tf.newaxis],
                                                    log_rate.shape + (20_000,)))
      def coroutine_model():
        log_rate = yield tfd.Normal(0., dtype(.2), name='log_rate')
        yield mk_20k_poisson(log_rate).copy(name='x')
      models[tfd.JointDistributionCoroutineAutoBatched] = coroutine_model

      models[tfd.JointDistributionSequentialAutoBatched] = [
          tfd.Normal(0., dtype(.2)), mk_20k_poisson
      ]

      models[tfd.JointDistributionNamedAutoBatched] = collections.OrderedDict((
          ('log_rate', tfd.Normal(0., dtype(.2))), ('x', mk_20k_poisson)))
      return models

    joint = jd_class(make_models(np.float32)[jd_class], validate_args=True,
                     experimental_use_kahan_sum=True)
    joint64 = jd_class(make_models(np.float64)[jd_class], validate_args=True)
    stream = test_util.test_seed_stream()
    nsamp = 7
    xs = self.evaluate(
        joint.sample(log_rate=tf.zeros([nsamp]), seed=stream()))
    if isinstance(xs, dict):
      xs['log_rate'] = tfd.Normal(0, .2).sample(nsamp, seed=stream())
    else:
      xs = (tfd.Normal(0, .2).sample(nsamp, seed=stream()), xs[1])
    xs64 = tf.nest.map_structure(lambda x: tf.cast(x, tf.float64), xs)
    lp = maybe_jit(joint.copy(validate_args=not jit).log_prob)(xs)
    lp64 = joint64.log_prob(xs64)
    lp, lp64 = self.evaluate((tf.cast(lp, tf.float64), lp64))
    # Without Kahan, example max-abs-diff: ~0.06
    self.assertAllClose(lp64, lp, rtol=0., atol=.01)


if __name__ == '__main__':
  # TODO(b/173158845): XLA:CPU reassociates away the Kahan correction term.
  os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'
  tf.test.main()
