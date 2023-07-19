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
"""Tests for the JointDistributionCoroutine."""

import collections
import functools
import warnings

# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import ldj_ratio
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import beta_binomial
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import dirichlet
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import inverse_gamma
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import logistic
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import multinomial
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.distributions import sample as sample_lib
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math import special
from tensorflow_probability.python.util import deferred_tensor

Root = jdc.JointDistributionCoroutine.Root


def basic_model_fn():
  yield Root(normal.Normal(0., 1.))
  e = yield Root(
      independent.Independent(
          exponential.Exponential(rate=[100, 120]),
          reinterpreted_batch_ndims=1,
          name='e'))
  yield gamma.Gamma(concentration=e[..., 0], rate=e[..., 1])


def basic_model_with_names_fn():
  yield Root(normal.Normal(0., 1., name='a'))
  e = yield Root(
      independent.Independent(
          exponential.Exponential(rate=[100, 120]),
          reinterpreted_batch_ndims=1,
          name='e'))
  yield gamma.Gamma(concentration=e[..., 0], rate=e[..., 1], name='x')


def nested_lists_with_names_model_fn():
  abc = yield Root(
      jds.JointDistributionSequential([
          mvn_diag.MultivariateNormalDiag([0., 0.], [1., 1.]),
          jds.JointDistributionSequential(
              [student_t.StudentT(3., -2., 5.),
               exponential.Exponential(4.)])
      ],
                                      name='abc'))
  a, (b, c) = abc
  yield jds.JointDistributionSequential([
      independent.Independent(
          normal.Normal(a * b, c), reinterpreted_batch_ndims=1),
      independent.Independent(
          normal.Normal(a + b, c), reinterpreted_batch_ndims=1)
  ],
                                        name='de')


def singleton_normal_model_fn():
  yield Root(normal.Normal(0., 1., name='x'))


def singleton_jds_model_fn():
  yield Root(
      jds.JointDistributionSequential(
          [normal.Normal(0., 1.), lambda x: poisson.Poisson(tf.exp(x))],
          name='x'))


def singleton_jdn_model_fn():
  yield Root(
      jdn.JointDistributionNamed(
          {
              'z': normal.Normal(0., 1.),
              'y': lambda z: poisson.Poisson(tf.exp(z))
          },
          name='x'))


@test_util.test_all_tf_execution_regimes
class JointDistributionCoroutineTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('_with_root', True),
      ('_without_root', False),
  )
  def test_batch_and_event_shape_no_plate(self, use_root):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #  (a)-->--(b)
    #    \      |
    #     \     v
    #      `->-(c)

    root = Root if use_root else lambda x: x

    def dist():
      a = yield root(bernoulli.Bernoulli(probs=0.5, dtype=tf.float32))
      b = yield bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32)
      yield normal.Normal(loc=a, scale=1. + b)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    # Properties `event_shape` and `batch_shape` should be defined
    # even before any sampling calls have occurred.
    self.assertAllEqual(joint.event_shape, [[], [], []])
    self.assertAllEqual(joint.batch_shape, [[], [], []])

    ds, _ = joint.sample_distributions(seed=test_util.test_seed())
    self.assertLen(ds, 3)
    self.assertIsInstance(ds[0], bernoulli.Bernoulli)
    self.assertIsInstance(ds[1], bernoulli.Bernoulli)
    self.assertIsInstance(ds[2], normal.Normal)

    is_event_scalar = joint.is_scalar_event()
    self.assertAllEqual(is_event_scalar[0], True)
    self.assertAllEqual(is_event_scalar[1], True)
    self.assertAllEqual(is_event_scalar[2], True)

    event_shape = joint.event_shape_tensor()
    self.assertAllEqual(event_shape[0], [])
    self.assertAllEqual(event_shape[1], [])
    self.assertAllEqual(event_shape[2], [])

    is_batch_scalar = joint.is_scalar_batch()
    self.assertAllEqual(is_batch_scalar[0], True)
    self.assertAllEqual(is_batch_scalar[1], True)
    self.assertAllEqual(is_batch_scalar[2], True)

    batch_shape = joint.batch_shape_tensor()
    self.assertAllEqual(batch_shape[0], [])
    self.assertAllEqual(batch_shape[1], [])
    self.assertAllEqual(batch_shape[2], [])

  @parameterized.named_parameters(
      ('_with_root', True),
      ('_without_root', False),
  )
  def test_batch_and_event_shape_with_plate(self, use_root):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #       +-----------+
    #  (g)--+-->--(loc) |
    #       |       |   |
    #       |       v   |
    # (df)--+-->---(x)  |
    #       +--------20-+

    root = Root if use_root else lambda x: x

    def dist():
      g = yield root(gamma.Gamma(2, 2))
      df = yield root(exponential.Exponential(1.))
      loc = yield sample_lib.Sample(normal.Normal(0, g), 20)
      yield independent.Independent(
          student_t.StudentT(tf.expand_dims(df, -1), loc, 1), 1)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    # Properties `event_shape` and `batch_shape` should be defined
    # even before any sampling calls have occurred.
    self.assertAllEqualNested(list(joint.event_shape), [[], [], [20], [20]])
    self.assertAllEqualNested(list(joint.batch_shape), [[], [], [], []])

    ds, _ = joint.sample_distributions(seed=test_util.test_seed())
    self.assertLen(ds, 4)
    self.assertIsInstance(ds[0], gamma.Gamma)
    self.assertIsInstance(ds[1], exponential.Exponential)
    self.assertIsInstance(ds[2], sample_lib.Sample)
    self.assertIsInstance(ds[3], independent.Independent)

    is_scalar = joint.is_scalar_event()
    self.assertAllEqual(is_scalar[0], True)
    self.assertAllEqual(is_scalar[1], True)
    self.assertAllEqual(is_scalar[2], False)
    self.assertAllEqual(is_scalar[3], False)

    event_shape = joint.event_shape_tensor()
    self.assertAllEqual(event_shape[0], [])
    self.assertAllEqual(event_shape[1], [])
    self.assertAllEqual(event_shape[2], [20])
    self.assertAllEqual(event_shape[3], [20])

    is_batch = joint.is_scalar_batch()
    self.assertAllEqual(is_batch[0], True)
    self.assertAllEqual(is_batch[1], True)
    self.assertAllEqual(is_batch[2], True)
    self.assertAllEqual(is_batch[3], True)

    batch_shape = joint.batch_shape_tensor()
    self.assertAllEqual(batch_shape[0], [])
    self.assertAllEqual(batch_shape[1], [])
    self.assertAllEqual(batch_shape[2], [])
    self.assertAllEqual(batch_shape[3], [])

  def test_sample_shape_no_plate(self):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #  (a)-->--(b)
    #    \      |
    #     \     v
    #      `->-(c)

    def dist():
      a = yield Root(bernoulli.Bernoulli(probs=0.5, dtype=tf.float32))
      b = yield bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32)
      yield normal.Normal(loc=a, scale=1. + b)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    z = joint.sample(seed=test_util.test_seed())

    self.assertAllEqual(tf.shape(z[0]), [])
    self.assertAllEqual(tf.shape(z[1]), [])
    self.assertAllEqual(tf.shape(z[2]), [])

    z = joint.sample(2, seed=test_util.test_seed())

    self.assertAllEqual(tf.shape(z[0]), [2])
    self.assertAllEqual(tf.shape(z[1]), [2])
    self.assertAllEqual(tf.shape(z[2]), [2])

    z = joint.sample([3, 2], seed=test_util.test_seed())

    self.assertAllEqual(tf.shape(z[0]), [3, 2])
    self.assertAllEqual(tf.shape(z[1]), [3, 2])
    self.assertAllEqual(tf.shape(z[2]), [3, 2])

  def test_sample_shape_with_plate(self):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #       +-----------+
    #  (g)--+-->--(loc) |
    #       |       |   |
    #       |       v   |
    # (df)--+-->---(x)  |
    #       +--------20-+

    def dist():
      g = yield Root(gamma.Gamma(2, 2))
      df = yield Root(exponential.Exponential(1.))
      loc = yield sample_lib.Sample(normal.Normal(0, g), 20)
      yield independent.Independent(
          student_t.StudentT(tf.expand_dims(df, -1), loc, 1), 1)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    z = joint.sample(seed=test_util.test_seed())

    self.assertAllEqual(tf.shape(z[0]), [])
    self.assertAllEqual(tf.shape(z[1]), [])
    self.assertAllEqual(tf.shape(z[2]), [20])
    self.assertAllEqual(tf.shape(z[3]), [20])

    z = joint.sample(2, seed=test_util.test_seed())

    self.assertAllEqual(tf.shape(z[0]), [2])
    self.assertAllEqual(tf.shape(z[1]), [2])
    self.assertAllEqual(tf.shape(z[2]), [2, 20])
    self.assertAllEqual(tf.shape(z[3]), [2, 20])

    z = joint.sample([3, 2], seed=test_util.test_seed())

    self.assertAllEqual(tf.shape(z[0]), [3, 2])
    self.assertAllEqual(tf.shape(z[1]), [3, 2])
    self.assertAllEqual(tf.shape(z[2]), [3, 2, 20])
    self.assertAllEqual(tf.shape(z[3]), [3, 2, 20])

  def test_log_prob_no_plate(self):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #  (a)-->--(b)
    #    \      |
    #     \     v
    #      `->-(c)

    def dist():
      a = yield Root(bernoulli.Bernoulli(probs=0.5, dtype=tf.float32))
      b = yield bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32)
      yield normal.Normal(loc=a, scale=1. + b)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    z = joint.sample(seed=test_util.test_seed())

    log_prob = joint.log_prob(z)

    a, b, c = z  # pylint: disable=unbalanced-tuple-unpacking

    expected_log_prob = (
        np.log(0.5) +
        tf.math.log(b * (0.25 + 0.5 * a) +
                    (1 - b) * (0.75 -0.5 * a)) +
        -0.5 * ((c - a) / (1. + b)) ** 2 -
        0.5 * np.log(2. * np.pi) -
        tf.math.log((1. + b)))

    self.assertAllClose(*self.evaluate([log_prob, expected_log_prob]))

  def test_log_prob_with_plate(self):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #       +-----------+
    #       |           |
    #  (a)--+--(b)->(c) |
    #       |           |
    #       +---------2-+

    def dist():
      a = yield Root(bernoulli.Bernoulli(probs=0.5, dtype=tf.float32))
      b = yield sample_lib.Sample(
          bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32), 2)
      yield independent.Independent(normal.Normal(loc=a, scale=1. + b), 1)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    z = joint.sample(seed=test_util.test_seed())
    a, b, c = z  # pylint: disable=unbalanced-tuple-unpacking

    log_prob = joint.log_prob(z)

    expected_log_prob = (
        np.log(0.5) +
        tf.reduce_sum(tf.math.log(b * (0.25 + 0.5 * a) +
                                  (1 - b) * (0.75 - 0.5 * a))) +
        tf.reduce_sum(-0.5 * ((c - a) / (1. + b))**2 -
                      0.5 * np.log(2. * np.pi) -
                      tf.math.log((1. + b))))

    self.assertAllClose(*self.evaluate([log_prob, expected_log_prob]))

  def test_sample_and_log_prob(self):

    # Define a bijector to detect if/when `inverse` is called.
    inverted_values = []

    class InverseTracingExp(exp.Exp):

      def _inverse(self, y):
        inverted_values.append(y)
        return tf.math.log(y)

    def coroutine_model():
      g = yield Root(InverseTracingExp()(normal.Normal(0., 1.), name='g'))
      df = yield Root(exponential.Exponential(1., name='df'))
      loc = yield sample_lib.Sample(normal.Normal(0, g), 20, name='loc')
      yield independent.Independent(
          student_t.StudentT(df[..., tf.newaxis], loc, 1, name='x'),
          reinterpreted_batch_ndims=1)

    joint = jdc.JointDistributionCoroutine(coroutine_model, validate_args=True)

    seed = test_util.test_seed(sampler_type='stateless')
    for sample_shape in ([], [5]):
      inverted_values.clear()
      x1, lp1 = self.evaluate(
          joint.experimental_sample_and_log_prob(
              sample_shape,
              seed=seed,
              df=2.7 * tf.ones(sample_shape)  # Check that kwargs are supported.
              ))
      x2 = self.evaluate(
          joint.sample(sample_shape,
                       seed=seed,
                       df=2.7 * tf.ones(sample_shape)))
      self.assertAllCloseNested(x1, x2)

      self.assertLen(inverted_values, 0)
      lp2 = joint.log_prob(x1)
      self.assertLen(inverted_values, 1)
      self.assertAllClose(lp1, lp2)

  def test_detect_missing_root(self):
    if not tf.executing_eagerly(): return
    # The joint distribution specified below is intended to
    # correspond to this graphical model
    #
    #       +-----------+
    #       |           |
    #  (a)--+--(b)->(c) |
    #       |           |
    #       +---------2-+

    def dist():
      a = yield bernoulli.Bernoulli(probs=0.5, dtype=tf.float32)  # Missing root
      b = yield sample_lib.Sample(
          bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32), 2)
      yield independent.Independent(normal.Normal(loc=a, scale=1. + b), 1)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    with self.assertRaisesRegexp(
        Exception,
        'must be wrapped in `Root`'):
      self.evaluate(joint.sample(2, seed=test_util.test_seed()))

  @parameterized.named_parameters(
      ('basic', basic_model_with_names_fn),
      ('nested_lists', nested_lists_with_names_model_fn),
      ('basic_unnamed', basic_model_fn))
  def test_can_call_log_prob_with_args_and_kwargs(self, model_fn):
    d = jdc.JointDistributionCoroutine(model_fn, validate_args=True)

    # Destructure vector-valued Tensors into Python lists, to mimic the values
    # a user might type.
    def _convert_ndarray_to_list(x):
      if isinstance(x, np.ndarray) and x.ndim > 0:
        return list(x)
      return x
    value = tf.nest.map_structure(
        _convert_ndarray_to_list,
        self.evaluate(d.sample(seed=test_util.test_seed())))
    value_with_names = list(zip(d._flat_resolve_names(), value))

    lp_value_positional = self.evaluate(d.log_prob(value))
    lp_value_named = self.evaluate(d.log_prob(value=value))
    self.assertAllEqual(lp_value_positional, lp_value_named)

    lp_args = self.evaluate(d.log_prob(*value))
    self.assertAllEqual(lp_value_positional, lp_args)

    lp_kwargs = self.evaluate(d.log_prob(**dict(value_with_names)))
    self.assertAllEqual(lp_value_positional, lp_kwargs)

    lp_args_then_kwargs = self.evaluate(d.log_prob(
        *value[:1], **dict(value_with_names[1:])))
    self.assertAllEqual(lp_value_positional, lp_args_then_kwargs)

    with self.assertRaisesRegexp(
        ValueError, r'Joint distribution expected values for [0-9] components'):
      d.log_prob(badvar=27.)

    with self.assertRaisesRegexp(ValueError, 'unexpected keyword argument'):
      d.log_prob(*value, extra_arg=27.)

  def test_log_prob_with_manual_kwargs(self):
    d = jdc.JointDistributionCoroutine(basic_model_fn, validate_args=True)
    x = d.sample(seed=test_util.test_seed())
    lp1 = d.log_prob(var0=x[0], e=x[1], var2=x[2])
    lp2 = d.log_prob(x)
    lp1_, lp2_ = self.evaluate([lp1, lp2])
    self.assertAllClose(lp1_, lp2_)

  def test_duplicate_names_error(self):

    @jdc.JointDistributionCoroutine
    def dist():
      yield Root(normal.Normal(0., 1., name='a'))
      yield Root(normal.Normal(0., 1., name='a'))

    with self.assertRaisesRegexp(ValueError, 'Duplicated distribution name: a'):
      dist.log_prob((1, 2))

  @parameterized.named_parameters(
      ('singleton_float', singleton_normal_model_fn),
      ('singleton_tuple', singleton_jds_model_fn),
      ('singleton_dict', singleton_jdn_model_fn))
  def test_singleton_model_works_with_args_and_kwargs(self, model_fn):
    d = jdc.JointDistributionCoroutine(model_fn)

    xs = self.evaluate(
        d.sample(seed=test_util.test_seed()))  # `xs` is a one-element list.

    lp_from_structure = self.evaluate(d.log_prob(xs))
    lp_from_structure_kwarg = self.evaluate(d.log_prob(value=xs))
    self.assertAllEqual(lp_from_structure, lp_from_structure_kwarg)

    lp_from_arg = self.evaluate(d.log_prob(xs[0]))
    self.assertAllEqual(lp_from_structure, lp_from_arg)

    lp_from_kwarg = self.evaluate(d.log_prob(x=xs[0]))
    self.assertAllEqual(lp_from_structure, lp_from_kwarg)

  def test_check_sample_rank(self):
    def dist():
      g = yield Root(gamma.Gamma(2, 2))
      # The following line lacks a `Root` so that if a shape [3, 5]
      # sample is requested the distribution below will produce
      # samples that have too low a rank to be consistent with
      # the shape.
      df = yield exponential.Exponential(1.)
      loc = yield sample_lib.Sample(normal.Normal(0, g), 20)
      yield independent.Independent(
          student_t.StudentT(tf.expand_dims(df, -1), loc, 1), 1)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    with self.assertRaisesRegexp(
        Exception,
        'are not consistent with `sample_shape`'):
      self.evaluate(joint.sample([3, 5], seed=test_util.test_seed()))

  def test_check_sample_shape(self):
    def dist():
      g = yield Root(gamma.Gamma(2, 2))
      # The following line lacks a `Root` so that if a shape [3, 5]
      # sample is requested the following line will yield samples
      # with an appropriate rank but whose shape starts with [2, 2]
      # rather than [3, 5].
      df = yield exponential.Exponential([[1., 2.], [3., 4.]])
      loc = yield sample_lib.Sample(normal.Normal(0, g), 20)
      yield independent.Independent(
          student_t.StudentT(tf.expand_dims(df, -1), loc, 1), 1)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    with self.assertRaisesRegexp(
        Exception,
        'are not consistent with `sample_shape`'):
      self.evaluate(joint.sample([3, 5], seed=test_util.test_seed()))

  def test_log_prob_multiple_samples(self):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #  (a)-->--(b)
    #    \      |
    #     \     v
    #      `->-(c)

    def dist():
      a = yield Root(bernoulli.Bernoulli(probs=0.5, dtype=tf.float32))
      b = yield bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32)
      yield normal.Normal(loc=a, scale=1. + b)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    z = joint.sample(4, seed=test_util.test_seed())

    log_prob = joint.log_prob(z)

    a, b, c = z  # pylint: disable=unbalanced-tuple-unpacking

    expected_log_prob = (
        np.log(0.5) +
        tf.math.log(b * (0.25 + 0.5 * a) +
                    (1 - b) * (0.75 -0.5 * a)) +
        -0.5 * ((c - a) / (1. + b)) ** 2 -
        0.5 * np.log(2. * np.pi) -
        tf.math.log((1. + b)))

    self.assertAllClose(*self.evaluate([log_prob, expected_log_prob]))

  def test_prob_multiple_samples(self):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #  (a)-->--(b)
    #    \      |
    #     \     v
    #      `->-(c)

    def dist():
      a = yield Root(bernoulli.Bernoulli(probs=0.5, dtype=tf.float32))
      b = yield bernoulli.Bernoulli(probs=0.25 + 0.5 * a, dtype=tf.float32)
      yield normal.Normal(loc=a, scale=1. + b)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    z = joint.sample(4, seed=test_util.test_seed())

    prob = joint.prob(z)

    a, b, c = z  # pylint: disable=unbalanced-tuple-unpacking

    expected_prob = tf.exp(
        np.log(0.5) +
        tf.math.log(b * (0.25 + 0.5 * a) +
                    (1 - b) * (0.75 -0.5 * a)) +
        -0.5 * ((c - a) / (1. + b)) ** 2 -
        0.5 * np.log(2. * np.pi) -
        tf.math.log((1. + b)))

    self.assertAllClose(*self.evaluate([prob, expected_prob]))

  def test_log_prob_multiple_roots(self):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #       +-----------+
    #  (a)--+-->---(b)  |
    #       |       |   |
    #       |       v   |
    #  (c)--+-->---(d)  |
    #       +---------2-+

    def dist():
      a = yield Root(exponential.Exponential(1.))
      b = yield sample_lib.Sample(normal.Normal(a, 1.), 20)
      c = yield Root(exponential.Exponential(1.))
      yield independent.Independent(normal.Normal(b, tf.expand_dims(c, -1)), 1)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    z = joint.sample(seed=test_util.test_seed())

    a, b, c, d = z  # pylint: disable=unbalanced-tuple-unpacking

    expected_log_prob = (
        -a +
        tf.reduce_sum(-0.5 * (b - a)**2 + -0.5 * np.log(2. * np.pi), axis=-1) -
        c +
        tf.reduce_sum(-0.5 * (d - b)**2 / c**2 -
                      0.5 * tf.math.log(2. * np.pi * c**2),
                      axis=-1))

    log_prob = joint.log_prob(z)

    self.assertAllClose(*self.evaluate([log_prob, expected_log_prob]),
                        rtol=1e-5)

  def test_log_prob_multiple_roots_and_samples(self):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #       +-----------+
    #  (a)--+-->---(b)  |
    #       |       |   |
    #       |       v   |
    #  (c)--+-->---(d)  |
    #       +---------2-+

    def dist():
      a = yield Root(exponential.Exponential(1.))
      b = yield sample_lib.Sample(normal.Normal(a, 1.), 20)
      c = yield Root(exponential.Exponential(1.))
      yield independent.Independent(normal.Normal(b, tf.expand_dims(c, -1)), 1)

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    z = joint.sample([3, 5], seed=test_util.test_seed())

    a, b, c, d = z  # pylint: disable=unbalanced-tuple-unpacking

    expanded_c = tf.expand_dims(c, -1)
    expected_log_prob = (
        -a +
        tf.reduce_sum(-0.5 * (b - tf.expand_dims(a, -1))**2 -
                      0.5 * np.log(2. * np.pi),
                      axis=-1) -
        c +
        tf.reduce_sum(-0.5 * (d - b)**2 / expanded_c**2 -
                      0.5 * tf.math.log(2. * np.pi * expanded_c**2),
                      axis=-1))

    log_prob = joint.log_prob(z)

    self.assertAllClose(*self.evaluate([log_prob, expected_log_prob]),
                        rtol=1e-4, atol=1e-5)

  def test_sample_dtype_structures_output(self):
    def noncentered_horseshoe_prior(num_features):
      scale_variance = yield Root(inverse_gamma.InverseGamma(0.5, 0.5))
      scale_noncentered = yield Root(
          sample_lib.Sample(half_normal.HalfNormal(1.), num_features))
      scale = scale_noncentered * scale_variance[..., None]**0.5
      weights_noncentered = yield Root(
          sample_lib.Sample(normal.Normal(0., 1.), num_features))
      yield independent.Independent(
          deterministic.Deterministic(weights_noncentered * scale),
          reinterpreted_batch_ndims=1)

    # Currently sample_dtype is only used for `tf.nest.pack_structure_as`. In
    # the future we may use it for error checking and/or casting.
    sample_dtype = collections.namedtuple('Model', [
        'scale_variance',
        'scale_noncentered',
        'weights_noncentered',
        'weights',
    ])(*([None]*4))
    joint = jdc.JointDistributionCoroutine(
        lambda: noncentered_horseshoe_prior(4),
        sample_dtype=sample_dtype,
        validate_args=True)
    self.assertAllEqual(sorted(sample_dtype._fields),
                        sorted(joint.sample(
                            seed=test_util.test_seed())._fields))
    ds, xs = joint.sample_distributions([2, 3], seed=test_util.test_seed())
    tf.nest.assert_same_structure(sample_dtype, ds)
    tf.nest.assert_same_structure(sample_dtype, xs)
    self.assertEqual([3, 4], joint.log_prob(
        joint.sample([3, 4], seed=test_util.test_seed())).shape)

    # Check that a list dtype doesn't get corrupted by `tf.Module` wrapping.
    sample_dtype = [None, None, None, None]
    joint = jdc.JointDistributionCoroutine(
        lambda: noncentered_horseshoe_prior(4),
        sample_dtype=sample_dtype,
        validate_args=True)
    ds, xs = joint.sample_distributions([2, 3], seed=test_util.test_seed())
    self.assertEqual(type(sample_dtype), type(xs))
    self.assertEqual(type(sample_dtype), type(ds))
    tf.nest.assert_same_structure(sample_dtype, ds)
    tf.nest.assert_same_structure(sample_dtype, xs)

  def test_repr_with_custom_sample_dtype(self):
    def model():
      s = yield jdc.JointDistributionCoroutine.Root(
          sample_lib.Sample(inverse_gamma.InverseGamma(2, 2), 100))
      yield independent.Independent(normal.Normal(0, s), 1)
    sd = collections.namedtuple('Model', ['s', 'w'])(None, None)
    m = jdc.JointDistributionCoroutine(
        model, sample_dtype=sd, validate_args=True)
    self.assertEqual(
        ('tfp.distributions.JointDistributionCoroutine('
         '"JointDistributionCoroutine",'
         ' batch_shape=Model(s=[], w=[]),'
         ' event_shape=Model(s=[100], w=[100]),'
         ' dtype=Model(s=float32, w=float32))'),
        str(m))
    self.assertEqual(
        ('<tfp.distributions.JointDistributionCoroutine'
         ' \'JointDistributionCoroutine\''
         ' batch_shape=Model(s=[], w=[])'
         ' event_shape=Model(s=[100], w=[100])'
         ' dtype=Model(s=float32, w=float32)>'),
        repr(m))

  def test_converts_nested_lists_to_tensor(self):
    def dist():
      a = yield Root(mvn_diag.MultivariateNormalDiag([0., 0.], [1., 1.]))
      yield jds.JointDistributionSequential([
          jds.JointDistributionSequential([normal.Normal(a[..., 0], 1.)]),
          normal.Normal(a[..., 1], 1.)
      ])

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    x = [tf.convert_to_tensor([4., 2.]), [[1.], 3.]]
    x_with_tensor_as_list = [[4., 2.], [[1.], 3.]]
    lp = self.evaluate(joint.log_prob(x))
    lp_with_tensor_as_list = self.evaluate(
        joint.log_prob(x_with_tensor_as_list))
    self.assertAllClose(lp, lp_with_tensor_as_list)

  def test_matrix_factorization(self):
    # A matrix factorization model based on
    # Probabilistic Matrix Factorization by
    # Ruslan Salakhutdinov and Andriy Mnih
    # https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf
    #
    #       users
    # +-----3-+
    # | U     |
    # | | +-------+
    # | | |   |   |
    # | +-->R<--+ |
    # |   |   | | |
    # +---|---+ | |
    #     |     V |
    #     +-5-----+
    #       items
    def model():
      n_users = 3
      n_items = 5
      n_factors = 2

      user_trait_prior_scale = 10.
      item_trait_prior_scale = 10.
      observation_noise_prior_scale = 1.

      # U in paper
      user_traits = yield Root(
          sample_lib.Sample(
              normal.Normal(loc=0., scale=user_trait_prior_scale),
              sample_shape=[n_factors, n_users]))

      # V in paper
      item_traits = yield Root(
          sample_lib.Sample(
              normal.Normal(loc=0., scale=item_trait_prior_scale),
              sample_shape=[n_factors, n_items]))

      # R in paper
      yield independent.Independent(
          normal.Normal(
              loc=tf.matmul(user_traits, item_traits, adjoint_a=True),
              scale=observation_noise_prior_scale),
          reinterpreted_batch_ndims=2)

    dist = jdc.JointDistributionCoroutine(model)
    self.assertAllEqual(dist.event_shape, [[2, 3], [2, 5], [3, 5]])

    z = dist.sample(seed=test_util.test_seed())
    self.assertAllEqual(tf.shape(z[0]), [2, 3])
    self.assertAllEqual(tf.shape(z[1]), [2, 5])
    self.assertAllEqual(tf.shape(z[2]), [3, 5])
    lp = dist.log_prob(z)
    self.assertEqual(lp.shape, [])

    z = dist.sample((7, 9), seed=test_util.test_seed())
    self.assertAllEqual(tf.shape(z[0]), [7, 9, 2, 3])
    self.assertAllEqual(tf.shape(z[1]), [7, 9, 2, 5])
    self.assertAllEqual(tf.shape(z[2]), [7, 9, 3, 5])
    lp = dist.log_prob(z)
    self.assertEqual(lp.shape, [7, 9])

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_variable_test
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
    u = uniform.Uniform(low=-1., high=1.)
    alpha = deferred_tensor.TransformedVariable(
        u.sample([num_topics]), softplus.Softplus(), name='alpha')
    beta = tf.Variable(u.sample([num_topics, num_words]), name='beta')

    # LDA Model.
    # Note near 1:1 with mathematical specification. The main distinction is the
    # use of Independent--this lets us easily aggregate multinomials across
    # topics (and in any "shape" of documents).
    def lda_model():
      n = yield Root(poisson.Poisson(rate=avg_doc_length))
      theta = yield Root(dirichlet.Dirichlet(concentration=alpha))
      z = yield multinomial.Multinomial(total_count=n, probs=theta)
      yield independent.Independent(
          multinomial.Multinomial(total_count=z, logits=beta),
          reinterpreted_batch_ndims=1)

    lda = jdc.JointDistributionCoroutine(lda_model, validate_args=True)

    # Now, let's sample some "documents" and compute the log-prob of each.
    docs_shape = [2, 4]  # That is, 8 docs in the shape of [2, 4].
    [n, theta, z, x] = lda.sample(docs_shape, seed=test_util.test_seed())
    log_probs = lda.log_prob([n, theta, z, x])
    self.assertEqual(docs_shape, log_probs.shape)

    # Verify we correctly track trainable variables.
    self.assertLen(lda.trainable_variables, 2)
    self.assertIs(alpha.pretransformed_input, lda.trainable_variables[0])
    self.assertIs(beta, lda.trainable_variables[1])

    # Ensure we can compute gradients.
    with tf.GradientTape() as tape:
      # Note: The samples are not taped, hence implicitly "stop_gradient."
      negloglik = -lda.log_prob([n, theta, z, x])
    grads = tape.gradient(negloglik, lda.trainable_variables)

    self.assertLen(grads, 2)
    self.assertAllEqual((alpha.pretransformed_input.shape, beta.shape),
                        (grads[0].shape, grads[1].shape))
    self.assertAllNotNone(grads)

  @test_util.jax_disable_test_missing_functionality('Graph tensors')
  @test_util.numpy_disable_test_missing_functionality('Graph tensors')
  def test_cache_doesnt_leak_graph_tensors(self):
    if not tf.executing_eagerly():
      return

    def dist():
      random_rank = tf.cast(3.5 + tf.random.uniform(
          [], seed=test_util.test_seed()), tf.int32)
      yield Root(normal.Normal(loc=0., scale=tf.ones([random_rank])))

    joint = jdc.JointDistributionCoroutine(dist, validate_args=True)

    @tf.function(autograph=False)
    def get_batch_shapes():
      return joint.batch_shape_tensor()

    # Calling the tf.function will put graph Tensors in
    # `joint._single_sample_distributions`
    _ = get_batch_shapes()

    # Referring to sampled distributions in eager mode should produce an eager
    # result. Graph Tensors will throw an error.
    _ = [s.numpy() for s in joint.batch_shape_tensor()]

  def test_default_event_space_bijector(self):
    def dists():
      a = yield Root(exponential.Exponential(1., validate_args=True))
      b = yield independent.Independent(
          uniform.Uniform([-1., -2.], a, validate_args=True))
      yield logistic.Logistic(b, a, validate_args=True)

    jd = jdc.JointDistributionCoroutine(dists, validate_args=True)
    joint_bijector = jd.experimental_default_event_space_bijector()

    def _finite_difference_ldj(bijectors, transform_direction, xs, delta):
      transform_plus = [getattr(b, transform_direction)(x + delta)
                        for x, b in zip(xs, bijectors)]
      transform_minus = [getattr(b, transform_direction)(x - delta)
                         for x, b in zip(xs, bijectors)]
      ldj = tf.reduce_sum(
          [tf.reduce_sum(tf.math.log((p - m) / (2. * delta)))
           for p, m in zip(transform_plus, transform_minus)])
      return ldj

    def _get_support_bijectors(dists, xs=None, ys=None):
      index = 0
      dist_gen = dists()
      d = next(dist_gen).distribution
      samples = [] if ys is None else ys
      bijectors = []
      try:
        while True:
          b = d.experimental_default_event_space_bijector()
          y = ys[index] if xs is None else b(xs[index])
          if ys is None:
            y = b(xs[index])
            samples.append(y)
          else:
            y = ys[index]
          bijectors.append(b)
          d = dist_gen.send(y)
          index += 1
      except StopIteration:
        pass
      return bijectors, samples

    # define a sample in the unconstrained space and construct the component
    # distributions
    xs = type(jd.dtype)(*[tf.constant(w) for w in [0.2, [-1.3, 0.1], -2.]])
    bijectors, ys = _get_support_bijectors(dists, xs=xs)
    ys = type(jd.dtype)(*ys)

    # Test forward and inverse values.
    self.assertAllClose(joint_bijector.forward(xs), ys)
    self.assertAllClose(joint_bijector.inverse(ys), xs)

    # Test forward log det Jacobian via finite differences.
    event_ndims = [0, 1, 0]
    fldj = joint_bijector.forward_log_det_jacobian(xs, event_ndims)
    fldj_fd = _finite_difference_ldj(bijectors, 'forward', xs, delta=0.01)
    self.assertAllClose(self.evaluate(fldj), self.evaluate(fldj_fd), rtol=1e-5)

    # Test inverse log det Jacobian via finite differences.
    ildj = joint_bijector.inverse_log_det_jacobian(ys, event_ndims)
    bijectors, _ = _get_support_bijectors(dists, ys=ys)
    ildj_fd = _finite_difference_ldj(bijectors, 'inverse', ys, delta=0.001)
    self.assertAllClose(self.evaluate(ildj), self.evaluate(ildj_fd), rtol=1e-4)

    # test event shapes
    event_shapes = [[2, None], [2], [4]]
    self.assertAllEqualNested(
        [shape.as_list()
         for shape in joint_bijector.forward_event_shape(event_shapes)],
        [bijectors[i].forward_event_shape(event_shapes[i]).as_list()
         for i in range(3)])
    self.assertAllEqualNested(
        [shape.as_list()
         for shape in joint_bijector.inverse_event_shape(event_shapes)],
        [bijectors[i].inverse_event_shape(event_shapes[i]).as_list()
         for i in range(3)])

    event_shapes = [[3], [3, 2], []]
    forward_joint_event_shape = joint_bijector.forward_event_shape_tensor(
        event_shapes)
    inverse_joint_event_shape = joint_bijector.inverse_event_shape_tensor(
        event_shapes)
    for i in range(3):
      self.assertAllEqual(
          self.evaluate(forward_joint_event_shape[i]),
          self.evaluate(
              bijectors[i].forward_event_shape_tensor(event_shapes[i])))
      self.assertAllEqual(
          self.evaluate(inverse_joint_event_shape[i]),
          self.evaluate(bijectors[i].inverse_event_shape_tensor(
              event_shapes[i])))

  @parameterized.named_parameters(
      ('_scalar', []),
      ('_batched', [3]),
  )
  def test_default_event_space_bijector_ratio(self, sample_shape):

    def dists():
      a = yield Root(exponential.Exponential(1., validate_args=True))
      b = yield independent.Independent(
          uniform.Uniform([-1., -2.], a[..., tf.newaxis], validate_args=True),
          1)
      yield independent.Independent(
          logistic.Logistic(b, a[..., tf.newaxis], validate_args=True), 1)

    jd = jdc.JointDistributionCoroutine(dists, validate_args=True)
    joint_bijector = jd.experimental_default_event_space_bijector()

    seed1, seed2 = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'), 2)
    x1 = jd.sample(sample_shape, seed=seed1)
    x2 = jd.sample(sample_shape, seed=seed2)
    z1 = joint_bijector.inverse(x1)
    z2 = joint_bijector.inverse(x2)
    z1, z2 = tf.nest.map_structure(tf.identity, (z1, z2))

    event_ndims = tf.nest.map_structure(len, jd.event_shape)
    true_fldj_ratio = (
        joint_bijector.forward_log_det_jacobian(z1, event_ndims) -
        joint_bijector.forward_log_det_jacobian(z2, event_ndims))
    self.assertAllClose(
        true_fldj_ratio,
        ldj_ratio.forward_log_det_jacobian_ratio(joint_bijector, z1,
                                                 joint_bijector, z2,
                                                 event_ndims))
    true_ildj_ratio = (
        joint_bijector.inverse_log_det_jacobian(x1, event_ndims) -
        joint_bijector.inverse_log_det_jacobian(x2, event_ndims))
    self.assertAllClose(
        true_ildj_ratio,
        ldj_ratio.inverse_log_det_jacobian_ratio(joint_bijector, x1,
                                                 joint_bijector, x2,
                                                 event_ndims))

  @parameterized.named_parameters(
      ('_sample', lambda d, **kwargs: d.sample(**kwargs)),
      ('_sample_and_log_prob',
       lambda d, **kwargs: d.experimental_sample_and_log_prob(**kwargs)[0]),
  )
  def test_nested_partial_value(self, sample_fn):

    @jdc.JointDistributionCoroutine
    def innermost():
      a = yield Root(exponential.Exponential(1., name='a'))
      yield sample_lib.Sample(lognormal.LogNormal(a, a), [5], name='b')

    @jdc.JointDistributionCoroutine
    def inner():
      yield Root(exponential.Exponential(1., name='c'))
      yield Root(innermost.copy(name='d'))

    @jdc.JointDistributionCoroutine
    def outer():
      yield Root(exponential.Exponential(1., name='e'))
      yield Root(inner.copy(name='f'))

    seed = test_util.test_seed(sampler_type='stateless')
    true_xs = outer.sample(seed=seed)

    # These asserts work because we advance the stateless seed inside the model
    # whether or not a sample is actually generated.
    partial_xs = true_xs._replace(f=None)
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

    partial_xs = true_xs._replace(e=None)
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

    partial_xs = true_xs._replace(f=true_xs.f._replace(d=None))
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

    partial_xs = true_xs._replace(
        f=true_xs.f._replace(d=true_xs.f.d._replace(a=None)))
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

  def test_default_event_space_bijector_nested(self):

    @jdc.JointDistributionCoroutine
    def inner():
      a = yield Root(exponential.Exponential(1., name='a'))
      yield sample_lib.Sample(lognormal.LogNormal(a, a), [5], name='b')

    @jdc.JointDistributionCoroutine
    def outer():
      yield Root(inner)
      yield Root(inner)
      yield Root(inner)

    xs = outer.sample(seed=test_util.test_seed())

    outer_bij = outer.experimental_default_event_space_bijector()
    joint_ldj = outer_bij.forward_log_det_jacobian(xs, [(0, 1)] * len(xs))

    inner_bij = inner.experimental_default_event_space_bijector()
    inner_ldjs = [inner_bij.forward_log_det_jacobian(x, (0, 1)) for x in xs]

    # Evaluate both at once, to make sure we're using the same samples.
    joint_ldj_, inner_ldjs_ = self.evaluate((joint_ldj, inner_ldjs))
    self.assertAllClose(joint_ldj_, sum(inner_ldjs_))

  def test_sample_kwargs(self):

    @jdc.JointDistributionCoroutine
    def joint():
      a = yield Root(normal.Normal(0., 1., name='a'))
      b = yield normal.Normal(a, 1., name='b')
      yield normal.Normal(a + b, 1., name='c')

    seed = test_util.test_seed()
    tf.random.set_seed(seed)
    samples = joint.sample(seed=seed, a=tf.constant(1.))
    # Check the first value is actually 1.
    self.assertEqual(1., self.evaluate(samples[0]))

    # Check the sample is reproducible using the `value` argument.
    tf.random.set_seed(seed)
    samples_named = joint.sample(seed=seed, value=[1., None, None])
    self.assertAllEqual(self.evaluate(samples), self.evaluate(samples_named))

    # Make sure to throw an exception if strange keywords are passed.
    expected_error = (
        'Found unexpected keyword arguments. Distribution names are\n'
        'a, b, c\n'
        'but received\n'
        'z\n'
        'These names were invalid:\n'
        'z')
    with self.assertRaisesRegex(ValueError, expected_error):
      joint.sample(seed=seed, z=2.)

    # Raise if value and keywords are passed.
    with self.assertRaisesRegex(
        ValueError, r'Supplied both `value` and keyword arguments .*'):
      joint.sample(seed=seed, a=1., value={'a': 1})

  def test_named_dtype(self):
    """Test using names for component distributions."""

    def model_fn():
      c = yield Root(lognormal.LogNormal(0., 1., name='c'))
      b = yield normal.Normal(c, 1., name='b')
      yield normal.Normal(c + b, 1e-3, name='a')

    model = jdc.JointDistributionCoroutine(model_fn, validate_args=True)

    seed = test_util.test_seed_stream()

    sample = self.evaluate(model.sample(seed=seed()))
    sample_type = type(sample)
    # Note the order is not alphabetic, but rather than the call older in the
    # coroutine.
    self.assertEqual(['c', 'b', 'a'], list(sample._asdict().keys()))

    # Passing the sample by value or by kwargs should both work.
    lp_by_val = model.log_prob(sample)
    lp_by_args = model.log_prob(*sample)
    lp_by_kwargs = model.log_prob(**sample._asdict())
    self.assertAllEqual(lp_by_val, lp_by_kwargs)
    self.assertAllEqual(lp_by_val, lp_by_args)

    # Partially specifying the values works when sampling.
    sample_partial = model.sample(c=1., b=2., seed=seed())
    self.assertAllClose(1., sample_partial.c)
    self.assertAllClose(2., sample_partial.b)
    self.assertAllClose(3., sample_partial.a, atol=0.1)
    sample_partial = model.sample(value=sample_type(c=1., b=2.), seed=seed())
    self.assertAllClose(1., sample_partial.c)
    self.assertAllClose(2., sample_partial.b)
    self.assertAllClose(3., sample_partial.a, atol=0.1)

    # Check that the distribution properties return the expected type.
    dtype = model.dtype
    self.assertEqual(
        sample_type(a=tf.float32, b=tf.float32, c=tf.float32), dtype)

    def assert_equal_part(a, b):
      self.assertAllEqual(a, b)

    def assert_equal(a, b):
      self.assertAllAssertsNested(assert_equal_part, a, b, shallow=dtype)

    assert_equal(sample_type(a=[], b=[], c=[]), model.event_shape)
    assert_equal(sample_type(a=[], b=[], c=[]), model.event_shape_tensor())
    assert_equal(sample_type(a=[], b=[], c=[]), model.batch_shape)
    assert_equal(sample_type(a=[], b=[], c=[]), model.batch_shape_tensor())

    # Check the default bijector.
    b = model.experimental_default_event_space_bijector()
    sample2 = self.evaluate(b.forward(b.inverse(sample)))
    self.assertAllClose(sample2, sample)

    # Verify that event shapes are passed through and flattened/unflattened
    # correctly.
    forward_event_shapes = b.forward_event_shape(model.event_shape)
    inverse_event_shapes = b.inverse_event_shape(model.event_shape)
    self.assertEqual(forward_event_shapes, model.event_shape)
    self.assertEqual(inverse_event_shapes, model.event_shape)

    # Verify that the outputs of other methods have the correct structure.
    forward_event_shape_tensors = b.forward_event_shape_tensor(
        model.event_shape_tensor())
    inverse_event_shape_tensors = b.inverse_event_shape_tensor(
        model.event_shape_tensor())
    tf.nest.assert_same_structure(forward_event_shape_tensors,
                                  inverse_event_shape_tensors)

  def test_automatic_naming(self):
    """Test leaving some distributions unnamed."""

    def model_fn():
      c = yield Root(lognormal.LogNormal(0., 1., name='c'))
      b = yield normal.Normal(c, 1.)
      b = yield Root(normal.Normal(c, 1., name='a'))
      yield normal.Normal(c + b, 1.)

    model = jdc.JointDistributionCoroutine(model_fn, validate_args=True)

    sample = self.evaluate(model.sample(seed=test_util.test_seed()))
    self.assertEqual(['c', 'var1', 'a', 'var3'], list(sample._asdict().keys()))

  def test_target_log_prob_fn(self):
    """Test the construction `target_log_prob_fn` from a joint distribution."""

    def model_fn():
      c = yield Root(lognormal.LogNormal(0., 1., name='c'))
      b = yield normal.Normal(c, 1., name='b')
      yield normal.Normal(c + b, 1., name='a')

    model = jdc.JointDistributionCoroutine(model_fn, validate_args=True)

    def target_log_prob_fn(*args):
      return model.log_prob(args + (1.,))

    dtype = model.dtype[:-1]
    event_shape = model.event_shape[:-1]
    self.assertAllEqual(('c', 'b'), dtype._fields)
    self.assertAllEqual(('c', 'b'), event_shape._fields)

    test_point = tf.nest.map_structure(tf.zeros, event_shape, dtype)
    lp_manual = model.log_prob(test_point + (1.,))
    lp_tlp = nest_util.call_fn(target_log_prob_fn, test_point)

    self.assertAllClose(self.evaluate(lp_manual), self.evaluate(lp_tlp))

  @test_util.jax_disable_test_missing_functionality('stateful samplers')
  @test_util.numpy_disable_test_missing_functionality('stateful samplers')
  def test_legacy_dists(self):

    class StatefulNormal(normal.Normal):

      def _sample_n(self, n, seed=None):
        return self.loc + self.scale * tf.random.normal(
            tf.concat([[n], self.batch_shape_tensor()], axis=0),
            seed=seed)

    def dist():
      e = yield Root(
          independent.Independent(
              exponential.Exponential(rate=[100, 120]), 1, name='e'))
      loc = yield Root(StatefulNormal(loc=0, scale=2., name='loc'))
      scale = yield gamma.Gamma(
          concentration=e[..., 0], rate=e[..., 1], name='scale')
      m = yield normal.Normal(loc, scale, name='m')
      yield sample_lib.Sample(bernoulli.Bernoulli(logits=m), 12, name='x')

    d = jdc.JointDistributionCoroutine(dist, validate_args=True)

    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as w:
      d.sample(seed=test_util.test_seed())
    self.assertRegexpMatches(
        str(w[0].message),
        r'Falling back to stateful sampling for distribution #1.*'
        r'of type.*StatefulNormal.*component name "loc" and `dist.name` '
        r'".*loc"',
        msg=w)

  @test_util.jax_disable_test_missing_functionality('stateful samplers')
  @test_util.numpy_disable_test_missing_functionality('stateful samplers')
  def test_legacy_dists_stateless_seed_raises(self):

    class StatefulNormal(normal.Normal):

      def _sample_n(self, n, seed=None):
        return self.loc + self.scale * tf.random.normal(
            tf.concat([[n], self.batch_shape_tensor()], axis=0),
            seed=seed)

    def dist():
      e = yield Root(
          independent.Independent(
              exponential.Exponential(rate=[100, 120]), 1, name='e'))
      loc = yield Root(StatefulNormal(loc=0, scale=2., name='loc'))
      scale = yield gamma.Gamma(
          concentration=e[..., 0], rate=e[..., 1], name='scale')
      m = yield normal.Normal(loc, scale, name='m')
      yield sample_lib.Sample(bernoulli.Bernoulli(logits=m), 12, name='x')

    d = jdc.JointDistributionCoroutine(dist, validate_args=True)

    with self.assertRaisesRegexp(TypeError, r'Expected int for argument'):
      d.sample(seed=samplers.zeros_seed())

  def test_pinning(self):
    """Test pinning a component distribution."""

    def model_fn():
      c = yield Root(lognormal.LogNormal(0., 1., name='c'))
      b = yield normal.Normal(c, 1., name='b')
      yield normal.Normal(c + b, 1e-3, name='a')

    d = jdc.JointDistributionCoroutine(model_fn, validate_args=True)
    samp = self.evaluate(d.experimental_pin(b=1.5).sample_unpinned(
        seed=test_util.test_seed()))
    self.assertEqual(('c', 'a'), samp._fields)
    self.assertAllClose(samp.c + 1.5, samp.a, atol=3e-3)

    samp = self.evaluate(d.experimental_pin(None, .75).sample_unpinned(
        seed=test_util.test_seed()))
    self.assertAllClose(samp.c + .75, samp.a, atol=3e-3)

    samp = self.evaluate(d.experimental_pin([None, 7.5]).sample_unpinned(
        seed=test_util.test_seed()))
    self.assertAllClose(samp.c + 7.5, samp.a, atol=3e-3)

  @test_util.numpy_disable_gradient_test
  def test_unnormalized_log_prob(self):
    def model_fn():
      c1 = yield Root(gamma.Gamma(1.2, 1.3, name='c1'))
      c0 = yield Root(gamma.Gamma(1.4, 1.5, name='c0'))
      yield beta_binomial.BetaBinomial(
          concentration1=c1,
          concentration0=c0,
          total_count=100,
          name='successes')

    d = jdc.JointDistributionCoroutine(model_fn, validate_args=True)

    c1 = tf.constant(2.1)
    c0 = tf.constant(3.1)
    successes = tf.constant(30.)  # Treated as conditioning.

    def desired_unnorm_lp(c1, c0):
      c1_unnorm = tf.math.xlogy(1.2 - 1., c1) - 1.3 * c1
      c0_unnorm = tf.math.xlogy(1.4 - 1., c0) - 1.5 * c0
      bb_unnorm = (
          special.lbeta(c1 + successes, 100 + c0 - successes) -
          special.lbeta(c1, c0))
      return c1_unnorm + c0_unnorm + bb_unnorm

    self.assertAllCloseNested(
        gradient.value_and_gradient(
            lambda c1, c0: d.log_prob(c1, c0, successes), (c1, c0))[1],
        gradient.value_and_gradient(desired_unnorm_lp, (c1, c0))[1])

    # TODO(b/187925322): This portion is aspirational.
    # actual = d.unnormalized_log_prob(c1=c1, c0=c0, successes=successes)
    # self.assertAllClose(desired_unnorm_lp(c1, c0), actual)

    self.assertAllCloseNested(
        gradient.value_and_gradient(
            lambda c1, c0: d.log_prob(c1, c0, successes), (c1, c0))[1],
        gradient.value_and_gradient(
            lambda c1, c0: d.unnormalized_log_prob(c1, c0, successes),
            (c1, c0))[1])

  @test_util.numpy_disable_gradient_test
  def test_unnormalized_log_prob_trainable_prior(self):

    def model_fn(cprior):
      c1 = yield Root(gamma.Gamma(cprior, 1.3, name='c1'))
      c0 = yield Root(gamma.Gamma(cprior, 1.5, name='c0'))
      yield beta_binomial.BetaBinomial(
          concentration1=c1,
          concentration0=c0,
          total_count=100,
          name='successes')

    successes = tf.constant(30.)  # Treated as conditioning.

    def lp_fn(cprior, c1, c0):
      d = jdc.JointDistributionCoroutine(
          functools.partial(model_fn, cprior), validate_args=True)
      return d.log_prob(c1, c0, successes)

    def ulp_fn(cprior, c1, c0):
      d = jdc.JointDistributionCoroutine(
          functools.partial(model_fn, cprior), validate_args=True)
      return d.unnormalized_log_prob(c1, c0, successes)

    cprior = tf.constant(1.2)
    c1 = tf.constant(2.1)
    c0 = tf.constant(3.1)

    def desired_unnorm_lp(cprior, c1, c0):
      c1_unnorm = gamma.Gamma(cprior, 1.3).log_prob(c1)
      c0_unnorm = gamma.Gamma(cprior, 1.5).log_prob(c0)
      bb_unnorm = (
          special.lbeta(c1 + successes, 100 + c0 - successes) -
          special.lbeta(c1, c0))
      return c1_unnorm + c0_unnorm + bb_unnorm

    self.assertAllCloseNested(
        gradient.value_and_gradient(lp_fn, (cprior, c1, c0))[1],
        gradient.value_and_gradient(desired_unnorm_lp, (cprior, c1, c0))[1])

    # TODO(b/187925322): This portion is aspirational.
    # actual = d.unnormalized_log_prob(c1=c1, c0=c0, successes=successes)
    # self.assertAllClose(desired_unnorm_lp(cprior, c1, c0), actual)

    self.assertAllCloseNested(
        gradient.value_and_gradient(lp_fn, (cprior, c1, c0))[1],
        gradient.value_and_gradient(ulp_fn, (cprior, c1, c0))[1])

  @test_util.numpy_disable_test_missing_functionality('symbolic tracing')
  @test_util.jax_disable_test_missing_functionality(
      'https://github.com/google/jax/issues/7011')
  def test_symbolic_trace_dtype(self):
    # A model that will definitely OOM. (1 billion squared floats).
    @jdc.JointDistributionCoroutine
    def model():
      x = yield Root(
          mvn_diag.MultivariateNormalDiag(
              tf.zeros(int(1e9)), tf.ones(int(1e9)), name='x'))
      loc = tf.einsum('i,j->ij', x, x)
      yield independent.Independent(
          mvn_diag.MultivariateNormalDiag(loc, tf.ones(int(1e9))),
          reinterpreted_batch_ndims=1,
          name='y')
    self.assertEqual((tf.float32, tf.float32), model.dtype)

  @test_util.numpy_disable_test_missing_functionality('symbolic tracing')
  def test_symbolic_trace_is_cached(self):
    model_executions = []

    @jdc.JointDistributionCoroutine
    def model():
      x = yield Root(normal.Normal(0., 1., name='x'))
      y = yield normal.Normal(x, 1., name='y')
      model_executions.append(y)

    self.assertAllEqual(((), ()), model.event_shape)
    self.assertAllEqual(((), ()), model.batch_shape)
    self.assertAllEqual((tf.float32, tf.float32), model.dtype)
    self.assertAllEqual(('x', 'y'), model._flat_resolve_names())
    self.assertLen(model_executions, 1)

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason='Numpy and JAX have no notion of CompositeTensor.')
  def testCompositeTensor(self):
    def model_fn():
      c1 = yield Root(gamma.Gamma(1.2, 1.3, name='c1'))
      c0 = yield Root(gamma.Gamma(1.4, 1.5, name='c0'))
      yield beta_binomial.BetaBinomial(
          concentration1=c1,
          concentration0=c0,
          total_count=100,
          name='successes')

    d = jdc.JointDistributionCoroutine(model_fn, validate_args=True)

    flat = tf.nest.flatten(d, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(
        d, flat, expand_composites=True)
    self.assertIsInstance(unflat, jdc.JointDistributionCoroutine)

    x = self.evaluate(d.sample(3, seed=test_util.test_seed()))
    actual = self.evaluate(d.log_prob(x))
    self.assertAllClose(self.evaluate(unflat.log_prob(x)), actual)

    @tf.function
    def call_log_prob(d):
      return d.log_prob(x)
    self.assertAllClose(actual, call_log_prob(d))
    self.assertAllClose(actual, call_log_prob(unflat))

if __name__ == '__main__':
  test_util.main()
