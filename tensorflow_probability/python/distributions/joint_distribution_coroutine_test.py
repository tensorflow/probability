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


Root = tfd.JointDistributionCoroutine.Root


def basic_model_with_names_fn():
  yield Root(tfd.Normal(0., 1., name='a'))
  e = yield Root(tfd.Independent(
      tfd.Exponential(rate=[100, 120]),
      reinterpreted_batch_ndims=1, name='e'))
  yield tfd.Gamma(concentration=e[..., 0], rate=e[..., 1], name='x')


def nested_lists_with_names_model_fn():
  abc = yield Root(tfd.JointDistributionSequential([
      tfd.MultivariateNormalDiag([0., 0.], [1., 1.]),
      tfd.JointDistributionSequential([
          tfd.StudentT(3., -2., 5.),
          tfd.Exponential(4.)])], name='abc'))
  a, (b, c) = abc
  yield tfd.JointDistributionSequential([tfd.Normal(a * b, c),
                                         tfd.Normal(a + b, c)], name='de')


def singleton_normal_model_fn():
  yield Root(tfd.Normal(0., 1., name='x'))


def singleton_jds_model_fn():
  yield Root(tfd.JointDistributionSequential(
      [tfd.Normal(0., 1.), lambda x: tfd.Poisson(tf.exp(x))], name='x'))


def singleton_jdn_model_fn():
  yield Root(tfd.JointDistributionNamed(
      {'z': tfd.Normal(0., 1.), 'y': lambda z: tfd.Poisson(tf.exp(z))},
      name='x'))


@test_util.test_all_tf_execution_regimes
class JointDistributionCoroutineTest(test_util.TestCase):

  def test_batch_and_event_shape_no_plate(self):
    # The joint distribution specified below corresponds to this
    # graphical model
    #
    #  (a)-->--(b)
    #    \      |
    #     \     v
    #      `->-(c)

    def dist():
      a = yield Root(tfd.Bernoulli(probs=0.5,
                                   dtype=tf.float32))
      b = yield tfd.Bernoulli(probs=0.25 + 0.5*a,
                              dtype=tf.float32)
      yield tfd.Normal(loc=a, scale=1. + b)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

    # Properties `event_shape` and `batch_shape` should be defined
    # even before any sampling calls have occurred.
    self.assertAllEqual(joint.event_shape, [[], [], []])
    self.assertAllEqual(joint.batch_shape, [[], [], []])

    ds, _ = joint.sample_distributions(seed=test_util.test_seed())
    self.assertLen(ds, 3)
    self.assertIsInstance(ds[0], tfd.Bernoulli)
    self.assertIsInstance(ds[1], tfd.Bernoulli)
    self.assertIsInstance(ds[2], tfd.Normal)

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

  def test_batch_and_event_shape_with_plate(self):
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
      g = yield Root(tfd.Gamma(2, 2))
      df = yield Root(tfd.Exponential(1.))
      loc = yield tfd.Sample(tfd.Normal(0, g), 20)
      yield tfd.Independent(tfd.StudentT(tf.expand_dims(df, -1), loc, 1), 1)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

    # Properties `event_shape` and `batch_shape` should be defined
    # even before any sampling calls have occurred.
    self.assertAllEqual(joint.event_shape, [[], [], [20], [20]])
    self.assertAllEqual(joint.batch_shape, [[], [], [], []])

    ds, _ = joint.sample_distributions(seed=test_util.test_seed())
    self.assertLen(ds, 4)
    self.assertIsInstance(ds[0], tfd.Gamma)
    self.assertIsInstance(ds[1], tfd.Exponential)
    self.assertIsInstance(ds[2], tfd.Sample)
    self.assertIsInstance(ds[3], tfd.Independent)

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
      a = yield Root(tfd.Bernoulli(probs=0.5,
                                   dtype=tf.float32))
      b = yield tfd.Bernoulli(probs=0.25 + 0.5*a,
                              dtype=tf.float32)
      yield tfd.Normal(loc=a, scale=1. + b)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
      g = yield Root(tfd.Gamma(2, 2))
      df = yield Root(tfd.Exponential(1.))
      loc = yield tfd.Sample(tfd.Normal(0, g), 20)
      yield tfd.Independent(tfd.StudentT(tf.expand_dims(df, -1), loc, 1), 1)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
      a = yield Root(tfd.Bernoulli(probs=0.5,
                                   dtype=tf.float32))
      b = yield tfd.Bernoulli(probs=0.25 + 0.5*a,
                              dtype=tf.float32)
      yield tfd.Normal(loc=a, scale=1. + b)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
      a = yield Root(tfd.Bernoulli(probs=0.5,
                                   dtype=tf.float32))
      b = yield tfd.Sample(tfd.Bernoulli(probs=0.25 + 0.5*a,
                                         dtype=tf.float32), 2)
      yield tfd.Independent(tfd.Normal(loc=a, scale=1. + b), 1)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
      a = yield tfd.Bernoulli(probs=0.5, dtype=tf.float32)  # Missing root
      b = yield tfd.Sample(tfd.Bernoulli(probs=0.25 + 0.5*a,
                                         dtype=tf.float32), 2)
      yield tfd.Independent(tfd.Normal(loc=a, scale=1. + b), 1)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

    with self.assertRaisesRegexp(
        Exception,
        'must be wrapped in `Root`'):
      self.evaluate(joint.sample(seed=test_util.test_seed()))

  @parameterized.named_parameters(
      ('basic', basic_model_with_names_fn),
      ('nested_lists', nested_lists_with_names_model_fn))
  def test_can_call_log_prob_with_args_and_kwargs(self, model_fn):
    d = tfd.JointDistributionCoroutine(model_fn, validate_args=True)

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

    with self.assertRaisesRegexp(TypeError, 'unexpected keyword argument'):
      d.log_prob(*value, extra_arg=27.)

  @parameterized.named_parameters(
      ('singleton_float', singleton_normal_model_fn),
      ('singleton_tuple', singleton_jds_model_fn),
      ('singleton_dict', singleton_jdn_model_fn))
  def test_singleton_model_works_with_args_and_kwargs(self, model_fn):
    d = tfd.JointDistributionCoroutine(model_fn)

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
      g = yield Root(tfd.Gamma(2, 2))
      # The following line lacks a `Root` so that if a shape [3, 5]
      # sample is requested the distribution below will produce
      # samples that have too low a rank to be consistent with
      # the shape.
      df = yield tfd.Exponential(1.)
      loc = yield tfd.Sample(tfd.Normal(0, g), 20)
      yield tfd.Independent(tfd.StudentT(tf.expand_dims(df, -1), loc, 1), 1)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

    with self.assertRaisesRegexp(
        Exception,
        'are not consistent with `sample_shape`'):
      self.evaluate(joint.sample([3, 5], seed=test_util.test_seed()))

  def test_check_sample_shape(self):
    def dist():
      g = yield Root(tfd.Gamma(2, 2))
      # The following line lacks a `Root` so that if a shape [3, 5]
      # sample is requested the following line will yield samples
      # with an appropriate rank but whose shape starts with [2, 2]
      # rather than [3, 5].
      df = yield tfd.Exponential([[1., 2.], [3., 4.]])
      loc = yield tfd.Sample(tfd.Normal(0, g), 20)
      yield tfd.Independent(tfd.StudentT(tf.expand_dims(df, -1), loc, 1), 1)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
      a = yield Root(tfd.Bernoulli(probs=0.5,
                                   dtype=tf.float32))
      b = yield tfd.Bernoulli(probs=0.25 + 0.5*a,
                              dtype=tf.float32)
      yield tfd.Normal(loc=a, scale=1. + b)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
      a = yield Root(tfd.Bernoulli(probs=0.5,
                                   dtype=tf.float32))
      b = yield tfd.Bernoulli(probs=0.25 + 0.5*a,
                              dtype=tf.float32)
      yield tfd.Normal(loc=a, scale=1. + b)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
      a = yield Root(tfd.Exponential(1.))
      b = yield tfd.Sample(tfd.Normal(a, 1.), 20)
      c = yield Root(tfd.Exponential(1.))
      yield tfd.Independent(tfd.Normal(b, tf.expand_dims(c, -1)), 1)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
      a = yield Root(tfd.Exponential(1.))
      b = yield tfd.Sample(tfd.Normal(a, 1.), 20)
      c = yield Root(tfd.Exponential(1.))
      yield tfd.Independent(tfd.Normal(b, tf.expand_dims(c, -1)), 1)

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
      scale_variance = yield Root(
          tfd.InverseGamma(0.5, 0.5))
      scale_noncentered = yield Root(
          tfd.Sample(tfd.HalfNormal(1.), num_features))
      scale = scale_noncentered * scale_variance[..., None]**0.5
      weights_noncentered = yield Root(
          tfd.Sample(tfd.Normal(0., 1.), num_features))
      yield tfd.Independent(tfd.Deterministic(weights_noncentered * scale),
                            reinterpreted_batch_ndims=1)
    # Currently sample_dtype is only used for `tf.nest.pack_structure_as`. In
    # the future we may use it for error checking and/or casting.
    sample_dtype = collections.namedtuple('Model', [
        'scale_variance',
        'scale_noncentered',
        'weights_noncentered',
        'weights',
    ])(*([None]*4))
    joint = tfd.JointDistributionCoroutine(
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

  def test_repr_with_custom_sample_dtype(self):
    def model():
      s = yield tfd.JointDistributionCoroutine.Root(
          tfd.Sample(tfd.InverseGamma(2, 2), 100))
      yield tfd.Independent(tfd.Normal(0, s), 1)
    sd = collections.namedtuple('Model', ['s', 'w'])(None, None)
    m = tfd.JointDistributionCoroutine(
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
      a = yield Root(tfd.MultivariateNormalDiag([0., 0.], [1., 1.]))
      yield tfd.JointDistributionSequential([
          tfd.JointDistributionSequential([
              tfd.Normal(a[..., 0], 1.)]),
          tfd.Normal(a[..., 1], 1.)])
    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
          tfd.Sample(tfd.Normal(loc=0.,
                                scale=user_trait_prior_scale),
                     sample_shape=[n_factors, n_users]))

      # V in paper
      item_traits = yield Root(
          tfd.Sample(tfd.Normal(loc=0.,
                                scale=item_trait_prior_scale),
                     sample_shape=[n_factors, n_items]))

      # R in paper
      yield tfd.Independent(
          tfd.Normal(loc=tf.matmul(user_traits, item_traits,
                                   adjoint_a=True),
                     scale=observation_noise_prior_scale),
          reinterpreted_batch_ndims=2)
    dist = tfd.JointDistributionCoroutine(model)
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
    alpha = tfp.util.TransformedVariable(
        u.sample([num_topics]), tfb.Softplus(), name='alpha')
    beta = tf.Variable(u.sample([num_topics, num_words]), name='beta')

    # LDA Model.
    # Note near 1:1 with mathematical specification. The main distinction is the
    # use of Independent--this lets us easily aggregate multinomials across
    # topics (and in any "shape" of documents).
    def lda_model():
      n = yield Root(tfd.Poisson(rate=avg_doc_length))
      theta = yield Root(tfd.Dirichlet(concentration=alpha))
      z = yield tfd.Multinomial(total_count=n, probs=theta)
      yield tfd.Independent(tfd.Multinomial(total_count=z, logits=beta),
                            reinterpreted_batch_ndims=1)

    lda = tfd.JointDistributionCoroutine(lda_model, validate_args=True)

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

  @test_util.jax_disable_test_missing_functionality(
      'Graph tensors are unsupported in JAX backend.')
  def test_cache_doesnt_leak_graph_tensors(self):
    if not tf.executing_eagerly():
      return

    def dist():
      random_rank = tf.cast(3.5 + tf.random.uniform(
          [], seed=test_util.test_seed()), tf.int32)
      yield Root(tfd.Normal(loc=0., scale=tf.ones([random_rank])))

    joint = tfd.JointDistributionCoroutine(dist, validate_args=True)

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
      a = yield Root(tfd.Exponential(1., validate_args=True))
      b = yield tfd.Independent(
          tfd.Uniform([-1., -2.], a, validate_args=True))
      yield tfd.Logistic(b, a, validate_args=True)

    jd = tfd.JointDistributionCoroutine(dists, validate_args=True)
    joint_bijector = jd._experimental_default_event_space_bijector()

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
          b = d._experimental_default_event_space_bijector()
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
    xs = [tf.constant(w) for w in [0.2, [-1.3, 0.1], -2.]]
    bijectors, ys = _get_support_bijectors(dists, xs=xs)

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
    self.assertAllEqual(
        [shape.as_list()
         for shape in joint_bijector.forward_event_shape(event_shapes)],
        [bijectors[i].forward_event_shape(event_shapes[i]).as_list()
         for i in range(3)])
    self.assertAllEqual(
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
          self.evaluate(
              bijectors[i].inverse_event_shape_tensor(event_shapes[i])))

if __name__ == '__main__':
  tf.test.main()
