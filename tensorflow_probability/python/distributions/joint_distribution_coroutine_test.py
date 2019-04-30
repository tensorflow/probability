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
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


tfd = tfp.distributions


Root = tfd.JointDistributionCoroutine.Root


@test_util.run_all_in_graph_and_eager_modes
class JointDistributionCoroutineTest(tf.test.TestCase):

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

    # Neither `event_shape` nor `batch_shape` can be determined
    # without the underlying distributions being cached.
    self.assertAllEqual(joint.event_shape, None)
    self.assertAllEqual(joint.batch_shape, None)

    ds, _ = joint.sample_distributions()
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

    self.assertAllEqual(joint.event_shape, [[], [], []])

    is_batch_scalar = joint.is_scalar_batch()
    self.assertAllEqual(is_batch_scalar[0], True)
    self.assertAllEqual(is_batch_scalar[1], True)
    self.assertAllEqual(is_batch_scalar[2], True)

    batch_shape = joint.batch_shape_tensor()
    self.assertAllEqual(batch_shape[0], [])
    self.assertAllEqual(batch_shape[1], [])
    self.assertAllEqual(batch_shape[2], [])

    self.assertAllEqual(joint.batch_shape, [[], [], []])

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

    # Neither `event_shape` nor `batch_shape` can be determined
    # without the underlying distributions being cached.
    self.assertAllEqual(joint.event_shape, None)
    self.assertAllEqual(joint.batch_shape, None)

    ds, _ = joint.sample_distributions()
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

    self.assertAllEqual(joint.event_shape, [[], [], [20], [20]])

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

    self.assertAllEqual(joint.batch_shape, [[], [], [], []])

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

    z = joint.sample()

    self.assertAllEqual(tf.shape(input=z[0]), [])
    self.assertAllEqual(tf.shape(input=z[1]), [])
    self.assertAllEqual(tf.shape(input=z[2]), [])

    z = joint.sample(2)

    self.assertAllEqual(tf.shape(input=z[0]), [2])
    self.assertAllEqual(tf.shape(input=z[1]), [2])
    self.assertAllEqual(tf.shape(input=z[2]), [2])

    z = joint.sample([3, 2])

    self.assertAllEqual(tf.shape(input=z[0]), [3, 2])
    self.assertAllEqual(tf.shape(input=z[1]), [3, 2])
    self.assertAllEqual(tf.shape(input=z[2]), [3, 2])

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

    z = joint.sample()

    self.assertAllEqual(tf.shape(input=z[0]), [])
    self.assertAllEqual(tf.shape(input=z[1]), [])
    self.assertAllEqual(tf.shape(input=z[2]), [20])
    self.assertAllEqual(tf.shape(input=z[3]), [20])

    z = joint.sample(2)

    self.assertAllEqual(tf.shape(input=z[0]), [2])
    self.assertAllEqual(tf.shape(input=z[1]), [2])
    self.assertAllEqual(tf.shape(input=z[2]), [2, 20])
    self.assertAllEqual(tf.shape(input=z[3]), [2, 20])

    z = joint.sample([3, 2])

    self.assertAllEqual(tf.shape(input=z[0]), [3, 2])
    self.assertAllEqual(tf.shape(input=z[1]), [3, 2])
    self.assertAllEqual(tf.shape(input=z[2]), [3, 2, 20])
    self.assertAllEqual(tf.shape(input=z[3]), [3, 2, 20])

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

    z = joint.sample()

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

    z = joint.sample()
    a, b, c = z  # pylint: disable=unbalanced-tuple-unpacking

    log_prob = joint.log_prob(z)

    expected_log_prob = (
        np.log(0.5) +
        tf.reduce_sum(input_tensor=tf.math.log(b * (0.25 + 0.5 * a) +
                                               (1 - b) * (0.75 -0.5 * a))) +
        tf.reduce_sum(input_tensor=-0.5 * ((c - a) / (1. + b)) ** 2 -
                      0.5 * np.log(2. * np.pi) -
                      tf.math.log((1. + b))))

    self.assertAllClose(*self.evaluate([log_prob, expected_log_prob]))

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

    z = joint.sample(4)

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

    z = joint.sample(4)

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

    z = joint.sample()

    a, b, c, d = z  # pylint: disable=unbalanced-tuple-unpacking

    expected_log_prob = (- a +
                         tf.reduce_sum(
                             input_tensor=
                             -0.5 * (b - a) ** 2 +
                             -0.5 * np.log(2. * np.pi),
                             axis=-1) -
                         c +
                         tf.reduce_sum(
                             input_tensor=
                             -0.5 * (d - b) ** 2 / c ** 2 +
                             -0.5 * tf.math.log(2. * np.pi * c ** 2),
                             axis=-1))

    log_prob = joint.log_prob(z)

    self.assertAllClose(*self.evaluate([log_prob, expected_log_prob]))

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

    z = joint.sample([3, 5])

    a, b, c, d = z  # pylint: disable=unbalanced-tuple-unpacking

    expanded_c = tf.expand_dims(c, -1)
    expected_log_prob = (
        - a +
        tf.reduce_sum(
            input_tensor=
            -0.5 * (b - tf.expand_dims(a, -1)) ** 2 +
            -0.5 * np.log(2. * np.pi),
            axis=-1) -
        c +
        tf.reduce_sum(
            input_tensor=
            -0.5 * (d - b) ** 2 / expanded_c ** 2 +
            -0.5 * tf.math.log(2. * np.pi * expanded_c ** 2),
            axis=-1))

    log_prob = joint.log_prob(z)

    self.assertAllClose(*self.evaluate([log_prob, expected_log_prob]))

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
                        sorted(joint.sample()._fields))
    ds, xs = joint.sample_distributions([2, 3])
    tf.nest.assert_same_structure(sample_dtype, ds)
    tf.nest.assert_same_structure(sample_dtype, xs)
    self.assertEqual([3, 4], joint.log_prob(joint.sample([3, 4])).shape)

  def test_repr_with_custom_sample_dtype(self):
    def model():
      s = yield tfd.JointDistributionCoroutine.Root(
          tfd.Sample(tfd.InverseGamma(2, 2), 100))
      yield tfd.Independent(tfd.Normal(0, s), 1)
    sd = collections.namedtuple('Model', ['s', 'w'])(None, None)
    m = tfd.JointDistributionCoroutine(model, sample_dtype=sd)
    self.assertEqual(
        ('tfp.distributions.JointDistributionCoroutine('
         '"JointDistributionCoroutine/",'
         ' dtype=Model(s=?, w=?))'),
        str(m))
    self.assertEqual(
        ('<tfp.distributions.JointDistributionCoroutine'
         ' \'JointDistributionCoroutine/\''
         ' batch_shape=?'
         ' event_shape=?'
         ' dtype=Model(s=?, w=?)>'),
        repr(m))
    m.sample()
    self.assertEqual(
        ('tfp.distributions.JointDistributionCoroutine('
         '"JointDistributionCoroutine/",'
         ' batch_shape=Model(s=[], w=[]),'
         ' event_shape=Model(s=[100], w=[100]),'
         ' dtype=Model(s=float32, w=float32))'),
        str(m))
    self.assertEqual(
        ('<tfp.distributions.JointDistributionCoroutine'
         ' \'JointDistributionCoroutine/\''
         ' batch_shape=Model(s=[], w=[])'
         ' event_shape=Model(s=[100], w=[100])'
         ' dtype=Model(s=float32, w=float32)>'),
        repr(m))


if __name__ == '__main__':
  tf.test.main()
