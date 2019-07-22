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

# Dependency imports
from absl.testing import parameterized

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class JointDistributionSequentialTest(tf.test.TestCase, parameterized.TestCase):

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
                      in zip(d0.distribution_fn, d1.distribution_fn))
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
        in zip(d0.distribution_fn, d1.distribution_fn))
    actual_xent = d0.cross_entropy(d1)
    expected_xent_, actual_xent_ = self.evaluate([expected_xent, actual_xent])
    self.assertNear(actual_xent_, expected_xent_, err=1e-5)

  def test_norequired_args_maker(self):
    """Test that only non-default args are passed through."""
    d = tfd.JointDistributionSequential([tfd.Normal(0., 1.), tfd.Bernoulli])
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Must pass probs or logits, but not both.'):
      d.sample()

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
    expected = tuple(getattr(d_, attr)() for d_ in d.distribution_fn)
    actual = getattr(d, attr)()
    self.assertAllEqual(*self.evaluate([expected, actual]))

  @parameterized.parameters(('covariance',))
  def test_notimplemented_summary_statistic(self, attr):
    d = tfd.JointDistributionSequential([tfd.Normal(0., 1.), tfd.Bernoulli],
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
    d = tfd.JointDistributionSequential([tfd.Normal(0., 1.), tfd.Bernoulli],
                                        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionSequential'):
      getattr(d, attr)([0.]*len(d.distribution_fn))

  def test_copy(self):
    pgm = [tfd.Normal(0., 1.), tfd.Bernoulli]
    d = tfd.JointDistributionSequential(pgm, validate_args=True)
    d_copy = d.copy()
    self.assertAllEqual(
        {'distribution_fn': pgm,
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


if __name__ == '__main__':
  tf.test.main()
