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
"""Tests for the JointDistributionNamed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


# Defer creating test dists (by hiding them in functions) until we know what
# execution regime (eager/graph/tf-function) the test will run under.
def basic_ordered_model_fn():
  return collections.OrderedDict(
      (('a', tfd.Normal(0., 1.)),
       ('e',
        tfd.Independent(tfd.Exponential(rate=[100, 120]),
                        reinterpreted_batch_ndims=1)),
       ('x', lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1])),
      ))


def nested_lists_model_fn():
  return collections.OrderedDict((
      ('abc', tfd.JointDistributionSequential([
          tfd.MultivariateNormalDiag([0., 0.], [1., 1.]),
          tfd.JointDistributionSequential(
              [tfd.StudentT(3., -2., 5.),
               tfd.Exponential(4.)])])),
      ('de', lambda abc: tfd.JointDistributionSequential([  # pylint: disable=g-long-lambda
          tfd.Normal(abc[0] * abc[1][0], abc[1][1]),
          tfd.Normal(abc[0] + abc[1][0], abc[1][1])]))))


@test_util.test_all_tf_execution_regimes
class JointDistributionNamedTest(test_util.TestCase):

  def test_dict_sample_log_prob(self):
    # pylint: disable=bad-whitespace
    d = tfd.JointDistributionNamed(dict(
        e    =          tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
        scale=lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        loc  =          tfd.Normal(loc=0, scale=2.),
        m    =          tfd.Normal,
        x    =lambda m: tfd.Sample(tfd.Bernoulli(logits=m), 12)),
                                   validate_args=True)
    # pylint: enable=bad-whitespace

    self.assertEqual(
        (
            ('e', ()),
            ('scale', ('e',)),
            ('loc', ()),
            ('m', ('loc', 'scale')),
            ('x', ('m',)),
        ),
        d.resolve_graph())

    xs = d.sample(seed=test_util.test_seed())
    self.assertLen(xs, 5)
    # We'll verify the shapes work as intended when we plumb these back into the
    # respective log_probs.

    ds, _ = d.sample_distributions(value=xs, seed=test_util.test_seed())
    self.assertLen(ds, 5)
    self.assertIsInstance(ds['e'], tfd.Independent)
    self.assertIsInstance(ds['scale'], tfd.Gamma)
    self.assertIsInstance(ds['loc'], tfd.Normal)
    self.assertIsInstance(ds['m'], tfd.Normal)
    self.assertIsInstance(ds['x'], tfd.Sample)

    # Static properties.
    self.assertAllEqual(
        {'e': tf.float32, 'scale': tf.float32, 'loc': tf.float32,
         'm': tf.float32, 'x': tf.int32},
        d.dtype)

    batch_shape_tensor_, event_shape_tensor_ = self.evaluate([
        d.batch_shape_tensor(), d.event_shape_tensor()])

    expected_batch_shape = {
        'e': [], 'scale': [], 'loc': [], 'm': [], 'x': []}
    batch_tensorshape = d.batch_shape
    for k in expected_batch_shape:
      self.assertAllEqual(expected_batch_shape[k], batch_tensorshape[k])
      self.assertAllEqual(expected_batch_shape[k], batch_shape_tensor_[k])

    expected_event_shape = {
        'e': [2], 'scale': [], 'loc': [], 'm': [], 'x': [12]}
    event_tensorshape = d.event_shape
    for k in expected_event_shape:
      self.assertAllEqual(expected_event_shape[k], event_tensorshape[k])
      self.assertAllEqual(expected_event_shape[k], event_shape_tensor_[k])

    expected_jlp = sum(ds[k].log_prob(xs[k]) for k in ds.keys())
    actual_jlp = d.log_prob(xs)
    self.assertAllClose(*self.evaluate([expected_jlp, actual_jlp]),
                        atol=0., rtol=1e-4)

  def test_namedtuple_sample_log_prob(self):
    Model = collections.namedtuple('Model', ['e', 'scale', 'loc', 'm', 'x'])  # pylint: disable=invalid-name
    # pylint: disable=bad-whitespace
    model = Model(
        e    =          tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
        scale=lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        loc  =          tfd.Normal(loc=0, scale=2.),
        m    =          tfd.Normal,
        x    =lambda m: tfd.Sample(tfd.Bernoulli(logits=m), 12))
    # pylint: enable=bad-whitespace
    d = tfd.JointDistributionNamed(model, validate_args=True)

    self.assertEqual(
        (
            ('e', ()),
            ('scale', ('e',)),
            ('loc', ()),
            ('m', ('loc', 'scale')),
            ('x', ('m',)),
        ),
        d.resolve_graph())

    xs = d.sample(seed=test_util.test_seed())
    self.assertLen(xs, 5)
    # We'll verify the shapes work as intended when we plumb these back into the
    # respective log_probs.

    ds, _ = d.sample_distributions(value=xs, seed=test_util.test_seed())
    self.assertLen(ds, 5)
    self.assertIsInstance(ds.e, tfd.Independent)
    self.assertIsInstance(ds.scale, tfd.Gamma)
    self.assertIsInstance(ds.loc, tfd.Normal)
    self.assertIsInstance(ds.m, tfd.Normal)
    self.assertIsInstance(ds.x, tfd.Sample)

    # Static properties.
    self.assertAllEqual(Model(e=tf.float32, scale=tf.float32, loc=tf.float32,
                              m=tf.float32, x=tf.int32),
                        d.dtype)

    batch_shape_tensor_, event_shape_tensor_ = self.evaluate([
        d.batch_shape_tensor(), d.event_shape_tensor()])

    expected_batch_shape = Model(e=[], scale=[], loc=[], m=[], x=[])
    for (expected, actual_tensorshape, actual_shape_tensor_) in zip(
        expected_batch_shape, d.batch_shape, batch_shape_tensor_):
      self.assertAllEqual(expected, actual_tensorshape)
      self.assertAllEqual(expected, actual_shape_tensor_)

    expected_event_shape = Model(e=[2], scale=[], loc=[], m=[], x=[12])
    for (expected, actual_tensorshape, actual_shape_tensor_) in zip(
        expected_event_shape, d.event_shape, event_shape_tensor_):
      self.assertAllEqual(expected, actual_tensorshape)
      self.assertAllEqual(expected, actual_shape_tensor_)

    expected_jlp = sum(d.log_prob(x) for d, x in zip(ds, xs))
    actual_jlp = d.log_prob(xs)
    self.assertAllClose(*self.evaluate([expected_jlp, actual_jlp]),
                        atol=0., rtol=1e-4)

  def test_ordereddict_sample_log_prob(self):
    build_ordereddict = lambda e, scale, loc, m, x: collections.OrderedDict([  # pylint: disable=g-long-lambda
        ('e', e), ('scale', scale), ('loc', loc), ('m', m), ('x', x)])

    # pylint: disable=bad-whitespace
    model = build_ordereddict(
        e    =          tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
        scale=lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        loc  =          tfd.Normal(loc=0, scale=2.),
        m    =          tfd.Normal,
        x    =lambda m: tfd.Sample(tfd.Bernoulli(logits=m), 12))
    # pylint: enable=bad-whitespace
    d = tfd.JointDistributionNamed(model, validate_args=True)

    self.assertEqual(
        (
            ('e', ()),
            ('scale', ('e',)),
            ('loc', ()),
            ('m', ('loc', 'scale')),
            ('x', ('m',)),
        ),
        d.resolve_graph())

    xs = d.sample(seed=test_util.test_seed())
    self.assertLen(xs, 5)
    # We'll verify the shapes work as intended when we plumb these back into the
    # respective log_probs.

    ds, _ = d.sample_distributions(value=xs, seed=test_util.test_seed())
    self.assertLen(ds, 5)
    values = tuple(ds.values())
    self.assertIsInstance(values[0], tfd.Independent)
    self.assertIsInstance(values[1], tfd.Gamma)
    self.assertIsInstance(values[2], tfd.Normal)
    self.assertIsInstance(values[3], tfd.Normal)
    self.assertIsInstance(values[4], tfd.Sample)

    # Static properties.
    self.assertAllEqual(build_ordereddict(
        e=tf.float32, scale=tf.float32, loc=tf.float32,
        m=tf.float32, x=tf.int32), d.dtype)

    batch_shape_tensor_, event_shape_tensor_ = self.evaluate([
        d.batch_shape_tensor(), d.event_shape_tensor()])

    expected_batch_shape = build_ordereddict(e=[], scale=[], loc=[], m=[], x=[])
    for (expected, actual_tensorshape, actual_shape_tensor_) in zip(
        expected_batch_shape, d.batch_shape, batch_shape_tensor_):
      self.assertAllEqual(expected, actual_tensorshape)
      self.assertAllEqual(expected, actual_shape_tensor_)

    expected_event_shape = build_ordereddict(
        e=[2], scale=[], loc=[], m=[], x=[12])
    for (expected, actual_tensorshape, actual_shape_tensor_) in zip(
        expected_event_shape, d.event_shape, event_shape_tensor_):
      self.assertAllEqual(expected, actual_tensorshape)
      self.assertAllEqual(expected, actual_shape_tensor_)

    expected_jlp = sum(d.log_prob(x) for d, x in zip(ds.values(), xs.values()))
    actual_jlp = d.log_prob(xs)
    self.assertAllClose(*self.evaluate([expected_jlp, actual_jlp]),
                        atol=0., rtol=1e-4)

  def test_can_call_log_prob_with_kwargs(self):

    d = tfd.JointDistributionNamed({
        'e': tfd.Normal(0., 1.),
        'a': tfd.Independent(
            tfd.Exponential(rate=[100, 120]),
            reinterpreted_batch_ndims=1),
        'x': lambda a: tfd.Gamma(concentration=a[..., 0], rate=a[..., 1])
    }, validate_args=True)

    sample = self.evaluate(d.sample([2, 3], seed=test_util.test_seed()))
    e, a, x = sample['e'], sample['a'], sample['x']

    lp_value_positional = self.evaluate(d.log_prob({'e': e, 'a': a, 'x': x}))
    lp_value_named = self.evaluate(d.log_prob(value={'e': e, 'a': a, 'x': x}))
    # Assert all close (rather than equal) because order is not defined for
    # dicts, and reordering the computation can give subtly different results.
    self.assertAllClose(lp_value_positional, lp_value_named)

    lp_kwargs = self.evaluate(d.log_prob(a=a, e=e, x=x))
    self.assertAllClose(lp_value_positional, lp_kwargs)

    with self.assertRaisesRegexp(ValueError,
                                 'Joint distribution with unordered variables '
                                 "can't take positional args"):
      lp_kwargs = d.log_prob(e, a, x)

  @parameterized.named_parameters(
      ('basic', basic_ordered_model_fn),
      ('nested_lists', nested_lists_model_fn))
  def test_can_call_ordereddict_log_prob_with_args_and_kwargs(self, model_fn):
    # With an OrderedDict, we can pass keyword and/or positional args.
    d = tfd.JointDistributionNamed(model_fn(), validate_args=True)

    # Destructure vector-valued Tensors into Python lists, to mimic the values
    # a user might type.
    def _convert_ndarray_to_list(x):
      if isinstance(x, np.ndarray) and x.ndim > 0:
        return list(x)
      return x
    sample = tf.nest.map_structure(
        _convert_ndarray_to_list,
        self.evaluate(d.sample(seed=test_util.test_seed())))
    sample_dict = dict(sample)

    lp_value_positional = self.evaluate(d.log_prob(sample_dict))
    lp_value_named = self.evaluate(d.log_prob(value=sample_dict))
    self.assertAllClose(lp_value_positional, lp_value_named)

    lp_args = self.evaluate(d.log_prob(*sample.values()))
    self.assertAllClose(lp_value_positional, lp_args)

    lp_kwargs = self.evaluate(d.log_prob(**sample_dict))
    self.assertAllClose(lp_value_positional, lp_kwargs)

    lp_args_then_kwargs = self.evaluate(d.log_prob(
        *list(sample.values())[:1], **dict(list(sample.items())[1:])))
    self.assertAllClose(lp_value_positional, lp_args_then_kwargs)

  def test_can_call_namedtuple_log_prob_with_args_and_kwargs(self):
    # With an namedtuple, we can pass keyword and/or positional args.
    Model = collections.namedtuple('Model', ['e', 'a', 'x'])  # pylint: disable=invalid-name
    d = tfd.JointDistributionNamed(
        Model(e=tfd.Normal(0., 1.),
              a=tfd.Independent(
                  tfd.Exponential(rate=[100, 120]),
                  reinterpreted_batch_ndims=1),
              x=lambda a: tfd.Gamma(concentration=a[..., 0], rate=a[..., 1])),
        validate_args=True)

    sample = self.evaluate(d.sample([2, 3], seed=test_util.test_seed()))
    e, a, x = sample.e, sample.a, sample.x

    lp_value_positional = self.evaluate(d.log_prob(Model(e=e, a=a, x=x)))
    lp_value_named = self.evaluate(d.log_prob(value=Model(e=e, a=a, x=x)))
    self.assertAllClose(lp_value_positional, lp_value_named)

    lp_kwargs = self.evaluate(d.log_prob(e=e, a=a, x=x))
    self.assertAllClose(lp_value_positional, lp_kwargs)

    lp_args = self.evaluate(d.log_prob(e, a, x))
    self.assertAllClose(lp_value_positional, lp_args)

    lp_args_then_kwargs = self.evaluate(d.log_prob(e, a=a, x=x))
    self.assertAllClose(lp_value_positional, lp_args_then_kwargs)

  def test_kl_divergence(self):
    d0 = tfd.JointDistributionNamed(
        dict(e=tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
             x=tfd.Normal(loc=0, scale=2.)),
        validate_args=True)
    d1 = tfd.JointDistributionNamed(
        dict(e=tfd.Independent(tfd.Exponential(rate=[10, 12]), 1),
             x=tfd.Normal(loc=1, scale=1.)),
        validate_args=True)
    self.assertEqual(d0.model.keys(), d1.model.keys())
    expected_kl = sum(tfd.kl_divergence(d0.model[k], d1.model[k])
                      for k in d0.model.keys())
    actual_kl = tfd.kl_divergence(d0, d1)
    other_actual_kl = d0.kl_divergence(d1)
    expected_kl_, actual_kl_, other_actual_kl_ = self.evaluate([
        expected_kl, actual_kl, other_actual_kl])
    self.assertNear(expected_kl_, actual_kl_, err=1e-5)
    self.assertNear(expected_kl_, other_actual_kl_, err=1e-5)

  def test_cross_entropy(self):
    d0 = tfd.JointDistributionNamed(
        dict(e=tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
             x=tfd.Normal(loc=0, scale=2.)),
        validate_args=True)
    d1 = tfd.JointDistributionNamed(
        dict(e=tfd.Independent(tfd.Exponential(rate=[10, 12]), 1),
             x=tfd.Normal(loc=1, scale=1.)),
        validate_args=True)
    self.assertEqual(d0.model.keys(), d1.model.keys())
    expected_xent = sum(d0.model[k].cross_entropy(d1.model[k])
                        for k in d0.model.keys())
    actual_xent = d0.cross_entropy(d1)
    expected_xent_, actual_xent_ = self.evaluate([expected_xent, actual_xent])
    self.assertNear(actual_xent_, expected_xent_, err=1e-5)

  def test_norequired_args_maker(self):
    """Test that only non-default args are passed through."""
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Must pass probs or logits, but not both.'):
      tfd.JointDistributionNamed(dict(logits=tfd.Normal(0., 1.),
                                      x=tfd.Bernoulli))

  def test_graph_resolution(self):
    # pylint: disable=bad-whitespace
    d = tfd.JointDistributionNamed(dict(
        e    =          tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
        scale=lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        s    =          tfd.HalfNormal(2.5),
        loc  =lambda s: tfd.Normal(loc=0, scale=s),
        df   =          tfd.Exponential(2),
        x    =          tfd.StudentT),
                                   validate_args=True)
    # pylint: enable=bad-whitespace
    self.assertEqual(
        (
            ('e', ()),
            ('scale', ('e',)),
            ('s', ()),
            ('loc', ('s',)),
            ('df', ()),
            ('x', ('df', 'loc', 'scale'))
        ),
        d.resolve_graph())

  @parameterized.parameters('mean', 'mode', 'stddev', 'variance')
  def test_summary_statistic(self, attr):
    d = tfd.JointDistributionNamed(dict(logits=tfd.Normal(0., 1.),
                                        x=tfd.Bernoulli(logits=0.)),
                                   validate_args=True)
    expected = {k: getattr(d.model[k], attr)() for k in d.model.keys()}
    actual = getattr(d, attr)()
    self.assertAllEqual(*self.evaluate([expected, actual]))

  @parameterized.parameters(('covariance',))
  def test_notimplemented_summary_statistic(self, attr):
    d = tfd.JointDistributionNamed(dict(logits=tfd.Normal(0., 1.),
                                        x=tfd.Bernoulli(probs=0.5)),
                                   validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionNamed'):
      getattr(d, attr)()

  @parameterized.parameters(
      'log_cdf', 'cdf', 'log_survival_function', 'survival_function')
  def test_notimplemented_evaluative_statistic(self, attr):
    d = tfd.JointDistributionNamed(dict(logits=tfd.Normal(0., 1.),
                                        x=tfd.Bernoulli(probs=0.5)),
                                   validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionNamed'):
      getattr(d, attr)(dict(logits=0., x=0.5))

  def test_notimplemented_quantile(self):
    d = tfd.JointDistributionNamed(dict(logits=tfd.Normal(0., 1.),
                                        x=tfd.Bernoulli(probs=0.5)),
                                   validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        'quantile is not implemented: JointDistributionNamed'):
      d.quantile(0.5)

  def test_copy(self):
    pgm = dict(logits=tfd.Normal(0., 1.), probs=tfd.Bernoulli(logits=0.5))
    d = tfd.JointDistributionNamed(pgm, validate_args=True)
    d_copy = d.copy()
    self.assertAllEqual(
        {'model': pgm,
         'validate_args': True,
         'name': 'JointDistributionNamed'},
        d_copy.parameters)

  def test_batch_slicing(self):
    # pylint: disable=bad-whitespace
    d = tfd.JointDistributionNamed(
        dict(s=          tfd.Exponential(rate=[10, 12, 14]),
             n=lambda s: tfd.Normal(loc=0, scale=s),
             x=lambda:   tfd.Beta(concentration0=[3, 2, 1], concentration1=1)),
        validate_args=True)
    # pylint: enable=bad-whitespace

    d0, d1 = d[:1], d[1:]
    x0 = d0.sample(seed=test_util.test_seed())
    x1 = d1.sample(seed=test_util.test_seed())

    self.assertLen(x0, 3)
    self.assertEqual([1], x0['s'].shape)
    self.assertEqual([1], x0['n'].shape)
    self.assertEqual([1], x0['x'].shape)

    self.assertLen(x1, 3)
    self.assertEqual([2], x1['s'].shape)
    self.assertEqual([2], x1['n'].shape)
    self.assertEqual([2], x1['x'].shape)

  def test_sample_shape_propagation_default_behavior(self):
    # pylint: disable=bad-whitespace
    d = tfd.JointDistributionNamed(dict(
        e    =          tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
        scale=lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        s    =          tfd.HalfNormal(2.5),
        loc  =lambda s: tfd.Normal(loc=0, scale=s),
        df   =          tfd.Exponential(2),
        x    =          tfd.StudentT),
                                   validate_args=False)
    # pylint: enable=bad-whitespace
    x = d.sample([2, 3], seed=test_util.test_seed())
    self.assertLen(x, 6)
    self.assertEqual((2, 3, 2), x['e'].shape)
    self.assertEqual((2, 3), x['scale'].shape)
    self.assertEqual((2, 3), x['s'].shape)
    self.assertEqual((2, 3), x['loc'].shape)
    self.assertEqual((2, 3), x['df'].shape)
    self.assertEqual((2, 3), x['x'].shape)
    lp = d.log_prob(x)
    self.assertEqual((2, 3), lp.shape)

  def test_sample_shape_propagation_nondefault_behavior(self):
    # pylint: disable=bad-whitespace
    d = tfd.JointDistributionNamed(dict(
        e    =          tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
        scale=lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        s    =          tfd.HalfNormal(2.5),
        loc  =lambda s: tfd.Normal(loc=0, scale=s),
        df   =          tfd.Exponential(2),
        x    =          tfd.StudentT),
                                   validate_args=False)
    # pylint: enable=bad-whitespace
    # The following enables the nondefault sample shape behavior.
    d._always_use_specified_sample_shape = True
    sample_shape = (2, 3)
    x = d.sample(sample_shape, seed=test_util.test_seed())
    self.assertLen(x, 6)
    self.assertEqual(sample_shape + (2,), x['e'].shape)
    self.assertEqual(sample_shape * 2, x['scale'].shape)  # Has 1 arg.
    self.assertEqual(sample_shape * 1, x['s'].shape)      # Has 0 args.
    self.assertEqual(sample_shape * 2, x['loc'].shape)    # Has 1 arg.
    self.assertEqual(sample_shape * 1, x['df'].shape)     # Has 0 args.
    # Has 3 args, one being scalar.
    self.assertEqual(sample_shape * 3, x['x'].shape)
    lp = d.log_prob(x)
    self.assertEqual(sample_shape * 3, lp.shape)

  def test_sample_complex_dependency(self):
    # pylint: disable=bad-whitespace
    d = tfd.JointDistributionNamed(
        dict(
            y    =          tfd.StudentT,
            x    =          tfd.StudentT,
            df   =          tfd.Exponential(2),
            loc  =lambda s: tfd.Normal(loc=0, scale=s),
            s    =          tfd.HalfNormal(2.5),
            scale=lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
            e    =          tfd.Independent(tfd.Exponential(rate=[100, 120]), 1)
        ),
        validate_args=False)

    # pylint: enable=bad-whitespace

    self.assertEqual(
        (
            ('e', ()),
            ('scale', ('e',)),
            ('s', ()),
            ('loc', ('s',)),
            ('df', ()),
            ('y', ('df', 'loc', 'scale')),
            ('x', ('df', 'loc', 'scale')),
        ),
        d.resolve_graph())

    x = d.sample(seed=test_util.test_seed())
    self.assertLen(x, 7)

    ds, s = d.sample_distributions(seed=test_util.test_seed())
    self.assertEqual(ds['x'].parameters['df'], s['df'])
    self.assertEqual(ds['x'].parameters['loc'], s['loc'])
    self.assertEqual(ds['x'].parameters['scale'], s['scale'])
    self.assertEqual(ds['y'].parameters['df'], s['df'])
    self.assertEqual(ds['y'].parameters['loc'], s['loc'])
    self.assertEqual(ds['y'].parameters['scale'], s['scale'])

  def test_default_event_space_bijector(self):
    # pylint: disable=bad-whitespace
    d = tfd.JointDistributionNamed(dict(
        e    =          tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
        scale=lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        s    =          tfd.HalfNormal(2.5),
        loc  =lambda s: tfd.Normal(loc=0, scale=s),
        df   =          tfd.Exponential(2),
        x    =          tfd.StudentT),
                                   validate_args=True)
    # pylint: enable=bad-whitespace

    # The event space bijector is inherited from `JointDistributionSequential`
    # and is tested more thoroughly in the tests for that class.
    b = d._experimental_default_event_space_bijector()
    y = self.evaluate(d.sample(seed=test_util.test_seed()))
    y_ = self.evaluate(b.forward(b.inverse(y)))
    self.assertAllClose(y, y_)

    # Verify that event shapes are passed through and flattened/unflattened
    # correctly.
    forward_event_shapes = b.forward_event_shape(d.event_shape)
    inverse_event_shapes = b.inverse_event_shape(d.event_shape)
    self.assertEqual(forward_event_shapes, d.event_shape)
    self.assertEqual(inverse_event_shapes, d.event_shape)

    # Verify that the outputs of other methods have the correct dict structure.
    forward_event_shape_tensors = b.forward_event_shape_tensor(
        d.event_shape_tensor())
    inverse_event_shape_tensors = b.inverse_event_shape_tensor(
        d.event_shape_tensor())
    for item in [forward_event_shape_tensors, inverse_event_shape_tensors]:
      self.assertSetEqual(set(self.evaluate(item).keys()), set(d.model.keys()))


if __name__ == '__main__':
  tf.test.main()
