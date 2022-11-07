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

import collections

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import beta
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample as sample_lib
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import test_util


# Defer creating test dists (by hiding them in functions) until we know what
# execution regime (eager/graph/tf-function) the test will run under.
def basic_ordered_model_fn():
  return collections.OrderedDict((
      ('a', normal.Normal(0., 1.)),
      ('e',
       independent.Independent(
           exponential.Exponential(rate=[100, 120]),
           reinterpreted_batch_ndims=1)),
      ('x', lambda e: gamma.Gamma(concentration=e[..., 0], rate=e[..., 1])),
  ))


def nested_lists_model_fn():
  return collections.OrderedDict((
      ('abc',
       jds.JointDistributionSequential([
           mvn_diag.MultivariateNormalDiag([0., 0.], [1., 1.]),
           jds.JointDistributionSequential(
               [student_t.StudentT(3., -2., 5.),
                exponential.Exponential(4.)])
       ])),
      (
          'de',
          lambda abc: jds.JointDistributionSequential([  # pylint: disable=g-long-lambda
              independent.Independent(
                  normal.Normal(abc[0] * abc[1][0], abc[1][1]),
                  reinterpreted_batch_ndims=1),
              independent.Independent(
                  normal.Normal(abc[0] + abc[1][0], abc[1][1]),
                  reinterpreted_batch_ndims=1)
          ]))))


@test_util.test_all_tf_execution_regimes
class JointDistributionNamedTest(test_util.TestCase):

  def test_dict_sample_log_prob(self):
    # pylint: disable=bad-whitespace
    d = jdn.JointDistributionNamed(
        dict(
            e=independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1),
            scale=lambda e: gamma.Gamma(  # pylint: disable=g-long-lambda
                concentration=e[..., 0], rate=e[..., 1]),
            loc=normal.Normal(loc=0, scale=2.),
            m=normal.Normal,
            x=lambda m: sample_lib.Sample(bernoulli.Bernoulli(logits=m), 12)),
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
    self.assertIsInstance(ds['e'], independent.Independent)
    self.assertIsInstance(ds['scale'], gamma.Gamma)
    self.assertIsInstance(ds['loc'], normal.Normal)
    self.assertIsInstance(ds['m'], normal.Normal)
    self.assertIsInstance(ds['x'], sample_lib.Sample)

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
        e=independent.Independent(exponential.Exponential(rate=[100, 120]), 1),
        scale=lambda e: gamma.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        loc=normal.Normal(loc=0, scale=2.),
        m=normal.Normal,
        x=lambda m: sample_lib.Sample(bernoulli.Bernoulli(logits=m), 12))
    # pylint: enable=bad-whitespace
    d = jdn.JointDistributionNamed(model, validate_args=True)

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
    self.assertIsInstance(ds.e, independent.Independent)
    self.assertIsInstance(ds.scale, gamma.Gamma)
    self.assertIsInstance(ds.loc, normal.Normal)
    self.assertIsInstance(ds.m, normal.Normal)
    self.assertIsInstance(ds.x, sample_lib.Sample)

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
        e=independent.Independent(exponential.Exponential(rate=[100, 120]), 1),
        scale=lambda e: gamma.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        loc=normal.Normal(loc=0, scale=2.),
        m=normal.Normal,
        x=lambda m: sample_lib.Sample(bernoulli.Bernoulli(logits=m), 12))
    # pylint: enable=bad-whitespace
    d = jdn.JointDistributionNamed(model, validate_args=True)

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
    self.assertIsInstance(values[0], independent.Independent)
    self.assertIsInstance(values[1], gamma.Gamma)
    self.assertIsInstance(values[2], normal.Normal)
    self.assertIsInstance(values[3], normal.Normal)
    self.assertIsInstance(values[4], sample_lib.Sample)

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

    d = jdn.JointDistributionNamed(
        {
            'e':
                normal.Normal(0., 1.),
            'a':
                independent.Independent(
                    exponential.Exponential(rate=[100, 120]),
                    reinterpreted_batch_ndims=1),
            'x':
                lambda a: gamma.Gamma(concentration=a[..., 0], rate=a[..., 1])
        },
        validate_args=True)

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
      ('_sample', lambda d, **kwargs: d.sample(**kwargs)),
      ('_sample_and_log_prob',
       lambda d, **kwargs: d.experimental_sample_and_log_prob(**kwargs)[0]),
  )
  def test_nested_partial_value(self, sample_fn):
    innermost = jdn.JointDistributionNamed({
        'a': exponential.Exponential(1.),
        'b': lambda a: sample_lib.Sample(lognormal.LogNormal(a, a), [5]),
    })

    inner = jdn.JointDistributionNamed({
        'c': exponential.Exponential(1.),
        'd': innermost,
    })

    outer = jdn.JointDistributionNamed({
        'e': exponential.Exponential(1.),
        'f': inner,
    })

    seed = test_util.test_seed(sampler_type='stateless')
    true_xs = outer.sample(seed=seed)

    def _update(dict_, **kwargs):
      dict_.copy().update(**kwargs)
      return dict_

    # These asserts work because we advance the stateless seed inside the model
    # whether or not a sample is actually generated.
    partial_xs = _update(true_xs, f=None)
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

    partial_xs = _update(true_xs, e=None)
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

    partial_xs = _update(true_xs, f=_update(true_xs['f'], d=None))
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

    partial_xs = _update(
        true_xs, f=_update(true_xs['f'], d=_update(true_xs['f']['d'], a=None)))
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

  @parameterized.named_parameters(
      ('basic', basic_ordered_model_fn),
      ('nested_lists', nested_lists_model_fn))
  def test_can_call_ordereddict_log_prob_with_args_and_kwargs(self, model_fn):
    # With an OrderedDict, we can pass keyword and/or positional args.
    d = jdn.JointDistributionNamed(model_fn(), validate_args=True)

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
    d = jdn.JointDistributionNamed(
        Model(
            e=normal.Normal(0., 1.),
            a=independent.Independent(
                exponential.Exponential(rate=[100, 120]),
                reinterpreted_batch_ndims=1),
            x=lambda a: gamma.Gamma(concentration=a[..., 0], rate=a[..., 1])),
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
    d0 = jdn.JointDistributionNamed(
        dict(
            e=independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1),
            x=normal.Normal(loc=0, scale=2.)),
        validate_args=True)
    d1 = jdn.JointDistributionNamed(
        dict(
            e=independent.Independent(
                exponential.Exponential(rate=[10, 12]), 1),
            x=normal.Normal(loc=1, scale=1.)),
        validate_args=True)
    self.assertEqual(d0.model.keys(), d1.model.keys())
    expected_kl = sum(
        kullback_leibler.kl_divergence(d0.model[k], d1.model[k])
        for k in d0.model.keys())
    actual_kl = kullback_leibler.kl_divergence(d0, d1)
    other_actual_kl = d0.kl_divergence(d1)
    expected_kl_, actual_kl_, other_actual_kl_ = self.evaluate([
        expected_kl, actual_kl, other_actual_kl])
    self.assertNear(expected_kl_, actual_kl_, err=1e-5)
    self.assertNear(expected_kl_, other_actual_kl_, err=1e-5)

  def test_cross_entropy(self):
    d0 = jdn.JointDistributionNamed(
        dict(
            e=independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1),
            x=normal.Normal(loc=0, scale=2.)),
        validate_args=True)
    d1 = jdn.JointDistributionNamed(
        dict(
            e=independent.Independent(
                exponential.Exponential(rate=[10, 12]), 1),
            x=normal.Normal(loc=1, scale=1.)),
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
      jdn.JointDistributionNamed(
          dict(logits=normal.Normal(0., 1.), x=bernoulli.Bernoulli))

  def test_dist_fn_takes_kwargs(self):
    dist = jdn.JointDistributionNamed(
        {
            'positive':
                exponential.Exponential(rate=1.),
            'negative':
                scale.Scale(-1.)(exponential.Exponential(rate=1.)),
            'b':
                lambda **kwargs: normal.Normal(  # pylint: disable=g-long-lambda
                    loc=kwargs['negative'],
                    scale=kwargs['positive'],
                    validate_args=True),
            'a':
                lambda **kwargs: scale.Scale(kwargs['b'])(  # pylint: disable=g-long-lambda
                    gamma.Gamma(
                        concentration=-kwargs['negative'],
                        rate=kwargs['positive'],
                        validate_args=True))
        },
        validate_args=True)
    lp = dist.log_prob(dist.sample(5, seed=test_util.test_seed()))
    self.assertAllEqual(lp.shape, [5])

  def test_graph_resolution(self):
    # pylint: disable=bad-whitespace
    d = jdn.JointDistributionNamed(
        dict(
            e=independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1),
            scale=lambda e: gamma.Gamma(  # pylint: disable=g-long-lambda
                concentration=e[..., 0], rate=e[..., 1]),
            s=half_normal.HalfNormal(2.5),
            loc=lambda s: normal.Normal(loc=0, scale=s),
            df=exponential.Exponential(2),
            x=student_t.StudentT),
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
    d = jdn.JointDistributionNamed(
        dict(logits=normal.Normal(0., 1.), x=bernoulli.Bernoulli(logits=0.)),
        validate_args=True)
    expected = {k: getattr(d.model[k], attr)() for k in d.model.keys()}
    actual = getattr(d, attr)()
    self.assertAllEqual(*self.evaluate([expected, actual]))

  @parameterized.parameters(('covariance',))
  def test_notimplemented_summary_statistic(self, attr):
    d = jdn.JointDistributionNamed(
        dict(logits=normal.Normal(0., 1.), x=bernoulli.Bernoulli(probs=0.5)),
        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionNamed'):
      getattr(d, attr)()

  @parameterized.parameters(
      'log_cdf', 'cdf', 'log_survival_function', 'survival_function')
  def test_notimplemented_evaluative_statistic(self, attr):
    d = jdn.JointDistributionNamed(
        dict(logits=normal.Normal(0., 1.), x=bernoulli.Bernoulli(probs=0.5)),
        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionNamed'):
      getattr(d, attr)(dict(logits=0., x=0.5))

  def test_notimplemented_quantile(self):
    d = jdn.JointDistributionNamed(
        dict(logits=normal.Normal(0., 1.), x=bernoulli.Bernoulli(probs=0.5)),
        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        'quantile is not implemented: JointDistributionNamed'):
      d.quantile(0.5)

  def test_copy(self):
    pgm = dict(
        logits=normal.Normal(0., 1.), probs=bernoulli.Bernoulli(logits=0.5))
    d = jdn.JointDistributionNamed(pgm, validate_args=True)
    d_copy = d.copy()
    self.assertEqual(d_copy.parameters['model'], pgm)
    self.assertEqual(d_copy.parameters['validate_args'], True)
    self.assertEqual(d_copy.parameters['name'], 'JointDistributionNamed')

  def test_batch_slicing(self):
    # pylint: disable=bad-whitespace
    d = jdn.JointDistributionNamed(
        dict(
            s=exponential.Exponential(rate=[10, 12, 14]),
            n=lambda s: normal.Normal(loc=0, scale=s),
            x=lambda: beta.Beta(concentration0=[3, 2, 1], concentration1=1)),
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
    d = jdn.JointDistributionNamed(
        dict(
            e=independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1),
            scale=lambda e: gamma.Gamma(  # pylint: disable=g-long-lambda
                concentration=e[..., 0], rate=e[..., 1]),
            s=half_normal.HalfNormal(2.5),
            loc=lambda s: normal.Normal(loc=0, scale=s),
            df=exponential.Exponential(2),
            x=student_t.StudentT),
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

  def test_sample_complex_dependency(self):
    # pylint: disable=bad-whitespace
    d = jdn.JointDistributionNamed(
        dict(
            y=student_t.StudentT,
            x=student_t.StudentT,
            df=exponential.Exponential(2),
            loc=lambda s: normal.Normal(loc=0, scale=s),
            s=half_normal.HalfNormal(2.5),
            scale=lambda e: gamma.Gamma(  # pylint: disable=g-long-lambda
                concentration=e[..., 0], rate=e[..., 1]),
            e=independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1)),
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
    d = jdn.JointDistributionNamed(
        dict(
            e=independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1),
            scale=lambda e: gamma.Gamma(  # pylint: disable=g-long-lambda
                concentration=e[..., 0], rate=e[..., 1]),
            s=half_normal.HalfNormal(2.5),
            loc=lambda s: normal.Normal(loc=0, scale=s),
            df=exponential.Exponential(2),
            x=student_t.StudentT),
        validate_args=True)
    # pylint: enable=bad-whitespace

    # The event space bijector is inherited from `JointDistributionSequential`
    # and is tested more thoroughly in the tests for that class.
    b = d.experimental_default_event_space_bijector()
    y = self.evaluate(d.sample(seed=test_util.test_seed()))
    y_ = self.evaluate(b.forward(b.inverse(y)))
    self.assertAllCloseNested(y, y_)

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

  def test_sample_kwargs(self):
    joint = jdn.JointDistributionNamed(
        dict(
            a=normal.Normal(0., 1.),
            b=lambda a: normal.Normal(a, 1.),
            c=lambda a, b: normal.Normal(a + b, 1.)))

    seed = test_util.test_seed()
    tf.random.set_seed(seed)
    samples = joint.sample(seed=seed, a=1.)
    # Check the first value is actually 1.
    self.assertEqual(1., self.evaluate(samples['a']))

    # Check the sample is reproducible using the `value` argument.
    tf.random.set_seed(seed)
    samples_named = joint.sample(seed=seed, value={'a': 1.})
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

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='Numpy has no notion of CompositeTensor/Pytree.')
  def testCompositeTensorOrPytree(self):
    d = jdn.JointDistributionNamed(
        dict(
            e=independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1),
            scale=lambda e: gamma.Gamma(  # pylint: disable=g-long-lambda
                concentration=e[..., 0], rate=e[..., 1]),
            loc=normal.Normal(loc=0, scale=2.),
            m=normal.Normal,
            x=lambda m: sample_lib.Sample(bernoulli.Bernoulli(logits=m), 12)),
        validate_args=True)

    flat = tf.nest.flatten(d, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(
        d, flat, expand_composites=True)
    self.assertIsInstance(unflat, jdn.JointDistributionNamed)
    self.assertIs(type(d.model), type(unflat.model))

    x = self.evaluate(d.sample(3, seed=test_util.test_seed()))
    actual = self.evaluate(d.log_prob(x))

    self.assertAllClose(self.evaluate(unflat.log_prob(x)), actual)

    @tf.function
    def call_log_prob(d):
      return d.log_prob(x)
    self.assertAllClose(actual, call_log_prob(d))
    self.assertAllClose(actual, call_log_prob(unflat))

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason='Numpy and JAX do not have type spec serialization.')
  def testCompositeTensorSerialization(self):
    encodable_jd = jdn.JointDistributionNamed(  # No lambdas.
        dict(
            e=independent.Independent(
                exponential.Exponential(rate=[10, 12]), 1),
            loc=normal.Normal(loc=0, scale=2.),
            m=normal.Normal(loc=-1., scale=1.)),
        validate_args=True)

    enc = tf.__internal__.saved_model.encode_structure(encodable_jd._type_spec)
    dec = tf.__internal__.saved_model.decode_proto(enc)
    flat = tf.nest.flatten(encodable_jd, expand_composites=True)
    deserialized_flat = dec._to_components(encodable_jd)
    unflat = tf.nest.pack_sequence_as(
        encodable_jd, flat, expand_composites=True)
    deserialized_unflat = dec._from_components(deserialized_flat)

    self.assertEqual(dec, encodable_jd._type_spec)
    self.assertAllEqualNested(
        flat, tf.nest.flatten(deserialized_flat, expand_composites=True))
    self.assertIsInstance(deserialized_unflat, jdn.JointDistributionNamed)
    self.assertIs(type(unflat.model), type(deserialized_unflat.model))

    non_ct_jd = jdn.JointDistributionNamed(
        dict(
            e=normal.Normal(loc=0, scale=2.),
            m=transformed_distribution.TransformedDistribution(
                normal.Normal(loc=-1., scale=1.),
                test_util.NonCompositeTensorExp()),
        ))
    self.assertNotIsInstance(non_ct_jd, tf.__internal__.CompositeTensor)


if __name__ == '__main__':
  test_util.main()
