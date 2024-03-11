# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for JointDistributionPinned."""

import collections
import functools

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import pad
from tensorflow_probability.python.distributions import beta
from tensorflow_probability.python.distributions import binomial
from tensorflow_probability.python.distributions import cholesky_lkj
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import multinomial
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.experimental.distributions import increment_log_prob as ilp
from tensorflow_probability.python.experimental.distributions import joint_distribution_pinned as jdp
from tensorflow_probability.python.internal import structural_tuple
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import generic

Root = jdc.JointDistributionCoroutine.Root


def part_dists():
  d0 = gamma.Gamma(1., 1., name='w')
  d1 = gamma.Gamma(1., 2., name='x')
  d2 = lambda x: sample.Sample(  # pylint:disable=g-long-lambda
      uniform.Uniform(x, x + np.float32(10.)), 2, name='y')
  def d3(y, x, w):
    return independent.Independent(
        cholesky_lkj.CholeskyLKJ(4, concentration=y + (w + x)[..., tf.newaxis]),
        reinterpreted_batch_ndims=1,
        name='z')
  return d0, d1, d2, d3


def jd_coroutine():
  d0, d1, d2, d3 = part_dists()

  root = jdc.JointDistributionCoroutine.Root

  @jdc.JointDistributionCoroutine
  def model():
    w = yield root(d0)
    x = yield root(d1)
    y = yield d2(x)
    yield d3(y, x, w)
  return model


def jd_coroutine_autobatched():
  d0, d1, d2, d3 = part_dists()

  root = jdab.JointDistributionCoroutineAutoBatched.Root

  @jdab.JointDistributionCoroutineAutoBatched
  def model():
    w = yield root(d0)
    x = yield root(d1)
    y = yield d2(x)
    yield d3(y, x, w)
  return model


def jd_sequential(model_from_seq=tuple):
  return jds.JointDistributionSequential(model_from_seq(part_dists()))


def jd_sequential_autobatched(model_from_seq=tuple):
  return jdab.JointDistributionSequentialAutoBatched(
      model_from_seq(part_dists()))


def jd_named():
  d0, d1, d2, d3 = part_dists()
  return jdn.JointDistributionNamed(dict(w=d0, x=d1, y=d2, z=d3))


def jd_named_autobatched():
  d0, d1, d2, d3 = part_dists()
  return jdab.JointDistributionNamedAutoBatched(dict(w=d0, x=d1, y=d2, z=d3))


def jd_named_ordered():
  d0, d1, d2, d3 = part_dists()
  return jdn.JointDistributionNamed(
      collections.OrderedDict((('w', d0), ('x', d1), ('y', d2), ('z', d3))))


def jd_named_namedtuple():
  d0, d1, d2, d3 = part_dists()
  nt_type = collections.namedtuple('Foo', 'w,x,y,z')
  return jdn.JointDistributionNamed(nt_type(w=d0, x=d1, y=d2, z=d3))


@test_util.test_graph_and_eager_modes
@parameterized.named_parameters(
    *(dict(testcase_name='_{}_{}'.format(jd_factory.__name__,  # pylint: disable=g-complex-comprehension
                                         '_'.join(map(str, sample_shape))),
           jd_factory=jd_factory,
           sample_shape=sample_shape)
      for jd_factory in (jd_coroutine, jd_coroutine_autobatched, jd_sequential,
                         jd_sequential_autobatched, jd_named,
                         jd_named_autobatched, jd_named_ordered,
                         jd_named_namedtuple)
      # TODO(b/168139745): Add support for: [13], [13, 1], [1, 13]
      for sample_shape in ([],)))
class JointDistributionPinnedParameterizedTest(test_util.TestCase):

  def test_pinned_distribution_seq_args(self, jd_factory, sample_shape):
    s = jd_coroutine().sample(
        sample_shape, seed=test_util.test_seed(sampler_type='stateless'))
    x, z = s.x, s.z
    underlying = jd_factory()

    tuple_args = (None, x,), (None, x, None, None), (None, x, None, z)
    if jd_factory is jd_named or jd_factory is jd_named_autobatched:
      # JDNamed does not support unnamed args unless model is ordered.
      for args in tuple_args:
        with self.assertRaisesRegex(ValueError, r'unordered'):
          jdp.JointDistributionPinned(underlying, args)
        with self.assertRaisesRegex(ValueError, r'unordered'):
          jdp.JointDistributionPinned(underlying, *args)
      tuple_args = ()

    for args in tuple_args:
      # Use as args[0].
      pinned = jdp.JointDistributionPinned(underlying, args)
      self._check_pinning(pinned, sample_shape)

      # Use as *args.
      pinned = jdp.JointDistributionPinned(underlying, *args)
      self._check_pinning(pinned, sample_shape)

  def test_pinned_distribution_kwargs(self, jd_factory, sample_shape):
    s = jd_coroutine().sample(
        sample_shape, seed=test_util.test_seed(sampler_type='stateless'))
    x, z = s.x, s.z
    underlying = jd_factory()

    for dict_arg in dict(x=x), dict(w=None, x=x), dict(x=x, z=z):
      # Use as args[0].
      pinned = jdp.JointDistributionPinned(underlying, dict_arg)
      self._check_pinning(pinned, sample_shape)

      # Use as **kwargs.
      pinned = jdp.JointDistributionPinned(underlying, **dict_arg)
      self._check_pinning(pinned, sample_shape)

  def _check_pinning(self, pinned, sample_shape):
    self.evaluate(tf.nest.map_structure(tf.convert_to_tensor,
                                        pinned.event_shape_tensor()))

    s0 = pinned.sample_unpinned(
        sample_shape, seed=test_util.test_seed(sampler_type='stateless'))
    w0 = self.evaluate(pinned.log_weight(s0))

    s0dict = (s0 if isinstance(s0, dict) else
              dict(zip(pinned._flat_resolve_names(), s0)))
    self.assertAllClose(
        pinned.distribution.log_prob(**dict(pinned.pins, **s0dict)),
        pinned.unnormalized_log_prob(s0))

    self.assertAllClose(
        sum(tf.nest.flatten(pinned.unnormalized_log_prob_parts(s0).pinned)),
        w0)

    s1, w1 = pinned.sample_and_log_weight(
        sample_shape, seed=test_util.test_seed(sampler_type='stateless'))
    self.assertAllClose(
        w1 + sum(tf.nest.flatten(
            pinned.unnormalized_log_prob_parts(s1).unpinned)),
        pinned.unnormalized_log_prob(s1))

    bij = pinned.experimental_default_event_space_bijector()
    pullback_event_shp = bij.inverse_event_shape(pinned.event_shape)
    self.assertAllEqualNested(
        pinned.event_shape, bij.forward_event_shape(pullback_event_shp))

    pullback_s0 = tf.nest.map_structure(tf.identity, bij.inverse(s0))
    self.assertAllCloseNested(s0, bij.forward(pullback_s0))

    init = tf.nest.map_structure(
        lambda s: tf.random.uniform(s, -2., 2., seed=test_util.test_seed()),
        pullback_event_shp)
    self.evaluate(pinned.unnormalized_log_prob(bij.forward(init)))
    self.evaluate(pinned.unnormalized_log_prob_parts(bij.forward(init)))


@test_util.test_graph_and_eager_modes
class JointDistributionPinnedTest(test_util.TestCase):

  def test_constructor_docstr(self):
    jd = jds.JointDistributionSequential([
        normal.Normal(0., 1., name='z'),
        normal.Normal(0., 1., name='y'),
        lambda y, z: normal.Normal(y + z, 1., name='x')
    ],
                                         validate_args=True)

    # The following `__init__` styles are all permissible and produce
    # `JointDistributionPinned` objects behaving identically.
    PartialXY = collections.namedtuple('PartialXY', 'x,y')
    PartialX = collections.namedtuple('PartialX', 'x')
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, x=2.).pins,
        jdp.JointDistributionPinned(jd, x=2., z=None).pins)
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, x=2.).pins,
        jdp.JointDistributionPinned(jd, dict(x=2.)).pins)
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, x=2.).pins,
        jdp.JointDistributionPinned(jd, dict(x=2., y=None)).pins)
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, x=2.).pins,
        jdp.JointDistributionPinned(jd, collections.OrderedDict(x=2.)).pins)
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, x=2.).pins,
        jdp.JointDistributionPinned(
            jd, collections.OrderedDict((('x', 2.), ('y', None)))).pins)
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, x=2.).pins,
        jdp.JointDistributionPinned(jd, PartialXY(x=2., y=None)).pins)
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, x=2.).pins,
        jdp.JointDistributionPinned(jd, PartialX(x=2.)).pins)
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, x=2.).pins,
        jdp.JointDistributionPinned(jd, None, None, 2.).pins)
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, x=2.).pins,
        jdp.JointDistributionPinned(jd, [None, None, 2.]).pins)

    # The 'care is taken to resolve any potential ambiguity' section...
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, 2.).pins,
        jdp.JointDistributionPinned(jd, [2.]).pins)
    self.assertAllEqualNested(
        jdp.JointDistributionPinned(jd, np.array([2.])).pins,
        jdp.JointDistributionPinned(jd, z=[2.]).pins)

  def test_dtype_and_structure_jd_seq(self):
    self.assertEqual((tf.float32, tf.float32, tf.float32),
                     jdp.JointDistributionPinned(jd_sequential(), 1.).dtype)
    self.assertEqual((tf.float32, tf.float32),
                     jdp.JointDistributionPinned(jd_sequential(), None, 1.,
                                                 [2., 3]).dtype)
    self.assertEqual([tf.float32, tf.float32],
                     jdp.JointDistributionPinned(jd_sequential(list),
                                                 [1., 1.]).dtype)

  def test_event_shape_and_structure_jd_seq(self):
    self.assertEqual(([], [2], [2, 4, 4]),
                     jdp.JointDistributionPinned(jd_sequential(),
                                                 1.).event_shape)
    self.assertEqual(([], [2, 4, 4]),
                     jdp.JointDistributionPinned(jd_sequential(), None, 1.,
                                                 [2., 3]).event_shape)
    self.assertEqual([[2], [2, 4, 4]],
                     jdp.JointDistributionPinned(jd_sequential(list),
                                                 [1., 1.]).event_shape)

  def test_dtype_and_structure_jd_named(self):
    self.assertEqual(
        dict(x=tf.float32, y=tf.float32, z=tf.float32),
        jdp.JointDistributionPinned(jd_named(), w=1.).dtype)
    self.assertEqual(
        collections.OrderedDict((('w', tf.float32), ('z', tf.float32))),
        jdp.JointDistributionPinned(jd_named_ordered(), None, 1.,
                                    [2., 3]).dtype)
    self.assertEqual(
        collections.OrderedDict(
            (('w', tf.float32), ('x', tf.float32), ('z', tf.float32))),
        jdp.JointDistributionPinned(jd_named_ordered(), y=[2., 3]).dtype)
    self.assertEqual(
        structural_tuple.structtuple(['w', 'y', 'z'])(tf.float32, tf.float32,
                                                      tf.float32),
        jdp.JointDistributionPinned(jd_named_namedtuple(), x=2.).dtype)

  def test_event_shape_and_structure_jd_named(self):
    self.assertEqual(
        dict(x=[], y=[2], z=[2, 4, 4]),
        jdp.JointDistributionPinned(jd_named(), w=1.).event_shape)
    self.assertEqual(
        collections.OrderedDict((('w', []), ('z', [2, 4, 4]))),
        jdp.JointDistributionPinned(jd_named_ordered(), None, 1.,
                                    [2., 3]).event_shape)
    self.assertEqual(
        collections.OrderedDict((('w', []), ('x', []), ('z', [2, 4, 4]))),
        jdp.JointDistributionPinned(jd_named_ordered(), y=[2., 3]).event_shape)
    self.assertEqual(
        structural_tuple.structtuple(['w', 'y', 'z'])([], [2], [2, 4, 4]),
        jdp.JointDistributionPinned(jd_named_namedtuple(), x=2.).event_shape)

  def test_dtype_and_structure_jd_coroutine(self):
    self.assertEqual(
        structural_tuple.structtuple(['x', 'y', 'z'])(tf.float32, tf.float32,
                                                      tf.float32),
        jdp.JointDistributionPinned(jd_coroutine(), w=1.).dtype)
    self.assertEqual(
        structural_tuple.structtuple(['w', 'z'])(tf.float32, tf.float32),
        jdp.JointDistributionPinned(jd_coroutine(), None, 1., [2., 3]).dtype)
    self.assertEqual(
        structural_tuple.structtuple(['w', 'y', 'z'])(tf.float32, tf.float32,
                                                      tf.float32),
        jdp.JointDistributionPinned(jd_coroutine(), x=2.).dtype)

    obs = jd_coroutine().sample(seed=test_util.test_seed())[-1:]
    self.assertEqual(obs._fields, ('z',))
    self.assertEqual(
        structural_tuple.structtuple(['w', 'x', 'y'])(tf.float32, tf.float32,
                                                      tf.float32),
        jdp.JointDistributionPinned(jd_coroutine(), obs).dtype)

  def test_event_shape_and_structure_jd_coroutine(self):
    self.assertEqual(
        structural_tuple.structtuple(['x', 'y', 'z'])([], [2], [2, 4, 4]),
        jdp.JointDistributionPinned(jd_coroutine(), w=1.).event_shape)
    self.assertEqual(
        structural_tuple.structtuple(['w', 'z'])([], [2, 4, 4]),
        jdp.JointDistributionPinned(jd_coroutine(), None, 1.,
                                    [2., 3]).event_shape)
    self.assertEqual(
        structural_tuple.structtuple(['w', 'y', 'z'])([], [2], [2, 4, 4]),
        jdp.JointDistributionPinned(jd_coroutine(), x=2.).event_shape)

    obs = jd_coroutine().sample(seed=test_util.test_seed())[-1:]
    self.assertEqual(obs._fields, ('z',))
    self.assertEqual(
        structural_tuple.structtuple(['w', 'x', 'y'])([], [], [2]),
        jdp.JointDistributionPinned(jd_coroutine(), obs).event_shape)

  def test_bijector(self):
    jd = jds.JointDistributionSequential([
        uniform.Uniform(-1., 1.), lambda a: uniform.Uniform(  # pylint:disable=g-long-lambda
            a + tf.ones_like(a), a + tf.constant(2, a.dtype)),
        lambda b, a: uniform.Uniform(a, b, name='c')
    ])
    bij = jd.experimental_default_event_space_bijector(a=-.5, b=1.)
    test_input = (0.5,)
    self.assertIs(type(jd.dtype), type(bij.inverse(test_input)))
    self.assertAllClose(2/3, tf.math.sigmoid(bij.inverse(test_input)[0]))

    @jdc.JointDistributionCoroutine
    def model():
      root = jdc.JointDistributionCoroutine.Root
      x = yield root(normal.Normal(0., 1., name='x'))
      y = yield root(gamma.Gamma(1., 1., name='y'))
      yield normal.Normal(x, y, name='z')
    bij = model.experimental_default_event_space_bijector(
        model.sample(seed=test_util.test_seed())[-1:])
    self.assertAllCloseNested(
        structural_tuple.structtuple(['x', 'y'])(1., 2.),
        bij.forward((1., generic.softplus_inverse(2.))))

  def test_bijector_unconstrained_shapes(self):
    pinned = jdp.JointDistributionPinned(jd_coroutine(), x=1., y=[1., 1])
    bij = pinned.experimental_default_event_space_bijector()
    self.assertEqual(
        structural_tuple.structtuple(['w', 'z'])([], [2, 6]),
        bij.inverse_event_shape(pinned.event_shape))

  def test_subsidiary_pin(self):
    pinned0 = jdp.JointDistributionPinned(jd_coroutine(), x=1., y=[1., 1])
    pinned1 = jdp.JointDistributionPinned(
        jd_coroutine(), x=1.).experimental_pin(None, [1., 1])  # w=None, y=[1,1]
    self.assertAllEqualNested(pinned0.pins, pinned1.pins)

  def test_subsidiary_bijector_pins(self):
    pinned0 = jdp.JointDistributionPinned(
        jd_coroutine(), x=1., z=tf.eye(4, batch_shape=[2]))
    pinned1 = jdp.JointDistributionPinned(jd_coroutine(), x=1.)
    bij0 = pinned0.experimental_default_event_space_bijector()
    bij1 = pinned1.experimental_default_event_space_bijector(
        z=tf.eye(4, batch_shape=[2]))
    self.assertEqual(bij0.inverse_event_shape(pinned0.event_shape),
                     bij1.inverse_event_shape(pinned0.event_shape))
    self.assertEqual(bij0.inverse_event_shape(pinned1.event_shape[:-1]),
                     bij1.inverse_event_shape(pinned1.event_shape[:-1]))
    init = self.evaluate(tf.nest.map_structure(
        lambda shp: tf.random.uniform(shp, -2., 2., seed=test_util.test_seed()),
        bij0.inverse_event_shape(pinned0.event_shape)))
    self.assertAllCloseNested(bij0.forward(init), bij1.forward(init))

  @test_util.numpy_disable_test_missing_functionality('vectorized_map')
  @parameterized.named_parameters(
      ('scalar_scalar', [], []),
      ('scalar_batch', [], [4]),
      ('batch_scalar', [3], []),
      ('batch_batch', [3], [4]))
  def test_bijector_for_autobatched_model(self, pin_batch_shape, sample_shape):
    if not tf1.control_flow_v2_enabled():
      self.skipTest('b/183994961')

    @jdab.JointDistributionCoroutineAutoBatched
    def model():
      a = yield normal.Normal(0., 1., name='a')
      yield uniform.Uniform(low=0., high=tf.exp(a * tf.ones([2])), name='b')

    pinned = jdp.JointDistributionPinned(
        model,
        a=tf.random.stateless_normal(
            pin_batch_shape,
            seed=test_util.test_seed(sampler_type='stateless')))
    ys = self.evaluate(
        pinned.sample_unpinned(
            sample_shape,
            seed=test_util.test_seed(sampler_type='stateless')))
    bij = pinned.experimental_default_event_space_bijector()
    self.assertAllCloseNested(
        ys,
        bij.forward(
            tf.nest.map_structure(
                tf.identity,  # Bypass bijector cache.
                bij.inverse(ys))))

  @test_util.numpy_disable_test_missing_functionality('vectorized map')
  def test_pin_broadcast_value_autobatched(self):

    def model():
      c0 = yield gamma.Gamma(1., rate=1., name='c0')  # []
      c1 = yield gamma.Gamma(1., rate=1, name='c1')  # []
      probs = yield sample.Sample(beta.Beta(c1, c0), 14, name='probs')  # [14]
      yield binomial.Binomial(total_count=30, probs=probs, name='obs')  # [14]

    pinned = jdab.JointDistributionCoroutineAutoBatched(model).experimental_pin(
        obs=15. * tf.ones([14]))
    pinned_with_broadcast_value = jdab.JointDistributionCoroutineAutoBatched(
        model).experimental_pin(obs=15.)  # Scalar value for vector-valued RV.
    xs = self.evaluate(pinned.sample_unpinned([5], seed=test_util.test_seed()))
    self.assertAllClose(pinned.log_prob(xs),
                        pinned_with_broadcast_value.log_prob(xs))

  def test_str(self):

    @jdc.JointDistributionCoroutine
    def model():
      x = yield Root(
          mvn_diag.MultivariateNormalDiag(
              tf.zeros([10, 3]), tf.ones(3), name='x'))
      yield multinomial.Multinomial(
          logits=pad.Pad([[0, 1]])(x), total_count=13, name='y')

    self.assertEqual(
        'tfp.distributions.JointDistributionPinned('
        '"PinnedJointDistributionCoroutine", '
        'batch_shape=StructTuple(\n  x=[10]\n), '
        'event_shape=StructTuple(\n  x=[3]\n), '
        'dtype=StructTuple(\n  x=float32\n))',
        str(jdp.JointDistributionPinned(model, y=tf.zeros([10, 4]))))

    @functools.partial(
        jdab.JointDistributionCoroutineAutoBatched, batch_ndims=1)
    def model_ab():
      x = yield Root(
          mvn_diag.MultivariateNormalDiag(
              tf.zeros([10, 3]), tf.ones(3), name='x'))
      yield multinomial.Multinomial(
          logits=pad.Pad([[0, 1]])(x), total_count=13, name='y')

    self.assertEqual(
        'tfp.distributions.JointDistributionPinned('
        '"PinnedJointDistributionCoroutineAutoBatched", '
        'batch_shape=[10], '
        'event_shape=StructTuple(\n  x=[3]\n), '
        'dtype=StructTuple(\n  x=float32\n))',
        str(jdp.JointDistributionPinned(model_ab, y=tf.zeros([10, 4]))))

  def test_log_prob_parts_with_improper_base_dists(self):
    root = jdc.JointDistributionCoroutine.Root

    @jdc.JointDistributionCoroutine
    def model():
      x = yield root(normal.Normal(0., 1., name='x'))
      yield ilp.IncrementLogProb(normal.Normal(x, 2.).log_prob(0.), name='y')

    p = model.experimental_pin(y=tf.zeros([0]))

    parts = p.unnormalized_log_prob_parts(x=2.)
    self.assertAllCloseNested(
        normal.Normal(0., 1.).log_prob(2.), parts.unpinned.x)
    self.assertAllCloseNested(
        normal.Normal(2., 2.).log_prob(0.), parts.pinned.y)

  def test_unpin(self):
    root = jdc.JointDistributionCoroutine.Root

    @jdc.JointDistributionCoroutine
    def model():
      loc = yield root(normal.Normal(0., 1., name='loc'))
      scl = yield root(gamma.Gamma(1., 1., name='scl'))
      _ = yield normal.Normal(loc, scl, name='y')

    p = model.experimental_pin(y=0.5)
    self.assertIs(model, p.unpin('y'))

    p = model.experimental_pin(scl=1., y=0.5)
    self.assertEqual(['scl'], list(p.unpin('y').pins))
    self.assertEqual(['y'], list(p.unpin('scl').pins))
    self.assertIs(model, p.unpin('y', 'scl'))
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"Unrecognized.*'foo'.*['scl', 'y']"):
      p.unpin('foo', 'y')


if __name__ == '__main__':
  test_util.main()
