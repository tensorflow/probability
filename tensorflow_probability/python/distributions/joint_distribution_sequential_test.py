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

import collections

# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import joint_distribution_sequential
from tensorflow_probability.python.internal import test_util

from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


tfb = tfp.bijectors
tfd = tfp.distributions


# Defer creating test dists (by hiding them in functions) until we know what
# execution regime (eager/graph/tf-function) the test will run under.
def basic_model_fn():
  return [
      tfd.Normal(0., 1., name='a'),
      tfd.Independent(tfd.Exponential(rate=[100, 120]),
                      reinterpreted_batch_ndims=1),
      lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1])
  ]


def nested_lists_model_fn():
  return [
      tfd.JointDistributionSequential([
          tfd.MultivariateNormalDiag([0., 0.], [1., 1.]),
          tfd.JointDistributionSequential([
              tfd.StudentT(3., -2., 5.),
              tfd.Exponential(4.)])], name='abc'),
      lambda abc: tfd.JointDistributionSequential([  # pylint: disable=g-long-lambda
          tfd.Normal(abc[0] * abc[1][0], abc[1][1]),
          tfd.Normal(abc[0] + abc[1][0], abc[1][1])], name='de')
  ]


class Dummy(object):
  """Dummy object to ensure `tf_inspect.getfullargspec` works for `__init__`."""

  # To ensure no code is keying on the unspecial name "self", we use "me".
  def __init__(me, arg1, arg2, arg3=None, **named):  # pylint: disable=no-self-argument
    pass


@test_util.test_all_tf_execution_regimes
class JointDistributionSequentialTest(test_util.TestCase):

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
        d.resolve_graph())

    xs = d.sample(seed=test_util.test_seed())
    self.assertLen(xs, 5)
    # We'll verify the shapes work as intended when we plumb these back into the
    # respective log_probs.

    ds, _ = d.sample_distributions(value=xs, seed=test_util.test_seed())
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
                      in zip(d0.model, d1.model))
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
        in zip(d0.model, d1.model))
    actual_xent = d0.cross_entropy(d1)
    expected_xent_, actual_xent_ = self.evaluate([expected_xent, actual_xent])
    self.assertNear(actual_xent_, expected_xent_, err=1e-5)

  def test_norequired_args_maker(self):
    """Test that only non-default args are passed through."""
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Must pass probs or logits, but not both.'):
      tfd.JointDistributionSequential([tfd.Normal(0., 1.), tfd.Bernoulli])

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
        d.resolve_graph())

  @parameterized.parameters('mean', 'mode', 'stddev', 'variance')
  def test_summary_statistic(self, attr):
    d = tfd.JointDistributionSequential(
        [tfd.Normal(0., 1.), tfd.Bernoulli(logits=0.)],
        validate_args=True)
    expected = tuple(getattr(d_, attr)() for d_ in d.model)
    actual = getattr(d, attr)()
    self.assertAllEqual(*self.evaluate([expected, actual]))

  @parameterized.parameters(('covariance',))
  def test_notimplemented_summary_statistic(self, attr):
    d = tfd.JointDistributionSequential([tfd.Normal(0., 1.),
                                         tfd.Bernoulli(probs=0.5)],
                                        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionSequential'):
      getattr(d, attr)()

  @parameterized.parameters(
      'log_cdf', 'cdf', 'log_survival_function', 'survival_function')
  def test_notimplemented_evaluative_statistic(self, attr):
    d = tfd.JointDistributionSequential([tfd.Normal(0., 1.),
                                         tfd.Bernoulli(probs=0.5)],
                                        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionSequential'):
      getattr(d, attr)([0.]*len(d.model))

  def test_notimplemented_quantile(self):
    d = tfd.JointDistributionSequential([tfd.Normal(0., 1.),
                                         tfd.Bernoulli(probs=0.5)],
                                        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        'quantile is not implemented: JointDistributionSequential'):
      d.quantile(0.5)

  def test_copy(self):
    pgm = [tfd.Normal(0., 1.), tfd.Bernoulli(probs=0.5)]
    d = tfd.JointDistributionSequential(pgm, validate_args=True)
    d_copy = d.copy()
    self.assertAllEqual(
        {'model': pgm,
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
    x0 = d0.sample(seed=test_util.test_seed())
    x1 = d1.sample(seed=test_util.test_seed())

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
    x = d.sample([2, 3], seed=test_util.test_seed())
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
    x = d.sample(sample_shape, seed=test_util.test_seed())
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

  def test_argspec(self):
    argspec = tf_inspect.getfullargspec(Dummy)
    self.assertAllEqual(['me', 'arg1', 'arg2', 'arg3'], argspec.args)
    self.assertIs(None, argspec.varargs)
    self.assertIs('named', argspec.varkw)
    self.assertAllEqual((None,), argspec.defaults)

  def test_invalid_structure_raises_error(self):
    with self.assertRaisesWithPredicateMatch(
        TypeError, 'Unable to unflatten like `model` with type "model".'):
      tfd.JointDistributionSequential(
          collections.namedtuple('model',
                                 'a b')(a=tfd.Normal(0, 1), b=tfd.Normal(1, 2)),
          validate_args=True)

  def test_simple_example_with_dynamic_shapes(self):
    dist = tfd.JointDistributionSequential([
        tfd.Normal(tf1.placeholder_with_default(0., shape=None),
                   tf1.placeholder_with_default(1., shape=None)),
        lambda a: tfd.Normal(a, 1.)], validate_args=True)
    lp = dist.log_prob(dist.sample(5, seed=test_util.test_seed()))
    self.assertAllEqual(self.evaluate(lp).shape, [5])

  @parameterized.named_parameters(
      ('basic', basic_model_fn),
      ('nested_lists', nested_lists_model_fn))
  def test_can_call_log_prob_with_args_and_kwargs(self, model_fn):
    d = tfd.JointDistributionSequential(
        model_fn(), validate_args=True)

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

  def test_can_call_prob_with_args_and_kwargs(self):
    d = tfd.JointDistributionSequential(basic_model_fn(), validate_args=True)
    a, e, x = self.evaluate(d.sample([2, 3], seed=test_util.test_seed()))
    prob_value_positional = self.evaluate(d.prob([a, e, x]))
    prob_value_named = self.evaluate(d.prob(value=[a, e, x]))
    self.assertAllEqual(prob_value_positional, prob_value_named)

    prob_args = self.evaluate(d.prob(a, e, x))
    self.assertAllEqual(prob_value_positional, prob_args)

    prob_kwargs = self.evaluate(d.prob(a=a, e=e, x=x))
    self.assertAllEqual(prob_value_positional, prob_kwargs)

    prob_args_then_kwargs = self.evaluate(d.prob(a, e=e, x=x))
    self.assertAllEqual(prob_value_positional, prob_args_then_kwargs)

  def test_uses_structure_to_convert_nested_lists(self):
    joint = tfd.JointDistributionSequential([
        tfd.MultivariateNormalDiag([0., 0.], [1., 1.]),
        lambda a: tfd.JointDistributionSequential([  # pylint: disable=g-long-lambda
            tfd.JointDistributionSequential([
                tfd.Normal(a[..., 0], 1.)]),
            tfd.Normal(a[..., 1], 1.)])
    ])

    x = [tf.convert_to_tensor([4., 2.]), [[1.], 3.]]
    x_with_tensor_as_list = [[4., 2.], [[1.], 3.]]
    lp = self.evaluate(joint.log_prob(x))
    lp_with_tensor_as_list = self.evaluate(
        joint.log_prob(x_with_tensor_as_list))
    self.assertAllClose(lp, lp_with_tensor_as_list, rtol=3e-7, atol=5e-7)

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
    n_users = 3
    n_items = 5
    n_factors = 2

    user_trait_prior_scale = 10.
    item_trait_prior_scale = 10.
    observation_noise_prior_scale = 1.

    dist = tfd.JointDistributionSequential([
        tfd.Sample(tfd.Normal(loc=0.,
                              scale=user_trait_prior_scale),
                   sample_shape=[n_factors, n_users]),  # U

        tfd.Sample(tfd.Normal(loc=0.,
                              scale=item_trait_prior_scale),
                   sample_shape=[n_factors, n_items]),  # V

        lambda item_traits, user_traits: tfd.Independent(  # pylint: disable=g-long-lambda
            tfd.Normal(loc=tf.matmul(user_traits, item_traits,  # R
                                     adjoint_a=True),
                       scale=observation_noise_prior_scale),
            reinterpreted_batch_ndims=2)], validate_args=True)
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
        u.sample([num_topics], seed=test_util.test_seed()),
        tfb.Softplus(), name='alpha')
    beta = tf.Variable(u.sample([num_topics, num_words],
                                seed=test_util.test_seed()), name='beta')

    # LDA Model.
    # Note near 1:1 with mathematical specification. The main distinction is the
    # use of Independent--this lets us easily aggregate multinomials across
    # topics (and in any "shape" of documents).
    lda = tfd.JointDistributionSequential(
        [
            tfd.Poisson(rate=avg_doc_length),  # n
            tfd.Dirichlet(concentration=alpha),  # theta
            lambda theta, n: tfd.Multinomial(total_count=n, probs=theta),  # z
            lambda z: tfd.Independent(  # x  pylint: disable=g-long-lambda
                tfd.Multinomial(total_count=z, logits=beta),
                reinterpreted_batch_ndims=1),
        ],
        validate_args=True)

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

  def test_poisson_switchover_graphical_model(self):
    # Build a pretend dataset.
    seed = test_util.test_seed_stream(salt='poisson')
    n = [43, 31]
    count_data = tf.cast(
        tf.concat([
            tfd.Poisson(rate=15.).sample(n[0], seed=seed()),
            tfd.Poisson(rate=25.).sample(n[1], seed=seed()),
        ], axis=0),
        dtype=tf.float32)
    count_data = self.evaluate(count_data)
    n = np.sum(n)

    # Make model.
    gather = lambda tau, lambda_: tf.gather(  # pylint: disable=g-long-lambda
        lambda_,
        indices=tf.cast(
            tau[..., tf.newaxis] < tf.linspace(0., 1., n),
            dtype=tf.int32),
        # TODO(b/139204153): Remove static value hack after bug closed.
        batch_dims=int(tf.get_static_value(tf.rank(tau))))

    alpha = tf.math.reciprocal(tf.reduce_mean(count_data))

    joint = tfd.JointDistributionSequential([
        tfd.Sample(tfd.Exponential(rate=alpha),
                   sample_shape=[2]),
        tfd.Uniform(),
        lambda tau, lambda_: tfd.Independent(  # pylint: disable=g-long-lambda
            tfd.Poisson(rate=gather(tau, lambda_)),
            reinterpreted_batch_ndims=1),
    ], validate_args=True)

    # Verify model correctly "compiles".
    batch_shape = [3, 4]
    self.assertEqual(
        batch_shape,
        joint.log_prob(
            joint.sample(batch_shape, seed=test_util.test_seed())).shape)

  def test_default_event_space_bijector(self):
    dist_fns = [
        tfd.LogNormal(0., 1., validate_args=True),
        lambda h: tfd.Independent(  # pylint: disable=g-long-lambda
            tfd.Uniform([0., 0.], h, validate_args=True)),
        lambda s: tfd.Normal(0., s, validate_args=True)
    ]

    jd = tfd.JointDistributionSequential(dist_fns, validate_args=True)
    joint_bijector = jd._experimental_default_event_space_bijector()

    # define a sample in the unconstrained space and construct the component
    # distributions
    x = [tf.constant(w) for w in [-0.2, [0.3, 0.1], -1.]]
    bijectors = []
    y = []

    b = dist_fns[0]._experimental_default_event_space_bijector()
    bijectors.append(b)
    y.append(b(x[0]))
    for i in range(1, 3):
      b = dist_fns[i](y[i - 1])._experimental_default_event_space_bijector()
      y.append(b(x[i]))
      bijectors.append(b)

    # Test forward and inverse values.
    self.assertAllClose(joint_bijector.forward(x), y)
    self.assertAllClose(joint_bijector.inverse(y), x)

    # Test forward log det Jacobian via finite differences.
    event_ndims = [0, 1, 0]
    delta = 0.01

    fldj = joint_bijector.forward_log_det_jacobian(x, event_ndims)
    forward_plus = [b.forward(x[i] + delta) for i, b in enumerate(bijectors)]
    forward_minus = [b.forward(x[i] - delta) for i, b in enumerate(bijectors)]
    fldj_fd = tf.reduce_sum(
        [tf.reduce_sum(tf.math.log((p - m) / (2. * delta)))
         for p, m in zip(forward_plus, forward_minus)])
    self.assertAllClose(self.evaluate(fldj), self.evaluate(fldj_fd), rtol=1e-5)

    # Test inverse log det Jacobian via finite differences.
    delta = 0.001
    y = [tf.constant(w) for w in [0.8, [0.4, 0.3], -0.05]]
    ildj = joint_bijector.inverse_log_det_jacobian(y, event_ndims)

    bijectors = []
    bijectors.append(dist_fns[0]._experimental_default_event_space_bijector())
    for i in range(1, 3):
      bijectors.append(
          dist_fns[i](y[i - 1])._experimental_default_event_space_bijector())

    inverse_plus = [b.inverse(y[i] + delta) for i, b in enumerate(bijectors)]
    inverse_minus = [b.inverse(y[i] - delta) for i, b in enumerate(bijectors)]
    ildj_fd = tf.reduce_sum(
        [tf.reduce_sum(tf.math.log((p - m) / (2. * delta)))
         for p, m in zip(inverse_plus, inverse_minus)])
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


class ResolveDistributionNamesTest(test_util.TestCase):

  def test_dummy_names_are_unique(self):

    dist_names = joint_distribution_sequential._resolve_distribution_names(
        dist_fn_args=[None, None, None],
        dist_names=None,
        leaf_name='x',
        instance_names=[None, None, None])
    self.assertAllEqual(dist_names, ['x2', 'x1', 'x'])

    dist_names = joint_distribution_sequential._resolve_distribution_names(
        dist_fn_args=[None, None, None],
        dist_names=None,
        leaf_name='x',
        instance_names=['x', 'x1', None])
    self.assertAllEqual(dist_names, ['x', 'x1', 'x2'])

  def test_ignores_trivial_names(self):

    # Should ignore a trivial reference downstream of the real name `z`.
    dist_names = joint_distribution_sequential._resolve_distribution_names(
        dist_fn_args=[None, ['z'], ['w', '_']],
        dist_names=None,
        leaf_name='y',
        instance_names=[None, None, None])
    self.assertAllEqual(dist_names, ['z', 'w', 'y'])

    # Trivial reference upstream of the real name `z`.
    dist_names = joint_distribution_sequential._resolve_distribution_names(
        dist_fn_args=[None, ['_'], ['w', 'z']],
        dist_names=None,
        leaf_name='y',
        instance_names=[None, None, None])
    self.assertAllEqual(dist_names, ['z', 'w', 'y'])

    # The only direct reference is trivial, but we have an instance name.
    dist_names = joint_distribution_sequential._resolve_distribution_names(
        dist_fn_args=[None, ['_']],
        dist_names=None,
        leaf_name='y',
        instance_names=['z', None])
    self.assertAllEqual(dist_names, ['z', 'y'])

  def test_inconsistent_names_raise_error(self):
    with self.assertRaisesRegexp(ValueError, 'Inconsistent names'):
      # Refers to first variable as both `z` and `x`.
      joint_distribution_sequential._resolve_distribution_names(
          dist_fn_args=[None, ['z'], ['x', 'w']],
          dist_names=None,
          leaf_name='y',
          instance_names=[None, None, None])

    with self.assertRaisesRegexp(ValueError, 'Inconsistent names'):
      # Refers to first variable as `x`, but it was explicitly named `z`.
      joint_distribution_sequential._resolve_distribution_names(
          dist_fn_args=[None, ['x']],
          dist_names=None,
          leaf_name='y',
          instance_names=['z', None])

if __name__ == '__main__':
  tf.test.main()
