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

import collections
import inspect

# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import beta as beta_lib
from tensorflow_probability.python.distributions import dirichlet
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import multinomial
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.util import deferred_tensor


# Defer creating test dists (by hiding them in functions) until we know what
# execution regime (eager/graph/tf-function) the test will run under.
def basic_model_fn():
  return [
      normal.Normal(0., 1., name='a'),
      independent.Independent(
          exponential.Exponential(rate=[100, 120]),
          reinterpreted_batch_ndims=1),
      lambda e: gamma.Gamma(concentration=e[..., 0], rate=e[..., 1])
  ]


def nested_lists_model_fn():
  return [
      jds.JointDistributionSequential([
          mvn_diag.MultivariateNormalDiag([0., 0.], [1., 1.]),
          jds.JointDistributionSequential(
              [student_t.StudentT(3., -2., 5.),
               exponential.Exponential(4.)])
      ],
                                      name='abc'),
      lambda abc: jds.JointDistributionSequential(  # pylint: disable=g-long-lambda
          [
              independent.Independent(
                  normal.Normal(abc[0] * abc[1][0], abc[1][1]),
                  reinterpreted_batch_ndims=1),
              independent.Independent(
                  normal.Normal(abc[0] + abc[1][0], abc[1][1]),
                  reinterpreted_batch_ndims=1)
          ],
          name='de')
  ]


class Dummy(object):
  """Dummy object to ensure `tf_inspect.getfullargspec` works for `__init__`."""

  # To ensure no code is keying on the unspecial name "self", we use "me".
  def __init__(me, arg1, arg2, arg3=None, **named):  # pylint: disable=no-self-argument
    pass


@test_util.test_all_tf_execution_regimes
class JointDistributionSequentialTest(test_util.TestCase):

  def test_sample_log_prob(self):
    d = jds.JointDistributionSequential(
        [
            independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1),
            lambda e: gamma.Gamma(concentration=e[..., 0], rate=e[..., 1]),
            normal.Normal(loc=0, scale=2.),
            normal
            .Normal,  # Or, `lambda loc, scale: normal.Normal(loc, scale)`.
            lambda m: sample.Sample(bernoulli.Bernoulli(logits=m), 12),
        ],
        validate_args=True)

    keys = ('e', 'scale', 'loc', 'm', 'x')
    deps = ((), ('e',), (), ('loc', 'scale'), ('m',))
    self.assertEqual(tuple(zip(keys, deps)), d.resolve_graph())

    xs = d.sample(seed=test_util.test_seed())
    self.assertLen(xs, 5)
    # We'll verify the shapes work as intended when we plumb these back into the
    # respective log_probs.

    ds, _ = d.sample_distributions(value=xs, seed=test_util.test_seed())
    self.assertLen(ds, 5)
    self.assertIsInstance(ds[0], independent.Independent)
    self.assertIsInstance(ds[1], gamma.Gamma)
    self.assertIsInstance(ds[2], normal.Normal)
    self.assertIsInstance(ds[3], normal.Normal)
    self.assertIsInstance(ds[4], sample.Sample)

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

    expected_lp_parts = [d_.log_prob(x) for d_, x in zip(ds, xs)]
    expected_jlp = sum(expected_lp_parts)
    actual_jlp = d.log_prob(xs)
    self.assertAllEqual(*self.evaluate([expected_jlp, actual_jlp]))
    # Verify different log_prob_parts calling conventions.
    self.assertAllCloseNested(expected_lp_parts, d.log_prob_parts(xs))
    self.assertAllCloseNested(expected_lp_parts, d.log_prob_parts(*xs))
    self.assertAllCloseNested(expected_lp_parts,
                              d.log_prob_parts(**dict(zip(keys, xs))))

  def test_kl_divergence(self):
    d0 = jds.JointDistributionSequential([
        independent.Independent(exponential.Exponential(rate=[100, 120]), 1),
        normal.Normal(loc=0, scale=2.),
    ],
                                         validate_args=True)
    d1 = jds.JointDistributionSequential([
        independent.Independent(exponential.Exponential(rate=[10, 12]), 1),
        normal.Normal(loc=1, scale=1.),
    ],
                                         validate_args=True)
    expected_kl = sum(kullback_leibler.kl_divergence(d0_, d1_) for d0_, d1_
                      in zip(d0.model, d1.model))
    actual_kl = kullback_leibler.kl_divergence(d0, d1)
    other_actual_kl = d0.kl_divergence(d1)
    expected_kl_, actual_kl_, other_actual_kl_ = self.evaluate([
        expected_kl, actual_kl, other_actual_kl])
    self.assertNear(expected_kl_, actual_kl_, err=1e-5)
    self.assertNear(expected_kl_, other_actual_kl_, err=1e-5)

  def test_cross_entropy(self):
    d0 = jds.JointDistributionSequential([
        independent.Independent(exponential.Exponential(rate=[100, 120]), 1),
        normal.Normal(loc=0, scale=2.),
    ],
                                         validate_args=True)
    d1 = jds.JointDistributionSequential([
        independent.Independent(exponential.Exponential(rate=[10, 12]), 1),
        normal.Normal(loc=1, scale=1.),
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
      d = jds.JointDistributionSequential(
          [normal.Normal(0., 1.), bernoulli.Bernoulli])
      d.sample(seed=test_util.test_seed())

  def test_graph_resolution(self):
    d = jds.JointDistributionSequential([
        independent.Independent(exponential.Exponential(rate=[100, 120]), 1),
        lambda e: gamma.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        half_normal.HalfNormal(2.5),
        lambda s: normal.Normal(loc=0, scale=s),
        exponential.Exponential(2),
        lambda df, loc, _, scale: student_t.StudentT(df, loc, scale),
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
    d = jds.JointDistributionSequential(
        [normal.Normal(0., 1.),
         bernoulli.Bernoulli(logits=0.)],
        validate_args=True)
    expected = tuple(getattr(d_, attr)() for d_ in d.model)
    actual = getattr(d, attr)()
    self.assertAllEqual(*self.evaluate([expected, actual]))

  @parameterized.parameters(('covariance',))
  def test_notimplemented_summary_statistic(self, attr):
    d = jds.JointDistributionSequential(
        [normal.Normal(0., 1.),
         bernoulli.Bernoulli(probs=0.5)],
        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionSequential'):
      getattr(d, attr)()

  @parameterized.parameters(
      'log_cdf', 'cdf', 'log_survival_function', 'survival_function')
  def test_notimplemented_evaluative_statistic(self, attr):
    d = jds.JointDistributionSequential(
        [normal.Normal(0., 1.),
         bernoulli.Bernoulli(probs=0.5)],
        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        attr + ' is not implemented: JointDistributionSequential'):
      getattr(d, attr)([0.]*len(d.model))

  def test_notimplemented_quantile(self):
    d = jds.JointDistributionSequential(
        [normal.Normal(0., 1.),
         bernoulli.Bernoulli(probs=0.5)],
        validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        NotImplementedError,
        'quantile is not implemented: JointDistributionSequential'):
      d.quantile(0.5)

  def test_copy(self):
    pgm = [normal.Normal(0., 1.), bernoulli.Bernoulli(probs=0.5)]
    d = jds.JointDistributionSequential(pgm, validate_args=True)
    d_copy = d.copy()
    self.assertAllEqual(d_copy.parameters['model'], pgm)
    self.assertAllEqual(d_copy.parameters['validate_args'], True)

  def test_batch_slicing(self):
    d = jds.JointDistributionSequential([
        exponential.Exponential(rate=[10, 12, 14]),
        lambda s: normal.Normal(loc=0, scale=s),
        lambda: beta_lib.Beta(concentration0=[3, 2, 1], concentration1=1),
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
    d = jds.JointDistributionSequential([
        independent.Independent(exponential.Exponential(rate=[100, 120]), 1),
        lambda e: gamma.Gamma(concentration=e[..., 0], rate=e[..., 1]),
        half_normal.HalfNormal(2.5),
        lambda s: normal.Normal(loc=0, scale=s),
        exponential.Exponential(2),
        lambda df, loc, _, scale: student_t.StudentT(df, loc, scale),
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

  def test_argspec(self):
    argspec = inspect.getfullargspec(Dummy)
    self.assertAllEqual(['me', 'arg1', 'arg2', 'arg3'], argspec.args)
    self.assertIs(None, argspec.varargs)
    self.assertIs('named', argspec.varkw)
    self.assertAllEqual((None,), argspec.defaults)

  def test_invalid_structure_raises_error(self):
    with self.assertRaisesWithPredicateMatch(
        TypeError, 'Unable to unflatten like `model` with type "model".'):
      jds.JointDistributionSequential(
          collections.namedtuple('model', 'a b')(
              a=normal.Normal(0, 1), b=normal.Normal(1, 2)),
          validate_args=True)

  def test_simple_example_with_dynamic_shapes(self):
    dist = jds.JointDistributionSequential([
        normal.Normal(
            tf1.placeholder_with_default(0., shape=None),
            tf1.placeholder_with_default(1., shape=None)),
        lambda a: normal.Normal(a, 1.)
    ],
                                           validate_args=True)
    lp = dist.log_prob(dist.sample(5, seed=test_util.test_seed()))
    self.assertAllEqual(self.evaluate(lp).shape, [5])

  def test_dist_fn_takes_varargs(self):
    dist = jds.JointDistributionSequential(
        [
            scale.Scale(-1.)(exponential.Exponential(1.)),  # Negative.
            lambda *args: exponential.Exponential(tf.exp(args[0])),  # Positive.
            lambda *args: normal.Normal(  # pylint: disable=g-long-lambda
                loc=args[1],
                scale=args[0],  # Must be positive.
                validate_args=True)
        ],
        validate_args=True)
    lp = dist.log_prob(dist.sample(5, seed=test_util.test_seed()))
    self.assertAllEqual(lp.shape, [5])

  @parameterized.named_parameters(
      ('_sample', lambda d, **kwargs: d.sample(**kwargs)),
      ('_sample_and_log_prob',
       lambda d, **kwargs: d.experimental_sample_and_log_prob(**kwargs)[0]),
  )
  def test_nested_partial_value(self, sample_fn):
    innermost = jds.JointDistributionSequential((
        exponential.Exponential(1.),
        lambda a: sample.Sample(lognormal.LogNormal(a, a), [5]),
    ))

    inner = jds.JointDistributionSequential((
        exponential.Exponential(1.),
        innermost,
    ))

    outer = jds.JointDistributionSequential((
        exponential.Exponential(1.),
        inner,
    ))

    seed = test_util.test_seed(sampler_type='stateless')
    true_xs = outer.sample(seed=seed)

    def _update(tuple_, index, value):
      res = list(tuple_)
      res[index] = value
      return tuple(res)

    # These asserts work because we advance the stateless seed inside the model
    # whether or not a sample is actually generated.
    partial_xs = _update(true_xs, 1, None)
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

    partial_xs = _update(true_xs, 0, None)
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

    partial_xs = _update(true_xs, 1, _update(true_xs[1], 1, None))
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

    partial_xs = _update(
        true_xs, 1, _update(true_xs[1], 1, _update(true_xs[1][1], 0, None)))
    xs = sample_fn(outer, value=partial_xs, seed=seed)
    self.assertAllCloseNested(true_xs, xs)

  @parameterized.named_parameters(
      ('basic', basic_model_fn),
      ('nested_lists', nested_lists_model_fn))
  def test_can_call_log_prob_with_args_and_kwargs(self, model_fn):
    d = jds.JointDistributionSequential(model_fn(), validate_args=True)

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

  def test_can_call_prob_with_args_and_kwargs(self):
    d = jds.JointDistributionSequential(basic_model_fn(), validate_args=True)
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
    joint = jds.JointDistributionSequential([
        mvn_diag.MultivariateNormalDiag([0., 0.], [1., 1.]),
        lambda a: jds.JointDistributionSequential([  # pylint: disable=g-long-lambda
            jds.JointDistributionSequential([normal.Normal(a[..., 0], 1.)]),
            normal.Normal(a[..., 1], 1.)
        ])
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

    dist = jds.JointDistributionSequential(
        [
            sample.Sample(
                normal.Normal(loc=0., scale=user_trait_prior_scale),
                sample_shape=[n_factors, n_users]),  # U
            sample.Sample(
                normal.Normal(loc=0., scale=item_trait_prior_scale),
                sample_shape=[n_factors, n_items]),  # V
            lambda item_traits, user_traits: independent.Independent(  # pylint: disable=g-long-lambda
                normal.Normal(
                    loc=tf.matmul(
                        user_traits,
                        item_traits,  # R
                        adjoint_a=True),
                    scale=observation_noise_prior_scale),
                reinterpreted_batch_ndims=2)
        ],
        validate_args=True)
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
        u.sample([num_topics], seed=test_util.test_seed()),
        softplus.Softplus(),
        name='alpha')
    beta = tf.Variable(u.sample([num_topics, num_words],
                                seed=test_util.test_seed()), name='beta')

    # LDA Model.
    # Note near 1:1 with mathematical specification. The main distinction is the
    # use of Independent--this lets us easily aggregate multinomials across
    # topics (and in any "shape" of documents).
    lda = jds.JointDistributionSequential(
        [
            poisson.Poisson(rate=avg_doc_length),  # n
            dirichlet.Dirichlet(concentration=alpha),  # theta
            lambda theta, n: multinomial.Multinomial(  # pylint: disable=g-long-lambda
                total_count=n, probs=theta),  # z
            lambda z: independent.Independent(  # x  pylint: disable=g-long-lambda
                multinomial.Multinomial(total_count=z, logits=beta),
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
            poisson.Poisson(rate=15.).sample(n[0], seed=seed()),
            poisson.Poisson(rate=25.).sample(n[1], seed=seed()),
        ],
                  axis=0),
        dtype=tf.float32)
    count_data = self.evaluate(count_data)
    n = np.sum(n)

    # Make model.
    gather = lambda tau, lambda_: tf.gather(  # pylint: disable=g-long-lambda
        lambda_,
        indices=tf.cast(
            tau[..., tf.newaxis] < tf.linspace(0., 1., n),
            dtype=tf.int32),
        batch_dims=ps.rank(tau))

    alpha = tf.math.reciprocal(tf.reduce_mean(count_data))

    joint = jds.JointDistributionSequential(
        [
            sample.Sample(
                exponential.Exponential(rate=alpha), sample_shape=[2]),
            uniform.Uniform(),
            lambda tau, lambda_: independent.Independent(  # pylint: disable=g-long-lambda
                poisson.Poisson(rate=gather(tau, lambda_)),
                reinterpreted_batch_ndims=1),
        ],
        validate_args=True)

    # Verify model correctly "compiles".
    batch_shape = [3, 4]
    self.assertEqual(
        batch_shape,
        joint.log_prob(
            joint.sample(batch_shape, seed=test_util.test_seed())).shape)

  def test_default_event_space_bijector(self):
    # Define dist parameters that also parameterize the event space
    # bijector outside of the distribution constructor to ensure that
    # bijector caching works.
    low = tf.constant([0., 0.], dtype=tf.float32)
    dist_fns = [
        lognormal.LogNormal(0., 1., validate_args=True),
        lambda h: independent.Independent(  # pylint: disable=g-long-lambda
            uniform.Uniform(low, h, validate_args=True)),
        lambda s: normal.Normal(0., s, validate_args=True)
    ]

    jd = jds.JointDistributionSequential(dist_fns, validate_args=True)
    joint_bijector = jd.experimental_default_event_space_bijector()

    # define a sample in the unconstrained space and construct the component
    # distributions
    x = [tf.constant(w) for w in [-0.2, [0.3, 0.1], -1.]]
    bijectors = []
    y = []

    b = dist_fns[0].experimental_default_event_space_bijector()
    bijectors.append(b)
    y.append(b(x[0]))
    for i in range(1, 3):
      b = dist_fns[i](y[i - 1]).experimental_default_event_space_bijector()
      y.append(b(x[i]))
      bijectors.append(b)

    # Test forward and inverse values.
    self.assertAllCloseNested(joint_bijector.forward(x), y)
    self.assertAllCloseNested(joint_bijector.inverse(y), x)

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
    bijectors.append(dist_fns[0].experimental_default_event_space_bijector())
    for i in range(1, 3):
      bijectors.append(
          dist_fns[i](y[i - 1]).experimental_default_event_space_bijector())

    inverse_plus = [b.inverse(y[i] + delta) for i, b in enumerate(bijectors)]
    inverse_minus = [b.inverse(y[i] - delta) for i, b in enumerate(bijectors)]
    ildj_fd = tf.reduce_sum(
        [tf.reduce_sum(tf.math.log((p - m) / (2. * delta)))
         for p, m in zip(inverse_plus, inverse_minus)])
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
          self.evaluate(
              bijectors[i].inverse_event_shape_tensor(event_shapes[i])))

    # test shared cache
    joint_bijector_2 = jd.experimental_default_event_space_bijector()
    y_1 = joint_bijector.forward(x)
    y_2 = joint_bijector_2.forward(x)
    for a, b in zip(y_1, y_2):
      self.assertIs(a, b)

    x_1 = joint_bijector.inverse(y_1)
    x_2 = joint_bijector_2.inverse(y_1)
    for a, b in zip(x_1, x_2):
      self.assertIs(a, b)

  def test_sample_kwargs(self):
    joint = jds.JointDistributionSequential([
        normal.Normal(0., 1.), lambda a: normal.Normal(a, 1.),
        lambda b, a: normal.Normal(a + b, 1.)
    ])

    seed = test_util.test_seed()
    tf.random.set_seed(seed)
    samples = joint.sample(seed=seed, a=1.)
    # Check the first value is actually 1.
    self.assertEqual(1., self.evaluate(samples[0]))

    # Check the sample is reproducible using the `value` argument.
    tf.random.set_seed(seed)
    samples_tuple = joint.sample(seed=seed, value=[1., None, None])
    self.assertAllEqual(self.evaluate(samples), self.evaluate(samples_tuple))

    # Make sure to throw an exception if strange keywords are passed.
    expected_error = (
        'Found unexpected keyword arguments. Distribution names are\n'
        'a, b, x\n'
        'but received\n'
        'z\n'
        'These names were invalid:\n'
        'z')
    with self.assertRaisesRegex(ValueError, expected_error):
      joint.sample(seed=seed, z=2.)

    # Also raise if value and keywords are passed
    with self.assertRaisesRegex(
        ValueError, r'Supplied both `value` and keyword arguments .*'):
      joint.sample(seed=seed, a=1., value=[1., None, None])

  def test_creates_valid_coroutine(self):
    joint = jds.JointDistributionSequential(
        [
            poisson.Poisson(rate=100.),
            dirichlet.Dirichlet(concentration=[1., 1.]),
            lambda theta, n: multinomial.Multinomial(  # pylint: disable=g-long-lambda
                total_count=n, probs=theta),
            lambda z: independent.Independent(  # pylint: disable=g-long-lambda
                multinomial.Multinomial(
                    total_count=z, logits=[[0., 1., 2.], [3., 4., 5.]]),
                reinterpreted_batch_ndims=1),
        ],
        validate_args=True)
    sample_shapes = [
        x.shape for x in joint._model_flatten(
            joint.sample([5], seed=test_util.test_seed()))]

    jd = jdc.JointDistributionCoroutine(joint._model_coroutine)
    jdc_sample_shapes = [
        x.shape for x in jd._model_flatten(
            jd.sample([5], seed=test_util.test_seed()))]
    self.assertAllEqualNested(sample_shapes, jdc_sample_shapes)

  def test_init_does_not_execute_model(self):
    model_traces = []
    def record_model_called(x):
      model_traces.append(x)
      return x

    model = jds.JointDistributionSequential([
        normal.Normal(0., 1.),
        lambda z: normal.Normal(record_model_called(z), 1.)
    ],
                                            validate_args=True)
    # Model should not be called from init.
    self.assertLen(model_traces, 0)
    model.sample(seed=test_util.test_seed())
    # The first sample call will run the model twice (for shape
    # inference + actually sampling).
    self.assertLen(model_traces, 2)
    # Subsequent calls should only run the model once.
    model.sample([2], seed=test_util.test_seed())
    self.assertLen(model_traces, 3)

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='Numpy has no notion of CompositeTensor/Pytree.')
  def testCompositeTensorOrPytree(self):
    d = jds.JointDistributionSequential(
        tuple([
            independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1),
            lambda e: gamma.Gamma(concentration=e[..., 0], rate=e[..., 1]),
            normal.Normal(loc=0, scale=2.),
            normal
            .Normal,  # Or, `lambda loc, scale: normal.Normal(loc, scale)`.
            lambda m: sample.Sample(bernoulli.Bernoulli(logits=m), 12),
        ]),
        validate_args=True)

    flat = tf.nest.flatten(d, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(
        d, flat, expand_composites=True)
    self.assertIsInstance(unflat, jds.JointDistributionSequential)
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
      reason='Numpy and JAX have no notion of type spec serialization.')
  def testCompositeTensorSerialization(self):
    encodable_jd = jds.JointDistributionSequential(  # no lambdas
        [
            independent.Independent(
                exponential.Exponential(rate=[100, 120]), 1),
            normal.Normal(loc=0, scale=2.),
            sample.Sample(bernoulli.Bernoulli(logits=0.), 12),
        ],
        validate_args=True)

    enc = tf.__internal__.saved_model.encode_structure(encodable_jd._type_spec)
    dec = tf.__internal__.saved_model.decode_proto(enc)
    self.assertEqual(dec, encodable_jd._type_spec)

    non_ct_jd = jds.JointDistributionSequential([
        normal.Normal(loc=0., scale=1.),
        bijector_test_util.NonCompositeTensorExp()(
            normal.Normal(loc=0., scale=1.)),
    ],
                                                validate_args=True)
    self.assertNotIsInstance(non_ct_jd, tf.__internal__.CompositeTensor)


class ResolveDistributionNamesTest(test_util.TestCase):

  def test_dummy_names_are_unique(self):

    dist_names = jds._resolve_distribution_names(
        dist_fn_args=[None, None, None],
        dist_names=None,
        leaf_name='x',
        instance_names=[None, None, None])
    self.assertAllEqual(dist_names, ['x2', 'x1', 'x'])

    dist_names = jds._resolve_distribution_names(
        dist_fn_args=[None, None, None],
        dist_names=None,
        leaf_name='x',
        instance_names=['x', 'x1', None])
    self.assertAllEqual(dist_names, ['x', 'x1', 'x2'])

  def test_ignores_trivial_names(self):

    # Should ignore a trivial reference downstream of the real name `z`.
    dist_names = jds._resolve_distribution_names(
        dist_fn_args=[None, ['z'], ['w', '_']],
        dist_names=None,
        leaf_name='y',
        instance_names=[None, None, None])
    self.assertAllEqual(dist_names, ['z', 'w', 'y'])

    # Trivial reference upstream of the real name `z`.
    dist_names = jds._resolve_distribution_names(
        dist_fn_args=[None, ['_'], ['w', 'z']],
        dist_names=None,
        leaf_name='y',
        instance_names=[None, None, None])
    self.assertAllEqual(dist_names, ['z', 'w', 'y'])

    # The only direct reference is trivial, but we have an instance name.
    dist_names = jds._resolve_distribution_names(
        dist_fn_args=[None, ['_']],
        dist_names=None,
        leaf_name='y',
        instance_names=['z', None])
    self.assertAllEqual(dist_names, ['z', 'y'])

  def test_inconsistent_names_raise_error(self):
    with self.assertRaisesRegexp(ValueError, 'Inconsistent names'):
      # Refers to first variable as both `z` and `x`.
      jds._resolve_distribution_names(
          dist_fn_args=[None, ['z'], ['x', 'w']],
          dist_names=None,
          leaf_name='y',
          instance_names=[None, None, None])

    with self.assertRaisesRegexp(ValueError, 'Inconsistent names'):
      # Refers to first variable as `x`, but it was explicitly named `z`.
      jds._resolve_distribution_names(
          dist_fn_args=[None, ['x']],
          dist_names=None,
          leaf_name='y',
          instance_names=['z', None])


if __name__ == '__main__':
  test_util.main()
