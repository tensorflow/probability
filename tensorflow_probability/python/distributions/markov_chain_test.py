# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for MarkovChain."""

from absl.testing import parameterized

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import split
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import dirichlet
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import inverse_gamma
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import linear_gaussian_ssm as lgssm_lib
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.distributions import markov_chain as markov_chain_lib
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.distributions import wishart
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


@test_util.test_graph_and_eager_modes
class MarkovChainTest(test_util.TestCase):

  def test_error_when_transition_modifies_batch_shape(self):
    loses_batch_shape = markov_chain_lib.MarkovChain(
        initial_state_prior=normal.Normal(loc=0., scale=[1., 1.]),
        transition_fn=lambda _, x: independent.Independent(  # pylint: disable=g-long-lambda
            normal.Normal(loc=0., scale=tf.ones_like(x)),
            reinterpreted_batch_ndims=1),
        num_steps=5)
    x = self.evaluate(loses_batch_shape.sample([2], seed=test_util.test_seed()))
    with self.assertRaisesRegex(ValueError, 'batch shape is incorrect'):
      loses_batch_shape.log_prob(x)

    gains_batch_shape = markov_chain_lib.MarkovChain(
        initial_state_prior=independent.Independent(
            normal.Normal(loc=0., scale=[1., 1.]), reinterpreted_batch_ndims=1),
        transition_fn=lambda _, x: normal.Normal(loc=0., scale=tf.ones_like(x)),
        num_steps=5)
    x = self.evaluate(gains_batch_shape.sample([2], seed=test_util.test_seed()))
    with self.assertRaisesRegex(ValueError, 'batch shape is incorrect'):
      gains_batch_shape.log_prob(x)

  def test_log_prob_matches_linear_gaussian_ssm(self):
    dim = 2
    batch_shape = [3, 1]
    seed, *model_seeds = samplers.split_seed(test_util.test_seed(), n=6)

    # Sample a random linear Gaussian process.
    prior_loc = self.evaluate(
        normal.Normal(0., 1.).sample(batch_shape + [dim], seed=model_seeds[0]))
    prior_scale = self.evaluate(
        inverse_gamma.InverseGamma(1., 1.).sample(
            batch_shape + [dim], seed=model_seeds[1]))
    transition_matrix = self.evaluate(
        normal.Normal(0., 1.).sample([dim, dim], seed=model_seeds[2]))
    transition_bias = self.evaluate(
        normal.Normal(0., 1.).sample(batch_shape + [dim], seed=model_seeds[3]))
    transition_scale_tril = self.evaluate(
        tf.linalg.cholesky(
            wishart.WishartTriL(
                df=dim, scale_tril=tf.eye(dim)).sample(seed=model_seeds[4])))

    initial_state_prior = mvn_diag.MultivariateNormalDiag(
        loc=prior_loc, scale_diag=prior_scale, name='initial_state_prior')

    lgssm = lgssm_lib.LinearGaussianStateSpaceModel(
        num_timesteps=7,
        transition_matrix=transition_matrix,
        transition_noise=mvn_tril.MultivariateNormalTriL(
            loc=transition_bias, scale_tril=transition_scale_tril),
        # Trivial observation model to pass through the latent state.
        observation_matrix=tf.eye(dim),
        observation_noise=mvn_diag.MultivariateNormalDiag(
            loc=tf.zeros(dim), scale_diag=tf.zeros(dim)),
        initial_state_prior=initial_state_prior)

    markov_chain = markov_chain_lib.MarkovChain(
        initial_state_prior=initial_state_prior,
        transition_fn=lambda _, x: mvn_tril.MultivariateNormalTriL(  # pylint: disable=g-long-lambda
            loc=tf.linalg.matvec(transition_matrix, x) + transition_bias,
            scale_tril=transition_scale_tril),
        num_steps=7)

    x = markov_chain.sample(5, seed=seed)
    self.assertAllClose(lgssm.log_prob(x), markov_chain.log_prob(x), rtol=5e-4)

  @test_util.numpy_disable_test_missing_functionality(
      'JointDistributionNamedAutoBatched')
  def test_docstring_example_autoregressive_process(self):

    def transition_fn(_, previous_state):
      return jdab.JointDistributionNamedAutoBatched(
          # The previous state may include batch dimensions. Since the log scale
          # is a scalar quantity, its shape is the batch shape.
          batch_ndims=ps.rank(previous_state['log_scale']),
          model={
              # The autoregressive coefficients and the `log_scale` each follow
              # an independent slow-moving random walk.
              'coefs':
                  normal.Normal(loc=previous_state['coefs'], scale=0.01),
              'log_scale':
                  normal.Normal(loc=previous_state['log_scale'], scale=0.01),
              # The level is a linear combination of the previous *two* levels,
              # with additional noise of scale `exp(log_scale)`.
              'level':
                  lambda coefs, log_scale: normal.Normal(  # pylint: disable=g-long-lambda
                      loc=(coefs[..., 0] * previous_state['level'] + coefs[
                          ..., 1] * previous_state['previous_level']),
                      scale=tf.exp(log_scale)),
              # Store the previous level to access at the next step.
              'previous_level':
                  deterministic.Deterministic(previous_state['level'])
          })

    process = markov_chain_lib.MarkovChain(
        # For simplicity, define the prior as a 'transition' from fixed values.
        initial_state_prior=transition_fn(
            0,
            previous_state={
                'coefs': [0.7, -0.2],
                'log_scale': -1.,
                'level': 0.,
                'previous_level': 0.
            }),
        transition_fn=transition_fn,
        num_steps=100)
    self.assertAllEqualNested(process.event_shape,
                              {'coefs': [100, 2], 'log_scale': [100],
                               'level': [100], 'previous_level': [100]})
    self.assertAllEqual(process.batch_shape, [])

    x = process.sample(5, seed=test_util.test_seed())
    self.assertAllEqual(x['coefs'].shape, [5, 100, 2])
    self.assertAllEqual(x['log_scale'].shape, [5, 100])
    self.assertAllEqual(x['level'].shape, [5, 100])
    self.assertAllEqual(x['previous_level'].shape, [5, 100])

    lp = process.log_prob(x)
    self.assertAllEqual(lp.shape, [5])

    x2, lp2 = process.experimental_sample_and_log_prob(
        2, seed=test_util.test_seed())
    self.assertAllClose(lp2, process.log_prob(x2))

  @parameterized.named_parameters(
      ('float32_dynamic', tf.float32, True),
      ('float64_static', tf.float64, False))
  def test_docstring_example_batch_gaussian_walk(self,
                                                 float_dtype,
                                                 use_dynamic_shapes):
    if tf.executing_eagerly() and use_dynamic_shapes:
      self.skipTest('No dynamic shapes in eager mode.')
    def _as_tensor(x, dtype=None):
      x = ps.cast(x, dtype=dtype if dtype else float_dtype)
      if use_dynamic_shapes:
        x = tf1.placeholder_with_default(x, shape=None)
      return x

    scales = _as_tensor([0.5, 0.3, 0.2, 0.2, 0.3, 0.2, 0.7])
    batch_gaussian_walk = markov_chain_lib.MarkovChain(
        # The prior distribution determines the batch shape for the chain.
        # Transitions must respect this batch shape.
        initial_state_prior=normal.Normal(
            loc=_as_tensor([-10., 0., 10.]), scale=_as_tensor([1., 1., 1.])),
        transition_fn=lambda t, x: normal.Normal(  # pylint: disable=g-long-lambda
            loc=x,
            # The `num_steps` dimension will always be leftmost in `x`, so we
            # pad the scale to the same rank as `x` so that the shapes line up.
            scale=tf.reshape(
                tf.gather(scales, t),
                ps.concat([[-1], ps.ones(ps.rank(x) - 1, dtype=tf.int32)],
                          axis=0))),
        # Limit to eight steps since we only specified scales for seven
        # transitions.
        num_steps=8)
    self.assertAllEqual(batch_gaussian_walk.event_shape_tensor(), [8])
    self.assertAllEqual(batch_gaussian_walk.batch_shape_tensor(), [3])

    x = batch_gaussian_walk.sample(5, seed=test_util.test_seed())
    self.assertAllEqual(ps.shape(x), [5, 3, 8])
    lp = batch_gaussian_walk.log_prob(x)
    self.assertAllEqual(ps.shape(lp), [5, 3])

    x2, lp2 = batch_gaussian_walk.experimental_sample_and_log_prob(
        [2], seed=test_util.test_seed())
    self.assertAllClose(lp2, batch_gaussian_walk.log_prob(x2))

  def test_docstring_example_gaussian_walk(self):
    gaussian_walk = markov_chain_lib.MarkovChain(
        initial_state_prior=normal.Normal(loc=0., scale=1.),
        transition_fn=lambda _, x: normal.Normal(loc=x, scale=1.),
        num_steps=100)
    self.assertAllEqual(gaussian_walk.event_shape, [100])
    self.assertAllEqual(gaussian_walk.batch_shape, [])

    x = gaussian_walk.sample(5, seed=test_util.test_seed())
    self.assertAllEqual(x.shape, [5, 100])
    lp = gaussian_walk.log_prob(x)
    self.assertAllEqual(lp.shape, [5])

    n = normal.Normal(0., 1.)
    expected_lp = (n.log_prob(x[:, 0]) +
                   tf.reduce_sum(n.log_prob(x[:, 1:] - x[:, :-1]), axis=-1))
    self.assertAllClose(lp, expected_lp)

    x2, lp2 = gaussian_walk.experimental_sample_and_log_prob(
        [2], seed=test_util.test_seed())
    self.assertAllClose(lp2, gaussian_walk.log_prob(x2))

  def test_non_autobatched_joint_distribution(self):

    def transition_fn(_, previous_state):
      return jdn.JointDistributionNamed({
          # The autoregressive coefficients and the `log_scale` each follow
          # an independent slow-moving random walk.
          'coefs':
              independent.Independent(
                  normal.Normal(loc=previous_state['coefs'], scale=0.01),
                  reinterpreted_batch_ndims=1),
          'log_scale':
              normal.Normal(loc=previous_state['log_scale'], scale=0.01),
          # The level is a linear combination of the previous *two* levels,
          # with additional noise of scale `exp(log_scale)`.
          'level':
              lambda coefs, log_scale: normal.Normal(  # pylint: disable=g-long-lambda
                  loc=(coefs[..., 0] * previous_state['level'] + coefs[..., 1] *
                       previous_state['previous_level']),
                  scale=tf.exp(log_scale)),
          # Store the previous level to access at the next step.
          'previous_level':
              deterministic.Deterministic(previous_state['level'])
      })

    process = markov_chain_lib.MarkovChain(
        # For simplicity, define the prior as a 'transition' from fixed values.
        initial_state_prior=transition_fn(
            0,
            previous_state={
                'coefs': [0.7, -0.2],
                'log_scale': -1.,
                'level': 0.,
                'previous_level': 0.
            }),
        transition_fn=transition_fn,
        num_steps=100)
    self.assertAllEqualNested(process.event_shape,
                              {'coefs': [100, 2], 'log_scale': [100],
                               'level': [100], 'previous_level': [100]})
    self.assertAllEqual(process.batch_shape,
                        {'coefs': [], 'log_scale': [],
                         'level': [], 'previous_level': []})

    x = process.sample(5, seed=test_util.test_seed())
    self.assertAllEqual(x['coefs'].shape, [5, 100, 2])
    self.assertAllEqual(x['log_scale'].shape, [5, 100])
    self.assertAllEqual(x['level'].shape, [5, 100])
    self.assertAllEqual(x['previous_level'].shape, [5, 100])

    lp = process.log_prob(x)
    self.assertAllEqual(lp.shape, [5])

    x2, lp2 = process.experimental_sample_and_log_prob(
        2, seed=test_util.test_seed())
    self.assertAllClose(lp2, process.log_prob(x2))

  def test_log_prob_ratio(self):
    p = markov_chain_lib.MarkovChain(
        initial_state_prior=normal.Normal(0., 1.),
        transition_fn=lambda _, x: normal.Normal(x, tf.nn.softplus(x)),
        num_steps=10)
    q = markov_chain_lib.MarkovChain(
        initial_state_prior=normal.Normal(-10, 3.),
        transition_fn=lambda _, x: normal.Normal(x, tf.abs(x)),
        num_steps=10)
    x = self.evaluate(p.sample(4, seed=test_util.test_seed()))
    y = self.evaluate(q.sample(4, seed=test_util.test_seed()))
    self.assertAllClose(
        p.log_prob(x) - q.log_prob(y),
        log_prob_ratio.log_prob_ratio(p, x, q, y), atol=1e-5)

  def test_unexpected_num_steps_raises(self):
    p = markov_chain_lib.MarkovChain(
        initial_state_prior=normal.Normal(0., 1.),
        transition_fn=lambda _, x: normal.Normal(x, tf.nn.softplus(x)),
        num_steps=10,
        validate_args=True)
    with self.assertRaisesRegex(
        (ValueError, tf.errors.InvalidArgumentError),
        'does not match the expected num_steps'):
      p.log_prob(tf.zeros([11]))


@test_util.test_graph_and_eager_modes
class MarkovChainBijectorTest(test_util.TestCase):

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      dict(
          testcase_name='deterministic_prior',
          prior_fn=lambda: deterministic.Deterministic([-100., 0., 100.]),
          transition_fn=lambda _, x: normal.Normal(loc=x, scale=1.)),
      dict(
          testcase_name='deterministic_transition',
          prior_fn=lambda: normal.Normal(loc=[-100., 0., 100.], scale=1.),
          transition_fn=lambda _, x: deterministic.Deterministic(x)),
      dict(
          testcase_name='fully_deterministic',
          prior_fn=lambda: deterministic.Deterministic([-100., 0., 100.]),
          transition_fn=lambda _, x: deterministic.Deterministic(x)),
      dict(
          testcase_name='mvn_diag',
          prior_fn=(lambda: mvn_diag.MultivariateNormalDiag(
              loc=[[2.], [2.]], scale_diag=[1.])),
          transition_fn=lambda _, x: deterministic.VectorDeterministic(x)),
      dict(
          testcase_name='docstring_dirichlet',
          prior_fn=lambda: jdab.JointDistributionNamedAutoBatched(
              {'probs': dirichlet.Dirichlet([1., 1.])}),
          transition_fn=lambda _, x: jdab.JointDistributionNamedAutoBatched(
              {
                  'probs':
                      mvn_diag.MultivariateNormalDiag(
                          loc=x['probs'], scale_diag=[0.1, 0.1])
              },
              batch_ndims=ps.rank(x['probs']))),
      dict(
          testcase_name='uniform_step',
          prior_fn=lambda: exponential.Exponential(tf.ones([4, 1])),
          transition_fn=lambda _, x: uniform.Uniform(low=x, high=x + 1.)),
      dict(
          testcase_name='joint_distribution',
          prior_fn=lambda: jdab.JointDistributionNamedAutoBatched(
              batch_ndims=2,
              model={
                  'a':
                      gamma.Gamma(tf.zeros([5]), 1.),
                  'b':
                      lambda a: (reshape.Reshape(
                          event_shape_in=[4, 3], event_shape_out=[2, 3, 2])
                                 (independent.Independent(
                                     normal.Normal(
                                         loc=tf.zeros([5, 4, 3]),
                                         scale=a[..., tf.newaxis, tf.newaxis]),
                                     reinterpreted_batch_ndims=2)))
              }),
          transition_fn=lambda _, x: jdab.JointDistributionNamedAutoBatched(
              batch_ndims=ps.rank_from_shape(x['a'].shape),
              model={
                  'a':
                      normal.Normal(loc=x['a'], scale=1.),
                  'b':
                      lambda a: deterministic.Deterministic(x['b'] + a[
                          ..., tf.newaxis, tf.newaxis, tf.newaxis])
              })),
      dict(
          testcase_name='nested_chain',
          prior_fn=lambda: markov_chain_lib.MarkovChain(
              initial_state_prior=split.Split(2)
              (mvn_diag.MultivariateNormalDiag(0., [1., 2.])),
              transition_fn=lambda _, x: split.Split(2)
              (mvn_diag.MultivariateNormalDiag(x[0], [1., 2.])),
              num_steps=6),
          transition_fn=(
              lambda _, x: jdab.JointDistributionSequentialAutoBatched(
                  [
                      mvn_diag.MultivariateNormalDiag(x[0], [1.]),
                      mvn_diag.MultivariateNormalDiag(x[1], [1.])
                  ],
                  batch_ndims=ps.rank(x[0])))))
  # pylint: enable=g-long-lambda
  def test_default_bijector(self, prior_fn, transition_fn):
    chain = markov_chain_lib.MarkovChain(
        initial_state_prior=prior_fn(),
        transition_fn=transition_fn,
        num_steps=7)

    y = self.evaluate(chain.sample(seed=test_util.test_seed()))
    bijector = chain.experimental_default_event_space_bijector()

    self.assertAllEqual(chain.batch_shape_tensor(),
                        bijector.experimental_batch_shape_tensor())

    x = bijector.inverse(y)
    yy = bijector.forward(
        tf.nest.map_structure(tf.identity, x))  # Bypass bijector cache.
    self.assertAllCloseNested(y, yy)

    chain_event_ndims = tf.nest.map_structure(
        ps.rank_from_shape, chain.event_shape_tensor())
    self.assertAllEqualNested(bijector.inverse_min_event_ndims,
                              chain_event_ndims)

    ildj = bijector.inverse_log_det_jacobian(
        tf.nest.map_structure(tf.identity, y),  # Bypass bijector cache.
        event_ndims=chain_event_ndims)
    if not bijector.is_constant_jacobian:
      self.assertAllEqual(ildj.shape, chain.batch_shape)
    fldj = bijector.forward_log_det_jacobian(
        tf.nest.map_structure(tf.identity, x),  # Bypass bijector cache.
        event_ndims=bijector.inverse_event_ndims(chain_event_ndims))
    self.assertAllClose(ildj, -fldj)

    # Verify that event shapes are passed through and flattened/unflattened
    # correctly.
    inverse_event_shapes = bijector.inverse_event_shape(chain.event_shape)
    x_event_shapes = tf.nest.map_structure(
        lambda t, nd: t.shape[ps.rank(t) - nd:],
        x, bijector.forward_min_event_ndims)
    self.assertAllEqualNested(inverse_event_shapes, x_event_shapes)
    forward_event_shapes = bijector.forward_event_shape(inverse_event_shapes)
    self.assertAllEqualNested(forward_event_shapes, chain.event_shape)

    # Verify that the outputs of other methods have the correct structure.
    inverse_event_shape_tensors = bijector.inverse_event_shape_tensor(
        chain.event_shape_tensor())
    self.assertAllEqualNested(inverse_event_shape_tensors, x_event_shapes)
    forward_event_shape_tensors = bijector.forward_event_shape_tensor(
        inverse_event_shape_tensors)
    self.assertAllEqualNested(forward_event_shape_tensors,
                              chain.event_shape_tensor())


if __name__ == '__main__':
  test_util.main()
