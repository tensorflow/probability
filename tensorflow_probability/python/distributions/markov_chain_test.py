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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


@test_util.test_graph_and_eager_modes
class MarkovChainTest(test_util.TestCase):

  def test_error_when_transition_modifies_batch_shape(self):
    loses_batch_shape = tfd.MarkovChain(
        initial_state_prior=tfd.Normal(loc=0., scale=[1., 1.]),
        transition_fn=lambda _, x: tfd.Independent(  # pylint: disable=g-long-lambda
            tfd.Normal(loc=0., scale=tf.ones_like(x)),
            reinterpreted_batch_ndims=1),
        num_steps=5)
    x = self.evaluate(loses_batch_shape.sample([2], seed=test_util.test_seed()))
    with self.assertRaisesRegexp(ValueError, 'batch shape is incorrect'):
      loses_batch_shape.log_prob(x)

    gains_batch_shape = tfd.MarkovChain(
        initial_state_prior=tfd.Independent(
            tfd.Normal(loc=0., scale=[1., 1.]),
            reinterpreted_batch_ndims=1),
        transition_fn=lambda _, x: tfd.Normal(loc=0., scale=tf.ones_like(x)),
        num_steps=5)
    x = self.evaluate(gains_batch_shape.sample([2], seed=test_util.test_seed()))
    with self.assertRaisesRegexp(ValueError, 'batch shape is incorrect'):
      gains_batch_shape.log_prob(x)

  def test_log_prob_matches_linear_gaussian_ssm(self):
    dim = 2
    batch_shape = [3, 1]
    seed, *model_seeds = samplers.split_seed(test_util.test_seed(), n=6)

    # Sample a random linear Gaussian process.
    prior_loc = self.evaluate(
        tfd.Normal(0., 1.).sample(batch_shape + [dim], seed=model_seeds[0]))
    prior_scale = self.evaluate(
        tfd.InverseGamma(1., 1.).sample(batch_shape + [dim],
                                        seed=model_seeds[1]))
    transition_matrix = self.evaluate(
        tfd.Normal(0., 1.).sample([dim, dim], seed=model_seeds[2]))
    transition_bias = self.evaluate(
        tfd.Normal(0., 1.).sample(batch_shape + [dim], seed=model_seeds[3]))
    transition_scale_tril = self.evaluate(
        tf.linalg.cholesky(
            tfd.WishartTriL(df=dim, scale_tril=tf.eye(dim)).sample(
                seed=model_seeds[4])))

    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=prior_loc, scale_diag=prior_scale, name='initial_state_prior')

    lgssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=7,
        transition_matrix=transition_matrix,
        transition_noise=tfd.MultivariateNormalTriL(
            loc=transition_bias, scale_tril=transition_scale_tril),
        # Trivial observation model to pass through the latent state.
        observation_matrix=tf.eye(dim),
        observation_noise=tfd.MultivariateNormalDiag(loc=tf.zeros(dim),
                                                     scale_diag=tf.zeros(dim)),
        initial_state_prior=initial_state_prior)

    markov_chain = tfd.MarkovChain(
        initial_state_prior=initial_state_prior,
        transition_fn=lambda _, x: tfd.MultivariateNormalTriL(  # pylint: disable=g-long-lambda
            loc=tf.linalg.matvec(transition_matrix, x) + transition_bias,
            scale_tril=transition_scale_tril),
        num_steps=7)

    x = markov_chain.sample(5, seed=seed)
    self.assertAllClose(lgssm.log_prob(x), markov_chain.log_prob(x))

  @test_util.numpy_disable_test_missing_functionality(
      'JointDistributionNamedAutoBatched')
  def test_docstring_example_autoregressive_process(self):

    def transition_fn(_, previous_state):
      return tfd.JointDistributionNamedAutoBatched(
          # The previous state may include batch dimensions. Since the log scale
          # is a scalar quantity, its shape is the batch shape.
          batch_ndims=ps.rank(previous_state['log_scale']),
          model={
              # The autoregressive coefficients and the `log_scale` each follow
              # an independent slow-moving random walk.
              'coefs': tfd.Normal(loc=previous_state['coefs'], scale=0.01),
              'log_scale': tfd.Normal(loc=previous_state['log_scale'],
                                      scale=0.01),
              # The level is a linear combination of the previous *two* levels,
              # with additional noise of scale `exp(log_scale)`.
              'level': lambda coefs, log_scale: tfd.Normal(  # pylint: disable=g-long-lambda
                  loc=(coefs[..., 0] * previous_state['level'] +
                       coefs[..., 1] * previous_state['previous_level']),
                  scale=tf.exp(log_scale)),
              # Store the previous level to access at the next step.
              'previous_level': tfd.Deterministic(previous_state['level'])})

    process = tfd.MarkovChain(
        # For simplicity, define the prior as a 'transition' from fixed values.
        initial_state_prior=transition_fn(
            0, previous_state={
                'coefs': [0.7, -0.2],
                'log_scale': -1.,
                'level': 0.,
                'previous_level': 0.}),
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
    batch_gaussian_walk = tfd.MarkovChain(
        # The prior distribution determines the batch shape for the chain.
        # Transitions must respect this batch shape.
        initial_state_prior=tfd.Normal(loc=_as_tensor([-10., 0., 10.]),
                                       scale=_as_tensor([1., 1., 1.])),
        transition_fn=lambda t, x: tfd.Normal(  # pylint: disable=g-long-lambda
            loc=x,
            # The `num_steps` dimension will always be leftmost in `x`, so we
            # pad the scale to the same rank as `x` so that the shapes line up.
            scale=tf.reshape(
                tf.gather(scales, t),
                ps.concat([[-1],
                           ps.ones(ps.rank(x) - 1, dtype=tf.int32)], axis=0))),
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
    gaussian_walk = tfd.MarkovChain(
        initial_state_prior=tfd.Normal(loc=0., scale=1.),
        transition_fn=lambda _, x: tfd.Normal(loc=x, scale=1.),
        num_steps=100)
    self.assertAllEqual(gaussian_walk.event_shape, [100])
    self.assertAllEqual(gaussian_walk.batch_shape, [])

    x = gaussian_walk.sample(5, seed=test_util.test_seed())
    self.assertAllEqual(x.shape, [5, 100])
    lp = gaussian_walk.log_prob(x)
    self.assertAllEqual(lp.shape, [5])

    n = tfd.Normal(0., 1.)
    expected_lp = (n.log_prob(x[:, 0]) +
                   tf.reduce_sum(n.log_prob(x[:, 1:] - x[:, :-1]), axis=-1))
    self.assertAllClose(lp, expected_lp)

    x2, lp2 = gaussian_walk.experimental_sample_and_log_prob(
        [2], seed=test_util.test_seed())
    self.assertAllClose(lp2, gaussian_walk.log_prob(x2))

  def test_non_autobatched_joint_distribution(self):

    def transition_fn(_, previous_state):
      return tfd.JointDistributionNamed(
          {
              # The autoregressive coefficients and the `log_scale` each follow
              # an independent slow-moving random walk.
              'coefs': tfd.Independent(
                  tfd.Normal(loc=previous_state['coefs'], scale=0.01),
                  reinterpreted_batch_ndims=1),
              'log_scale': tfd.Normal(loc=previous_state['log_scale'],
                                      scale=0.01),
              # The level is a linear combination of the previous *two* levels,
              # with additional noise of scale `exp(log_scale)`.
              'level': lambda coefs, log_scale: tfd.Normal(  # pylint: disable=g-long-lambda
                  loc=(coefs[..., 0] * previous_state['level'] +
                       coefs[..., 1] * previous_state['previous_level']),
                  scale=tf.exp(log_scale)),
              # Store the previous level to access at the next step.
              'previous_level': tfd.Deterministic(previous_state['level'])})

    process = tfd.MarkovChain(
        # For simplicity, define the prior as a 'transition' from fixed values.
        initial_state_prior=transition_fn(
            0, previous_state={
                'coefs': [0.7, -0.2],
                'log_scale': -1.,
                'level': 0.,
                'previous_level': 0.}),
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
    p = tfd.MarkovChain(
        initial_state_prior=tfd.Normal(0., 1.),
        transition_fn=lambda _, x: tfd.Normal(x, tf.nn.softplus(x)),
        num_steps=10)
    q = tfd.MarkovChain(
        initial_state_prior=tfd.Normal(-10, 3.),
        transition_fn=lambda _, x: tfd.Normal(x, tf.abs(x)),
        num_steps=10)
    x = self.evaluate(p.sample(4, seed=test_util.test_seed()))
    y = self.evaluate(q.sample(4, seed=test_util.test_seed()))
    self.assertAllClose(
        p.log_prob(x) - q.log_prob(y),
        log_prob_ratio.log_prob_ratio(p, x, q, y), atol=1e-5)

  def test_unexpected_num_steps_raises(self):
    p = tfd.MarkovChain(
        initial_state_prior=tfd.Normal(0., 1.),
        transition_fn=lambda _, x: tfd.Normal(x, tf.nn.softplus(x)),
        num_steps=10,
        validate_args=True)
    with self.assertRaisesRegex(
        (ValueError, tf.errors.InvalidArgumentError),
        'does not match the expected num_steps'):
      p.log_prob(tf.zeros([11]))

if __name__ == '__main__':
  tf.test.main()
