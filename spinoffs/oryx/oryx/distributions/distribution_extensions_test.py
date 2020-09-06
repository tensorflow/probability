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
# Lint as: python3
"""Tests for tensorflow_probability.spinoffs.oryx.distributions.distributions_extensions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as np
import numpy as onp
from oryx import core
from oryx import distributions as bd
from oryx.core import ppl


# Use lambdas to defer construction of distributions
# pylint: disable=g-long-lambda
DISTRIBUTIONS = [
    ('normal_scalar_args', bd.Normal, lambda: (0., 1.), lambda: {},
     0., [0., 1.]),
    ('normal_scalar_kwargs', bd.Normal, lambda: (),
     lambda: {'loc': 0., 'scale': 1.}, 0., [0., 1.]),
    ('mvn_diag_args', bd.MultivariateNormalDiag,
     lambda: (onp.zeros(5, dtype=onp.float32), onp.ones(5, dtype=onp.float32)),
     lambda: {},
     onp.zeros(5, dtype=onp.float32),
     [onp.zeros(5, dtype=onp.float32), onp.ones(5, dtype=onp.float32)]),
    ('mvn_diag_kwargs', bd.MultivariateNormalDiag, lambda: (),
     lambda: {'loc': onp.zeros(5, dtype=onp.float32),
              'scale_diag': onp.ones(5, dtype=onp.float32)},
     onp.zeros(5, dtype=onp.float32),
     [onp.zeros(5, dtype=onp.float32),
      onp.ones(5, dtype=onp.float32)]),
    ('independent_normal_args', bd.Independent, lambda: (bd.Normal(
        onp.zeros(5, dtype=onp.float32), onp.ones(5, dtype=onp.float32)),),
     lambda: {'reinterpreted_batch_ndims': 1}, onp.zeros(5, dtype=onp.float32),
     [onp.zeros(5, dtype=onp.float32),
      onp.ones(5, dtype=onp.float32)]),
    ('independent_normal_args2', bd.Independent, lambda: (bd.Normal(
        loc=onp.zeros(5, dtype=onp.float32),
        scale=onp.ones(5, dtype=onp.float32)),), lambda: {
            'reinterpreted_batch_ndims': 1
        }, onp.zeros(5, dtype=onp.float32),
     [onp.zeros(5, dtype=onp.float32),
      onp.ones(5, dtype=onp.float32)]),
    ('independent_normal_kwargs', bd.Independent, lambda: (), lambda: {
        'reinterpreted_batch_ndims': 1,
        'distribution': bd.Normal(onp.zeros(5, dtype=onp.float32),
                                  onp.ones(5, dtype=onp.float32))},
     onp.zeros(5, dtype=onp.float32),
     [onp.zeros(5, dtype=onp.float32), onp.ones(5, dtype=onp.float32)]),
]


class DistributionsExtensionsTest(parameterized.TestCase):

  @parameterized.named_parameters(DISTRIBUTIONS)
  def test_sample(self, dist, args, kwargs, out, flat):
    del out, flat
    args = args()
    kwargs = kwargs()
    p = dist(*args, **kwargs)
    p.sample(seed=random.PRNGKey(0))

  @parameterized.named_parameters(DISTRIBUTIONS)
  def test_log_prob(self, dist, args, kwargs, out, flat):
    del flat
    args = args()
    kwargs = kwargs()
    p = dist(*args, **kwargs)
    p.log_prob(out)

  @parameterized.named_parameters(DISTRIBUTIONS)
  def test_flatten(self, dist, args, kwargs, out, flat):
    del out
    args = args()
    kwargs = kwargs()
    p = dist(*args, **kwargs)
    flat_p, _ = jax.tree_flatten(p)
    self.assertEqual(len(flat_p), len(flat))
    for e1, e2 in zip(flat_p, flat):
      onp.testing.assert_allclose(e1, e2)

  @parameterized.named_parameters(DISTRIBUTIONS)
  def test_log_prob_transformation(self, dist, args, kwargs, out, flat):
    del out, flat
    args = args()
    kwargs = kwargs()
    p = dist(*args, **kwargs)

    def sample(key):
      return ppl.random_variable(p)(key)

    self.assertEqual(
        p.log_prob(sample(random.PRNGKey(0))),
        ppl.log_prob(sample)(sample(random.PRNGKey(0))))

  @parameterized.named_parameters(DISTRIBUTIONS)
  def test_unzip_transformation(self, dist, args, kwargs, out, flat):
    del out, flat
    args = args()
    kwargs = kwargs()
    p = dist(*args, **kwargs)

    def model(key):
      return ppl.random_variable(p, name='x')(key)

    init = core.unzip(model, tag=ppl.RANDOM_VARIABLE)(random.PRNGKey(0))[0]
    self.assertLen(init(random.PRNGKey(0)), 1)

  def test_joint_distribution(self):

    def model(key):
      k1, k2 = random.split(key)
      z = ppl.random_variable(bd.Normal(0., 1.), name='z')(k1)
      x = ppl.random_variable(bd.Normal(z, 1.), name='x')(k2)
      return x

    with self.assertRaises(ValueError):
      core.log_prob(model)(0.1)
    sample = ppl.joint_sample(model)
    self.assertEqual(
        core.log_prob(sample)({
            'z': 1.,
            'x': 2.
        }),
        bd.Normal(0., 1.).log_prob(1.) + bd.Normal(1., 1.).log_prob(2.))

  def test_eight_schools(self):
    treatment_stddevs = np.array([15, 10, 16, 11, 9, 11, 10, 18],
                                 dtype=np.float32)

    def eight_schools(key):
      ae_key, as_key, se_key, te_key = random.split(key, 4)
      avg_effect = ppl.random_variable(
          bd.Normal(loc=0., scale=10.), name='avg_effect')(
              ae_key)
      avg_stddev = ppl.random_variable(
          bd.Normal(loc=5., scale=1.), name='avg_stddev')(
              as_key)
      school_effects_standard = ppl.random_variable(
          bd.Independent(
              bd.Normal(loc=np.zeros(8), scale=np.ones(8)),
              reinterpreted_batch_ndims=1),
          name='se_standard')(
              se_key)
      treatment_effects = ppl.random_variable(
          bd.Independent(
              bd.Normal(
                  loc=(avg_effect[..., np.newaxis] +
                       np.exp(avg_stddev[..., np.newaxis]) *
                       school_effects_standard),
                  scale=treatment_stddevs),
              reinterpreted_batch_ndims=1),
          name='te')(
              te_key)
      return treatment_effects

    jd_sample = ppl.joint_sample(eight_schools)
    jd_log_prob = core.log_prob(jd_sample)
    jd_log_prob(jd_sample(random.PRNGKey(0)))

  def test_bnn(self):

    def dense(dim_out, name):

      def forward(key, x):
        dim_in = x.shape[-1]
        w_key, b_key = random.split(key)
        w = ppl.random_variable(
            bd.Sample(bd.Normal(0., 1.), sample_shape=(dim_out, dim_in)),
            name=f'{name}_w')(
                w_key)
        b = ppl.random_variable(
            bd.Sample(bd.Normal(0., 1.), sample_shape=(dim_out,)),
            name=f'{name}_b')(
                b_key)
        return np.dot(w, x) + b

      return forward

    def mlp(hidden_size, num_layers):

      def forward(key, x):
        for i in range(num_layers):
          key, subkey = random.split(key)
          x = dense(hidden_size, 'dense_{}'.format(i + 1))(subkey, x)
          x = jax.nn.relu(x)
        logits = dense(10, 'dense_{}'.format(num_layers + 1))(key, x)
        return logits

      return forward

    def predict(mlp):

      def forward(key, x):
        k1, k2 = random.split(key)
        logits = mlp(k1, x)
        return bd.Categorical(logits=logits).sample(seed=k2, name='y')

      return forward

    sample = ppl.joint_sample(predict(mlp(200, 2)))
    core.log_prob(sample)(sample(random.PRNGKey(0), np.ones(784)), np.ones(784))


if __name__ == '__main__':
  absltest.main()
