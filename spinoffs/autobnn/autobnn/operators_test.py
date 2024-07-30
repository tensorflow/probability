# Copyright 2023 The TensorFlow Probability Authors.
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
"""Tests for operators.py."""

from absl.testing import parameterized
import bayeux as bx
import jax
import jax.numpy as jnp
import numpy as np
from autobnn import kernels
from autobnn import operators
from autobnn import util
from tensorflow_probability.substrates.jax.distributions import distribution as distribution_lib
from absl.testing import absltest


KERNELS = [
    operators.Add(
        bnns=(kernels.OneLayerBNN(width=50), kernels.OneLayerBNN(width=50))
    ),
    operators.Add(
        bnns=(kernels.OneLayerBNN(width=50), kernels.OneLayerBNN(width=100))
    ),
    operators.Add(
        bnns=(
            kernels.PeriodicBNN(width=50, period=0.1),
            kernels.OneLayerBNN(width=50),
        )
    ),
    operators.WeightedSum(
        bnns=(kernels.OneLayerBNN(width=50), kernels.OneLayerBNN(width=50))
    ),
    operators.WeightedSum(
        bnns=(
            kernels.PeriodicBNN(width=50, period=0.1),
            kernels.OneLayerBNN(width=50),
        ),
        alpha=2.0,
    ),
    operators.WeightedSum(
        bnns=(
            kernels.ExponentiatedQuadraticBNN(width=50),
            kernels.ExponentiatedQuadraticBNN(width=50),
        )
    ),
    operators.Multiply(
        bnns=(
            kernels.OneLayerBNN(width=50, going_to_be_multiplied=True),
            kernels.OneLayerBNN(width=50, going_to_be_multiplied=True),
        ),
    ),
    operators.Multiply(
        bnns=(
            kernels.OneLayerBNN(width=50, going_to_be_multiplied=True),
            kernels.OneLayerBNN(width=50, going_to_be_multiplied=True),
            kernels.OneLayerBNN(width=50, going_to_be_multiplied=True),
        )
    ),
    operators.Multiply(
        bnns=(
            operators.Add(
                bnns=(
                    kernels.OneLayerBNN(width=50, going_to_be_multiplied=True),
                    kernels.OneLayerBNN(width=50, going_to_be_multiplied=True),
                ),
                going_to_be_multiplied=True
            ),
            operators.Add(
                bnns=(
                    kernels.OneLayerBNN(width=50, going_to_be_multiplied=True),
                    kernels.OneLayerBNN(width=50, going_to_be_multiplied=True),
                ),
                going_to_be_multiplied=True
            ),
        )
    ),
    operators.ChangePoint(
        bnns=(kernels.OneLayerBNN(width=50), kernels.OneLayerBNN(width=50)),
        change_point=5.0,
        slope=1.0,
    ),
    operators.LearnableChangePoint(
        bnns=(kernels.OneLayerBNN(width=50), kernels.OneLayerBNN(width=50)),
        time_series_xs=np.linspace(0., 5., 100),
    ),
]


NAMES = [
    "(OneLayer#OneLayer)",
    "(OneLayer#OneLayer)",
    "(Periodic(period=0.10)#OneLayer)",
    "(OneLayer+OneLayer)",
    "(Periodic(period=0.10)+OneLayer)",
    "(RBF+RBF)",
    "(OneLayer*OneLayer)",
    "(OneLayer*OneLayer*OneLayer)",
    "((OneLayer#OneLayer)*(OneLayer#OneLayer))",
    "(OneLayer<[5.0]OneLayer)",
    "(OneLayer<OneLayer)",
]


class OperatorsTest(parameterized.TestCase):

  @parameterized.product(shape=[(5,), (5, 1), (5, 5)], kernel=KERNELS)
  def test_operators(self, shape, kernel):
    params = kernel.init(jax.random.PRNGKey(0), jnp.zeros(shape))
    lp = kernel.log_prior(params)
    self.assertLess(lp, 0.0)
    output = kernel.apply(params, jnp.ones(shape))
    self.assertEqual(shape[:-1] + (1,), output.shape)

    self.assertEqual(
        jax.tree_util.tree_structure(
            kernel.get_all_distributions(),
            is_leaf=lambda x: isinstance(x, distribution_lib.Distribution),
        ),
        jax.tree_util.tree_structure(params["params"]),
    )

  @parameterized.parameters(KERNELS)
  def test_likelihood(self, kernel):
    params = kernel.init(jax.random.PRNGKey(1), jnp.zeros(1))
    data = jnp.array([[0], [1], [2], [3], [4], [5]], dtype=jnp.float32)
    obs = jnp.array([1, 0, 1, 0, 1, 0], dtype=jnp.float32)
    ll = kernel.log_likelihood(params, data, obs)
    lp = kernel.log_prob(params, data, obs)
    self.assertLess(jnp.sum(lp), jnp.sum(ll))

  @parameterized.parameters(zip(KERNELS, NAMES))
  def test_summarize(self, kernel, expected):
    self.assertEqual(expected, kernel.summarize())

  def test_add_of_onelayers_has_penultimate(self):
    add2 = operators.Add(
        bnns=(kernels.OneLayerBNN(width=50), kernels.OneLayerBNN(width=50))
    )
    params = add2.init(jax.random.PRNGKey(0), jnp.zeros(5))
    lp = add2.log_prior(params)
    self.assertLess(lp, 0.0)
    h = add2.apply(params, jnp.ones(5), method=add2.penultimate)
    self.assertEqual((50,), h.shape)

  def test_learnable_changepoint_distribution(self):
    # checking that the prior on the changepoint has some reasonable properties
    lcp = operators.LearnableChangePoint(
        time_series_xs=jnp.linspace(-3., 5., 100))
    cp_distribution = lcp.distributions()["change_point"]

    # zero at endpoints
    self.assertAlmostEqual(cp_distribution.prob(-3.), 0.)
    self.assertAlmostEqual(cp_distribution.prob(5.), 0.)

    # Greatest at midpoint
    self.assertLess(cp_distribution.prob(0.), cp_distribution.prob(1.))

    # Flattish in middle
    early_slope = cp_distribution.prob(-1.) - cp_distribution.prob(-2.)
    mid_slope = cp_distribution.prob(1.) - cp_distribution.prob(0.)
    self.assertLess(mid_slope, early_slope)

  def test_add_of_adds_has_penultimate(self):
    add2 = operators.Add(
        bnns=(kernels.OneLayerBNN(width=50), kernels.OneLayerBNN(width=50))
    )
    add2more = operators.Add(
        bnns=(kernels.OneLayerBNN(width=50), kernels.OneLayerBNN(width=50))
    )
    addtree = operators.Add(bnns=(add2, add2more))
    params = addtree.init(jax.random.PRNGKey(0), jnp.zeros(5))
    lp = addtree.log_prior(params)
    self.assertLess(lp, 0.0)
    h = addtree.apply(params, jnp.ones(5), method=addtree.penultimate)
    self.assertEqual((50,), h.shape)

  def test_multiply_can_be_trained(self):
    seed = jax.random.PRNGKey(20231018)
    x_train, y_train = util.load_fake_dataset()

    leaf1 = kernels.PeriodicBNN(width=5, period=0.1,
                                going_to_be_multiplied=True)
    leaf2 = kernels.LinearBNN(width=5, going_to_be_multiplied=True)
    bnn = operators.Multiply(bnns=[leaf1, leaf2])

    init_seed, seed = jax.random.split(seed)
    init_params = bnn.init(init_seed, x_train)

    def train_density(params):
      return bnn.log_prob(params, x_train, y_train)

    transform, inverse_transform, _ = util.make_transforms(bnn)
    mix_model = bx.Model(
        train_density,
        init_params,
        transform_fn=transform,
        inverse_transform_fn=inverse_transform,
    )

    self.assertTrue(
        mix_model.optimize.optax_adam.debug(seed=seed, verbosity=10)
    )


if __name__ == "__main__":
  absltest.main()
