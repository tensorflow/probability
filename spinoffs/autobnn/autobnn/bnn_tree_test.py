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
"""Tests for bnn_tree.py."""

from absl.testing import parameterized
from flax import linen as nn
import jax
import jax.numpy as jnp
from autobnn import bnn_tree
from autobnn import kernels
from absl.testing import absltest


class TreeTest(parameterized.TestCase):

  def test_list_of_all_depth0(self):
    l0 = bnn_tree.list_of_all(jnp.linspace(0.0, 100.0, 100), 0)
    # With no periods, there should be three kernels.
    self.assertLen(l0, 3)
    for k in l0:
      self.assertFalse(k.going_to_be_multiplied)

    l0 = bnn_tree.list_of_all(100, 0, 50, [20.0, 40.0], parent_is_multiply=True)
    # With two periods specified, 3+2 = 5.
    self.assertLen(l0, 5)
    for k in l0:
      self.assertTrue(k.going_to_be_multiplied)

  def test_list_of_all_depth1(self):
    l1 = bnn_tree.list_of_all(jnp.linspace(0.0, 100.0, 100), 1)
    # With no periods, there should be
    # choose(3+1, 2) = 6 trees with a Multiply top node,
    # choose(3, 2) = 3 trees with a WeightedSum top node, and
    # 3*3 = 9 trees with a LearnableChangePoint top node.
    self.assertLen(l1, 18)

    # Check that all of the BNNs in the tree can be trained.
    for k in l1:
      params = k.init(jax.random.PRNGKey(0), jnp.zeros(5))
      lp = k.log_prior(params)
      self.assertLess(lp, 0.0)
      output = k.apply(params, jnp.ones(5))
      self.assertEqual((1,), output.shape)

    l1 = bnn_tree.list_of_all(
        jnp.linspace(0.0, 100.0, 100),
        1,
        50,
        [20.0, 40.0],
        parent_is_multiply=True,
    )
    # With 2 periods and parent_is_multiply, there are only WeightedSum top
    # nodes, with 4*5/2 = 10 trees.
    self.assertLen(l1, 10)

  def test_list_of_all_depth2(self):
    l2 = bnn_tree.list_of_all(jnp.linspace(0.0, 100.0, 100), 2)
    # There are 18 trees of depth 1, of which 3 are safe to multiply.
    # choose(3+1, 2) = 6 trees with a Multiply top node,
    # choose(18, 2) = 153 trees with a WeightedSum top node, and
    # 18*18 = 324 trees with a LearnableChangePoint top node.
    self.assertLen(l2, 483)

  @parameterized.parameters(0, 1)  # depth=2 segfaults on my desktop :(
  def test_weighted_sum_of_all(self, depth):
    soa = bnn_tree.weighted_sum_of_all(
        jnp.linspace(0.0, 1.0, 100), jnp.ones(100), depth=depth
    )
    params = soa.init(jax.random.PRNGKey(0), jnp.zeros(5))
    lp = soa.log_prior(params)
    self.assertLess(lp, 0.0)
    output = soa.apply(params, jnp.ones(5))
    self.assertEqual((1,), output.shape)

  def test_random_tree(self):
    r0 = bnn_tree.random_tree(
        jax.random.PRNGKey(0), depth=0, width=50, period=7
    )
    self.assertIsInstance(r0, kernels.OneLayerBNN)
    params = r0.init(jax.random.PRNGKey(1), jnp.zeros(5))
    lp = r0.log_prior(params)
    self.assertLess(lp, 0.0)
    output = r0.apply(params, jnp.ones(5))
    self.assertEqual((1,), output.shape)

    r1 = bnn_tree.random_tree(
        jax.random.PRNGKey(0), depth=1, width=50, period=24
    )
    self.assertIsInstance(r1, nn.Module)
    params = r1.init(jax.random.PRNGKey(1), jnp.zeros(5))
    lp = r1.log_prior(params)
    self.assertLess(lp, 0.0)
    output = r1.apply(params, jnp.ones(5))
    self.assertEqual((1,), output.shape)

    r2 = bnn_tree.random_tree(
        jax.random.PRNGKey(0), depth=2, width=50, period=52
    )
    self.assertIsInstance(r2, nn.Module)
    params = r2.init(jax.random.PRNGKey(1), jnp.zeros(5))
    lp = r2.log_prior(params)
    self.assertLess(lp, 0.0)
    output = r2.apply(params, jnp.ones(5))
    self.assertEqual((1,), output.shape)


if __name__ == '__main__':
  absltest.main()
