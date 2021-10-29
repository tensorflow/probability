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
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discussion.pathfinder import pathfinder
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


@pytest.mark.parametrize("dim, history", [(12, 2), (12, 0), (4, 4)])
def test_bfgs_inverse_hessian(dim, history):
  position_diffs = jnp.ones((dim, history))
  gradient_diffs = jnp.ones((dim, history))
  alpha, beta, gamma = pathfinder.bfgs_inverse_hessian(
      updates_of_position_differences=position_diffs,
      updates_of_gradient_differences=gradient_diffs)
  assert alpha.shape == (dim,)
  assert beta.shape == (dim, 2 * history)
  assert gamma.shape == (2 * history, 2 * history)


def test_cov_estimate():
  positions = np.random.randn(10, 2)
  gradients = -2 * positions  # x^2 + y^2
  ret = pathfinder.cov_estimate(
      optimization_path=positions, optimization_path_grads=gradients, history=3)
  assert len(ret) == 9


def test_lbfgs():
  target = tfd.Independent(
      tfd.Normal(jnp.array([1., 2.]), jnp.ones(2)),
      reinterpreted_batch_ndims=1).log_prob
  optimization_path, _ = pathfinder.lbfgs(
      log_target_density=target,
      initial_value=jnp.array([-1.0, -4.0]),
      max_iters=10)
  assert jnp.all(optimization_path[-1] == jnp.array([1.0, 2.0]))


def test_bfgs_sample():
  dim, history = 12, 2
  position_diffs = jnp.ones((dim, history))
  gradient_diffs = jnp.ones((dim, history))
  diagonal_estimate, thin_factors, scaling_outer_product = pathfinder.bfgs_inverse_hessian(
      updates_of_position_differences=position_diffs,
      updates_of_gradient_differences=gradient_diffs)
  log_density = lambda x: jnp.sum(-x * x)
  grad_log_density = jax.grad(log_density)

  value = jnp.ones(dim)

  key = jax.random.PRNGKey(0)
  num_draws = 5
  draws, probs = pathfinder.bfgs_sample(
      value=value,
      grad_density=grad_log_density(value),
      diagonal_estimate=diagonal_estimate,
      thin_factors=thin_factors,
      scaling_outer_product=scaling_outer_product,
      num_draws=num_draws,
      key=key,
  )
  assert draws.shape[0] == num_draws
  assert draws.shape[1] == dim
  assert probs.shape[0] == num_draws


def test_pathfinder():

  target = tfd.Independent(
      tfd.StudentT(3., loc=jnp.array([1., 2]), scale=jnp.array([1., 1.])),
      reinterpreted_batch_ndims=1).log_prob

  key = jax.random.PRNGKey(3)
  init = jnp.zeros(2)
  draws, _ = pathfinder.pathfinder(
      target_density=target,
      initial_value=init,
      lbfgs_max_iters=20,
      num_draws=1000,
      key=key,
  )
  assert (jnp.linalg.norm(
      jnp.mean(draws, axis=0) - jnp.array([1., 2.]), ord=2) < 0.1)


def test_multipath_pathfinder():
  target = tfd.Independent(
      tfd.StudentT(2., loc=jnp.array([1., 2]), scale=jnp.array([1., 1.])),
      reinterpreted_batch_ndims=1).log_prob

  key, init_key = jax.random.split(jax.random.PRNGKey(3))
  init = jax.random.normal(init_key, (5, 2))
  draws = pathfinder.multipath_pathfinder(
      target_density=target,
      initial_values=init,
      key=key,
      lbfgs_max_iters=20,
      num_pathfinder_draws=10,
      num_draws=10,
  )
  assert draws.shape == (10, 2)

if __name__ == "__main__":
  pytest.main([__file__])
