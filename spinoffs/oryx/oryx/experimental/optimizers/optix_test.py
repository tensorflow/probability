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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.optimizers.optix."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from oryx.core import state
from oryx.experimental.optimizers import optix
from oryx.internal import test_util


class OptixTest(test_util.TestCase):

  def test_clip_should_clip_updates(self):
    self.assertEqual(optix.clip(1.)(0.5, 0.), .5)
    self.assertEqual(optix.clip(1.)(1.5, 0.), 1.)

  def test_clip_by_global_norm_should_clip_correctly(self):
    params = [jnp.zeros(6), jnp.zeros(10)]
    updates = [jnp.ones(6), jnp.ones(10)]
    np.testing.assert_array_equal(
        optix.clip_by_global_norm(1.)(updates, params)[0], .25 * jnp.ones(6))
    np.testing.assert_array_equal(
        optix.clip_by_global_norm(1.)(updates, params)[1], .25 * jnp.ones(10))
    np.testing.assert_array_equal(
        optix.clip_by_global_norm(4.)(updates, params)[0], jnp.ones(6))
    np.testing.assert_array_equal(
        optix.clip_by_global_norm(4.)(updates, params)[1], jnp.ones(10))

  def test_trace_should_keep_track_of_momentum(self):
    params = jnp.zeros(6)
    updates = jnp.ones(6)
    opt = state.init(optix.trace(0.99, False))(random.PRNGKey(0),
                                               updates, params)
    np.testing.assert_array_equal(opt.trace, jnp.zeros(6))
    opt = opt.update(updates, params)
    np.testing.assert_array_equal(opt.trace, jnp.ones(6))
    np.testing.assert_array_equal(opt(updates, params), 1.99 * jnp.ones(6))

  def test_trace_should_keep_track_of_momentum_with_nesterov(self):
    params = jnp.zeros(6)
    updates = jnp.ones(6)
    opt = state.init(optix.trace(0.99, True))(random.PRNGKey(0),
                                              updates, params)
    np.testing.assert_array_equal(opt.trace, jnp.zeros(6))
    opt = opt.update(updates, params)
    np.testing.assert_array_equal(opt.trace, jnp.ones(6))
    np.testing.assert_array_equal(
        opt(updates, params), (1.99 + 0.99**2) * jnp.ones(6))

  def test_scale_by_rms_should_scale_by_rms(self):
    params = jnp.zeros(9)
    updates = 2 * jnp.ones(9)
    opt = state.init(optix.scale_by_rms(0.5, 0.))(random.PRNGKey(0), updates,
                                                  params)
    np.testing.assert_array_equal(opt.nu, jnp.zeros(9))
    opt = opt.update(updates, params)
    np.testing.assert_array_equal(opt.nu, 2 * jnp.ones(9))
    np.testing.assert_array_equal(
        opt(updates, params), 2 * jnp.ones(9) / jnp.sqrt(3.))

  def test_scale_by_stddev_should_scale_by_stddev(self):
    params = jnp.zeros(9)
    updates = 2 * jnp.ones(9)
    opt = state.init(optix.scale_by_stddev(0.5, 0.))(random.PRNGKey(0), updates,
                                                     params)
    np.testing.assert_array_equal(opt.mu, jnp.zeros(9))
    np.testing.assert_array_equal(opt.nu, jnp.zeros(9))
    opt = opt.update(updates, params)
    np.testing.assert_array_equal(opt.mu, jnp.ones(9))
    np.testing.assert_array_equal(opt.nu, 2 * jnp.ones(9))
    np.testing.assert_array_equal(
        opt(updates, params), 2 * jnp.ones(9) / jnp.sqrt(0.75))

  def test_scale_by_adam_should_scale_by_adam(self):
    params = jnp.zeros(9)
    updates = 2 * jnp.ones(9)
    opt = state.init(optix.scale_by_adam(0.5, 0.5, 0.))(random.PRNGKey(0),
                                                        updates, params)
    np.testing.assert_array_equal(opt.count, 0.)
    np.testing.assert_array_equal(opt.mu, jnp.zeros(9))
    np.testing.assert_array_equal(opt.nu, jnp.zeros(9))
    opt = opt.update(updates, params)
    np.testing.assert_array_equal(opt.count, 1.)
    np.testing.assert_array_equal(opt.mu, jnp.ones(9))
    np.testing.assert_array_equal(opt.nu, 2 * jnp.ones(9))
    np.testing.assert_array_equal(
        opt(updates, params), 1.5 / 0.75 * jnp.ones(9) / jnp.sqrt(3 / 0.75))

  def test_scale_by_schedule_should_update_scale(self):
    params = 0.
    updates = 1.

    def schedule(t):
      return 1. / ((t + 1)**2)

    opt = state.init(optix.scale_by_schedule(schedule))(random.PRNGKey(0),
                                                        updates, params)
    np.testing.assert_array_equal(opt.count, 0.)
    opt = opt.update(updates, params)
    np.testing.assert_array_equal(opt.count, 1.)
    np.testing.assert_array_equal(opt(updates, params), 0.25)

  def test_add_noise_should_add_noise(self):
    params = 0.
    updates = 1.
    opt = state.init(optix.add_noise(1., 0., 0))(random.PRNGKey(0), updates,
                                                 params)
    np.testing.assert_array_equal(opt.count, 0.)
    np.testing.assert_array_equal(opt.rng_key, random.PRNGKey(0))
    value, opt = opt.call_and_update(updates, params)
    np.testing.assert_array_equal(
        value, 1. + random.normal(random.split(random.PRNGKey(0))[1], ()))

  def test_apply_every_should_delay_updates(self):
    params = 0.
    updates = 1.
    opt = state.init(optix.apply_every(5))(random.PRNGKey(0), updates, params)
    np.testing.assert_array_equal(opt.count, 0.)
    np.testing.assert_array_equal(opt.grad_acc, 0.)
    for _ in range(4):
      value, opt = opt.call_and_update(updates, params)
      np.testing.assert_array_equal(value, 0.)
    value = opt(updates, params)
    np.testing.assert_array_equal(value, 5.)


UPDATES = [
    ('sgd', optix.sgd(1e-1)),
    ('sgd_with_momentum', optix.sgd(1e-1, 0.9)),
    ('sgd_with_nesterov_momentum', optix.sgd(1e-1, 0.9, nesterov=True)),
    ('noisy_sgd', optix.noisy_sgd(1e-1)),
    ('adam', optix.adam(1e-1)),
    ('rmsprop', optix.rmsprop(1e-1)),
]


class OptimizerTest(test_util.TestCase):

  @parameterized.named_parameters(UPDATES)
  def test_optimizer(self, update):

    def loss(x):
      return jnp.sum(x**2)

    x = jnp.array([3., 4.])
    opt = state.init(optix.optimize(loss, update, 500))(random.PRNGKey(0), x)
    x = jax.jit(opt.call)(x)
    np.testing.assert_allclose(jnp.zeros(2), x, atol=1e-1, rtol=1e-1)


if __name__ == '__main__':
  absltest.main()
