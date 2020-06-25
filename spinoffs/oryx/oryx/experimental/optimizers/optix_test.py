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
import jax.numpy as np
import numpy as onp

from oryx.core import state
from oryx.experimental.optimizers import optix


class OptixTest(absltest.TestCase):

  def test_clip_should_clip_updates(self):
    self.assertEqual(optix.clip(1.)(0., .5), .5)
    self.assertEqual(optix.clip(1.)(0., 1.5), 1.)

  def test_clip_by_global_norm_should_clip_correctly(self):
    params = [np.zeros(6), np.zeros(10)]
    updates = [np.ones(6), np.ones(10)]
    onp.testing.assert_array_equal(
        optix.clip_by_global_norm(1.)(params, updates)[0], .25 * np.ones(6))
    onp.testing.assert_array_equal(
        optix.clip_by_global_norm(1.)(params, updates)[1], .25 * np.ones(10))
    onp.testing.assert_array_equal(
        optix.clip_by_global_norm(4.)(params, updates)[0], np.ones(6))
    onp.testing.assert_array_equal(
        optix.clip_by_global_norm(4.)(params, updates)[1], np.ones(10))

  def test_trace_should_keep_track_of_momentum(self):
    params = np.zeros(6)
    updates = np.ones(6)
    opt = state.init(optix.trace(0.99, False))(random.PRNGKey(0), params,
                                               updates)
    onp.testing.assert_array_equal(opt.trace, np.zeros(6))
    opt = opt.update(params, updates)
    onp.testing.assert_array_equal(opt.trace, np.ones(6))
    onp.testing.assert_array_equal(opt(params, updates), 1.99 * np.ones(6))

  def test_trace_should_keep_track_of_momentum_with_nesterov(self):
    params = np.zeros(6)
    updates = np.ones(6)
    opt = state.init(optix.trace(0.99, True))(random.PRNGKey(0), params,
                                              updates)
    onp.testing.assert_array_equal(opt.trace, np.zeros(6))
    opt = opt.update(params, updates)
    onp.testing.assert_array_equal(opt.trace, np.ones(6))
    onp.testing.assert_array_equal(
        opt(params, updates), (1.99 + 0.99**2) * np.ones(6))

  def test_scale_by_rms_should_scale_by_rms(self):
    params = np.zeros(9)
    updates = 2 * np.ones(9)
    opt = state.init(optix.scale_by_rms(0.5, 0.))(random.PRNGKey(0), params,
                                                  updates)
    onp.testing.assert_array_equal(opt.nu, np.zeros(9))
    opt = opt.update(params, updates)
    onp.testing.assert_array_equal(opt.nu, 2 * np.ones(9))
    onp.testing.assert_array_equal(
        opt(params, updates), 2 * np.ones(9) / np.sqrt(3.))

  def test_scale_by_stddev_should_scale_by_stddev(self):
    params = np.zeros(9)
    updates = 2 * np.ones(9)
    opt = state.init(optix.scale_by_stddev(0.5, 0.))(random.PRNGKey(0), params,
                                                     updates)
    onp.testing.assert_array_equal(opt.mu, np.zeros(9))
    onp.testing.assert_array_equal(opt.nu, np.zeros(9))
    opt = opt.update(params, updates)
    onp.testing.assert_array_equal(opt.mu, np.ones(9))
    onp.testing.assert_array_equal(opt.nu, 2 * np.ones(9))
    onp.testing.assert_array_equal(
        opt(params, updates), 2 * np.ones(9) / np.sqrt(0.75))

  def test_scale_by_adam_should_scale_by_adam(self):
    params = np.zeros(9)
    updates = 2 * np.ones(9)
    opt = state.init(optix.scale_by_adam(0.5, 0.5, 0.))(random.PRNGKey(0),
                                                        params, updates)
    onp.testing.assert_array_equal(opt.count, 0.)
    onp.testing.assert_array_equal(opt.mu, np.zeros(9))
    onp.testing.assert_array_equal(opt.nu, np.zeros(9))
    opt = opt.update(params, updates)
    onp.testing.assert_array_equal(opt.count, 1.)
    onp.testing.assert_array_equal(opt.mu, np.ones(9))
    onp.testing.assert_array_equal(opt.nu, 2 * np.ones(9))
    onp.testing.assert_array_equal(
        opt(params, updates), 1.5 / 0.75 * np.ones(9) / np.sqrt(3 / 0.75))

  def test_scale_by_schedule_should_update_scale(self):
    params = 0.
    updates = 1.

    def schedule(t):
      return 1. / ((t + 1)**2)

    opt = state.init(optix.scale_by_schedule(schedule))(random.PRNGKey(0),
                                                        params, updates)
    onp.testing.assert_array_equal(opt.count, 0.)
    opt = opt.update(params, updates)
    onp.testing.assert_array_equal(opt.count, 1.)
    onp.testing.assert_array_equal(opt(params, updates), 0.25)

  def test_add_noise_should_add_noise(self):
    params = 0.
    updates = 1.
    opt = state.init(optix.add_noise(1., 0., 0))(
        random.PRNGKey(0), params, updates)
    onp.testing.assert_array_equal(opt.count, 0.)
    onp.testing.assert_array_equal(opt.rng_key, random.PRNGKey(0))
    value, opt = opt.call_and_update(params, updates)
    onp.testing.assert_array_equal(
        value, 1. + random.normal(random.split(random.PRNGKey(0))[1], ()))

  def test_apply_every_should_delay_updates(self):
    params = 0.
    updates = 1.
    opt = state.init(optix.apply_every(5))(random.PRNGKey(0), params, updates)
    onp.testing.assert_array_equal(opt.count, 0.)
    onp.testing.assert_array_equal(opt.grad_acc, 0.)
    for _ in range(4):
      value, opt = opt.call_and_update(params, updates)
      onp.testing.assert_array_equal(value, 0.)
    value = opt(params, updates)
    onp.testing.assert_array_equal(value, 5.)


UPDATES = [
    ('sgd', optix.sgd(1e-1)),
    ('sgd_with_momentum', optix.sgd(1e-1, 0.9)),
    ('sgd_with_nesterov_momentum', optix.sgd(1e-1, 0.9, nesterov=True)),
    ('noisy_sgd', optix.noisy_sgd(1e-1)),
    ('adam', optix.adam(1e-1)),
    ('rmsprop', optix.rmsprop(1e-1)),
]


class OptimizerTest(parameterized.TestCase):

  @parameterized.named_parameters(UPDATES)
  def test_optimizer(self, update):

    def loss(x):
      return np.sum(x**2)

    x = np.array([3., 4.])
    opt = state.init(
        optix.optimize(loss, update, 500))(random.PRNGKey(0), x)
    x = jax.jit(opt.call)(x)
    onp.testing.assert_allclose(np.zeros(2), x, atol=1e-1, rtol=1e-1)


if __name__ == '__main__':
  absltest.main()
