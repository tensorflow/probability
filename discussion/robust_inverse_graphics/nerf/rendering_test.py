# Copyright 2024 The TensorFlow Probability Authors.
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
"""Tests for rendering."""

import jax
import jax.numpy as jnp
import numpy as np
from discussion.robust_inverse_graphics.nerf import rendering
from discussion.robust_inverse_graphics.util import test_util


class RenderingTest(test_util.TestCase):

  def testRenderMip(self):
    num_samples = [6, 4]

    def rf_fn(ray_sample):
      self.assertEqual(ray_sample.covariance.shape, [3, 3])
      return (0.0, np.zeros(3)), ()

    ray = rendering.Ray(
        origin=np.zeros(3),
        direction=np.ones(3),
        viewdir=np.ones(3) / np.sqrt(3),
        radius=np.array([0.1]),
    )

    @jax.jit
    def run(seed):
      return rendering.ray_trace_mip_radiance_field(
          rf_fn=rf_fn,
          ray=ray,
          num_samples=num_samples,
          near=1.0,
          far=2.0,
          seed=seed,
      )

    rgb, extra = run(jax.random.PRNGKey(0))

    self.assertEqual([3], rgb.shape)

    for level, (ns, extra_l) in enumerate(zip(num_samples, extra.levels)):
      self.assertEqual([ns + 1], extra_l.z_vals.shape, msg=f'{level=}')
      self.assertEqual([ns, 3], extra_l.means.shape, msg=f'{level=}')
      self.assertEqual([ns, 3, 3], extra_l.covariances.shape, msg=f'{level=}')
      self.assertEqual([ns], extra_l.density.shape, msg=f'{level=}')
      self.assertEqual([ns], extra_l.composed_density.shape, msg=f'{level=}')

      self.assertEqual([], extra_l.acc.shape)
      self.assertEqual([], extra_l.mean_distance.shape)
      self.assertEqual([], extra_l.p95_distance.shape)
      self.assertEqual([], extra_l.composed_acc.shape)

    ns = num_samples[-1]
    self.assertEqual([ns + 1], extra.z_vals.shape, msg='last level')
    self.assertEqual([ns, 3], extra.means.shape, msg='last level')
    self.assertEqual([ns, 3, 3], extra.covariances.shape, msg='last level')
    self.assertEqual([ns], extra.density.shape, msg='last level')
    self.assertEqual([ns], extra.composed_density.shape, msg='last level')

    self.assertEqual([], extra.acc.shape)
    self.assertEqual([], extra.mean_distance.shape)
    self.assertEqual([], extra.p95_distance.shape)
    self.assertEqual([], extra.composed_acc.shape)

  def testRenderMipComposed(self):
    num_nerfs = 3
    num_samples = [6, 4]

    def rf_fn(ray_sample):
      self.assertEqual(ray_sample.covariance.shape, [3, 3])
      return (np.zeros(num_nerfs), np.zeros((num_nerfs, 3))), ()

    ray = rendering.Ray(
        origin=np.zeros(3),
        direction=np.ones(3),
        viewdir=np.ones(3) / np.sqrt(3),
        radius=np.array([0.1]),
    )

    @jax.jit
    def run(seed):
      return rendering.ray_trace_mip_radiance_field(
          rf_fn=rf_fn,
          ray=ray,
          num_samples=num_samples,
          near=1.0,
          far=2.0,
          seed=seed,
      )

    rgb, extra = run(jax.random.PRNGKey(0))

    self.assertEqual([3], rgb.shape)

    for level, (ns, extra_l) in enumerate(zip(num_samples, extra.levels)):
      self.assertEqual([ns + 1], extra_l.z_vals.shape, msg=f'{level=}')
      self.assertEqual([ns, 3], extra_l.means.shape, msg=f'{level=}')
      self.assertEqual([ns, 3, 3], extra_l.covariances.shape, msg=f'{level=}')
      self.assertEqual([ns, num_nerfs], extra_l.density.shape, msg=f'{level=}')
      self.assertEqual([ns], extra_l.composed_density.shape, msg=f'{level=}')

      self.assertEqual([num_nerfs], extra_l.acc.shape)
      self.assertEqual([], extra_l.mean_distance.shape)
      self.assertEqual([], extra_l.p95_distance.shape)
      self.assertEqual([], extra_l.composed_acc.shape)

    ns = num_samples[-1]
    self.assertEqual([ns + 1], extra.z_vals.shape, msg='last level')
    self.assertEqual([ns, 3], extra.means.shape, msg='last level')
    self.assertEqual([ns, 3, 3], extra.covariances.shape, msg='last level')
    self.assertEqual([ns, num_nerfs], extra.density.shape, msg='last level')
    self.assertEqual([ns], extra.composed_density.shape, msg='last level')

    self.assertEqual([num_nerfs], extra.acc.shape)
    self.assertEqual([], extra.mean_distance.shape)
    self.assertEqual([], extra.p95_distance.shape)
    self.assertEqual([], extra.composed_acc.shape)

  def test_render_fn(self):
    rays = rendering.Ray(
        origin=jnp.zeros([32, 32, 3]),
        radius=jnp.ones((32, 32)),
        direction=jnp.ones([32, 32, 3]),
        viewdir=jnp.ones([32, 32, 3]),
    )
    rf_fn = lambda ray_sample: ((jnp.zeros([]), jnp.zeros(3)), ())

    rgb, extra = rendering.render_rf(
        rf_fn,
        rays=rays,
        near=1.0,
        far=2.0,
        num_samples=(3, 5),
        seed=jax.random.PRNGKey(0),
    )

    self.assertEqual(rgb.shape, [32, 32, 3])
    self.assertEqual(extra.composed_acc.shape, [32, 32])

  def test_conditional_render_fn(self):
    rays = rendering.Ray(
        origin=jnp.zeros([32, 32, 3]),
        radius=jnp.ones((32, 32)),
        direction=jnp.ones([32, 32, 3]),
        viewdir=jnp.ones([32, 32, 3]),
    )

    def cond_rf_fn(ray_sample, cond):
      del ray_sample
      return (jnp.zeros([]), jnp.zeros(3)), ({'cond_sum': cond.sum()},)

    cond_dim = 7
    rgb, extra = rendering.render_rf(
        cond_rf_fn,
        rays=rays,
        near=1.0,
        far=2.0,
        num_samples=(3, 5),
        seed=jax.random.PRNGKey(0),
        cond_kwargs={'cond': jnp.ones([32, 32, cond_dim])},
    )

    self.assertEqual(rgb.shape, [32, 32, 3])
    self.assertEqual(extra.composed_acc.shape, [32, 32])
    self.assertAllEqual(extra.extra[0]['cond_sum'], cond_dim)


if __name__ == '__main__':
  test_util.main()
