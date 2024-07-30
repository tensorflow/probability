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
import functools
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from discussion.robust_inverse_graphics import models
from discussion.robust_inverse_graphics import probnerf
from discussion.robust_inverse_graphics.nerf import rendering
from discussion.robust_inverse_graphics.util import test_util


class TestNerf(nn.Module):

  @nn.compact
  def __call__(
      self, ray_sample: rendering.RaySample
  ) -> tuple[tuple[rendering.Density, rendering.RGB], Any]:
    init_fn = functools.partial(jax.random.uniform, shape=[3])
    rgb = jax.nn.sigmoid(self.param('rgb', init_fn))
    density = jnp.ones([])

    return (density, rgb), ()


class ModelsTest(test_util.TestCase):

  def test_fields_with_view_axis(self):
    rays_shape = 2, 3
    ones = jnp.ones(rays_shape)
    zeros = jnp.zeros(rays_shape)
    example_1 = models.Example(
        rays=rendering.Ray(zeros, ones, ones), rgb=zeros,
        scene_id=jnp.zeros([]),
    )
    example_2 = jax.tree.map(lambda x: x, example_1)
    example = jax.tree.map(
        lambda x_1, x_2: None,
        example_1.fields_with_view_axis(),
        example_2.fields_with_view_axis(),
    ).value
    self.assertIsNone(example.rays.origin)
    self.assertIsNone(example.rgb)
    self.assertEqual(example.scene_id, 0)

  def test_rgb_log_likelihood(self):
    ll, extra = models.rgb_log_likelihood(
        recon_rgb=jnp.zeros(3),
        rgb=jnp.zeros(3),
        obs_scale=jnp.ones([]),
    )

    self.assertEqual(ll.shape, [])
    self.assertEqual(extra.rgb_loss.shape, [])
    self.assertEqual(extra.per_channel_ll.shape, [3])

  def test_sinh_arcsinh_rgb_log_likelihood(self):
    ll, extra = models.sinh_arcsinh_rgb_log_likelihood(
        recon_rgb=jnp.zeros(3),
        rgb=jnp.zeros(3),
        obs_scale=jnp.ones([]),
    )

    self.assertEqual(ll.shape, [])
    self.assertEqual(extra.rgb_loss.shape, [])
    self.assertEqual(extra.per_channel_ll.shape, [3])

  def test_chunked_render_latents_fn(self):
    rgb = jnp.linspace(0.0, 1.0, 40 * 3).reshape([-1, 3])
    example = models.Example(rgb=rgb, scene_id=jnp.zeros([]))
    latents = jnp.array([1.0])

    def render_latents_fn(latents, example, _):
      self.assertEqual(example.rgb.shape, (2, 3))
      return example.rgb + latents, ()

    rendered_rgb, _ = models.chunked_render_latents(
        render_latents_fn=render_latents_fn,
        latents=latents,
        example=example,
        seed=jax.random.PRNGKey(0),
        num_chunks=20,
    )
    self.assertAllClose(rgb + latents, rendered_rgb)

  def test_make_nerf_model(self):
    model_fn = functools.partial(
        models.make_nerf_model,
        TestNerf(),
        near=1.0,
        far=2.0,
        num_samples=(4, 4),
        obs_scales=(1.0, 1.0),
    )
    model = model_fn()

    init_latents, _ = model.init_latents_fn(jax.random.PRNGKey(0))
    example = models.Example(
        rgb=jnp.zeros([16, 3]),
        rays=rendering.Ray(
            origin=jnp.zeros([16, 3]),
            direction=jnp.ones([16, 3]),
            viewdir=jnp.ones([16, 3]),
            radius=jnp.ones([16]),
        ),
    )
    ll, _ = model.log_likelihood_fn(
        init_latents,
        models.LikelihoodInputs(example=example, seed=jax.random.PRNGKey(1)),
    )
    self.assertEqual(ll.shape, [])

  def test_make_probnerf_model(self):
    num_latent = 2
    grid_size = 3
    obs_scale = 0.1
    total_num_views = 10
    im_height, im_width = 4, 4
    num_rays = total_num_views * im_width * im_height

    nerf = probnerf.TwoPartNerf(grid_size=grid_size)
    nerf_variables = nerf.init(
        jax.random.PRNGKey(0),
        rendering.RaySample.test_sample(),
    )
    hypernet = probnerf.DecoderHypernet(nerf_variables)
    realnvp = probnerf.RealNVPStack(num_latent)

    model = models.make_probnerf_model(
        nerf,
        hypernet,
        realnvp,
        num_rays=num_rays,
        obs_scales=(obs_scale,),
    )

    init_params, _ = model.init_params_fn(jax.random.PRNGKey(0))

    (
        rgb_seed,
        origin_seed,
        direction_seed,
        camera_world_matrix_seed,
        latents_seed,
    ) = jax.random.split(jax.random.PRNGKey(0), 5)
    rgb = jax.random.normal(rgb_seed, [total_num_views, im_height, im_width, 3])
    origin = jax.random.normal(
        origin_seed, [total_num_views, im_height, im_width, 3]
    )
    direction = jax.random.normal(
        direction_seed, [total_num_views, im_height, im_width, 3]
    )
    viewdir = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    camera_world_matrix = jax.random.normal(
        camera_world_matrix_seed, [total_num_views, 4, 4]
    )
    latents = jax.random.normal(latents_seed, [num_latent])
    example = models.Example(
        rgb=rgb,
        rays=rendering.Ray(
            origin=origin,
            direction=direction,
            viewdir=viewdir,
            radius=jnp.ones([total_num_views, im_height, im_width]),
        ),
        camera_world_matrix=camera_world_matrix,
    )

    ll = model.log_likelihood_fn(
        init_params,
        latents,
        models.LikelihoodInputs(example=example, seed=jax.random.PRNGKey(1)),
    )[0]
    self.assertEqual(ll.shape, [])

  def test_make_probnerf_guide(self):
    num_latent = 2
    grid_size = 3
    total_num_views = 10
    im_height, im_width = 4, 4

    nerf = probnerf.TwoPartNerf(grid_size=grid_size)
    nerf_variables = nerf.init(
        jax.random.PRNGKey(0),
        rendering.RaySample.test_sample(),
    )
    guide = probnerf.Guide(num_latent, nerf_variables)
    probnerf_guide = models.make_probnerf_guide(guide, im_height, im_width)

    init_params, _ = probnerf_guide.init_params_fn(jax.random.PRNGKey(0))

    (
        rgb_seed,
        origin_seed,
        direction_seed,
        camera_world_matrix_seed,
    ) = jax.random.split(jax.random.PRNGKey(0), 4)
    rgb = jax.random.normal(rgb_seed, [total_num_views, im_height, im_width, 3])
    origin = jax.random.normal(
        origin_seed, [total_num_views, im_height, im_width, 3]
    )
    direction = jax.random.normal(
        direction_seed, [total_num_views, im_height, im_width, 3]
    )
    viewdir = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    camera_world_matrix = jax.random.normal(
        camera_world_matrix_seed, [total_num_views, 4, 4]
    )
    example = models.Example(
        rgb=rgb,
        rays=rendering.Ray(
            origin=origin,
            direction=direction,
            viewdir=viewdir,
        ),
        camera_world_matrix=camera_world_matrix,
    )

    latents, extra = probnerf_guide.guide_sample_fn(
        init_params,
        example,
        jax.random.PRNGKey(1),
    )

    self.assertEqual(latents.shape, (num_latent,))
    assert extra.log_prob_stop_grad_params
    self.assertEqual(extra.log_prob_stop_grad_params.shape, ())


if __name__ == '__main__':
  test_util.main()
