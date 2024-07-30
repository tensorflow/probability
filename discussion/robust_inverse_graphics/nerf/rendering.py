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
"""Radiance field rendering."""

from collections.abc import Sequence
import functools
import operator
from typing import Any, Callable, NamedTuple, Optional

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from camp_zipnerf.internal import coord as mipnerf_coord
from camp_zipnerf.internal import math as mipnerf_math
from camp_zipnerf.internal import render as mipnerf_render
from camp_zipnerf.internal import stepfun as mipnerf_stepfun

__all__ = [
    'Ray',
    'RaySample',
    'ray_trace_mip_radiance_field',
    'RayTraceMipExtra',
]

# World-space coordinates. Shape: [3]
WorldSpace = jnp.ndarray
# Density at point (positive real). Shape: []
Density = jnp.ndarray
# Color at a point (elements in [0, 1]). Shape: [3]
RGB = jnp.ndarray


class RaySample(NamedTuple):
  """A sample from a ray.

  Attributes:
    position: Sample position. Shape: [3]
    viewdir: View directions (from camera). Shape: [3]
    covariance: Covariance of a Gaussian fit to the truncated cone associated
      with this ray sample. Shape: [3, 3].
  """
  position: WorldSpace
  viewdir: jnp.ndarray
  covariance: Optional[jnp.ndarray] = None

  @classmethod
  def test_sample(cls) -> 'RaySample':
    return cls(
        position=jnp.zeros(3), viewdir=jnp.ones(3), covariance=jnp.eye(3)
    )


class Ray(NamedTuple):
  """A ray.

  Attributes:
    origin: Origin of the ray.
    direction: Un-normalized direction of the ray. The magnitude encodes how
      fast to move along the ray.
    viewdir: The normalized direction of the ray.
    radius: Radius of the truncated cone associated with the ray at the origin.
  """
  origin: WorldSpace
  direction: jnp.ndarray
  viewdir: jnp.ndarray
  radius: Optional[jnp.ndarray] = None


@struct.dataclass
class RayTraceMipLevelExtra:
  """Extra-outputs from ray_trace_mip_radiance_field for each sampling level.

  `num_rfs` is the number of radiance fields, and could be an empty dimension.

  Attributes:
    z_vals: Values between the `near` and `far` which span the truncated cones
      that are then fit with gaussians. Shape: [num_samples + 1]
    means: Means of the gaussians where `rf_fn` was evaluated. Shape:
      [num_samples, 3]
    covariances: Covariances of the gaussians where `rf_fn` was evaluated.
      Shape: [num_samples, 3, 3]
    density: `rf_fn` density at each sample point: Shape: [num_samples, num_rfs]
    composed_density: Composed density. Shape: [num_samples]
    rgb: RGB outputs for this level. Shape: [3]
    acc: Accumulated alpha, pre-composed. Shape: [num_rfs]
    composed_acc: Accumulated alpha. Shape: []
    mean_distance: Mean distance from the camera to the scattering point along
      the ray. Shape: []
    p95_distance: 95th percentile distance from the camera to the scattering
      point along the ray. Shape: []
    extra: Extra outputs from `rf_fn`, for debugging and other uses.
  """
  z_vals: jnp.ndarray
  means: WorldSpace
  covariances: jnp.ndarray

  density: Density
  composed_density: Density

  rgb: RGB
  acc: jnp.ndarray
  composed_acc: jnp.ndarray
  mean_distance: jnp.ndarray
  p95_distance: jnp.ndarray

  extra: Any


@struct.dataclass
class RayTraceMipExtra(RayTraceMipLevelExtra):
  """Extra-outputs from ray_trace_mip_radiance_field.

  `num_rfs` is the number of radiance fields, and could be an empty dimension.

  Attributes:
    levels: Extra-outputs for each sampling level. The remaining fields
      correspond to the last sampling level.
  """

  levels: Sequence[RayTraceMipLevelExtra]


def ray_trace_mip_radiance_field(
    rf_fn: Callable[[RaySample], tuple[tuple[Density, RGB], Any]],
    ray: Ray,
    num_samples: Sequence[int],
    near: float,
    far: float,
    jitter: str = 'correlated',
    ray_shape: str = 'cone',
    background_color: jnp.ndarray = np.ones(3, np.float32),
    weight_bias: float = 0.01,
    epsilon: float = 1e-10,
    ray_warp_fn: str | Callable[[jax.Array], jax.Array] | None = None,
    seed: Optional[jax.Array] = None,
) -> tuple[RGB, RayTraceMipExtra]:
  """Given a single ray, and a radiance field, computes the color.

  `rf_fn` is a mapping `(ray_sample) -> ((density, rgb), extra)`.

  `rf_fn` is allowed to return values with a leading dimension, which is taken
  to indicate multiple RFs. Their colors and densities will be composed by
  summing the densities and blending the colors appropriately.

  The ray samples are generated by sampling a `z_val` from a distribution over
  `[near, far]` and computing a truncated cone boundaries as:
  `position = z_val * ray_direction + ray_origin`.

  This function uses multi-level sampling, where the `rf_fn` is first evaluated
  on a set of coarse positions and the densities computed at those locations are
  used to construct an improved sampling distribution. The number of levels is
  determined by the length of the `num_samples` argument.

  This function is intended to be used with Integrated Positional Encoding from
  [1]. It computes both positions(means) and covariances of gaussians that
  approximate the truncated cones along the ray.

  Args:
    rf_fn: Radiance field function.
    ray: The ray.
    num_samples: A sequence of numbers of samples for each level.
    near: Minimum z-val.
    far: Maximum z-val.
    jitter: Type of jitter to use. Can be `stratified`, `correlated` or `none`.
      This corresponds to generating independent samples within each bucket,
      generating the same sample for each bucket and no jitter at all.
    ray_shape: Can be either `cone` or `cylinder`.
    background_color: Background color to use.
    weight_bias: Bias to add to the sampling weights, so as to sample from the
      empty regions implied by `rf_fn` to make sure they stay empty.
    epsilon: Used to stabilize composition of multiple radiance fields.
    ray_warp_fn: Ray warp function. See Equation 11 in
      https://arxiv.org/abs/2111.12077.
    seed: Optional random seed used for jitter. Can be omitted if using the
      `none` jitter.

  Returns:
    rgb: Composed color.
    extra: An array of `RayTraceMipExtra`, one per level.

  #### References

  1. Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R.,
    & Srinivasan, P. P. (2021). Mip-NeRF: A Multiscale Representation for
    Anti-Aliasing Neural Radiance Fields. In arXiv [cs.CV]. arXiv.
    http://arxiv.org/abs/2103.13415
  """
  # We warp the [near, far] range into [0, 1], with the linear re-scaling by
  # default.
  _, s_to_z = mipnerf_coord.construct_ray_warps(ray_warp_fn, near, far)
  s_vals = jnp.array([0.0, 1.0])
  weights = jnp.ones(1)

  extras = []

  for level_num_samples in num_samples:
    # Fix logits for samples which have equal values.
    logits_resample = jnp.where(
        s_vals[..., 1:] > s_vals[..., :-1],
        mipnerf_math.safe_log(weights + weight_bias),
        -jnp.inf,
    )

    match jitter:
      case 'none':
        sample_seed = None
        single_jitter = False
      case 'correlated':
        sample_seed, seed = jax.random.split(seed)
        single_jitter = True
      case 'stratified':
        sample_seed, seed = jax.random.split(seed)
        single_jitter = False
      case _:
        raise ValueError(f'Unknown jitter type: {jitter}')

    # Sample values along the ray.
    s_vals = mipnerf_stepfun.sample_intervals(
        sample_seed,
        s_vals,
        w_logits=logits_resample,
        num_samples=level_num_samples,
        single_jitter=single_jitter,
        domain=(0.0, 1.0),
    )

    # For training stability.
    # TODO(siege): Make this configurable?
    s_vals = jax.lax.stop_gradient(s_vals)

    z_vals = s_to_z(s_vals)

    # Compute the means/covariances of the approximating gaussians.
    means, covs = mipnerf_render.cast_rays(
        tdist=z_vals,
        origins=ray.origin,
        directions=ray.direction,
        radii=ray.radius,
        ray_shape=ray_shape,
        diag=False,
    )

    # Evaluate `rf_fn` at the gaussian.
    (density, rgb), extra = jax.vmap(
        lambda pos, cov: rf_fn(  # pylint: disable=g-long-lambda
            RaySample(position=pos, covariance=cov, viewdir=ray.viewdir)
        )
    )(means, covs)

    # Compose across RFs, if necessary.
    if len(density.shape) == 1:
      rf_weights = density
      composed_density = density
      composed_rgb = rgb
    else:
      chex.assert_rank(density, 2)

      rf_weights = density / (density.sum(1, keepdims=True) + epsilon)
      composed_density = density.sum(1)
      composed_rgb = (rgb * rf_weights[..., jnp.newaxis]).sum(1)

    # Perform volumetric rendering. The weights computed here are used to inform
    # future levels.
    weights = mipnerf_render.compute_alpha_weights(
        density=composed_density,
        tdist=z_vals,
        dirs=ray.direction,
    )

    rendering = mipnerf_render.volumetric_rendering(
        rgbs=composed_rgb,
        weights=weights,
        tdist=z_vals,
        bg_rgbs=jnp.asarray(background_color),
        compute_extras=True,
    )

    composed_acc = rendering['acc']
    if len(density.shape) == 1:
      acc = composed_acc
    else:
      acc = (weights[..., jnp.newaxis] * rf_weights).sum(0)

    rt_extra = RayTraceMipLevelExtra(
        z_vals=z_vals,
        means=means,
        covariances=covs,
        composed_density=composed_density,
        density=density,
        rgb=rendering['rgb'],
        acc=acc,
        composed_acc=composed_acc,
        mean_distance=rendering['distance_mean'],
        p95_distance=rendering['distance_percentile_95'],
        extra=extra,
    )
    extras.append(rt_extra)

  return extras[-1].rgb, RayTraceMipExtra(
      levels=extras,
      z_vals=extras[-1].z_vals,
      means=extras[-1].means,
      covariances=extras[-1].covariances,
      composed_density=extras[-1].composed_density,
      density=extras[-1].density,
      rgb=extras[-1].rgb,
      acc=extras[-1].acc,
      composed_acc=extras[-1].composed_acc,
      mean_distance=extras[-1].mean_distance,
      p95_distance=extras[-1].p95_distance,
      extra=extras[-1].extra,
  )


@functools.partial(
    jax.jit, static_argnames=('rf_fn', 'num_samples', 'ray_warp_fn')
)
def render_rf(
    rf_fn: Callable[
        [RaySample, Any],
        tuple[tuple[Density, RGB], Any],
    ],
    rays: Ray,
    near: float,
    far: float,
    num_samples: Sequence[int],
    seed: jax.Array,
    ray_warp_fn: str | Callable[[jax.Array], jax.Array] | None = None,
    cond_kwargs: Any | None = None,
    **kwargs: Any,
) -> tuple[RGB, Any]:
  """Render a radiance field using Mip rendering.

  This is a thin wrapper around `ray_trace_mip_radiance_field` to handle
  arbitrary batch shapes and do proper seed handling.

  Args:
    rf_fn: Radiance field.
    rays: Rays.
    near: Near plane.
    far: Far plane.
    num_samples: See `ray_trace_mip_radiance_field`.
    seed: Random seed.
    ray_warp_fn: Ray warp function. See Equation 11 in
      https://arxiv.org/abs/2111.12077.
    cond_kwargs: If given, conditional data from which to derive rf_fn.
    **kwargs: Passed to `ray_trace_mip_radiance_field`.

  Returns:
    Same as `ray_trace_mip_radiance_field`.
  """
  if cond_kwargs is None:
    cond_kwargs = {}
  batch_shape = rays.origin.shape[:-1]
  batch_ndims = len(batch_shape)
  batch_size = functools.reduce(operator.mul, batch_shape)

  tree_flatten = functools.partial(
      jax.tree.map, lambda x: x.reshape((-1,) + x.shape[batch_ndims:])
  )

  @jax.vmap
  def _render(ray, cond_kwargs, seed):
    return ray_trace_mip_radiance_field(
        rf_fn=functools.partial(rf_fn, **cond_kwargs),
        ray=ray,
        num_samples=num_samples,
        near=near,
        far=far,
        seed=seed,
        ray_warp_fn=ray_warp_fn,
        **kwargs,
    )

  rgb, extra = _render(
      tree_flatten(rays),
      tree_flatten(cond_kwargs),
      jax.random.split(seed, batch_size),
  )

  return jax.tree.map(
      lambda x: x.reshape(batch_shape + x.shape[1:]), (rgb, extra)
  )
