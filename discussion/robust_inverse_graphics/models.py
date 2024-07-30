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
"""Probabilistic models."""

import abc
from collections.abc import Callable
import dataclasses
import functools
import operator
from typing import Any

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
from discussion.robust_inverse_graphics import diffusion
from discussion.robust_inverse_graphics import saving
from discussion.robust_inverse_graphics.nerf import rendering
from discussion.robust_inverse_graphics.util import tree_util
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
pbroadcast = tfp.internal.distribute_lib.rwb_pbroadcast
reduce_mean = tfp.internal.distribute_lib.reduce_mean

__all__ = [
    'chunked_render_latents',
    'Dataset',
    'EvaluationGuideParams',
    'Example',
    'ExamplesIterator',
    'Extra',
    'Guide',
    'GuideParams',
    'GuideSampleExtra',
    'Latents',
    'LikelihoodInputs',
    'make_mean_field_guide',
    'make_nerf_model',
    'make_probnerf_guide',
    'make_probnerf_model',
    'MeanFieldGuideParams',
    'Model',
    'ModelParams',
    'ParameterizedGuide',
    'ParameterizedModel',
    'rgb_log_likelihood',
    'sinh_arcsinh_rgb_log_likelihood',
    'RGBExtra',
    'UnprocessedExample',
]

ModelParams = Any
GuideParams = Any
Latents = Any
Extra = Any
UnprocessedExample = Any


@saving.register
@struct.dataclass
class GuideSampleExtra:
  log_prob: jax.Array | None = None
  log_prob_stop_grad_params: jax.Array | None = None
  dist_params: Any | None = None


# TODO(siege): Annotate these fields in the class itself.
# The selectors need to live outside, so that different DataclassViews compare
# equal.
def _fields_with_view_axis_selector(f: str) -> bool:
  return f not in ['scene_id', 'scene_embedding']


def _fields_with_ray_axis_selector(f: str) -> bool:
  return f in ['rays', 'rgb', 'depth', 'segmentation']


@struct.dataclass
class Example:
  """Inference data/example.

  Attributes:
    rays: The rays used to generate the image(s).
    rgb: The colors of the pixels of the image(s). Shape: [3]
    time: Optional time value, for video data. Shape: []
    depth: Optional depth value, increasing with forward distance from the
      camera. Shape: []
    segmentation: Optional integer-valued segmentation mask, with distinct
      values identifying distinct objects in the scene. Shape: []
    camera_world_matrix: Optional camera world matrix. Shape: [4, 4]
    camera_intrinsics: Optional pinhole camera intrinsic parameters. Shape: [3,
      3]
    scene_id: Optional scene index. Shape: []
    scene_embedding: Optional scene embedding. Shape [...]
  """

  rays: rendering.Ray | None = None
  rgb: Any | None = None
  time: Any | None = None
  depth: Any | None = None
  segmentation: Any | None = None
  camera_world_matrix: Any | None = None
  camera_intrinsics: Any | None = None
  scene_id: Any | None = None
  scene_embedding: Any | None = None

  def fields_with_view_axis(self) -> tree_util.DataclassView['Example']:
    """Returns a view with only fields that typically have the view axis."""
    return tree_util.DataclassView(self, _fields_with_view_axis_selector)

  def fields_with_ray_axis(self) -> tree_util.DataclassView['Example']:
    """Returns a view with only fields that typically have the ray axis."""
    return tree_util.DataclassView(self, _fields_with_ray_axis_selector)

  @classmethod
  def test_example(
      cls, num_views: int = 5, im_height: int = 128, im_width: int = 128
  ) -> 'Example':
    shape = num_views, im_height, im_width, 3
    ones = jnp.ones(shape)
    zeros = jnp.zeros(shape)
    return cls(rays=rendering.Ray(zeros, ones, ones), rgb=zeros)

  @classmethod
  def view_axis(cls) -> 'Example':
    """Returns the example with fields replaced with the view axis location."""
    axes = {}
    for f in dataclasses.fields(cls):
      if f.name in ['scene_id', 'scene_embedding']:
        axis = None
      else:
        axis = 0
      axes[f.name] = axis
    return cls(**axes)


class ExamplesIterator(metaclass=abc.ABCMeta):
  """A sized iterator for examples."""

  @abc.abstractmethod
  def __next__(self) -> UnprocessedExample:
    pass

  def __iter__(self) -> 'ExamplesIterator':
    return self

  @abc.abstractmethod
  def _size(self) -> int:
    """Epoch size."""

  @property
  def size(self) -> int:
    return self._size()

  @abc.abstractmethod
  def save(self, checkpoint_dir: str):
    """Save the checkpoint iteration state."""

  @abc.abstractmethod
  def load(self, checkpoint_dir: str):
    """Load the checkpoint iteration state."""


@dataclasses.dataclass
class Dataset:
  train_examples_fn: Callable[[], ExamplesIterator]
  test_examples_fn: Callable[[], ExamplesIterator]
  process_example_fn: Callable[[UnprocessedExample], Example] = lambda x: x


@struct.dataclass
class RGBExtra:
  """Extras from `rgb_log_likelihood`.

  S below refers to the shape of RGB, which is at least a vector.

  Attributes:
    per_channel_ll: Per-channel likelihood. Shape: [S]
    rgb_loss: L2 reconstruction loss. Shape: []
  """

  per_channel_ll: jax.Array
  rgb_loss: jax.Array


def rgb_log_likelihood(
    recon_rgb: rendering.RGB,
    rgb: rendering.RGB,
    obs_scale: jax.Array,
) -> tuple[jax.Array, RGBExtra]:
  """RGB per-channel likelihood with Gaussian noise.

  S below refers to the shape of RGB, which is at least a vector.

  Args:
    recon_rgb: Reconstructed RGB. Shape: [S, 3]
    rgb: Target RGB. Shape: [S, 3]
    obs_scale: Observation scale. Shape: []

  Returns:
    A tuple of:
      The likelihood value.
      RGBExtra, for extra outputs.
  """
  rgb_loss = ((recon_rgb - rgb) ** 2).mean()
  per_channel_ll = tfd.Normal(recon_rgb, obs_scale).log_prob(rgb)
  ll = per_channel_ll.sum()

  extra = RGBExtra(
      per_channel_ll=per_channel_ll,
      rgb_loss=rgb_loss,
  )

  return ll, extra


def sinh_arcsinh_rgb_log_likelihood(
    recon_rgb: rendering.RGB,
    rgb: rendering.RGB,
    obs_scale: jax.Array,
    reversion_to_mean: jax.Array | float = 0.8,
    obs_scale_factor: jax.Array | float = 2.1,
    tailweight: jax.Array | float = 1.1,
) -> tuple[jax.Array, RGBExtra]:
  """RGB per-channel likelihood with a skewed, SinhArcsinh distributed noise.

  S below refers to the shape of RGB, which is at least a vector.

  Args:
    recon_rgb: Reconstructed RGB. Shape: [S, 3]
    rgb: Target RGB. Shape: [S, 3]
    obs_scale: Observation scale. Shape: []
    reversion_to_mean: Reversion to mean factor. Shape: []
    obs_scale_factor: Observation scale factor. Shape: []
    tailweight: Tailweight. Shape: []

  Returns:
    A tuple of:
      The likelihood value.
      RGBExtra, for extra outputs.
  """
  rgb_loss = ((recon_rgb - rgb) ** 2).mean()
  per_channel_ll = tfd.SinhArcsinh(
      0.5 + reversion_to_mean * (recon_rgb - 0.5),
      obs_scale * obs_scale_factor,
      (0.5 - recon_rgb),
      tailweight,
  ).log_prob(rgb)
  ll = per_channel_ll.sum()

  extra = RGBExtra(
      per_channel_ll=per_channel_ll,
      rgb_loss=rgb_loss,
  )

  return ll, extra


@functools.partial(jax.jit, static_argnames=('render_latents_fn', 'num_chunks'))
def chunked_render_latents(
    render_latents_fn: Callable[
        [Latents, Example, jax.Array],
        tuple[rendering.RGB, Extra],
    ],
    latents: Latents,
    example: 'Example',
    seed: jax.Array,
    num_chunks: int = 40,
) -> tuple[rendering.RGB, Extra]:
  """Renders latents in a chunked way, to save on memory."""
  example_view = jax.tree.map(
      lambda x: x.reshape((num_chunks, x.shape[0] // num_chunks) + x.shape[1:]),
      example.fields_with_ray_axis(),
  )

  rgb, extra = jax.lax.map(
      lambda example_seed: render_latents_fn(  # pylint: disable=g-long-lambda
          latents, example_seed[0].value, example_seed[1]
      ),
      (example_view, jax.random.split(seed, num_chunks)),
  )

  return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), (rgb, extra))


@struct.dataclass
class LikelihoodInputs:
  """Additional inputs to the likelihood function.

  Attributes:
    example: The example.
    seed: The random seed.
    step: The inference step.
  """

  example: Example | None = None
  seed: jax.Array | None = None
  step: jax.Array | None = None


@saving.register
@struct.dataclass
class LikelihoodExtra:
  psnr: jax.Array | None = None
  render_extra: Any | None = None
  rgb_mse: jax.Array | None = None


@struct.dataclass
class Model:
  """A model."""

  init_latents_fn: Callable[[jax.Array], tuple[Latents, Extra]]
  render_latents_fn: Callable[
      [Latents, Example, jax.Array],
      tuple[rendering.RGB, Extra],
  ]
  log_likelihood_fn: Callable[
      [Latents, LikelihoodInputs], tuple[jax.Array, Extra]
  ]
  prior_log_prob_fn: Callable[[Latents], tuple[jax.Array, Extra]]
  prior_sample_fn: Callable[[jax.Array], tuple[Latents, Extra]]
  reduce_extra_fn: (
      Callable[[Extra, Extra, str | None], tuple[Extra, Extra]] | None
  ) = None


@struct.dataclass
class SSDNeRFModel(Model):
  """SSDNeRF model."""

  denoise_fn: Callable[[jax.Array, jax.Array], tuple[jax.Array, Any]] | None = (
      None
  )
  denoise_output: diffusion.DenoiseOutputType | None = None
  log_snr_fn: Callable[[jax.Array], jax.Array] | None = None
  num_rays: int | None = None
  near: float | None = None
  far: float | None = None
  num_samples: tuple[int, ...] | None = None
  obs_scales: tuple[float, ...] | None = None
  prior_map_bwd_fn: Callable[[jax.Array], jax.Array] = lambda x: x
  prior_map_fwd_fn: Callable[[jax.Array], jax.Array] = lambda x: x


@struct.dataclass
class GenSRTModel(Model):
  """GenSRT model."""

  denoise_fn: Callable[[jax.Array, jax.Array], tuple[jax.Array, Any]] | None = (
      None
  )
  denoise_output: diffusion.DenoiseOutputType | None = None
  log_snr_fn: Callable[[jax.Array], jax.Array] | None = None
  num_rays: int | None = None


def make_nerf_model(
    nerf: nn.Module,
    near: float,
    far: float,
    num_samples: tuple[int, ...],
    obs_scales: tuple[float, ...],
    anneal_nerf: bool = False,
    num_rays: int = 0,
    rgb_log_likelihood_fn: Callable[
        [rendering.RGB, rendering.RGB, jax.Array],
        tuple[jax.Array, RGBExtra],
    ] = rgb_log_likelihood,
    ray_warp_fn: str | Callable[[jax.Array], jax.Array] | None = None,
) -> Model:
  """Creates a model from a NeRF.

  Args:
    nerf: A NeRF model. The call method should have the signature of
      `(RaySample) -> ((Density, RGB), Extra)`
    near: The near plane.
    far: The far plane.
    num_samples: Number of samples for Mip-NeRF rendering.
    obs_scales: Scales for the likelihoods for each Mip-NeRF level.
    anneal_nerf: If True, also pass in a `step` argument to the NeRF model,
      typically for annealing purposes.
    num_rays: If non-zero, subsample rays to this amount.
    rgb_log_likelihood_fn: Pixel-level likelihood function to use.
    ray_warp_fn: Ray warp function. See Equation 11 in
      https://arxiv.org/abs/2111.12077.

  Returns:
    A Model.
  """

  def init_latents_fn(seed):
    if anneal_nerf:
      kwargs = {'step': 0}
    else:
      kwargs = {}
    return (
        nerf.init(
            seed,
            rendering.RaySample(
                position=jnp.zeros(3),
                covariance=jnp.ones((3, 3)),
                viewdir=jnp.ones(3),
            ),
            **kwargs,
        ),
        (),
    )

  def render_latents_fn(latents, example, seed, step=None):
    if anneal_nerf:
      kwargs = {'step': step}
    else:
      kwargs = {}
    return rendering.render_rf(
        rf_fn=functools.partial(nerf.apply, latents, **kwargs),
        rays=example.rays,
        near=near,
        far=far,
        num_samples=num_samples,
        seed=seed,
        ray_warp_fn=ray_warp_fn,
    )

  def log_likelihood_fn(latents, inputs, return_render_extra=False):
    seed = inputs.seed

    if num_rays > 0:
      seed, subsample_seed = jax.random.split(seed)
      rays, rgb = subsample_rays(inputs.example, num_rays, subsample_seed)
      total_num_rays = functools.reduce(
          operator.mul, inputs.example.rgb.shape[:-1]
      )
      ll_factor = total_num_rays / num_rays
    else:
      rays = inputs.example.rays
      rgb = inputs.example.rgb
      ll_factor = 1

    _, extra = render_latents_fn(latents, Example(rays=rays), seed, inputs.step)
    ll = 0.0
    for extra_l, obs_scale in zip(extra.levels, obs_scales):
      one_ll, ll_extra = rgb_log_likelihood_fn(
          extra_l.rgb,
          rgb,
          obs_scale,
      )
      ll += one_ll
    # Grab the ll_extra from the last level, as that corresponds to the final
    # reconstruction.
    rgb_mse = ll_extra.rgb_loss
    if not return_render_extra:
      extra = None
    return ll * ll_factor, LikelihoodExtra(
        psnr=None, rgb_mse=rgb_mse, render_extra=extra
    )

  def prior_log_prob_fn(latents):
    del latents  # Unused.
    return 0.0, ()

  def prior_sample_fn(seed):
    return init_latents_fn(seed)

  def reduce_extra_fn(prior_extra, likelihood_extra, example_axis_name=None):
    rgb_mse = reduce_mean(
        likelihood_extra.rgb_mse, named_axis=example_axis_name
    )
    return prior_extra, likelihood_extra.replace(
        rgb_mse=rgb_mse, psnr=-10 * jnp.log10(rgb_mse)
    )

  return Model(
      init_latents_fn=init_latents_fn,
      render_latents_fn=render_latents_fn,
      log_likelihood_fn=log_likelihood_fn,
      prior_log_prob_fn=prior_log_prob_fn,
      prior_sample_fn=prior_sample_fn,
      reduce_extra_fn=reduce_extra_fn,
  )


@struct.dataclass
class ParameterizedModel:
  """A parameterized model."""

  init_params_fn: Callable[[jax.Array], tuple[ModelParams, Extra]]
  prior_sample_fn: Callable[[ModelParams, jax.Array], tuple[Latents, Extra]]
  prior_log_prob_fn: Callable[[ModelParams, Latents], tuple[jax.Array, Extra]]
  render_latents_fn: Callable[
      [ModelParams, Latents, Example, jax.Array],
      tuple[rendering.RGB, Extra],
  ]
  log_likelihood_fn: Callable[
      [ModelParams, Latents, LikelihoodInputs],
      tuple[jax.Array, Extra],
  ]
  reduce_extra_fn: (
      Callable[
          [Extra, Extra, str | None],
          tuple[Extra, Extra],
      ]
      | None
  ) = None


@struct.dataclass
class ParameterizedSSDNeRFModel(ParameterizedModel):
  """A parameterized SSDNeRF model."""

  denoise_fn: (
      Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, Any]] | None
  ) = None
  log_snr_fn: Callable[[jax.Array], jax.Array] | None = None
  num_rays: int | None = None
  near: float | None = None
  far: float | None = None
  num_samples: tuple[int, ...] | None = None
  obs_scales: tuple[float, ...] | None = None
  denoise_output: diffusion.DenoiseOutputType | None = None
  prior_map_fwd_fn: Callable[[jax.Array], jax.Array] = lambda x: x
  prior_map_bwd_fn: Callable[[jax.Array], jax.Array] = lambda x: x


@struct.dataclass
class ParameterizedGenSRTModel(ParameterizedModel):
  """A parameterized GenSRT model."""

  denoise_fn: (
      Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, Any]] | None
  ) = None
  log_snr_fn: Callable[[jax.Array], jax.Array] | None = None
  num_rays: int | None = None
  denoise_output: diffusion.DenoiseOutputType | None = None


@struct.dataclass
class Guide:
  """A guide."""

  guide_sample_fn: Callable[
      [Example, jax.Array],
      tuple[Latents, GuideSampleExtra],
  ]
  guide_log_prob_fn: (
      Callable[
          [Example, Latents],
          tuple[jax.Array, Extra],
      ]
      | None
  ) = None
  reduce_extra_fn: (
      Callable[
          [GuideSampleExtra, Extra, str | None],
          tuple[GuideSampleExtra, Extra],
      ]
      | None
  ) = None


@struct.dataclass
class ParameterizedGuide:
  """A parameterized guide."""

  init_params_fn: Callable[[jax.Array], tuple[GuideParams, Extra]]
  guide_sample_fn: Callable[
      [GuideParams, Example, jax.Array],
      tuple[Latents, GuideSampleExtra],
  ]
  guide_log_prob_fn: (
      Callable[
          [GuideParams, Example, Latents],
          tuple[jax.Array, Extra],
      ]
      | None
  ) = None
  reduce_extra_fn: (
      Callable[
          [GuideSampleExtra, Extra, str | None],
          tuple[GuideSampleExtra, Extra],
      ]
      | None
  ) = None


@saving.register
@struct.dataclass
class ProbNeRFModelParams:
  hypernet_params: Any
  realnvp_params: Any
  corruption_params: Any | None = None


def make_probnerf_model(
    nerf: nn.Module,
    hypernet: nn.Module,
    realnvp: nn.Module,
    corruption_nerf: nn.Module | None = None,
    num_rays: int = 1024,
    near: float = 0.2,
    far: float = 1.5,
    num_samples: tuple[int, ...] = (48, 48),
    obs_scales: tuple[float, ...] = (1.0, 0.1),
    rgb_log_likelihood_fn: Callable[
        [rendering.RGB, rendering.RGB, jax.Array],
        tuple[jax.Array, RGBExtra],
    ] = rgb_log_likelihood,
) -> ParameterizedModel:
  """Creates a ProbNeRF model [1].

  Args:
    nerf: Base NeRF.
    hypernet: Hypernetwork mapping from z to nerf weights.
    realnvp: RealNVP prior.
    corruption_nerf: Optional corruption nerf.
    num_rays: Number of rays to subsample after view subsampling.
    near: The near plane.
    far: The far plane.
    num_samples: Number of samples for Mip-NeRF rendering.
    obs_scales: Scales for the likelihoods for each Mip-NeRF level.
    rgb_log_likelihood_fn: Pixel-level likelihood function to use.

  Returns:
    A ProbNeRF model.

  #### References

  [1] Hoffman, M. D., Le, T. A., Sountsov, P., Suter, C., Lee, B., Mansinghka,
    V. K., & Saurous, R. A. (2023). ProbNeRF: Uncertainty-Aware Inference of 3D
    Shapes from 2D Images. International Conference on Artificial Intelligence
    and Statistics. https://arxiv.org/abs/2210.17415
  """
  num_latent = realnvp.ndims

  def init_params_fn(seed):
    hypernet_seed, corruption_nerf_seed, realnvp_seed = jax.random.split(
        seed, 3
    )

    if corruption_nerf is None:
      corruption_params = None
    else:
      corruption_params = corruption_nerf.init(
          corruption_nerf_seed,
          rendering.RaySample(
              position=jnp.zeros(3),
              covariance=jnp.ones((3, 3)),
              viewdir=jnp.ones(3),
          ),
      )

    return (
        ProbNeRFModelParams(
            hypernet_params=hypernet.init(
                hypernet_seed, jnp.zeros((num_latent,))
            ),
            realnvp_params=realnvp.init(realnvp_seed, jnp.zeros((num_latent,))),
            corruption_params=corruption_params,
        ),
        (),
    )

  def prior_sample_fn(params, seed):
    pulled_back_latents = tfd.MultivariateNormalDiag(
        0.0, jnp.ones(num_latent)
    ).sample(seed=seed)
    latents, _ = realnvp.apply(
        params.realnvp_params, pulled_back_latents, forward=True
    )
    return latents, ()

  def prior_log_prob_fn(params, latents):
    pulled_back_latents, ildj = realnvp.apply(
        params.realnvp_params, latents, forward=False
    )
    return (
        tfd.MultivariateNormalDiag(0.0, jnp.ones(latents.shape[-1])).log_prob(
            pulled_back_latents
        )
        + ildj,
        (),
    )

  @jax.jit
  def mipnerf_render_rays_from_weights(
      nerf_params, corruption_params, rays, seed
  ):
    if corruption_params is None:
      rf = lambda ray_sample: nerf.apply(nerf_params, ray_sample)
    else:
      if corruption_nerf is None:
        raise ValueError(
            'corruption_nerf is None, but corruption_params are not?'
        )
      if nerf_params is None:
        rf = lambda ray_sample: corruption_nerf.apply(
            corruption_params, ray_sample
        )
      else:
        rf = lambda ray_sample: jax.tree.map(  # pylint: disable=g-long-lambda
            lambda *x: jnp.stack(x, 0),
            nerf.apply(nerf_params, ray_sample),
            corruption_nerf.apply(corruption_params, ray_sample),
        )
    return rendering.render_rf(
        rf_fn=rf,
        rays=rays,
        near=near,
        far=far,
        num_samples=num_samples,
        seed=seed,
    )

  def render_latents_fn(
      params,
      latents,
      example,
      seed,
      render_corruption=True,
      render_parts=('scene', 'corruption'),
  ):
    """Renders latents from views specified by example.

    Args:
      params: ProbNeRF parameters.
      latents: ProbNeRF latent.
      example: Example.
      seed: Random seed.
      render_corruption: Whether to render the corruption model.
      render_parts: Which parts to render (can be 'scene' or 'corruption').

    Returns:
      rendered_rgb: Rendered RGB.
      extra: Extra.
    """
    render_parts = set(render_parts)
    if not render_corruption:
      render_parts.remove('corruption')
    nerf_weights = (
        hypernet.apply(params.hypernet_params, latents)
        if 'scene' in render_parts
        else None
    )
    corruption_params = (
        params.corruption_params if 'corruption' in render_parts else None
    )
    rendered_rgb, extra = mipnerf_render_rays_from_weights(
        nerf_weights, corruption_params, example.rays, seed
    )
    return rendered_rgb, extra

  def log_likelihood_fn(params, latents, inputs):
    """Computes the likelihood.

    Assumes view subsampling. Subsamples rays.

    Args:
      params: ProbNeRF model params.
      latents: Latents.
      inputs: Likelihood inputs. Shape (num_views, ...).

    Returns:
      ll: Evidence lower bound, averaged over the batch. Shape ().
      extra: Extra, averaged over the batch. Shape ().
    """
    subsample_seed, render_seed = jax.random.split(inputs.seed)

    subsampled_rays, subsampled_rgb = subsample_rays(
        inputs.example, num_rays, subsample_seed
    )

    _, extra = render_latents_fn(
        params, latents, Example(rays=subsampled_rays), render_seed
    )
    ll = 0.0
    for extra_l, mipnerf_obs_scale in zip(extra.levels, obs_scales):
      one_ll, ll_extra = rgb_log_likelihood_fn(
          extra_l.rgb,
          subsampled_rgb,
          mipnerf_obs_scale,
      )
      ll += one_ll
    # Grab the ll_extra from the last level, as that corresponds to the
    # final reconstruction.

    total_num_rays = functools.reduce(
        operator.mul, inputs.example.rgb.shape[:-1]
    )
    rgb_mse = ll_extra.rgb_loss
    return ll * total_num_rays / num_rays, LikelihoodExtra(
        psnr=None, rgb_mse=rgb_mse
    )

  def reduce_extra_fn(prior_extra, likelihood_extra, example_axis_name=None):
    rgb_mse = reduce_mean(
        likelihood_extra.rgb_mse, named_axis=example_axis_name
    )
    return prior_extra, likelihood_extra.replace(
        rgb_mse=rgb_mse, psnr=-10 * jnp.log10(rgb_mse)
    )

  return ParameterizedModel(
      init_params_fn=init_params_fn,
      prior_sample_fn=prior_sample_fn,
      prior_log_prob_fn=prior_log_prob_fn,
      render_latents_fn=render_latents_fn,
      log_likelihood_fn=log_likelihood_fn,
      reduce_extra_fn=reduce_extra_fn,
  )


def make_probnerf_guide(
    guide: nn.Module,
    im_height: int = 128,
    im_width: int = 128,
) -> ParameterizedGuide:
  """Creates a ProbNeRF guide [1].

  Assumes view subsampling.

  Args:
    guide: Variational approximation to the posterior.
    im_height: Image height.
    im_width: Image width.

  Returns:
    A ProbNeRF guide.

  #### References

  [1] Hoffman, M. D., Le, T. A., Sountsov, P., Suter, C., Lee, B., Mansinghka,
    V. K., & Saurous, R. A. (2023). ProbNeRF: Uncertainty-Aware Inference of 3D
    Shapes from 2D Images. International Conference on Artificial Intelligence
    and Statistics. https://arxiv.org/abs/2210.17415
  """

  def init_params_fn(seed):
    num_views = 10
    dummy_rgb = jnp.zeros((num_views, im_height, im_width, 3))
    dummy_camera_world_matrix = jnp.zeros((num_views, 4, 4))
    return (
        guide.init(
            seed,
            dummy_rgb,
            dummy_camera_world_matrix,
            jax.random.PRNGKey(0),
        ),
        (),
    )

  def guide_sample_fn(params, example, seed):
    latents, (_, _), log_prob_stop_grad_params = guide.apply(
        params, example.rgb, example.camera_world_matrix, seed
    )
    return latents, GuideSampleExtra(
        log_prob_stop_grad_params=log_prob_stop_grad_params
    )

  def reduce_extra_fn(sample_extra, log_prob_extra, example_axis_name=None):
    del example_axis_name
    return sample_extra, log_prob_extra

  return ParameterizedGuide(
      init_params_fn=init_params_fn,
      guide_sample_fn=guide_sample_fn,
      reduce_extra_fn=reduce_extra_fn,
  )


@saving.register
@struct.dataclass
class EvaluationGuideParams:
  loc: jax.Array
  log_scale: jax.Array


def make_probnerf_evaluation_guide(
    init_loc_fn: Callable[[jax.Array], tuple[jax.Array, Extra]],
) -> ParameterizedGuide:
  """Creates a guide for evaluating a ProbNeRF on a single example.

  Used for maximizing an ELBO which is a lower bound to p(image | model params)
  with respect to model params, marginalizing over ProbNeRF latents.

  Model params don't necessarily correspond to ProbNeRF model params. For
  example, we can fix ProbNeRF model params while defining optimizable model
  params to be parameters of another NeRF representing the corruption.

  Args:
    init_loc_fn: Callable to sample the initial locations.

  Returns:
    A guide q(latents; guide_params) over ProbNeRF scene latents which ignores
    the example. Parameterized as a diagonal multivariate Normal.
  """

  def init_params_fn(seed):
    loc_seed, log_scale_seed = jax.random.split(seed)
    init_loc, _ = init_loc_fn(loc_seed)
    num_latent = init_loc.shape[0]

    # Sample from N(-5, 1) since sampling from N(0, 1) makes RealNVP's
    # log prob nan out.
    init_log_scale = jax.random.normal(log_scale_seed, (num_latent,)) - 5.0

    return EvaluationGuideParams(init_loc, init_log_scale), ()

  def guide_sample_fn(params, example, seed):
    del example
    dist = tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=jnp.exp(params.log_scale)
    )
    dist_stop_grad_params = tfd.MultivariateNormalDiag(
        loc=jax.lax.stop_gradient(params.loc),
        scale_diag=jax.lax.stop_gradient(jnp.exp(params.log_scale)),
    )
    latents = dist.sample(seed=seed)
    log_prob_stop_grad_params = dist_stop_grad_params.log_prob(latents)
    return latents, GuideSampleExtra(
        log_prob_stop_grad_params=log_prob_stop_grad_params
    )

  return ParameterizedGuide(
      init_params_fn=init_params_fn,
      guide_sample_fn=guide_sample_fn,
  )


@saving.register
@struct.dataclass
class MeanFieldGuideParams:
  loc: Any
  isp_scale: Any


def make_mean_field_guide(
    init_loc_fn: Callable[[jax.Array], tuple[Latents, Any]],
    init_scale: float | jax.Array = 1e-2,
) -> ParameterizedGuide:
  """Creates a mean field guide for a general model.

  Args:
    init_loc_fn: Callable to sample the initial locations.
    init_scale: Initial scale multiplier.

  Returns:
    A guide q(latents; guide_params). Parameterized as a diagonal multivariate
    Normal.
  """

  def init_params_fn(seed):
    init_loc, _ = init_loc_fn(seed)
    init_isp_scale = tfp.math.softplus_inverse(init_scale)
    isp_scale = jax.tree.map(
        lambda l: jnp.full(l.shape, init_isp_scale, dtype=l.dtype),
        init_loc,
    )

    return MeanFieldGuideParams(loc=init_loc, isp_scale=isp_scale), ()

  def guide_sample_fn(params, example, seed):
    del example

    leaves, treedef = jax.tree_util.tree_flatten(params.loc)
    num_seeds = len(leaves)
    seeds = jax.tree_util.tree_unflatten(
        treedef, jax.random.split(seed, num_seeds)
    )

    def sample_part(loc, isp_scale, seed):
      return tfd.Normal(loc, jax.nn.softplus(isp_scale)).sample(seed=seed)

    def log_prob_part(latent, loc, isp_scale):
      return (
          tfd.Normal(
              jax.lax.stop_gradient(loc),
              jax.lax.stop_gradient(jax.nn.softplus(isp_scale)),
          )
          .log_prob(latent)
          .sum()
      )

    latents = jax.tree.map(sample_part, params.loc, params.isp_scale, seeds)
    log_prob_stop_grad_params = jax.tree.map(
        log_prob_part, latents, params.loc, params.isp_scale
    )
    log_prob_stop_grad_params = functools.reduce(
        lambda x, y: x + y, jax.tree_util.tree_leaves(log_prob_stop_grad_params)
    )

    return latents, GuideSampleExtra(
        log_prob_stop_grad_params=log_prob_stop_grad_params
    )

  return ParameterizedGuide(
      init_params_fn=init_params_fn,
      guide_sample_fn=guide_sample_fn,
  )


def subsample_rays(
    example: Example,
    num_rays: int,
    seed: jax.Array,
) -> tuple[rendering.Ray, rendering.RGB]:
  """Subsample rays from an example."""
  # Flatten inputs
  assert example.rgb is not None
  rays_shape = example.rgb.shape[:-1]
  rays_ndims = len(rays_shape)
  total_num_rays = functools.reduce(operator.mul, rays_shape)
  flat_rays = jax.tree.map(
      lambda x: x.reshape((-1,) + x.shape[rays_ndims:]),
      example.rays,
  )
  flat_rgb = example.rgb.reshape((-1, 3))

  # Subsample rays
  indices = jax.random.choice(seed, total_num_rays, (num_rays,), False)
  take_fn = lambda x: jax.tree.map(lambda y: y[indices], x)
  subsampled_rays = take_fn(flat_rays)
  subsampled_rgb = take_fn(flat_rgb)
  return subsampled_rays, subsampled_rgb
