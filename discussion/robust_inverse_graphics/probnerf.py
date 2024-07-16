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
"""ProbNeRF."""

from typing import Any

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import tensorflow_probability.substrates.jax as tfp


tfd = tfp.distributions


__all__ = [
    'TwoPartNerf',
    'DecoderHypernet',
    'RealNVPStack',
    'Guide',
]


# Taken from
# <https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/model_utils.py>
def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
  """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

  Instead of computing [sin(x), cos(x)], we use the trig identity
  cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    min_deg: int, the minimum (inclusive) degree of the encoding.
    max_deg: int, the maximum (exclusive) degree of the encoding.
    legacy_posenc_order: bool, keep the same ordering as the original tf code.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if min_deg == max_deg:
    return x
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])
  if legacy_posenc_order:
    xb = x[..., None, :] * scales[:, None]
    four_feat = jnp.reshape(
        jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], -2)),
        list(x.shape[:-1]) + [-1],
    )
  else:
    xb = jnp.reshape(
        (x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1]
    )
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  return jnp.concatenate([x] + [four_feat], axis=-1)


class DensityNerf(nn.Module):
  """Density NeRF."""

  scale: float = 1.0

  min_degree: int = 0
  max_degree: int = 10

  @nn.compact
  def __call__(self, position):
    pos_encoding = posenc(
        position / self.scale * jnp.pi, self.min_degree, self.max_degree
    )
    x = pos_encoding
    x = nn.Dense(64)(x)
    x = nn.relu(x)
    x = nn.Dense(64)(x)
    x = nn.relu(x)
    density = nn.softplus(nn.Dense(1)(x))[0]
    return density, x


class AppearanceNerf(nn.Module):
  """Appearance NeRF."""

  scale: float = 1.0

  min_degree: int = 0
  max_degree: int = 10

  @nn.compact
  def __call__(self, position, viewdir, density):
    pos_encoding = posenc(
        position / self.scale * jnp.pi, self.min_degree, self.max_degree
    )
    view_encoding = posenc(viewdir * jnp.pi, self.min_degree, self.max_degree)
    x = jnp.concatenate([pos_encoding, view_encoding, density], -1)
    x = nn.Dense(64)(x)
    x = nn.relu(x)
    x = nn.Dense(64)(x)
    x = nn.relu(x)
    rgb = nn.sigmoid(nn.Dense(3)(x))
    return rgb


class TwoPartNerf(nn.Module):
  """Combines the density and appearance NeRF."""

  grid_size: int = 128
  scale: float = 1.0

  min_degree: int = 0
  max_degree: int = 10
  appearance_min_degree: int = 0
  appearance_max_degree: int = 10

  @nn.compact
  def __call__(self, ray_sample):
    density, _ = DensityNerf(self.scale, self.min_degree, self.max_degree)(
        ray_sample.position
    )
    rgb = AppearanceNerf(
        self.scale, self.appearance_min_degree, self.appearance_max_degree
    )(ray_sample.position, ray_sample.viewdir, density[jnp.newaxis])

    # For foam rendering, convert density to alpha assuming constant density.
    return (-jnp.expm1(-density / self.grid_size), rgb), ()


def _map_to_params(flat_params, template):
  leaves, treedef = jax.tree_util.tree_flatten(template)
  new_leaves = []
  param_index = 0
  for p in leaves:
    num_params = np.prod(p.shape)
    new_leaves.append(
        flat_params[param_index : param_index + num_params].reshape(p.shape)
    )
    param_index += num_params
  return jax.tree_util.tree_unflatten(treedef, new_leaves)


def _map_hidden_to_params(h, template):
  leaves = jax.tree_util.tree_leaves(template)
  total_num_params = sum([np.prod(p.shape) for p in leaves])
  flat_params = nn.Dense(total_num_params)(h)
  return _map_to_params(flat_params, template)


class Hypernet(nn.Module):
  """Hypernet mapping from latent to NeRF weights."""

  # The parameter shapes for the network the hypernetwork makes.
  output_template: Any
  width: int = 512
  depth: int = 2
  num_outputs: int = 1

  @nn.compact
  def __call__(self, latent):
    """Maps from latent vector to parameters to a neural net."""

    for _ in range(self.depth):
      latent = nn.relu(nn.Dense(self.width)(latent))
    outputs = tuple([
        _map_hidden_to_params(latent, self.output_template)
        for _ in range(self.num_outputs)
    ])
    if self.num_outputs == 1:
      return outputs[0]
    else:
      return outputs


class DecoderHypernet(nn.Module):
  """Decoder hypernet."""

  # The parameter shapes for the decoder.
  decoder_template: Any

  density_width: int = 512
  density_depth: int = 2

  appearance_width: int = 512
  appearance_depth: int = 2

  @nn.compact
  def __call__(self, latent):
    """Maps from latent vector to parameters to `TwoPartNerf`."""

    # Rescale latent elementwise to make it easier to match a N(0, I) prior.
    latent = latent * jnp.exp(
        self.param('latent_scale', nn.initializers.zeros, latent.shape[-1])
    )
    density_latent, appearance_latent = latent.reshape([2, -1])

    density_params, _ = Hypernet(
        self.decoder_template['params']['DensityNerf_0'],
        self.density_width,
        self.density_depth,
        2,
    )(density_latent)

    appearance_latent = jnp.concatenate([density_latent, appearance_latent], -1)
    appearance_params, _ = Hypernet(
        self.decoder_template['params']['AppearanceNerf_0'],
        self.appearance_width,
        self.appearance_depth,
        2,
    )(appearance_latent)

    params = self.decoder_template.unfreeze()
    params['params']['DensityNerf_0'] = density_params
    params['params']['AppearanceNerf_0'] = appearance_params

    return nn.FrozenDict(params)


def _split(x):
  flat_split_x = jnp.transpose(x.reshape([-1, 2, x.shape[-1] // 2]), [1, 0, 2])
  split_x = flat_split_x.reshape([2, *x.shape[:-1], x.shape[-1] // 2])
  x1, x2 = split_x
  return x1, x2


class RealNVPLayer(nn.Module):
  """One RealNVP layer."""

  hidden_width: int
  ndims: int

  def setup(self):
    kernel_init = nn.initializers.normal(0.01)
    self.hid1 = nn.Dense(self.hidden_width, kernel_init=kernel_init)
    self.hid2 = nn.Dense(self.hidden_width, kernel_init=kernel_init)
    self.shift_and_scale1 = nn.Dense(self.ndims, kernel_init=kernel_init)
    self.shift_and_scale2 = nn.Dense(self.ndims, kernel_init=kernel_init)

  @nn.compact
  def __call__(self, x, forward=True):
    permutation = jax.random.permutation(jax.random.PRNGKey(0), x.shape[-1])
    if not forward:
      permutation = jnp.argsort(permutation)
      x = x[..., permutation]

    x1, x2 = _split(x)
    ldj = 0.0
    if forward:
      h = nn.relu(self.hid1(x1))
      shift, log_scale = _split(self.shift_and_scale1(h))
      x2 = x2 * jnp.exp(log_scale)
      x2 = x2 + shift
      ldj += log_scale.sum(-1)

      h = nn.relu(self.hid2(x2))
      shift, log_scale = _split(self.shift_and_scale2(h))
      x1 = x1 * jnp.exp(log_scale)
      x1 = x1 + shift
      ldj += log_scale.sum(-1)

      x = jnp.concatenate([x1, x2], -1)
      x = x[..., permutation]
    else:
      h = nn.relu(self.hid2(x2))
      shift, log_scale = _split(self.shift_and_scale2(h))
      x1 = x1 - shift
      x1 = x1 * jnp.exp(-log_scale)
      ldj -= log_scale.sum(-1)

      h = nn.relu(self.hid1(x1))
      shift, log_scale = _split(self.shift_and_scale1(h))
      x2 = x2 - shift
      x2 = x2 * jnp.exp(-log_scale)
      ldj -= log_scale.sum(-1)

      x = jnp.concatenate([x1, x2], -1)
    return x, ldj


class RealNVPStack(nn.Module):
  """Stack of RealNVPs."""

  ndims: int
  hidden_width: int = 512
  depth: int = 2

  def setup(self):
    self.layers = [
        RealNVPLayer(self.hidden_width, self.ndims) for _ in range(self.depth)
    ]

  @nn.compact
  def __call__(self, x, forward=True):
    ldj = 0.0
    layers = self.layers if forward else self.layers[::-1]
    for layer in layers:
      x, new_ldj = layer(x, forward=forward)
      ldj += new_ldj
    return x, ldj


class Guide(nn.Module):
  """Guide / recognition model."""

  latent_dim: int
  decoder_template: Any

  @nn.compact
  def __call__(self, images, cameras, seed):
    # Encode images
    h = images
    h = nn.Conv(16, (3, 3), (2, 2))(h)
    h = nn.relu(h)
    h = nn.Conv(32, (3, 3), (2, 2))(h)
    h = nn.relu(h)
    h = nn.Conv(64, (3, 3), (2, 2))(h)
    h = nn.relu(h)
    h = nn.Conv(128, (3, 3), (2, 2))(h)
    h = nn.relu(h)
    h = nn.Conv(256, (3, 3), (2, 2))(h)
    h = nn.avg_pool(h, (2, 2), (2, 2))
    image_h = h.reshape([h.shape[0], -1])

    h = cameras.reshape([cameras.shape[0], -1])
    h = nn.Dense(512)(h)
    h = nn.relu(h)
    camera_h = nn.Dense(512)(h)

    h = jnp.concatenate([image_h, camera_h], -1)

    # Probabilistic aggregation.
    def aggregate(locs, log_precisions):
      log_precisions = jnp.minimum(10.0, log_precisions)
      precisions = jnp.exp(log_precisions)
      loc = (locs * precisions).sum(0) / precisions.sum(0)
      log_scale = -0.5 * jsp.special.logsumexp(log_precisions, 0)
      return loc, log_scale

    z_latent_params = nn.Dense(2 * self.latent_dim)(h).reshape(
        [-1, 2, self.latent_dim]
    )
    z_locs, z_log_precisions = (
        z_latent_params[..., 0, :],
        z_latent_params[..., 1, :],
    )

    # Add potential for the realnvp prior.
    prior_loc = self.param('prior_loc', nn.initializers.zeros, self.latent_dim)
    prior_log_precision = self.param(
        'prior_log_precision', nn.initializers.zeros, self.latent_dim
    )
    z_locs = jnp.concatenate([z_locs, 0 * z_locs[:1] + prior_loc], 0)
    z_log_precisions = jnp.concatenate(
        [z_log_precisions, 0 * z_locs[:1] + prior_log_precision], 0
    )

    z_locs, z_log_scales = aggregate(z_locs, z_log_precisions)

    z = tfd.Normal(z_locs, jnp.exp(z_log_scales)).sample((), seed)
    z_log_prob = (
        tfd.Normal(
            lax.stop_gradient(z_locs), lax.stop_gradient(jnp.exp(z_log_scales))
        )
        .log_prob(z)
        .sum()
    )

    h = jnp.concatenate([h, z + jnp.zeros([h.shape[0], 1])], -1)

    return (z, (z_locs, z_log_scales), z_log_prob)
