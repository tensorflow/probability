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
r"""Diffusion/score matching utilities.

In this file, we are generally dealing with the following joint model:

Q(x) Q(z_0 | x) \prod_{t=1}^T Q(z_t | z_{t - 1})

T could be infinite depending on the model used. The goal is to learn a model
for Q(x) which we only have access to via samples from it. We do this by
learning P(z_{t - 1} | z_t; t), which we parameterize using a trainable
`denoise_fn(z_t, f(t))` with some function `f` (often log signal-to-noise
ratio).
"""

import enum
from typing import Any, Callable

from flax import struct
import jax
import jax.numpy as jnp
from discussion.robust_inverse_graphics import saving
from fun_mc import using_jax as fun_mc


__all__ = [
    "linear_log_snr",
    "variance_preserving_forward_process",
    "vdm_diffusion_loss",
    "vdm_sample",
    "VDMDiffusionLossExtra",
    "VDMSampleExtra",
    "DenoiseOutputType",
]


Extra = Any
LogSnrFn = Callable[[jnp.ndarray], jnp.ndarray]
DenoiseFn = Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, Extra]]


class DenoiseOutputType(enum.Enum):
  """How to interpret the output of `denoise_fn`.

  NOISE: The output is the predicted noise.
  ANGULAR_VELOCITY: The output is the angular velocity, defined as:
    `alpha_t * noise - sigma_t * x`.
  DIRECT: The output is the denoised value.
  """

  NOISE = "noise"
  ANGULAR_VELOCITY = "angular_velocity"
  DIRECT = "direct"


def variance_preserving_forward_process(
    z_0: jnp.ndarray, noise: jnp.ndarray, log_snr_t: jnp.ndarray
) -> jnp.ndarray:
  """Variance preserving forward process.

  This produces a sample from Q(z_t | z_0) given the desired level of noise and
  randomness.

  Args:
    z_0: Un-noised inputs.
    noise: Noise.
    log_snr_t: Log signal-to-noise ratio at time t.

  Returns:
    Value of z_t.
  """
  var_t = jax.nn.sigmoid(-log_snr_t)
  alpha_t = jnp.sqrt(jax.nn.sigmoid(log_snr_t))  # sqrt(1 - var_t)
  return alpha_t * z_0 + jnp.sqrt(var_t) * noise


@saving.register
@struct.dataclass
class VDMDiffusionLossExtra:
  """Extra outputs from `vdm_diffusion_loss`.

  Attributes:
    noise: The added noise.
    recon_noise: The reconstructed noise (only set if `denoise_output` is NOISE.
    target: Target value for the loss to reconstruct.
    recon: Output of `denoise_fn`.
    extra: Extra outputs from `denoise_fn`.
  """

  noise: jnp.ndarray
  recon_noise: jnp.ndarray | None
  target: jnp.ndarray
  recon: jnp.ndarray
  extra: Extra


def vdm_diffusion_loss(
    t: jnp.ndarray,
    num_steps: int | None,
    x: jnp.ndarray,
    log_snr_fn: LogSnrFn,
    denoise_fn: DenoiseFn,
    seed: jax.Array,
    denoise_output: DenoiseOutputType = DenoiseOutputType.NOISE,
) -> tuple[jnp.ndarray, VDMDiffusionLossExtra]:
  r"""The diffusion loss of the variational diffusion model (VDM).

  This uses the parameterization from [1]. The typical procedure minimizes the
  expectation of this function, averaging across examples (`z_0`) and times
  (sampled uniformly from [0, 1]).

  When `denoise_output` is NOISE, and when the loss is minimized,
  `denoise_fn(z_t, log_snr_t) \propto -grad log Q(z_t; log_snr_t)` where `z_t`
  is sampled from the forward process (`variance_preserving_forward_process`)
  and `Q(.; log_snr_t)` is the marginal density of `z_t`.

  Args:
    t: Time in [0, 1]
    num_steps: If None, use continuous time parameterization. Otherwise,
      discretize `t` to this many bins.
    x: Un-noised inputs.
    log_snr_fn: Takes in time in [0, 1] and returns the log signal-to-noise
      ratio.
    denoise_fn: Function that denoises `z_t` given the `log_snr_t`. Its output
      is interpreted based on the value of `denoise_output`.
    seed: Random seed.
    denoise_output: How to interpret the output of `denoise_fn`.

  Returns:
    A tuple of the loss and `VDMDiffusionLossExtra` extra outputs.

  #### References

  [1] Kingma, D. P., Salimans, T., Poole, B., & Ho, J. (2021). Variational
      Diffusion Models. In arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2107.00630
  """

  if num_steps is not None:
    t = jnp.ceil(t * num_steps) / num_steps

  log_snr_t = log_snr_fn(t)
  noise = jax.random.normal(seed, x.shape)
  z_t = variance_preserving_forward_process(x, noise, log_snr_t)

  recon, extra = denoise_fn(z_t, log_snr_t)

  match denoise_output:
    case DenoiseOutputType.NOISE:
      target = noise
      recon_noise = recon
      sq_error = 0.5 * jnp.square(target - recon).sum()
      if num_steps is None:
        log_snr_t_grad = jax.grad(log_snr_fn)(t)
        loss = -log_snr_t_grad * sq_error
      else:
        s = t - (1.0 / num_steps)
        log_snr_s = log_snr_fn(s)
        loss = num_steps * jnp.expm1(log_snr_s - log_snr_t) * sq_error
    case DenoiseOutputType.ANGULAR_VELOCITY:
      # Plug in x_hat = alpha_t * z_t - sigma_t * v into equation (13) or (15)
      # and simplify to get the loss being (SNR(s) - SNR(t)) sigma_t**2 * MSE
      # for discrete case and SNR'(t) * sigma_t**2 * MSE for the continuous
      # case.
      recon_noise = None
      var_t = jax.nn.sigmoid(-log_snr_t)
      sigma_t = jnp.sqrt(var_t)
      alpha_t_2 = jax.nn.sigmoid(log_snr_t)
      alpha_t = jnp.sqrt(alpha_t_2)
      v = alpha_t * noise - sigma_t * x

      target = v
      sq_error = 0.5 * jnp.square(target - recon).sum()
      if num_steps is None:
        log_snr_t_grad = jax.grad(log_snr_fn)(t)
        loss = -alpha_t_2 * log_snr_t_grad * sq_error
      else:
        s = t - (1.0 / num_steps)
        log_snr_s = log_snr_fn(s)
        loss = (
            num_steps * jnp.expm1(log_snr_s - log_snr_t) * alpha_t_2 * sq_error
        )
    case DenoiseOutputType.DIRECT:
      recon_noise = None
      target = x
      sq_error = 0.5 * jnp.square(target - recon).sum()
      if num_steps is None:
        snr_t_grad = jax.grad(lambda t: jnp.exp(log_snr_fn(t)))(t)
        loss = -snr_t_grad * sq_error
      else:
        s = t - (1.0 / num_steps)
        snr_t = jnp.exp(log_snr_t)
        # TODO(siege): Not sure this is more stable than doing snr_s - snr_t
        # directly.
        log_snr_s = log_snr_fn(s)
        loss = num_steps * snr_t * jnp.expm1(log_snr_s - log_snr_t) * sq_error
    case _:
      raise ValueError(f"Unknown denoise_output: {denoise_output}")

  return loss, VDMDiffusionLossExtra(
      noise=noise,
      recon_noise=recon_noise,
      target=target,
      recon=recon,
      extra=extra,
  )


def _vdm_sample_step(
    z_t: jnp.ndarray,
    step: jnp.ndarray,
    num_steps: int,
    log_snr_fn: LogSnrFn,
    denoise_fn: DenoiseFn,
    seed: jax.Array,
    denoise_output: DenoiseOutputType,
    t_start: jnp.ndarray,
) -> tuple[jnp.ndarray, Extra]:
  """One step of the sampling process."""
  t = t_start * (step / num_steps)
  s = t_start * ((step - 1) / num_steps)

  log_snr_t = log_snr_fn(t)
  log_snr_s = log_snr_fn(s)
  recon, extra = denoise_fn(z_t, log_snr_t)

  zeta = jax.random.normal(seed, z_t.shape)

  alpha_s_2 = jax.nn.sigmoid(log_snr_s)
  alpha_s = jnp.sqrt(alpha_s_2)
  alpha_t_2 = jax.nn.sigmoid(log_snr_t)
  alpha_t = jnp.sqrt(alpha_t_2)
  var_t_s_div_var_t = -jnp.expm1(log_snr_t - log_snr_s)
  var_s = jax.nn.sigmoid(-log_snr_s)
  var_t = jax.nn.sigmoid(-log_snr_t)
  sigma_t = jnp.sqrt(var_t)

  match denoise_output:
    case DenoiseOutputType.NOISE:
      recon_noise = recon
      mu = jnp.sqrt(alpha_s_2 / alpha_t_2) * (
          z_t - sigma_t * var_t_s_div_var_t * recon_noise
      )
    case DenoiseOutputType.ANGULAR_VELOCITY:
      # We use the expression for q(z_s | z_t, x) directly with x_hat
      # substituted for x.
      # TODO(siege): Try simplifying this further for better numerics.
      x_hat = alpha_t * z_t - sigma_t * recon
      alpha_t_s = alpha_t / alpha_s

      mu = alpha_t_s * var_s / var_t * z_t + alpha_s * var_t_s_div_var_t * x_hat
    case DenoiseOutputType.DIRECT:
      x_hat = recon
      alpha_t_s = alpha_t / alpha_s

      mu = alpha_t_s * var_s / var_t * z_t + alpha_s * var_t_s_div_var_t * x_hat
    case _:
      raise ValueError(f"Unknown denoise_output: {denoise_output}")
  sigma = jnp.sqrt(var_t_s_div_var_t * var_s)
  z_s = mu + sigma * zeta
  return z_s, extra


@saving.register
@struct.dataclass
class VDMSampleExtra:
  """Extra outputs from `vdm_sample`.

  Attributes:
    z_s: A trace of samples.
  """

  z_s: jnp.ndarray | None


def vdm_sample(
    z_t: jnp.ndarray,
    num_steps: int,
    log_snr_fn: LogSnrFn,
    denoise_fn: DenoiseFn,
    seed: jax.Array,
    trace_z_s: bool = False,
    denoise_output: DenoiseOutputType = DenoiseOutputType.NOISE,
    t_start: jnp.ndarray | float = 1.0,
) -> tuple[jnp.ndarray, VDMSampleExtra]:
  """Generates a sample from the variational diffusion model (VDM).

  This uses the sampler from [1]. See `vdm_diffusion_loss` for the requirements
  on `denoise_fn`.

  Args:
    z_t: The initial noised sample. Should have the same distribution as
      `variance_preserving_forward_process(x, noise, log_snr_fn(t_start))`.
    num_steps: Number of steps to take. The more steps taken, then more accurate
      the sample. 1000 is a common value.
    log_snr_fn: Takes in time in [0, 1] and returns the log signal-to-noise
      ratio.
    denoise_fn: Function that denoises `z_t` given the `log_snr_t`. Its output
      is interpreted based on the value of `denoise_output`.
    seed: Random seed.
    trace_z_s: Whether to trace intermediate samples.
    denoise_output: How to interpret the output of `denoise_fn`.
    t_start: The value of t in z_t. Typically this is 1, signifying that z_t is
      a sample from a standard normal.

  Returns:
    A tuple of the sample and `VDMSampleExtra` extra outputs.


  #### References

  [1] Kingma, D. P., Salimans, T., Poole, B., & Ho, J. (2021). Variational
      Diffusion Models. In arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2107.00630
  """

  def body(z_t, step, seed):
    sample_seed, seed = jax.random.split(seed)
    z_s, _ = _vdm_sample_step(
        z_t=z_t,
        step=step,
        num_steps=num_steps,
        log_snr_fn=log_snr_fn,
        denoise_fn=denoise_fn,
        seed=sample_seed,
        denoise_output=denoise_output,
        t_start=t_start,
    )
    if trace_z_s:
      trace = {"z_s": z_s}
    else:
      trace = {}
    return (z_s, step - 1, seed), trace

  (z_0, _, _), trace = fun_mc.trace((z_t, num_steps, seed), body, num_steps)

  return z_0, VDMSampleExtra(z_s=trace.get("z_s"))


def linear_log_snr(
    t: jnp.ndarray,
    log_snr_start: jax.typing.ArrayLike = 6.0,
    log_snr_end: jax.typing.ArrayLike = -6.0,
) -> jnp.ndarray:
  """Linear log signal-to-noise ratio function."""
  return log_snr_start + (log_snr_end - log_snr_start) * t  # pytype: disable=bad-return-type  # numpy-scalars
