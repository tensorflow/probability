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

from absl.testing import parameterized
import jax
import jax.numpy as jnp

from discussion.robust_inverse_graphics import diffusion
from discussion.robust_inverse_graphics.util import test_util


class DiffusionTest(test_util.TestCase):

  def test_variance_preserving_forward_process(self):
    seeds = jax.random.split(self.test_seed())

    x = 1.0 + 2 * jax.random.normal(seeds[0], [10000])
    noise = jax.random.normal(seeds[1], [10000])

    z_small = diffusion.variance_preserving_forward_process(
        x,
        noise,
        log_snr_t=jnp.array(-20.0),
    )
    z_large = diffusion.variance_preserving_forward_process(
        x,
        noise,
        log_snr_t=jnp.array(20.0),
    )

    # At the SNR extremes, the outputs should either be all noise or all inputs.
    self.assertAllClose(z_small.mean(), 0.0, atol=1e-1)
    self.assertAllClose(z_small.std(), 1.0, rtol=1e-1)
    self.assertAllClose(z_large.mean(), 1.0, atol=1e-1)
    self.assertAllClose(z_large.std(), 2.0, rtol=1e-1)

  @parameterized.named_parameters(
      ("_continuous_noise", None, diffusion.DenoiseOutputType.NOISE),
      ("_discretized_noise", 1000, diffusion.DenoiseOutputType.NOISE),
      ("_continuous_v", None, diffusion.DenoiseOutputType.ANGULAR_VELOCITY),
      ("_discretized_v", 1000, diffusion.DenoiseOutputType.ANGULAR_VELOCITY),
      ("_continuous_direct", None, diffusion.DenoiseOutputType.DIRECT),
      ("_discretized_direct", 1000, diffusion.DenoiseOutputType.DIRECT),
  )
  def test_vdm_diffusion_loss(self, num_steps, denoise_output):
    # P(x) is a gaussian with these parameters.
    loc_0 = 1.0
    scale_0 = 3.0

    def denoise_fn(w, z_t, log_snr_t):
      # Since P(x) is a gaussian, we know the solution in closed form. See e.g.
      # Appendix L from Kingma et al. 2021 and also the form of the forward
      # process.
      var_t = jax.nn.sigmoid(-log_snr_t)
      sigma_t = jnp.sqrt(var_t)
      alpha_t = jnp.sqrt(jax.nn.sigmoid(log_snr_t))  # sqrt(1 - var_t)

      loc = alpha_t * loc_0
      var = (alpha_t * scale_0) ** 2 + var_t
      score = (loc - z_t) / var
      recon_noise = -score * jnp.sqrt(var_t) + w
      match denoise_output:
        case diffusion.DenoiseOutputType.NOISE:
          return recon_noise, ()
        case diffusion.DenoiseOutputType.ANGULAR_VELOCITY:
          predicted_x = (z_t - sigma_t * recon_noise) / alpha_t
          return (alpha_t * z_t - predicted_x) / sigma_t, ()
        case diffusion.DenoiseOutputType.DIRECT:
          predicted_x = (z_t - sigma_t * recon_noise) / alpha_t
          return predicted_x, ()

    def get_loss(w, x, t, seed):

      return diffusion.vdm_diffusion_loss(
          t=t,
          num_steps=num_steps,
          x=x,
          log_snr_fn=diffusion.linear_log_snr,
          denoise_fn=functools.partial(denoise_fn, w),
          seed=seed,
          denoise_output=denoise_output,
      )[0]

    x_seed, seed = jax.random.split(self.test_seed(), 2)

    x = jax.random.normal(x_seed, [100000]) * scale_0 + loc_0
    t = jnp.linspace(0., 1., 100000)
    seeds = jax.random.split(seed, 100000)

    grad = jax.grad(
        lambda w: jax.vmap(lambda x, t, seed: get_loss(w, x, t, seed))(  # pylint: disable=g-long-lambda
            x, t, seeds
        ).mean()
    )(jnp.zeros([]))

    self.assertAllClose(grad, 0., atol=2e-1)

  @parameterized.named_parameters(
      ("_noise", diffusion.DenoiseOutputType.NOISE),
      ("_v", diffusion.DenoiseOutputType.ANGULAR_VELOCITY),
      ("_direct", diffusion.DenoiseOutputType.DIRECT),
  )
  def test_vdm_sample(self, denoise_output):
    # P(x) is a gaussian with these parameters.
    loc_0 = 1.0
    scale_0 = 3.0

    def denoise_fn(z_t, log_snr_t):
      # Since P(x) is a gaussian, we know the solution in closed form. See e.g.
      # Appendix L from Kingma et al. 2021 and also the form of the forward
      # process.
      var_t = jax.nn.sigmoid(-log_snr_t)
      sigma_t = jnp.sqrt(var_t)
      alpha_t = jnp.sqrt(jax.nn.sigmoid(log_snr_t))  # sqrt(1 - var_t)

      loc = alpha_t * loc_0
      var = (alpha_t * scale_0) ** 2 + var_t
      score = (loc - z_t) / var
      recon_noise = -score * jnp.sqrt(var_t)
      match denoise_output:
        case diffusion.DenoiseOutputType.NOISE:
          return recon_noise, ()
        case diffusion.DenoiseOutputType.ANGULAR_VELOCITY:
          predicted_x = (z_t - sigma_t * recon_noise) / alpha_t
          return (alpha_t * z_t - predicted_x) / sigma_t, ()
        case diffusion.DenoiseOutputType.DIRECT:
          predicted_x = (z_t - sigma_t * recon_noise) / alpha_t
          return predicted_x, ()

    init_seed, sample_seed = jax.random.split(self.test_seed())
    z_t = jax.random.normal(init_seed, [10000])

    sample, _ = diffusion.vdm_sample(
        z_t=z_t,
        num_steps=1000,
        log_snr_fn=diffusion.linear_log_snr,
        denoise_fn=denoise_fn,
        seed=sample_seed,
        denoise_output=denoise_output,
    )
    self.assertAllClose(sample.mean(), loc_0, atol=2e-1)
    self.assertAllClose(sample.std(), scale_0, rtol=2e-1)


if __name__ == "__main__":
  test_util.main()
