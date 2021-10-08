# Lint as: python3
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
"""Plasma spectroscopy model."""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from inference_gym.internal import data
from inference_gym.targets import bayesian_model
from inference_gym.targets import model

tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    'PlasmaSpectroscopy',
    'SyntheticPlasmaSpectroscopy',
    'SyntheticPlasmaSpectroscopyWithBump',
]

Root = tfd.JointDistributionCoroutine.Root


class PlasmaSpectroscopy(bayesian_model.BayesianModel):
  """Plasma spectroscopy model.

  This implements the fusion plasma spectroscopy problem described in [1]. The
  goal of the model is to reconstruct the plasma configuration in a linear
  fusion reactor via spectroscopy. The plasma blob is instrumented by an array
  of regularly spaced spectrometers arranged in the transverse plane to the
  reactor. Each spectrometer can measure the emission spectrum integrated along
  the line of sight of the sensor. The sensor array can resolve shifts
  perpendicular to the their lines of sight, but cannot measure shifts along
  them. The plasma is parameterized via amplitude, temperature and velocity ,
  which are assumed to be radially symmetric about the plasma blob center and
  are spatially correlated.

  The overall model is as follows:

  ```none
  for r in range(num_shells):
    amplitude[r] ~ Normal(0, 1)
    temperature[r] ~ Normal(0, 1)
    velocity[r] ~ Normal(0, 1)
  shift ~ Normal(0, 1)

  mean_emission = forward_model(amplitude, temperature, velocity, shift)

  measurements[wavelength, sensor] ~ Normal(mean_emission,
    (absolute_noise_scale**2 + relative_noise_scale**2 * mean_emission)**0.5)
  ```

  where the `forward_model` describes the mechanism of the spectrometers. Note
  that the latent parameters are unconstrained and the correlations between them
  are baked into the forward model.

  This model is notable for poor conditioning and presence of multiple low
  probability modes, requiring careful initialization for MCMC to progress.

  #### References

  [1]: Langmore I, Dikovsky M, Geraedts S, Norgaard P, von Behren R. Hamiltonian
       Monte Carlo in Inverse Problems; Ill-Conditioning and Multi-Modality.
       2021. http://arxiv.org/abs/2103.07515
  """

  def __init__(
      self,
      measurements,
      wavelengths,
      center_wavelength,
      num_shells=16,
      outer_shell_radius=1.,
      sensor_span=1.5,
      num_integration_points=64,
      prior_diag_noise_variance=0.1,
      prior_length_scale=0.05,
      amplitude_scale=2.,
      temperature_scale=0.2,
      velocity_scale=0.5,
      absolute_noise_scale=0.5,
      relative_noise_scale=0.05,
      use_bump_function=False,
      name='plasma_spectroscopy',
      pretty_name='Plasma Spectroscopy',
  ):
    """Construct the PlasmaSpectroscopy model.

    Args:
      measurements: Float `Tensor` with shape [num_wavelengths, num_sensors].
        The spectrometer measurements.
      wavelengths: Float `Tensor` with shape [num_wavelengths]. Wavelengths
        measured by the spectrometers.
      center_wavelength: Float `Tensor` scalar. The center wavelength of the
        target emission line.
      num_shells: Python integer. Number of radial shells to model.
      outer_shell_radius: Python float. Outermost shell radius.
      sensor_span: Python float. Half-span of the sensors (i.e. the sensors are
        regularly placed on the interval `[-sensor_span, sensor_span]`.
      num_integration_points: Python integer. How many points to use to
        discretize the integral used to compute the integrated emission spectra.
      prior_diag_noise_variance: Float `Tensor` scalar. Diagonal variance of the
        latent variables.
      prior_length_scale: Float `Tensor` scalar. Length scale of the radial
        correlations between the latent variables.
      amplitude_scale: Float `Tensor` scalar. Scale factor of the constrained
        amplitude.
      temperature_scale: Float `Tensor` scalar. Scale factor of the constrained
        temperature.
      velocity_scale: Float `Tensor` scalar. Scale factor of the constrained
        velocity.
      absolute_noise_scale: Float `Tensor` scalar. Absolute noise scale of the
        observation noise.
      relative_noise_scale: Float `Tensor` scalar. Absolute noise scale of the
        observation noise.
      use_bump_function: Python `bool`. If True, use a bump function to smoothly
        decay the plasma blob to 0 at the edges.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.

    Raises:
      ValueError: If `measurements` and `wavelengths` have inconsistent shape.
    """
    if measurements.shape[0] != wavelengths.shape[0]:
      raise ValueError('Number of measurement rows does not match the number '
                       f'wavelengths {measurements.shape[0]} != '
                       f'{len(wavelengths)}')

    with tf.name_scope(name):

      self._measurements = measurements
      self._wavelengths = wavelengths
      self._center_wavelength = center_wavelength
      self._num_shells = num_shells
      self._outer_shell_radius = outer_shell_radius
      self._sensor_span = sensor_span
      self._num_integration_points = num_integration_points
      self._prior_diag_noise_variance = prior_diag_noise_variance
      self._prior_length_scale = prior_length_scale
      self._amplitude_scale = amplitude_scale
      self._temperature_scale = temperature_scale
      self._velocity_scale = velocity_scale
      self._absolute_noise_scale = absolute_noise_scale
      self._relative_noise_scale = relative_noise_scale
      self._use_bump_function = use_bump_function

      @tfd.JointDistributionCoroutine
      def prior():
        yield Root(tfd.Sample(tfd.Normal(0., 1.), num_shells, name='amplitude'))
        yield Root(
            tfd.Sample(tfd.Normal(0., 1.), num_shells, name='temperature'))
        yield Root(tfd.Sample(tfd.Normal(0., 1.), num_shells, name='velocity'))
        yield Root(tfd.Normal(0., 1., name='shift'))

      self._prior_dist = prior

      def observation_noise_fn(mean_measurement):
        """Creates the observation noise distribution."""
        return tfd.Independent(
            tfd.Normal(
                mean_measurement,
                tf.sqrt(absolute_noise_scale**2 +
                        relative_noise_scale**2 * mean_measurement**2)), 2)

      self._observation_noise_fn = observation_noise_fn

      def log_likelihood_fn(sample):
        """The log_likelihood function."""
        _, mean_measurement = self.forward_model(sample)
        return observation_noise_fn(mean_measurement).log_prob(measurements)

      self._log_likelihood_fn = log_likelihood_fn

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=lambda params: params,
                  pretty_name='Identity',
                  dtype=self._prior_dist.dtype,
              )
      }

    super(PlasmaSpectroscopy, self).__init__(
        default_event_space_bijector=type(self._prior_dist.dtype)(
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
        ),
        event_shape=self._prior_dist.event_shape,
        dtype=self._prior_dist.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  @property
  def num_sensors(self):
    return self._measurements.shape[1]

  @property
  def measurements(self):
    return self._measurements

  @property
  def wavelengths(self):
    return self._wavelengths

  @property
  def center_wavelength(self):
    return self._center_wavelength

  @property
  def num_shells(self):
    return self._num_shells

  @property
  def outer_shell_radius(self):
    return self._outer_shell_radius

  @property
  def sensor_span(self):
    return self._sensor_span

  @property
  def num_integration_points(self):
    return self._num_integration_points

  @property
  def prior_diag_noise_variance(self):
    return self._prior_diag_noise_variance

  @property
  def prior_length_scale(self):
    return self._prior_length_scale

  @property
  def amplitude_scale(self):
    return self._amplitude_scale

  @property
  def temperature_scale(self):
    return self._temperature_scale

  @property
  def velocity_scale(self):
    return self._velocity_scale

  @property
  def absolute_noise_scale(self):
    return self._absolute_noise_scale

  @property
  def relative_noise_scale(self):
    return self._relative_noise_scale

  @property
  def use_bump_function(self):
    return self._use_bump_function

  def forward_model(self, sample):
    """The forward model.

    Args:
      sample: A sample from the model.

    Returns:
      mesh_emissivity: Float `Tensor` with shape [num_wavelengths, num_sensors,
        num_integration_points]. The spatial spectral emissivity. The last two
        dimensions describe locations perpendicular and parallel to the
        spectrometer lines of sight, respectively. Those dimensions span
        [-sensor_span, sensor_span] and [-outer_shell_radius,
        outer_shell_radius] respectively.
      mean_measurement: Float `Tensor` with shape [num_wavelengths,
        num_sensors]. The measurement means.
    """
    wavelengths = tf.convert_to_tensor(self.wavelengths, tf.float32)
    center_wavelength = tf.convert_to_tensor(self.center_wavelength, tf.float32)

    shell_radii = tf.linspace(0., self.outer_shell_radius, self.num_shells)

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        length_scale=self.prior_length_scale)
    prior_cov = kernel.matrix(shell_radii[..., tf.newaxis],
                              shell_radii[..., tf.newaxis])
    prior_cov = (prior_cov + self.prior_diag_noise_variance * tf.eye(
        int(prior_cov.shape[-1]))) / (1 + self.prior_diag_noise_variance)
    prior_scale = tf.linalg.cholesky(prior_cov)

    amplitude = tf.linalg.matvec(prior_scale, sample.amplitude)
    temperature = tf.linalg.matvec(prior_scale, sample.temperature)
    velocity = tf.linalg.matvec(prior_scale, sample.velocity)

    # [1, num_shells]
    amplitude = (self.amplitude_scale *
                 tf.nn.softplus(amplitude))[..., tf.newaxis, :]
    # [1, num_shells]
    temperature = (self.temperature_scale *
                   tf.nn.softplus(temperature))[..., tf.newaxis, :]
    # [1, num_shells]
    velocity = (self.velocity_scale * velocity)[..., tf.newaxis, :]
    shift = sample.shift[..., tf.newaxis]

    doppler_shifted_center_wavelength = center_wavelength * (1 - velocity)
    bandwidth = center_wavelength * tf.sqrt(temperature)

    # [num_wavelengths, num_shells]
    emissivity = amplitude / (
        tf.constant(np.sqrt(2 * np.pi), bandwidth.dtype) * bandwidth) * tf.exp(
            -(wavelengths[:, tf.newaxis] - doppler_shifted_center_wavelength)**2
            / (2 * bandwidth**2))

    if self.use_bump_function:
      emissivity *= tfp.math.round_exponential_bump_function(
          tf.linspace(-1., 1., self.num_shells))

    x = tf.linspace(-self.outer_shell_radius, self.outer_shell_radius,
                    self.num_integration_points)
    y = tf.linspace(-self.sensor_span, self.sensor_span, self.num_sensors)

    mesh_x, mesh_y = tf.meshgrid(x, y)
    # [num_sensors, num_integration_points]
    mesh_y = -shift[..., tf.newaxis] + mesh_y
    mesh_x = tf.broadcast_to(mesh_x, mesh_y.shape)
    mesh_r = tf.linalg.norm(tf.stack([mesh_x, mesh_y], -1), axis=-1)

    # [num_wavelengths, num_sensors, num_integration_points]
    mesh_emissivity = tfp.math.batch_interp_regular_1d_grid(
        mesh_r[..., tf.newaxis, :, :],
        0.,
        self.outer_shell_radius,
        emissivity[..., :, tf.newaxis, :],
        fill_value=0.)

    # [num_wavelengths, num_sensors]
    mean_measurement = tfp.math.trapz(
        mesh_emissivity,
        tf.broadcast_to(mesh_x[..., tf.newaxis, :, :], mesh_emissivity.shape))
    return mesh_emissivity, mean_measurement

  @property
  def observation_noise_fn(self):
    return self._observation_noise_fn

  def _prior_distribution(self):
    return self._prior_dist

  def log_likelihood(self, value):
    return self._log_likelihood_fn(value)

  def _sample_dataset(self, seed):
    seeds = tfp.random.split_seed(seed, 2)
    sample = self.prior_distribution().sample(seed=seeds[0])
    _, mean_measurement = self.forward_model(sample)
    measurements = self.observation_noise_fn(mean_measurement).sample(
        seed=seeds[1])
    dataset = dict(
        wavelengths=self.wavelengths,
        center_wavelength=self.center_wavelength,
        measurements=measurements,
    )
    return sample, dataset


class SyntheticPlasmaSpectroscopy(PlasmaSpectroscopy):
  """Synthetic plasma spectroscopy model.

  This uses is a synthetic dataset sampled from the model prior. This
  parameterization produces discontinuous log-probabilities, which makes it
  challenging for typical gradient-based inference methods.
  """

  def __init__(self):
    dataset = data.synthetic_plasma_spectroscopy()
    super().__init__(
        name='synthetic_plasma_spectroscopy',
        pretty_name='Synthetic Plasma Spectroscopy',
        **dataset)


class SyntheticPlasmaSpectroscopyWithBump(PlasmaSpectroscopy):
  """Synthetic plasma spectroscopy model.

  This uses is a synthetic dataset sampled from the model prior. It also uses
  the bump function to smooth out the gradients. It is also smaller than
  the `SyntheticPlasmaSpectroscopy` model.

  This model posterior is multi-modal, in part induced by the small number of
  sensors used.
  """

  def __init__(self):
    dataset = data.synthetic_plasma_spectroscopy_with_bump()
    super().__init__(
        name='synthetic_plasma_spectroscopy_with_bump',
        pretty_name='Synthetic Plasma Spectroscopy With Bump',
        use_bump_function=True,
        **dataset)
