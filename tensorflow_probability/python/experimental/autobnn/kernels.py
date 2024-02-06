# Copyright 2023 The TensorFlow Probability Authors.
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
"""`Leaf` BNNs, most of which correspond to some known GP kernel."""

import functools
from flax import linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp
from tensorflow_probability.python.experimental.autobnn import bnn
from tensorflow_probability.substrates.jax.distributions import lognormal as lognormal_lib
from tensorflow_probability.substrates.jax.distributions import normal as normal_lib
from tensorflow_probability.substrates.jax.distributions import student_t as student_t_lib
from tensorflow_probability.substrates.jax.distributions import uniform as uniform_lib


Array = jnp.ndarray


SQRT_TWO = 1.41421356237309504880168872420969807856967187537694807317667


class MultipliableBNN(bnn.BNN):
  """Abstract base class for BNN's that can be multiplied."""
  width: int = 50
  going_to_be_multiplied: bool = False

  def penultimate(self, inputs):
    raise NotImplementedError('Subclasses of MultipliableBNN must define this.')


class IdentityBNN(MultipliableBNN):
  """A BNN that always predicts 1."""

  def penultimate(self, inputs):
    return jnp.ones(shape=inputs.shape[:-1] + (self.width,))

  def __call__(self, inputs, deterministic=True):
    out_shape = inputs.shape[:-1] + (self.likelihood_model.num_outputs(),)
    return jnp.ones(shape=out_shape)


class OneLayerBNN(MultipliableBNN):
  """A BNN with one hidden layer."""

  # Period is currently only used by the PeriodicBNN class, but we declare it
  # here so it can be passed to a "generic" OneLayerBNN instance.
  period: float = 0.0

  bias_scale: float = 1.0

  def setup(self):
    if not hasattr(self, 'input_warping'):
      self.input_warping = lambda x: x
    if not hasattr(self, 'activation_function'):
      self.activation_function = nn.relu
    if not hasattr(self, 'kernel_init'):
      self.kernel_init = initializers.lecun_normal()
    if not hasattr(self, 'bias_init'):
      self.bias_init = initializers.zeros_init()
    self.dense1 = nn.Dense(self.width,
                           kernel_init=self.kernel_init,
                           bias_init=self.bias_init)
    if not self.going_to_be_multiplied:
      self.dense2 = nn.Dense(
          self.likelihood_model.num_outputs(),
          kernel_init=nn.initializers.normal(1. / jnp.sqrt(self.width)),
          bias_init=nn.initializers.zeros)
    else:
      def fake_dense2(x):
        out_shape = x.shape[:-1] + (self.likelihood_model.num_outputs(),)
        return jnp.ones(out_shape)
      self.dense2 = fake_dense2
    super().setup()

  def distributions(self):
    # Strictly speaking, these distributions don't exactly correspond to
    # the initializations used in setup().  lecun_normal uses a truncated
    # normal, for example, and the zeros_init used for the bias certainly
    # isn't a sample from a normal.
    d = {
        'dense1': {
            'kernel': normal_lib.Normal(
                loc=0, scale=1.0 / jnp.sqrt(self.width)
            ),
            'bias': normal_lib.Normal(loc=0, scale=self.bias_scale),
        }
    }
    if not self.going_to_be_multiplied:
      d['dense2'] = {
          'kernel': normal_lib.Normal(loc=0, scale=1.0 / jnp.sqrt(self.width)),
          'bias': normal_lib.Normal(loc=0, scale=self.bias_scale),
      }
    return super().distributions() | d

  @functools.partial(jax.named_call, name='OneLayer::penultimate')
  def penultimate(self, inputs):
    y = self.input_warping(inputs)
    return self.activation_function(self.dense1(y))

  @functools.partial(jax.named_call, name='OneLayer::__call__')
  def __call__(self, inputs, deterministic=True):
    return self.dense2(self.penultimate(inputs))


class ExponentiatedQuadraticBNN(OneLayerBNN):
  """A BNN corresponding to the Radial Basis Function kernel."""
  amplitude_scale: float = 1.0
  length_scale_scale: float = 1.0

  def setup(self):
    if not hasattr(self, 'activation_function'):
      self.activation_function = lambda x: SQRT_TWO * jnp.sin(x)
    if not hasattr(self, 'input_warping'):
      self.input_warping = lambda x: x / self.length_scale
    self.kernel_init = nn.initializers.normal(1.0)
    def uniform_init(seed, shape, dtype):
      return nn.initializers.uniform(scale=2.0 * jnp.pi)(
          seed, shape, dtype=dtype) - jnp.pi
    self.bias_init = uniform_init
    super().setup()

  def distributions(self):
    d = super().distributions()
    return d | {
        'amplitude': lognormal_lib.LogNormal(loc=0, scale=self.amplitude_scale),
        'length_scale': lognormal_lib.LogNormal(
            loc=0, scale=self.length_scale_scale
        ),
        'dense1': {
            'kernel': normal_lib.Normal(loc=0, scale=1.0),
            'bias': uniform_lib.Uniform(low=-jnp.pi, high=jnp.pi),
        },
    }

  @functools.partial(jax.named_call, name='RBF::__call__')
  def __call__(self, inputs, deterministic=True):
    return self.amplitude * self.dense2(self.penultimate(inputs))

  def shortname(self) -> str:
    sn = super().shortname()
    return 'RBF' if sn == 'ExponentiatedQuadratic' else sn


class MaternBNN(ExponentiatedQuadraticBNN):
  """A BNN corresponding to the Matern kernel."""
  degrees_of_freedom: float = 2.5

  def setup(self):
    def kernel_init(seed, shape, unused_dtype):
      return student_t_lib.StudentT(
          df=2.0 * self.degrees_of_freedom, loc=0.0, scale=1.0
      ).sample(shape, seed=seed)
    self.kernel_init = kernel_init
    super().setup()

  def distributions(self):
    d = super().distributions()
    d['dense1']['kernel'] = student_t_lib.StudentT(
        df=2.0 * self.degrees_of_freedom, loc=0.0, scale=1.0)
    return d

  def summarize(self, params=None, full: bool = False) -> str:
    """Return a string summarizing the structure of the BNN."""
    return f'{self.shortname()}({self.degrees_of_freedom})'


class ExponentialBNN(MaternBNN):
  """Matern(0.5), also known as the absolute exponential kernel."""
  degrees_of_freedom: float = 0.5

  def summarize(self, params=None, full: bool = False) -> str:
    return self.shortname()


class PolynomialBNN(OneLayerBNN):
  """A BNN where samples are polynomial functions."""
  degree: int = 2
  shift_mean: float = 0.0
  shift_scale: float = 1.0
  amplitude_scale: float = 1.0
  bias_init_amplitude: float = 0.0

  def distributions(self):
    d = super().distributions()
    del d['dense1']
    for i in range(self.degree):
      # Do not scale these layers by 1/sqrt(width), because we also
      # multiply these weights by the learned `amplitude` parameter.
      d[f'hiddens_{i}'] = {
          'kernel': normal_lib.Normal(loc=0, scale=1.0),
          'bias': normal_lib.Normal(loc=0, scale=self.bias_scale),
      }
    return d | {
        'shift': normal_lib.Normal(loc=self.shift_mean, scale=self.shift_scale),
        'amplitude': lognormal_lib.LogNormal(loc=0, scale=self.amplitude_scale),
    }

  def setup(self):
    kernel_init = nn.initializers.normal(1.0)
    def bias_init(seed, shape, dtype=jnp.float32):
      return self.bias_init_amplitude * jax.random.normal(
          seed, shape, dtype=dtype)
    self.hiddens = [
        nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)
        for _ in range(self.degree)]
    super().setup()

  @functools.partial(jax.named_call, name='Polynomial::penultimate')
  def penultimate(self, inputs):
    x = inputs - self.shift
    ys = jnp.stack([h(x) for h in self.hiddens], axis=-1)
    return self.amplitude * jnp.prod(ys, axis=-1)

  def summarize(self, params=None, full: bool = False) -> str:
    """Return a string summarizing the structure of the BNN."""
    return f'{self.shortname()}(degree={self.degree})'


class LinearBNN(PolynomialBNN):
  """A BNN where samples are lines."""
  degree: int = 1

  def summarize(self, params=None, full: bool = False) -> str:
    return self.shortname()


class QuadraticBNN(PolynomialBNN):
  """A BNN where samples are parabolas."""

  degree: int = 2

  def summarize(self, params=None, full: bool = False) -> str:
    return self.shortname()


def make_periodic_input_warping(period, periodic_index, include_original):
  """Return an input warping function that adds Fourier features.

  Args:
    period: The added features will repeat this many time steps.
    periodic_index: Look for the time feature in input[..., periodic_index].
    include_original: If true, don't replace the time feature with the
      new Fourier features.

  Returns:
    A function that takes an input tensor of shape [..., n] and returns a
    tensor of shape [..., n+2] if include_original is True and of shape
    [..., n+1] if include_original is False.
  """
  def input_warping(x):
    time = x[..., periodic_index]
    y = 2.0 * jnp.pi * time / period
    features = [jnp.cos(y), jnp.sin(y)]
    if include_original:
      features.append(time)
    if jnp.ndim(x) == 1:
      features = jnp.array(features).T
    else:
      features = jnp.vstack(features).T
    return jnp.concatenate(
        [
            x[..., :periodic_index],
            features,
            x[..., periodic_index + 1:],
        ],
        -1,
    )

  return input_warping


class PeriodicBNN(ExponentiatedQuadraticBNN):
  """A BNN corresponding to a periodic kernel."""
  periodic_index: int = 0

  def setup(self):
    # TODO(colcarroll): Figure out how to assert that self.period is positive.

    self.input_warping = make_periodic_input_warping(
        self.period, self.periodic_index, include_original=False
    )
    super().setup()

  def summarize(self, params=None, full: bool = False) -> str:
    """Return a string summarizing the structure of the BNN."""
    return f'{self.shortname()}(period={self.period:.2f})'


class MultiLayerBNN(OneLayerBNN):
  """Multi-layer BNN that also has access to periodic features."""
  num_layers: int = 3
  periodic_index: int = 0

  def setup(self):
    if not hasattr(self, 'kernel_init'):
      self.kernel_init = initializers.lecun_normal()
    if not hasattr(self, 'bias_init'):
      self.bias_init = initializers.zeros_init()
    self.input_warping = make_periodic_input_warping(
        self.period, self.periodic_index, include_original=True
    )
    self.dense = [
        nn.Dense(
            self.width, kernel_init=self.kernel_init, bias_init=self.bias_init
        )
        for _ in range(self.num_layers)
    ]
    super().setup()

  def distributions(self):
    d = super().distributions()
    del d['dense1']
    for i in range(self.num_layers):
      d[f'dense_{i}'] = {
          'kernel': normal_lib.Normal(loc=0, scale=1.0 / jnp.sqrt(self.width)),
          'bias': normal_lib.Normal(loc=0, scale=self.bias_scale),
      }
    return d

  def penultimate(self, inputs):
    y = self.input_warping(inputs)
    for i in range(self.num_layers):
      y = self.activation_function(self.dense[i](y))
    return y

  def summarize(self, params=None, full: bool = False) -> str:
    """Return a string summarizing the structure of the BNN."""
    return (
        f'{self.shortname()}(num_layers={self.num_layers},period={self.period})'
    )
