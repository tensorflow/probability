# Copyright 2018 The TensorFlow Probability Authors.
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
"""Horseshoe Distribution Class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import half_cauchy
from tensorflow_probability.python.distributions.seed_stream import SeedStream
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization


__all__ = [
    "Horseshoe",
]


class Horseshoe(distribution.Distribution):
  r"""Horseshoe distribution.

  The so-called 'horseshoe' distribution is a Cauchy-Normal scale mixture,
  proposed as a sparsity-inducing prior for Bayesian regression. [1] It is
  symmetric around zero, has heavy (Cauchy-like) tails, so that large
  coefficients face relatively little shrinkage, but an infinitely tall spike at
  0, which pushes small coefficients towards zero. It is parameterized by a
  positive scalar `scale` parameter: higher values yield a weaker
  sparsity-inducing effect.

  #### Mathematical details

  The Horseshoe distribution is centered at zero, with scale parameter
  \\(\lambda\\). It is defined by: \
  \\(
  X \sim \text {Horseshoe}(scale=\lambda) \, \equiv \, X \sim \text{Normal}
  (0, \, \lambda \cdot \sigma) \quad \text{where} \quad \sigma \sim
  \text{HalfCauchy} (0, \,1)
  \\)

  The probability density function, \
  \\(
  \pi_\lambda(x) = \int_0^\infty \, \frac{1}{\sqrt{ 2\pi \lambda^2 t^2 }} \,
  \exp \left\{ -\frac{x^2}{2\lambda^2t^2} \right\} \,
  \frac{2}{\pi\left(1+t^2\right)} \mathrm{d} t
  \\)

  can be rewritten [1] as \
  \\(
  \pi_\lambda(x) = \frac{1}{\sqrt{2 \pi^3 \lambda^2}} \, \exp \left\{
  \frac{x^2}{2\lambda^2} \right\} \, E_1\left(\frac{x^2}{2\lambda^2}\right)
  \\)

  where E<sub>1</sub>(.) is the [exponential integral function][wiki1] which can
  be approximated by elementary functions. [2]

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Define a single scalar Horseshoe distribution.
  dist = tfp.distributions.Horseshoe(scale=3.0)

  # Evaluate the log_prob at 1, returning a scalar.
  dist.log_prob(1.)

  # Define a batch of two scalar valued Horseshoes.
  # The first has scale 11.0, the second 22.0
  dist = tfp.distributions.Horseshoe(scale=[11.0, 22.0])

  # Evaluate the log_prob of the first distribution on 1.0, and the second on
  # 1.5, returning a length two tensor.
  dist.log_prob([1.0, 1.5])

  # Evaluate the log_prob of both distributions at 2.0 and 2.5, returning a
  # 2 x 2 tensor.
  dist.log_prob([[2.0], [2.5]])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  #### References

  [1] Carvalho, Polson, Scott.
  [Handling Sparsity via the Horseshoe (2008)][link1].

  [2] Barry, Parlange, Li.
  [Approximation for the exponential integral (2000)][link2]. Formula from
  [Wikipedia][wiki2].

  [link1]:
      http://faculty.chicagobooth.edu/nicholas.polson/research/papers/Horse.pdf
  [link2]: https://doi.org/10.1016/S0022-1694(99)00184-5
  [wiki1]: https://en.wikipedia.org/wiki/Exponential_integral
  [wiki2]: https://en.wikipedia.org/wiki/Exponential_integral#cite_note-17
  """

  def __init__(self,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="Horseshoe"):
    """Construct a Horseshoe distribution with `scale`.

    Args:
      scale: Floating point tensor; the scales of the distribution(s).
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs. Default value: `False` (i.e., do not validate args).
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'Horseshoe'.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([scale], dtype_hint=tf.float32)
      scale = tf.convert_to_tensor(scale, name="scale", dtype=dtype)
      with tf.control_dependencies(
          [assert_util.assert_positive(scale)] if validate_args else []):
        self._scale = tf.identity(
            scale, name="scale")
    self._half_cauchy = half_cauchy.HalfCauchy(
        loc=tf.zeros([], dtype=dtype),
        scale=tf.ones([], dtype=dtype),
        allow_nan_stats=True)
    super(Horseshoe, self).__init__(
        dtype=self._scale.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._scale],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"scale": tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

  @classmethod
  def _params_event_ndims(cls):
    return dict(scale=0)

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._scale

  def _batch_shape_tensor(self):
    return tf.shape(self.scale)

  def _batch_shape(self):
    return self.scale.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    # The exact HalfCauchy-Normal marginal log-density is analytically
    # intractable; we compute a (relatively accurate) numerical
    # approximation. This is a log space version of ref[2] from class docstring.
    xx = (x / self.scale)**2 / 2
    g = 0.5614594835668851  # tf.exp(-0.5772156649015328606)
    b = 1.0420764938351215   # tf.sqrt(2 * (1-g) / (g * (2-g)))
    h_inf = 1.0801359952503342  #  (1-g)*(g*g-6*g+12) / (3*g * (2-g)**2 * b)
    q = 20. / 47. * xx**1.0919284281983377
    h = 1. / (1 + xx**(1.5)) + h_inf * q / (1 + q)
    c = -.5 * np.log(2 * np.pi**3) - tf.math.log(g * self._scale)
    return -tf.math.log1p((1 - g) / g * tf.exp(-xx / (1 - g))) + tf.math.log(
        tf.math.log1p(g / xx - (1 - g) / (h + b * xx)**2)) + c

  def _sample_n(self, n, seed=None):
    shape = tf.concat([[n], self._batch_shape_tensor()], axis=0)
    seed = SeedStream(seed, salt="random_horseshoe")
    local_shrinkage = self._half_cauchy.sample(shape, seed=seed())
    shrinkage = self.scale * local_shrinkage
    sampled = tf.random.normal(
        shape=shape, mean=0., stddev=1., dtype=self.scale.dtype, seed=seed())
    return sampled * shrinkage

  def _mean(self):
    return tf.zeros(self.batch_shape_tensor())

  def _mode(self):
    return self._mean()

  def _stddev(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    raise ValueError("`stddev` is undefined for Horseshoe distribution.")

  def _variance(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    raise ValueError(
        "`variance` is undefined for Horseshoe distribution.")
