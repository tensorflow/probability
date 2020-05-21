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
"""The PoissonLogNormalQuadratureCompound distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'PoissonLogNormalQuadratureCompound',
    'quadrature_scheme_lognormal_gauss_hermite',
    'quadrature_scheme_lognormal_quantiles',
]


def quadrature_scheme_lognormal_gauss_hermite(
    loc, scale, quadrature_size,
    validate_args=False, name=None):  # pylint: disable=unused-argument
  """Use Gauss-Hermite quadrature to form quadrature on positive-reals.

  Note: for a given `quadrature_size`, this method is generally less accurate
  than `quadrature_scheme_lognormal_quantiles`.

  Args:
    loc: `float`-like (batch of) scalar `Tensor`; the location parameter of
      the LogNormal prior.
    scale: `float`-like (batch of) scalar `Tensor`; the scale parameter of
      the LogNormal prior.
    quadrature_size: Python `int` scalar representing the number of quadrature
      points.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
    grid: (Batch of) length-`quadrature_size` vectors representing the
      `log_rate` parameters of a `Poisson`.
    probs: (Batch of) length-`quadrature_size` vectors representing the
      weight associate with each `grid` value.
  """
  with tf.name_scope(
      name or 'vector_diffeomixture_quadrature_gauss_hermite'):
    grid, probs = np.polynomial.hermite.hermgauss(deg=quadrature_size)
    npdt = dtype_util.as_numpy_dtype(loc.dtype)
    grid = grid.astype(npdt)
    probs = probs.astype(npdt)
    probs /= np.linalg.norm(probs, ord=1, keepdims=True)
    probs = tf.convert_to_tensor(probs, name='probs', dtype=loc.dtype)
    # The following maps the broadcast of `loc` and `scale` to each grid
    # point, i.e., we are creating several log-rates that correspond to the
    # different Gauss-Hermite quadrature points and (possible) batches of
    # `loc` and `scale`.
    grid = (loc[..., tf.newaxis] + np.sqrt(2.) * scale[..., tf.newaxis] * grid)
    return grid, probs


def quadrature_scheme_lognormal_quantiles(
    loc, scale, quadrature_size,
    validate_args=False, name=None):
  """Use LogNormal quantiles to form quadrature on positive-reals.

  Args:
    loc: `float`-like (batch of) scalar `Tensor`; the location parameter of
      the LogNormal prior.
    scale: `float`-like (batch of) scalar `Tensor`; the scale parameter of
      the LogNormal prior.
    quadrature_size: Python `int` scalar representing the number of quadrature
      points.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
    grid: (Batch of) length-`quadrature_size` vectors representing the
      `log_rate` parameters of a `Poisson`.
    probs: (Batch of) length-`quadrature_size` vectors representing the
      weight associate with each `grid` value.
  """
  with tf.name_scope(name or 'quadrature_scheme_lognormal_quantiles'):
    # Create a LogNormal distribution.
    dist = transformed_distribution.TransformedDistribution(
        distribution=normal.Normal(loc=loc, scale=scale),
        bijector=exp_bijector.Exp(),
        validate_args=validate_args)
    batch_ndims = tensorshape_util.rank(dist.batch_shape)
    if batch_ndims is None:
      batch_ndims = tf.shape(dist.batch_shape_tensor())[0]

    def _compute_quantiles():
      """Helper to build quantiles."""
      # Omit {0, 1} since they might lead to Inf/NaN.
      zero = tf.zeros([], dtype=dist.dtype)
      edges = tf.linspace(zero, 1., quadrature_size + 3)[1:-1]
      # Expand edges so its broadcast across batch dims.
      edges = tf.reshape(
          edges,
          shape=tf.concat(
              [[-1], tf.ones([batch_ndims], dtype=tf.int32)], axis=0))
      quantiles = dist.quantile(edges)
      # Cyclically permute left by one.
      perm = tf.concat([tf.range(1, 1 + batch_ndims), [0]], axis=0)
      quantiles = tf.transpose(a=quantiles, perm=perm)
      return quantiles
    quantiles = _compute_quantiles()

    # Compute grid as quantile midpoints.
    grid = (quantiles[..., :-1] + quantiles[..., 1:]) / 2.
    # Set shape hints.
    new_shape = tensorshape_util.concatenate(dist.batch_shape,
                                             [quadrature_size])
    tensorshape_util.set_shape(grid, new_shape)

    # By construction probs is constant, i.e., `1 / quadrature_size`. This is
    # important, because non-constant probs leads to non-reparameterizable
    # samples.
    probs = tf.fill(
        dims=[quadrature_size],
        value=tf.math.reciprocal(tf.cast(quadrature_size, dist.dtype)))

    return grid, probs


class PoissonLogNormalQuadratureCompound(distribution.Distribution):
  """`PoissonLogNormalQuadratureCompound` distribution.

  The `PoissonLogNormalQuadratureCompound` is an approximation to a
  Poisson-LogNormal [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution), i.e.,

  ```none
  p(k|loc, scale)
  = int_{R_+} dl LogNormal(l | loc, scale) Poisson(k | l)
  approx= sum{ prob[d] Poisson(k | lambda(grid[d])) : d=0, ..., deg-1 }
  ```

  By default, the `grid` is chosen as quantiles of the `LogNormal` distribution
  parameterized by `loc`, `scale` and the `prob` vector is
  `[1. / quadrature_size]*quadrature_size`.

  In the non-approximation case, a draw from the LogNormal prior represents the
  Poisson rate parameter. Unfortunately, the non-approximate distribution lacks
  an analytical probability density function (pdf). Therefore the
  `PoissonLogNormalQuadratureCompound` class implements an approximation based
  on [quadrature](https://en.wikipedia.org/wiki/Numerical_integration).

  Note: although the `PoissonLogNormalQuadratureCompound` is approximately the
  Poisson-LogNormal compound distribution, it is itself a valid distribution.
  Viz., it possesses a `sample`, `log_prob`, `mean`, `variance`, etc. which are
  all mutually consistent.

  #### Mathematical Details

  The `PoissonLogNormalQuadratureCompound` approximates a Poisson-LogNormal
  [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution). Using
  variable-substitution and [numerical quadrature](
  https://en.wikipedia.org/wiki/Numerical_integration) (default:
  based on `LogNormal` quantiles) we can redefine the distribution to be a
  parameter-less convex combination of `deg` different Poisson samples.

  That is, defined over positive integers, this distribution is parameterized
  by a (batch of) `loc` and `scale` scalars.

  The probability density function (pdf) is,

  ```none
  pdf(k | loc, scale, deg)
    = sum{ prob[d] Poisson(k | lambda=exp(grid[d]))
          : d=0, ..., deg-1 }
  ```

  #### Examples

  ```python
  tfd = tfp.distributions

  # Create two batches of PoissonLogNormalQuadratureCompounds, one with
  # prior `loc = 0.` and another with `loc = 1.` In both cases `scale = 1.`
  pln = tfd.PoissonLogNormalQuadratureCompound(
      loc=[0., -0.5],
      scale=1.,
      quadrature_size=10,
      validate_args=True)
  """

  def __init__(self,
               loc,
               scale,
               quadrature_size=8,
               quadrature_fn=quadrature_scheme_lognormal_quantiles,
               validate_args=False,
               allow_nan_stats=True,
               name='PoissonLogNormalQuadratureCompound'):
    """Constructs the PoissonLogNormalQuadratureCompound`.

    Note: `probs` returned by (optional) `quadrature_fn` are presumed to be
    either a length-`quadrature_size` vector or a batch of vectors in 1-to-1
    correspondence with the returned `grid`. (I.e., broadcasting is only
    partially supported.)

    Args:
      loc: `float`-like (batch of) scalar `Tensor`; the location parameter of
        the LogNormal prior.
      scale: `float`-like (batch of) scalar `Tensor`; the scale parameter of
        the LogNormal prior.
      quadrature_size: Python `int` scalar representing the number of quadrature
        points.
      quadrature_fn: Python callable taking `loc`, `scale`,
        `quadrature_size`, `validate_args` and returning `tuple(grid, probs)`
        representing the LogNormal grid and corresponding normalized weight.
        Default value: `quadrature_scheme_lognormal_quantiles`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `quadrature_grid` and `quadrature_probs` have different base
        `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      self._quadrature_fn = quadrature_fn
      dtype_util.assert_same_float_dtype([self._loc, self._scale])

      self._quadrature_size = quadrature_size

      super(PoissonLogNormalQuadratureCompound, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  def poisson_and_mixture_distributions(self):
    """Returns the Poisson and Mixture distribution parameterized by the quadrature grid and weights."""
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    quadrature_grid, quadrature_probs = tuple(self._quadrature_fn(
        loc, scale, self.quadrature_size, self.validate_args))
    dt = quadrature_grid.dtype
    if not dtype_util.base_equal(dt, quadrature_probs.dtype):
      raise TypeError('Quadrature grid dtype ({}) does not match quadrature '
                      'probs dtype ({}).'.format(
                          dtype_util.name(dt),
                          dtype_util.name(quadrature_probs.dtype)))

    dist = poisson.Poisson(
        log_rate=quadrature_grid,
        validate_args=self.validate_args,
        allow_nan_stats=self.allow_nan_stats)

    mixture_dist = categorical.Categorical(
        logits=tf.math.log(quadrature_probs),
        validate_args=self.validate_args,
        allow_nan_stats=self.allow_nan_stats)
    return dist, mixture_dist

  @property
  def loc(self):
    """Location parameter of the LogNormal prior."""
    return self._loc

  @property
  def scale(self):
    """Scale parameter of the LogNormal prior."""
    return self._scale

  @property
  def quadrature_size(self):
    return self._quadrature_size

  def _batch_shape_tensor(self, distributions=None):
    if distributions is None:
      distributions = self.poisson_and_mixture_distributions()
    dist, mixture_dist = distributions
    return tf.broadcast_dynamic_shape(
        dist.batch_shape_tensor(),
        prefer_static.shape(mixture_dist.logits))[:-1]

  def _batch_shape(self):
    dist, mixture_dist = self.poisson_and_mixture_distributions()
    return tf.broadcast_static_shape(
        dist.batch_shape,
        mixture_dist.logits.shape)[:-1]

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    # Get ids as a [n, batch_size]-shaped matrix, unless batch_shape=[] then get
    # ids as a [n]-shaped vector.
    distributions = self.poisson_and_mixture_distributions()
    dist, mixture_dist = distributions
    batch_size = tensorshape_util.num_elements(self.batch_shape)
    if batch_size is None:
      batch_size = tf.reduce_prod(
          self._batch_shape_tensor(distributions=distributions))
    # We need to 'sample extra' from the mixture distribution if it doesn't
    # already specify a probs vector for each batch coordinate.
    # We only support this kind of reduced broadcasting, i.e., there is exactly
    # one probs vector for all batch dims or one for each.
    mixture_seed, poisson_seed = samplers.split_seed(
        seed, salt='PoissonLogNormalQuadratureCompound')
    ids = mixture_dist.sample(
        sample_shape=concat_vectors(
            [n],
            distribution_util.pick_vector(
                mixture_dist.is_scalar_batch(),
                [batch_size],
                np.int32([]))),
        seed=mixture_seed)
    # We need to flatten batch dims in case mixture_dist has its own
    # batch dims.
    ids = tf.reshape(
        ids,
        shape=concat_vectors([n],
                             distribution_util.pick_vector(
                                 self.is_scalar_batch(), np.int32([]),
                                 np.int32([-1]))))

    # Stride `quadrature_size` for `batch_size` number of times.
    offset = tf.range(
        start=0,
        limit=batch_size * self._quadrature_size,
        delta=self._quadrature_size,
        dtype=ids.dtype)
    ids = ids + offset
    rate = tf.gather(tf.reshape(dist.rate_parameter(), shape=[-1]), ids)
    rate = tf.reshape(
        rate, shape=concat_vectors([n], self._batch_shape_tensor(
            distributions=distributions)))
    return samplers.poisson(
        shape=[], lam=rate, dtype=self.dtype, seed=poisson_seed)

  def _log_prob(self, x):
    dist, mixture_dist = self.poisson_and_mixture_distributions()
    return tf.reduce_logsumexp((mixture_dist.logits +
                                dist.log_prob(x[..., tf.newaxis])),
                               axis=-1)

  def _mean(self, distributions=None):
    if distributions is None:
      distributions = self.poisson_and_mixture_distributions()
    dist, mixture_dist = distributions
    return tf.exp(
        tf.reduce_logsumexp(
            mixture_dist.logits + dist.log_rate,
            axis=-1))

  def _variance(self):
    return tf.exp(self._log_variance())

  def _stddev(self):
    return tf.exp(0.5 * self._log_variance())

  def _log_variance(self):
    # Following calculation is based on law of total variance:
    #
    # Var[Z] = E[Var[Z | V]] + Var[E[Z | V]]
    #
    # where,
    #
    # Z|v ~ interpolate_affine[v](dist)
    # V ~ mixture_dist
    #
    # thus,
    #
    # E[Var[Z | V]] = sum{ prob[d] Var[d] : d=0, ..., deg-1 }
    # Var[E[Z | V]] = sum{ prob[d] (Mean[d] - Mean)**2 : d=0, ..., deg-1 }
    distributions = self.poisson_and_mixture_distributions()
    dist, mixture_dist = distributions
    v = tf.stack(
        [
            # log(dist.variance()) = log(Var[d]) = log(rate[d])
            dist.log_rate,
            # log((Mean[d] - Mean)**2)
            2. * tf.math.log(
                tf.abs(
                    dist.mean() -
                    self._mean(distributions=distributions)[..., tf.newaxis])),
        ],
        axis=-1)
    return tf.reduce_logsumexp(
        mixture_dist.logits[..., tf.newaxis] + v, axis=[-2, -1])

  def _default_event_space_bijector(self):
    return

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    return assertions


def concat_vectors(*args):
  """Concatenates input vectors, statically if possible."""
  args_ = [tf.get_static_value(x) for x in args]
  if any(vec is None for vec in args_):
    return tf.concat(args, axis=0)
  return [val for vec in args_ for val in vec]
