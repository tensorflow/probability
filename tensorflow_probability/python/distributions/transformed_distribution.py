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
"""A Transformed Distribution class."""
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import ldj_ratio
from tensorflow_probability.python.distributions import batch_broadcast
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'TransformedDistribution',
]


def _default_kwargs_split_fn(kwargs):
  """Default `kwargs` `dict` getter."""
  return (kwargs.get('distribution_kwargs', {}),
          kwargs.get('bijector_kwargs', {}))


class _TransformedDistribution(distribution_lib.Distribution):
  """A Transformed Distribution.

  A `TransformedDistribution` models `p(y)` given a base distribution `p(x)`,
  and a deterministic, invertible, differentiable transform, `Y = g(X)`. The
  transform is typically an instance of the `Bijector` class and the base
  distribution is typically an instance of the `Distribution` class.

  A `Bijector` is expected to implement the following functions:

  - `forward`,
  - `inverse`,
  - `inverse_log_det_jacobian`.

  The semantics of these functions are outlined in the `Bijector` documentation.

  We now describe how a `TransformedDistribution` alters the input/outputs of a
  `Distribution` associated with a random variable (rv) `X`.

  Write `cdf(Y=y)` for an absolutely continuous cumulative distribution function
  of random variable `Y`; write the probability density function
  `pdf(Y=y) := d^k / (dy_1,...,dy_k) cdf(Y=y)` for its derivative wrt to `Y`
  evaluated at `y`. Assume that `Y = g(X)` where `g` is a deterministic
  diffeomorphism, i.e., a non-random, continuous, differentiable, and invertible
  function.  Write the inverse of `g` as `X = g^{-1}(Y)` and `(J o g)(x)` for
  the Jacobian of `g` evaluated at `x`.

  A `TransformedDistribution` implements the following operations:

    * `sample`
      Mathematically:   `Y = g(X)`
      Programmatically: `bijector.forward(distribution.sample(...))`

    * `log_prob`
      Mathematically:   `(log o pdf)(Y=y) = (log o pdf o g^{-1})(y)
                         + (log o abs o det o J o g^{-1})(y)`
      Programmatically: `(distribution.log_prob(bijector.inverse(y))
                         + bijector.inverse_log_det_jacobian(y))`

    * `log_cdf`
      Mathematically:   `(log o cdf)(Y=y) = (log o cdf o g^{-1})(y)`
      Programmatically: `distribution.log_cdf(bijector.inverse(x))`

    * and similarly for: `cdf`, `prob`, `log_survival_function`,
     `survival_function`.

  Kullback-Leibler divergence is also well defined for `TransformedDistribution`
  instances that have matching bijectors.  Bijector matching is performed via
  the `Bijector.__eq__` method, e.g., `td1.bijector == td2.bijector`, as part
  of the KL calculation.  If the underlying bijectors do not match, a
  `NotImplementedError` is raised when calling `kl_divergence`.  This is the
  same behavior as calling `kl_divergence` when two distributions do not have
  a registered KL divergence.

  **Note** Due to the current constraints imposed on bijector equality testing,
  `kl_divergence` may behave differently in eager mode computation vs. traced
  computation.  For example, if a TD Bijector's parameters are `Tensor` objects,
  and are themselves derived from e.g. a Variable, some stateful operation, or
  from an argument to a `tf.function` then Bijector equality cannot be known
  during the call to `kl_divergence` and the bijectors are assumed unequal.
  In this case, calling `kl_divergence` may raise an exception in
  graph / tf.function mode, but work just fine in eager / numpy mode.

  A simple example constructing a Log-Normal distribution from a Normal
  distribution:

  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors
  log_normal = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.Exp(),
    name='LogNormalTransformedDistribution')
  ```

  A `LogNormal` made from callables:

  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors
  log_normal = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.Inline(
      forward_fn=tf.exp,
      inverse_fn=tf.log,
      inverse_log_det_jacobian_fn=(
        lambda y: -tf.reduce_sum(tf.log(y), axis=-1)),
    name='LogNormalTransformedDistribution')
  ```

  Another example constructing a Normal from a StandardNormal:

  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors
  normal = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.Shift(shift=-1.)(tfb.Scale(scale=2.)),
    name='NormalTransformedDistribution')
  ```

  A `TransformedDistribution`'s `batch_shape` is derived by *broadcasting* the
  batch shapes of the base distribution and the bijector. The base distribution
  is then itself implicitly lifted to the broadcast batch shape. For example, in

  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors
  batch_normal = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.Shift(shift=[-1., 0., 1.]),
    name='BatchNormalTransformedDistribution')
  ```

  the base distribution has batch shape `[]`, and the bijector applied to this
  distribution contributes a batch shape of `[3]` (obtained as
  `bijector.experimental_batch_shape(
  x_event_ndims=tf.rank(distribution.event_shape))`, yielding the broadcast
  shape `batch_normal.batch_shape == [3]`. Although sampling from the base
  distribution would ordinarily return just a single value, calling
  `batch_normal.sample()` will return a Tensor of 3 independent values, just as
  if the base distribution had explicitly followed the broadcast batch shape.

  The `event_shape` of a `TransformedDistribution` is the `forward_event_shape`
  of the bijector applied to the `event_shape` of the base distribution.

  `tfd.Sample` or `tfd.Independent` may be used to add extra IID dimensions to
  the `event_shape` of the base distribution before the bijector operates on it.
  The following example demonstrates how to construct a multivariate Normal as a
  `TransformedDistribution`, by adding a rank-1 IID dimension to the
  `event_shape` of a standard Normal and applying `tfb.ScaleMatvecTriL`.

  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors
  # We will create two MVNs with batch_shape = event_shape = 2.
  mean = [[-1., 0],      # batch:0
          [0., 1]]       # batch:1
  chol_cov = [[[1., 0],
               [0, 1]],  # batch:0
              [[1, 0],
               [2, 2]]]  # batch:1
  mvn1 = tfd.TransformedDistribution(
      distribution=tfd.Sample(
          tfd.Normal(loc=[0., 0], scale=1.),  # base_dist.batch_shape == [2]
          sample_shape=[2])                   # base_dist.event_shape == [2]
      bijector=tfb.Shift(shift=mean)(tfb.ScaleMatvecTriL(scale_tril=chol_cov)))
  mvn2 = ds.MultivariateNormalTriL(loc=mean, scale_tril=chol_cov)
  # mvn1.log_prob(x) == mvn2.log_prob(x)
  ```

  """

  def __init__(self,
               distribution,
               bijector,
               kwargs_split_fn=_default_kwargs_split_fn,
               validate_args=False,
               parameters=None,
               name=None):
    """Construct a Transformed Distribution.

    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      bijector: The object responsible for calculating the transformation.
        Typically an instance of `Bijector`.
      kwargs_split_fn: Python `callable` which takes a kwargs `dict` and returns
        a tuple of kwargs `dict`s for each of the `distribution` and `bijector`
        parameters respectively.
        Default value: `_default_kwargs_split_fn` (i.e.,
            `lambda kwargs: (kwargs.get('distribution_kwargs', {}),
                             kwargs.get('bijector_kwargs', {}))`)
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      parameters: Locals dict captured by subclass constructor, to be used for
        copy/slice re-instantiation operations.
      name: Python `str` name prefixed to Ops created by this class. Default:
        `bijector.name + distribution.name`.
    """
    parameters = dict(locals()) if parameters is None else parameters
    name = name or (('' if bijector is None else bijector.name) +
                    (distribution.name or ''))
    with tf.name_scope(name) as name:
      self._distribution = distribution
      self._bijector = bijector
      self._kwargs_split_fn = (_default_kwargs_split_fn
                               if kwargs_split_fn is None
                               else kwargs_split_fn)

      # For convenience we define some handy constants.
      self._zero = tf.constant(0, dtype=tf.int32, name='zero')

      # We don't just want to check isinstance(JointDistribution) because
      # TransformedDistributions with multipart bijectors are effectively
      # joint but don't inherit from JD. The 'duck-type' test is that
      # JDs have a structured dtype.
      dtype = self.bijector.forward_dtype(self.distribution.dtype)
      self._is_joint = tf.nest.is_nested(dtype)

      super(_TransformedDistribution, self).__init__(
          dtype=dtype,
          reparameterization_type=self._distribution.reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=self._distribution.allow_nan_stats,
          parameters=parameters,
          name=name)

  @property
  def distribution(self):
    """Base distribution, p(x)."""
    return self._distribution

  @property
  def bijector(self):
    """Function transforming x => y."""
    return self._bijector

  @property
  def experimental_is_sharded(self):
    raise NotImplementedError  # TODO(b/175084455): Handle bijector sharding.

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        distribution=parameter_properties.BatchedComponentProperties(),
        bijector=parameter_properties.BatchedComponentProperties(
            event_ndims=lambda td: tf.nest.map_structure(  # pylint: disable=g-long-lambda
                tensorshape_util.rank, td.distribution.event_shape),
            event_ndims_tensor=lambda td: tf.nest.map_structure(  # pylint: disable=g-long-lambda
                ps.rank_from_shape, td.distribution.event_shape_tensor())))

  def _event_shape_tensor(self):
    return self.bijector.forward_event_shape_tensor(
        self.distribution.event_shape_tensor())

  def _event_shape(self):
    # Since the `bijector` may change the `event_shape`, we then forward what we
    # know to the bijector. This allows the `bijector` to have final say in the
    # `event_shape`.
    return self.bijector.forward_event_shape(self.distribution.event_shape)

  def _batch_shape_tensor(self):
    base_batch_shape_tensor = self.distribution.batch_shape_tensor()
    if tf.nest.is_nested(base_batch_shape_tensor) and self._is_joint:
      # Pass-through rudimentary support for JDs with structured batch shape.
      # TODO(b/194742372): remove support for structured batch shape.
      return tf.nest.pack_sequence_as(
          self.dtype, tf.nest.flatten(base_batch_shape_tensor))
    return super()._batch_shape_tensor()

  def _batch_shape(self):
    batch_shape = self.distribution.batch_shape
    if tf.nest.is_nested(batch_shape) and self._is_joint:
      # Pass-through rudimentary support for JDs with structured batch shape.
      # TODO(b/194742372): remove support for structured batch shape.
      return tf.nest.pack_sequence_as(
          self.dtype, tf.nest.flatten(batch_shape))
    return super()._batch_shape()

  def _maybe_broadcast_distribution_batch_shape(self):
    """Returns the base distribution broadcast to the TD's full batch shape."""
    distribution_batch_shape = self.distribution.batch_shape
    if (tf.nest.is_nested(distribution_batch_shape) or
        tf.nest.is_nested(self.distribution.dtype)):
      # TODO(b/191674464): Support joint distributions in BatchBroadcast.
      return self.distribution

    overall_batch_shape = self.batch_shape
    if (tensorshape_util.is_fully_defined(overall_batch_shape) and
        distribution_batch_shape == overall_batch_shape):
      # No need to broadcast if the distribution already has full batch shape.
      return self.distribution

    if not tensorshape_util.is_fully_defined(overall_batch_shape):
      overall_batch_shape = self.batch_shape_tensor()
    return batch_broadcast.BatchBroadcast(
        self.distribution, with_shape=overall_batch_shape)

  def _call_sample_n(self, sample_shape, seed, **kwargs):
    # We override `_call_sample_n` rather than `_sample_n` so we can ensure that
    # the result of `self.bijector.forward` is not modified (and thus caching
    # works).
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)

    # First, generate samples from the base distribution.
    x = self._maybe_broadcast_distribution_batch_shape().sample(
        sample_shape=sample_shape, seed=seed, **distribution_kwargs)
    # Apply the bijector's forward transformation. For caching to
    # work, it is imperative that this is the last modification to the
    # returned result.
    return self.bijector.forward(x, **bijector_kwargs)

  def _sample_and_log_prob(self, sample_shape, seed, **kwargs):
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      # Computing log_prob with a non-injective bijector requires an explicit
      # inverse to get all points in the inverse image, so we can't get by
      # with just doing the forward pass.
      return super()._sample_and_log_prob(sample_shape, seed=seed, **kwargs)

    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x, base_distribution_log_prob = (
        self._maybe_broadcast_distribution_batch_shape(
            ).experimental_sample_and_log_prob(
                sample_shape, seed, **distribution_kwargs))
    y = self.bijector.forward(x, **bijector_kwargs)
    fldj = self.bijector.forward_log_det_jacobian(
        x,
        event_ndims=tf.nest.map_structure(
            ps.rank_from_shape,
            self.distribution.event_shape_tensor()),
        **bijector_kwargs)
    return y, (base_distribution_log_prob -
               tf.cast(fldj, base_distribution_log_prob.dtype))

  def _log_prob(self, y, **kwargs):
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)

    # For caching to work, it is imperative that the bijector is the first to
    # modify the input.
    x = self.bijector.inverse(y, **bijector_kwargs)
    event_ndims = tf.nest.map_structure(
        ps.rank_from_shape,
        self._event_shape_tensor(),
        self.event_shape)

    ildj = self.bijector.inverse_log_det_jacobian(
        y, event_ndims=event_ndims, **bijector_kwargs)
    if self.bijector._is_injective:  # pylint: disable=protected-access
      base_log_prob = self.distribution.log_prob(x, **distribution_kwargs)
      return base_log_prob + tf.cast(ildj, base_log_prob.dtype)

    # Compute log_prob on each element of the inverse image.
    lp_on_fibers = []
    for x_i, ildj_i in zip(x, ildj):
      base_log_prob = self.distribution.log_prob(x_i, **distribution_kwargs)
      lp_on_fibers.append(base_log_prob + tf.cast(ildj_i, base_log_prob.dtype))
    return tf.reduce_logsumexp(tf.stack(lp_on_fibers), axis=0)

  def _prob(self, y, **kwargs):
    if not hasattr(self.distribution, '_prob'):
      return tf.exp(self._log_prob(y, **kwargs))
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)

    x = self.bijector.inverse(y, **bijector_kwargs)
    event_ndims = tf.nest.map_structure(
        ps.rank_from_shape,
        self._event_shape_tensor(),
        self.event_shape
        )
    ildj = self.bijector.inverse_log_det_jacobian(
        y, event_ndims=event_ndims, **bijector_kwargs)
    if self.bijector._is_injective:  # pylint: disable=protected-access
      base_prob = self.distribution.prob(x, **distribution_kwargs)
      return base_prob * tf.exp(tf.cast(ildj, base_prob.dtype))

    # Compute prob on each element of the inverse image.
    prob_on_fibers = []
    for x_i, ildj_i in zip(x, ildj):
      base_prob = self.distribution.prob(x_i, **distribution_kwargs)
      prob_on_fibers.append(
          base_prob * tf.exp(tf.cast(ildj_i, base_prob.dtype)))
    return sum(prob_on_fibers)

  def _log_cdf(self, y, **kwargs):
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`log_cdf` is not implemented when '
                                '`bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    dist = self.distribution
    # TODO(b/141130733): Check/fix any gradient numerics issues.
    return ps.smart_where(
        self.bijector._internal_is_increasing(**bijector_kwargs),  # pylint: disable=protected-access
        lambda: dist.log_cdf(x, **distribution_kwargs),
        lambda: dist.log_survival_function(x, **distribution_kwargs))

  def _cdf(self, y, **kwargs):
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`cdf` is not implemented when '
                                '`bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    # TODO(b/141130733): Check/fix any gradient numerics issues.
    return ps.smart_where(
        self.bijector._internal_is_increasing(**bijector_kwargs),  # pylint: disable=protected-access
        lambda: self.distribution.cdf(x, **distribution_kwargs),
        lambda: self.distribution.survival_function(x, **distribution_kwargs))

  def _log_survival_function(self, y, **kwargs):
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`log_survival_function` is not implemented '
                                'when `bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    dist = self.distribution
    # TODO(b/141130733): Check/fix any gradient numerics issues.
    return ps.smart_where(
        self.bijector._internal_is_increasing(**bijector_kwargs),  # pylint: disable=protected-access
        lambda: dist.log_survival_function(x, **distribution_kwargs),
        lambda: dist.log_cdf(x, **distribution_kwargs))

  def _survival_function(self, y, **kwargs):
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`survival_function` is not implemented when '
                                '`bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    # TODO(b/141130733): Check/fix any gradient numerics issues.
    return ps.smart_where(
        self.bijector._internal_is_increasing(**bijector_kwargs),  # pylint: disable=protected-access
        lambda: self.distribution.survival_function(x, **distribution_kwargs),
        lambda: self.distribution.cdf(x, **distribution_kwargs))

  def _quantile(self, value, **kwargs):
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`quantile` is not implemented when '
                                '`bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    value = ps.smart_where(
        self.bijector._internal_is_increasing(**bijector_kwargs),  # pylint: disable=protected-access
        lambda: value,
        lambda: 1 - value)
    # x_q is the "qth quantile" of X iff q = P[X <= x_q].  Now, since X =
    # g^{-1}(Y), q = P[X <= x_q] = P[g^{-1}(Y) <= x_q] = P[Y <= g(x_q)],
    # implies the qth quantile of Y is g(x_q).
    inv_cdf = self.distribution.quantile(value, **distribution_kwargs)
    return self.bijector.forward(inv_cdf, **bijector_kwargs)

  def _mode(self, **kwargs):
    return self._mean_mode_impl('mode', kwargs)

  def _mean(self, **kwargs):
    return self._mean_mode_impl('mean', kwargs)

  def _mean_mode_impl(self, attr, kwargs):
    if not self.bijector.is_constant_jacobian:
      raise NotImplementedError(
          f'`{attr}` is not implemented for non-affine `bijectors`.')

    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = getattr(self.distribution, attr)(**distribution_kwargs)
    y = self.bijector.forward(x, **bijector_kwargs)

    sample_shape = tf.convert_to_tensor([], dtype=tf.int32, name='sample_shape')
    y = self._set_sample_static_shape(y, sample_shape)
    return y

  def _stddev(self, **kwargs):
    if not self.bijector.is_constant_jacobian:
      raise NotImplementedError('`stddev` is not implemented for non-affine '
                                '`bijectors`.')
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`stddev` is not implemented when '
                                '`bijector` is not injective.')
    if not (self.bijector._is_scalar  # pylint: disable=protected-access
            or self.bijector._is_permutation):  # pylint: disable=protected-access
      raise NotImplementedError('`stddev` is not implemented when `bijector` '
                                'is a multivariate transformation.')

    # A scalar affine bijector is of the form `forward(x) = scale * x + shift`,
    # where the standard deviation is invariant to the shift, so we extract the
    # shift and subtract it.
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x_stddev = self.distribution.stddev(**distribution_kwargs)
    y_stddev_plus_shift = self.bijector.forward(x_stddev, **bijector_kwargs)
    shift = self.bijector.forward(
        tf.nest.map_structure(
            tf.zeros_like, x_stddev),
        **bijector_kwargs)
    return tf.nest.map_structure(
        tf.abs,
        tf.nest.map_structure(tf.subtract, y_stddev_plus_shift, shift))

  def _covariance(self, **kwargs):
    if not self.bijector.is_constant_jacobian:
      raise NotImplementedError(
          '`covariance` is not implemented for non-affine `bijectors`.')
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError(
          '`covariance` is not implemented when `bijector` is not injective.')
    if (tf.nest.is_nested(self.bijector.forward_min_event_ndims) or
        self.bijector.forward_event_ndims(1) != 1):
      raise NotImplementedError(
          '`covariance` is only implemented when `bijector` takes vector '
          'inputs and produces vector outputs.')

    # An affine bijector is of the form `forward(x) = scale @ x + shift`,
    # where the covariance is invariant to the shift, so we extract the
    # shift and subtract it.
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    cov = self.distribution.covariance(**distribution_kwargs)
    zero_vector = tf.zeros_like(cov[..., 0])
    shift = self.bijector.forward(zero_vector, **bijector_kwargs)

    if shift is zero_vector:  # Short-circuit if bijector is tfb.Identity.
      return cov

    # Broadcast `cov` to full batch rank so we can treat its rows as an
    # additional batch dim. Note that we can't just call `forward(cov)` directly
    # because the user presumably lined up the bijector batch dimensions to work
    # when transforming vectors, not matrices.
    cov = tf.broadcast_to(
        cov, ps.broadcast_shape(ps.shape(cov),
                                ps.concat([ps.ones_like(ps.shape(shift)),
                                           [1]], axis=0)))
    ndims = ps.rank(cov)
    cov_rows = dist_util.move_dimension(  # No-op if cov has no batch dims.
        cov, source_idx=-2, dest_idx=0)
    tmp = self.bijector.forward(  # scale @ transpose(cov).
        cov_rows, **bijector_kwargs) - shift
    # Swap leftmost batch dim (rows) with event dim (columns).
    tmp_transpose = tf.transpose(  # cov @ transpose(scale).
        tmp, perm=ps.concat([[ndims - 1], ps.range(1, ndims - 1), [0]], axis=0))
    result_rows = self.bijector.forward(  # scale @ cov @ transpose(scale).
        tmp_transpose, **bijector_kwargs) - shift
    return dist_util.move_dimension(  # No-op if result has no batch dims.
        result_rows, source_idx=0, dest_idx=-2)

  def _entropy(self, **kwargs):
    if not self.bijector.is_constant_jacobian:
      raise NotImplementedError('`entropy` is not implemented.')
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`entropy` is not implemented when '
                                '`bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    # Suppose Y = g(X) where g is a diffeomorphism and X is a continuous rv. It
    # can be shown that:
    #   H[Y] = H[X] + E_X[(log o abs o det o J o g)(X)].
    # If is_constant_jacobian then:
    #   E_X[(log o abs o det o J o g)(X)] = (log o abs o det o J o g)(c)
    # where c can by anything.
    entropy = self.distribution.entropy(**distribution_kwargs)

    # Create a dummy event of zeros to pass to
    # `bijector.inverse_log_det_jacobian` to extract the constant Jacobian.
    event_shape_tensor = self._event_shape_tensor()
    event_ndims = tf.nest.map_structure(
        ps.rank_from_shape,
        event_shape_tensor, self.event_shape)
    dummy = tf.nest.map_structure(
        ps.zeros, event_shape_tensor, self.dtype)

    ildj = self.bijector.inverse_log_det_jacobian(
        dummy, event_ndims=event_ndims, **bijector_kwargs)

    entropy = entropy - tf.cast(ildj, entropy.dtype)
    tensorshape_util.set_shape(entropy, self.batch_shape)
    return entropy

  # pylint: disable=not-callable
  def _default_event_space_bijector(self):
    if self.distribution.experimental_default_event_space_bijector() is None:
      return None
    return self.bijector(
        self.distribution.experimental_default_event_space_bijector())
  # pylint: enable=not-callable


class TransformedDistribution(
    _TransformedDistribution, distribution_lib.AutoCompositeTensorDistribution):

  def __new__(cls, *args, **kwargs):
    """Maybe return a non-`CompositeTensor` `_TransformedDistribution`."""

    if cls is TransformedDistribution:
      if args:
        distribution = args[0]
      else:
        distribution = kwargs.get('distribution')
      if len(args) > 1:
        bijector = args[1]
      else:
        bijector = kwargs.get('bijector')

      if not (isinstance(distribution, tf.__internal__.CompositeTensor)
              and isinstance(bijector, tf.__internal__.CompositeTensor)):
        return _TransformedDistribution(*args, **kwargs)
    return super(TransformedDistribution, cls).__new__(cls)


TransformedDistribution.__doc__ = _TransformedDistribution.__doc__ + '\n' + (
    'If both `distribution` and `bijector` are `CompositeTensor`s, then the '
    'resulting `TransformedDistribution` instance is a `CompositeTensor` as '
    'well. Otherwise, a non-`CompositeTensor` `_TransformedDistribution` '
    'instance is created instead. Distribution subclasses that inherit from '
    '`TransformedDistribution` will also inherit from `CompositeTensor`.')


@kullback_leibler.RegisterKL(
    _TransformedDistribution, _TransformedDistribution)
def _kl_transformed_transformed(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Transformed.

  Args:
    a: instance of a TransformedDistribution object.
    b: instance of a TransformedDistribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e., `'kl_normal_normal'`).

  Returns:
    kl_div: Batchwise KL(a || b)

  Raises:
    NotImplementedError: If `a.bijector != b.bijector`.
  """
  with tf.name_scope(name or 'kl_transformed_transformed'):
    if a.bijector == b.bijector:
      return kullback_leibler.kl_divergence(a.distribution, b.distribution)
  raise NotImplementedError(
      'Unable to calculate KL divergence between {} and {} because '
      'their bijectors are not equal: {} vs. {}'.format(
          a, b, a.bijector, b.bijector))


@log_prob_ratio.RegisterLogProbRatio(TransformedDistribution)
def _transformed_log_prob_ratio(p, x, q, y, name=None):
  """Computes p.log_prob(x) - q.log_prob(y) for p and q both TDs."""
  with tf.name_scope(name or 'transformed_log_prob_ratio'):
    x_ = p.bijector.inverse(x)
    y_ = q.bijector.inverse(y)

    base_log_prob_ratio = log_prob_ratio.log_prob_ratio(
        p.distribution, x_, q.distribution, y_)

    event_ndims = tf.nest.map_structure(
        ps.rank_from_shape,
        p.event_shape_tensor,
        tf.nest.map_structure(tensorshape_util.merge_with,
                              p.event_shape, q.event_shape))
    ildj_ratio = ldj_ratio.inverse_log_det_jacobian_ratio(
        p.bijector, x, q.bijector, y, event_ndims)
    return base_log_prob_ratio + tf.cast(ildj_ratio, base_log_prob_ratio.dtype)

