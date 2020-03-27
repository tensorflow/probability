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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'TransformedDistribution',
]


def _default_kwargs_split_fn(kwargs):
  """Default `kwargs` `dict` getter."""
  return (kwargs.get('distribution_kwargs', {}),
          kwargs.get('bijector_kwargs', {}))


def _pick_scalar_condition(pred, cond_true, cond_false):
  """Convenience function which chooses the condition based on the predicate."""
  # Note: This function is only valid if all of pred, cond_true, and cond_false
  # are scalars. This means its semantics are arguably more like tf.cond than
  # tf.where even though we use tf.where to implement it.
  pred_ = tf.get_static_value(tf.convert_to_tensor(pred))
  if pred_ is None:
    return tf.where(pred, cond_true, cond_false)
  return cond_true if pred_ else cond_false


class TransformedDistribution(distribution_lib.Distribution):
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
    bijector=tfb.Affine(
      shift=-1.,
      scale_identity_multiplier=2.)
    name='NormalTransformedDistribution')
  ```

  A `TransformedDistribution`'s batch- and event-shape are implied by the base
  distribution unless explicitly overridden by `batch_shape` or `event_shape`
  arguments. Specifying an overriding `batch_shape` (`event_shape`) is
  permitted only if the base distribution has scalar batch-shape (event-shape).
  The bijector is applied to the distribution as if the distribution possessed
  the overridden shape(s). The following example demonstrates how to construct a
  multivariate Normal as a `TransformedDistribution`.

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
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.Affine(shift=mean, scale_tril=chol_cov),
      batch_shape=[2],  # Valid because base_distribution.batch_shape == [].
      event_shape=[2])  # Valid because base_distribution.event_shape == [].
  mvn2 = ds.MultivariateNormalTriL(loc=mean, scale_tril=chol_cov)
  # mvn1.log_prob(x) == mvn2.log_prob(x)
  ```

  """

  @deprecation.deprecated_args(
      '2020-06-01', '`batch_shape` and `event_shape` args are deprecated. '
      'Please use `tfd.Sample`, `tfd.Independent`, and broadcasted parameters '
      'of the base distribution instead. For example, replace '
      '`tfd.TransformedDistribution(tfd.Normal(0., 1.), tfb.Exp(), '
      'batch_shape=[2, 3], event_shape=[4])` with '
      '`tfd.TransformedDistrbution(tfd.Sample(tfd.Normal(tf.zeros([2, 3]), 1.),'
      'sample_shape=[4]), tfb.Exp())` or '
      '`tfd.TransformedDistribution(tfd.Independent(tfd.Normal('
      'tf.zeros([2, 3, 4]), 1.), reinterpreted_batch_ndims=1), tfb.Exp())`.',
      'batch_shape', 'event_shape')
  def __init__(self,
               distribution,
               bijector,
               batch_shape=None,
               event_shape=None,
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
      batch_shape: `integer` vector `Tensor` which overrides `distribution`
        `batch_shape`; valid only if `distribution.is_scalar_batch()`.
      event_shape: `integer` vector `Tensor` which overrides `distribution`
        `event_shape`; valid only if `distribution.is_scalar_event()`.
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

    Raises:
      ValueError: If `distribution` is a joint distribution and a `batch_shape`
        override is passed.
      ValueError: If `distribution` is a joint distribution and an `event_shape`
        override is passed.
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
      self._empty = tf.constant([], dtype=tf.int32, name='empty')

      # We don't just want to check isinstance(JointDistribution) because
      # TransformedDistributions with multipart bijectors are effectively
      # joint but don't inherit from JD. The 'duck-type' test is that
      # JDs have a structured dtype.
      self._base_is_joint = tf.nest.is_nested(self.distribution.dtype)
      if self._base_is_joint:
        if batch_shape:
          raise ValueError('Overriding the batch shape of a joint distribution'
                           ' ({}) is not currently supported.'.format(
                               self.distribution))
        if event_shape:
          raise ValueError('Overriding the event shape of a joint distribution'
                           ' ({}) is not currently supported.'.format(
                               self.distribution))

      override_batch_shape = self._empty if batch_shape is None else batch_shape
      self._override_batch_shape = tensor_util.convert_nonref_to_tensor(
          override_batch_shape, dtype=tf.int32, name='override_batch_shape')

      override_event_shape = self._empty if event_shape is None else event_shape
      self._override_event_shape = tensor_util.convert_nonref_to_tensor(
          override_event_shape, dtype=tf.int32, name='override_event_shape')

      # `_is_maybe_{batch, event}_override` is False if we know statically that
      # the batch/event shape is not being overridden; otherwise it is True.
      self._is_maybe_event_override = tensorshape_util.dims(
          self._override_event_shape.shape) != [0]
      self._is_maybe_batch_override = tensorshape_util.dims(
          self._override_batch_shape.shape) != [0]

      dtype = self.bijector.forward_dtype(self.distribution.dtype)
      self._is_joint = tf.nest.is_nested(dtype)

      super(TransformedDistribution, self).__init__(
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

  def __getitem__(self, slices):
    # Because slicing is parameterization-dependent, we only implement slicing
    # for instances of TD, not subclasses thereof.
    if type(self) is not TransformedDistribution:  # pylint: disable=unidiomatic-typecheck
      return super(TransformedDistribution, self).__getitem__(slices)

    if tensorshape_util.rank(self.distribution.batch_shape) is None:
      raise NotImplementedError(
          'Slicing TransformedDistribution with underlying distribution of '
          'unknown rank is not yet implemented.')
    overrides = {}
    if (tensorshape_util.rank(self.distribution.batch_shape) == 0 and
        self.parameters.get('batch_shape', None) is not None):
      overrides['batch_shape'] = tf.shape(
          tf.zeros(self.parameters['batch_shape'])[slices])
    elif self.parameters.get('distribution', None) is not None:
      overrides['distribution'] = self.distribution[slices]
    return self.copy(**overrides)

  def _event_shape_tensor(
      self, override_event_shape=None, base_event_shape_tensor=None):
    override_event_shape = (tf.convert_to_tensor(self._override_event_shape)
                            if override_event_shape is None
                            else override_event_shape)
    base_event_shape_tensor = (self.distribution.event_shape_tensor()
                               if base_event_shape_tensor is None
                               else base_event_shape_tensor)

    # If the base distribution is not joint, use the base event shape override,
    # if any.
    if not self._base_is_joint:
      base_event_shape_tensor = distribution_util.pick_vector(
          self._has_nonzero_rank(override_event_shape),
          override_event_shape,
          base_event_shape_tensor)
    return self.bijector.forward_event_shape_tensor(base_event_shape_tensor)

  def _event_shape(self):
    # If there's a chance that the event_shape has been overridden, we return
    # what we statically know about the `override_event_shape`. This works
    # because: `_is_maybe_event_override` means that the `constant_value()` of
    # `override_event_shape` is `None` or a non-empty list, i.e., we don't
    # statically know the `event_shape` or we do.
    #
    # Since the `bijector` may change the `event_shape`, we then forward what we
    # know to the bijector. This allows the `bijector` to have final say in the
    # `event_shape`.
    if self._is_maybe_event_override:
      shape = tensorshape_util.constant_value_as_shape(
          self._override_event_shape)
    else:
      shape = self.distribution.event_shape
    return self.bijector.forward_event_shape(shape)

  def _batch_shape_tensor(
      self, override_batch_shape=None, base_batch_shape_tensor=None):
    override_batch_shape = (tf.convert_to_tensor(self._override_batch_shape)
                            if override_batch_shape is None
                            else override_batch_shape)
    base_batch_shape_tensor = (self.distribution.batch_shape_tensor()
                               if base_batch_shape_tensor is None
                               else base_batch_shape_tensor)

    # The `batch_shape_tensor` of the transformed distribution is the same as
    # that of the base distribution in all cases except when the following are
    # both true:
    #   - the base distribution is joint with structured `batch_shape_tensor`
    #   - the transformed distribution is not joint.
    # In this case, the components of the base distribution's
    # `batch_shape_tensor` are broadcast to obtain the `batch_shape_tensor` of
    # the transformed distribution. Non-broadcasting components are not
    # supported. (Note that joint distributions may either have a single
    # `batch_shape_tensor` for all components, or a component-wise
    # `batch_shape_tensor` with the same nested structure as the distribution's
    # dtype.)
    if tf.nest.is_nested(base_batch_shape_tensor):
      if self._is_joint:
        return base_batch_shape_tensor

      base_batch_shape_tensor = functools.reduce(
          prefer_static.broadcast_shape,
          tf.nest.flatten(base_batch_shape_tensor))

    # If the batch shape has been overridden, return the override batch shape
    # instead.
    return distribution_util.pick_vector(
        self._has_nonzero_rank(override_batch_shape),
        override_batch_shape,
        base_batch_shape_tensor)

  def _batch_shape(self):
    # If there's a chance that the batch_shape has been overridden, we return
    # what we statically know about the `override_batch_shape`. This works
    # because: `_is_maybe_batch_override` means that the `constant_value()` of
    # `override_batch_shape` is `None` or a non-empty list, i.e., we don't
    # statically know the `batch_shape` or we do.
    #
    # Notice that this implementation parallels the `_event_shape` except that
    # the `bijector` doesn't get to alter the `batch_shape`. Recall that
    # `batch_shape` is a property of a distribution while `event_shape` is
    # shared between both the `distribution` instance and the `bijector`.
    if self._is_maybe_batch_override:
      return tensorshape_util.constant_value_as_shape(
          self._override_batch_shape)

    # As with `batch_shape_tensor`, if the base distribution is joint with
    # structured batch shape and the transformed distribution is not joint,
    # the batch shape components of the base distribution are broadcast to
    # obtain the batch shape of the transformed distribution.
    batch_shape = self.distribution.batch_shape
    if tf.nest.is_nested(batch_shape) and not self._is_joint:
      batch_shape = functools.reduce(
          tf.broadcast_static_shape, tf.nest.flatten(batch_shape))
    return batch_shape

  def _has_nonzero_rank(self, override_shape):
    return prefer_static.logical_not(
        prefer_static.equal(
            prefer_static.rank_from_shape(override_shape),
            self._zero))

  def _needs_rotation(
      self, override_event_shape, override_batch_shape, base_is_scalar_batch):
    # To convert a scalar distribution into a multivariate distribution we
    # will permute dims from the sample dims, which are otherwise iid. This is
    # easy to do except in the case that the base distribution has nonscalar
    # batch and we're overriding only event shape. Under these conditions, this
    # function returns `True`, indicating that event dims will incorrectly be to
    # the left of the batch dims and we'll need to cyclically permute left the
    # new dims (in `_maybe_rotate_dims`). If these conditions do not hold, this
    # function returns `False` and no rotation is needed.
    if self._base_is_joint:
      # `prefer_static` can't handle nested structures like
      # `base_is_scalar_batch` and shape overrides are not supported if the base
      # distribution is joint.
      return False
    return prefer_static.reduce_all([
        self._has_nonzero_rank(override_event_shape),
        prefer_static.logical_not(
            self._has_nonzero_rank(override_batch_shape)),
        prefer_static.logical_not(base_is_scalar_batch)])

  def _get_rotation_ndims(
      self, override_event_shape, override_batch_shape, base_is_scalar_batch):
    override_event_ndims = prefer_static.rank_from_shape(override_event_shape)
    return _pick_scalar_condition(
        self._needs_rotation(
            override_event_shape, override_batch_shape, base_is_scalar_batch),
        override_event_ndims, 0)

  def _reduce_event_indices(
      self, override_event_shape, override_batch_shape, base_is_scalar_batch):
    # We'll be reducing the leftmost dims (if at all), i.e., this will be []
    # if we don't need to reduce.
    rotate_ndims = self._get_rotation_ndims(
        override_event_shape, override_batch_shape, base_is_scalar_batch)
    override_event_ndims = prefer_static.rank_from_shape(override_event_shape)
    return prefer_static.range(
        rotate_ndims - override_event_ndims, rotate_ndims)

  def _sample_n(self, n, seed=None, **distribution_kwargs):
    override_event_shape = tf.convert_to_tensor(self._override_event_shape)
    override_batch_shape = tf.convert_to_tensor(self._override_batch_shape)
    base_is_scalar_batch = self.distribution.is_scalar_batch()

    needs_rotation = self._needs_rotation(
        override_event_shape, override_batch_shape, base_is_scalar_batch)
    sample_shape = prefer_static.concat([
        distribution_util.pick_vector(needs_rotation, self._empty, [n]),
        override_batch_shape,
        override_event_shape,
        distribution_util.pick_vector(needs_rotation, [n], self._empty),
    ], axis=0)
    x = self.distribution.sample(sample_shape=sample_shape, seed=seed,
                                 **distribution_kwargs)
    x = self._maybe_rotate_dims(
        x, override_event_shape, override_batch_shape, base_is_scalar_batch)
    # We'll apply the bijector in the `_call_sample_n` function.
    return x

  def _call_sample_n(self, sample_shape, seed, name, **kwargs):
    # We override `_call_sample_n` rather than `_sample_n` so we can ensure that
    # the result of `self.bijector.forward` is not modified (and thus caching
    # works).
    with self._name_and_control_scope(name):
      sample_shape = tf.convert_to_tensor(
          sample_shape, dtype=tf.int32, name='sample_shape')
      sample_shape, n = self._expand_sample_shape_to_vector(
          sample_shape, 'sample_shape')

      distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)

      # First, generate samples. We will possibly generate extra samples in the
      # event that we need to reinterpret the samples as part of the
      # event_shape.
      x = self._sample_n(n, seed, **distribution_kwargs)

      # Next, we reshape `x` into its final form. We do this prior to the call
      # to the bijector to ensure that the bijector caching works.
      def reshape_sample_shape(t):
        batch_event_shape = tf.shape(t)[1:]
        final_shape = tf.concat([sample_shape, batch_event_shape], 0)
        return tf.reshape(t, final_shape)
      x = tf.nest.map_structure(reshape_sample_shape, x)

      # Finally, we apply the bijector's forward transformation. For caching to
      # work, it is imperative that this is the last modification to the
      # returned result.
      y = self.bijector.forward(x, **bijector_kwargs)
      y = self._set_sample_static_shape(y, sample_shape)

      return y

  def _log_prob(self, y, **kwargs):
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    override_event_shape = tf.convert_to_tensor(self._override_event_shape)
    override_batch_shape = tf.convert_to_tensor(self._override_batch_shape)
    base_is_scalar_batch = self.distribution.is_scalar_batch()

    # For caching to work, it is imperative that the bijector is the first to
    # modify the input.
    x = self.bijector.inverse(y, **bijector_kwargs)
    event_ndims = tf.nest.map_structure(
        prefer_static.rank_from_shape,
        self._event_shape_tensor(override_event_shape=override_event_shape),
        self.event_shape)

    ildj = self.bijector.inverse_log_det_jacobian(
        y, event_ndims=event_ndims, **bijector_kwargs)
    if self.bijector._is_injective:  # pylint: disable=protected-access
      return self._finish_log_prob_for_one_fiber(
          y, x, ildj, event_ndims, override_event_shape, override_batch_shape,
          base_is_scalar_batch, **distribution_kwargs)

    lp_on_fibers = [
        self._finish_log_prob_for_one_fiber(  # pylint: disable=g-complex-comprehension
            y, x_i, ildj_i, event_ndims, override_event_shape,
            override_batch_shape, base_is_scalar_batch, **distribution_kwargs)
        for x_i, ildj_i in zip(x, ildj)]
    return tf.reduce_logsumexp(tf.stack(lp_on_fibers), axis=0)

  def _finish_log_prob_for_one_fiber(
      self, y, x, ildj, event_ndims, override_event_shape, override_batch_shape,
      base_is_scalar_batch, **distribution_kwargs):
    """Finish computation of log_prob on one element of the inverse image."""
    x = self._maybe_rotate_dims(
        x, override_event_shape, override_batch_shape, base_is_scalar_batch,
        rotate_right=True)
    log_prob = self.distribution.log_prob(x, **distribution_kwargs)
    if self._is_maybe_event_override:
      log_prob = tf.reduce_sum(
          log_prob, axis=self._reduce_event_indices(
              override_event_shape, override_batch_shape, base_is_scalar_batch))
    log_prob = log_prob + tf.cast(ildj, log_prob.dtype)
    if self._is_maybe_event_override and isinstance(event_ndims, int):
      tensorshape_util.set_shape(
          log_prob,
          tf.broadcast_static_shape(y.shape[:-event_ndims], self.batch_shape))
    return log_prob

  def _prob(self, y, **kwargs):
    override_event_shape = tf.convert_to_tensor(self._override_event_shape)
    override_batch_shape = tf.convert_to_tensor(self._override_batch_shape)
    base_is_scalar_batch = self.distribution.is_scalar_batch()
    if not hasattr(self.distribution, '_prob'):
      return tf.exp(self._log_prob(y, **kwargs))
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)

    x = self.bijector.inverse(y, **bijector_kwargs)
    event_ndims = tf.nest.map_structure(
        prefer_static.rank_from_shape,
        self._event_shape_tensor(override_event_shape=override_event_shape),
        self.event_shape
        )
    ildj = self.bijector.inverse_log_det_jacobian(
        y, event_ndims=event_ndims, **bijector_kwargs)
    if self.bijector._is_injective:  # pylint: disable=protected-access
      return self._finish_prob_for_one_fiber(
          y, x, ildj, event_ndims, override_event_shape, override_batch_shape,
          base_is_scalar_batch, **distribution_kwargs)

    prob_on_fibers = [
        self._finish_prob_for_one_fiber(  # pylint: disable=g-complex-comprehension
            y, x_i, ildj_i, event_ndims, override_event_shape,
            override_batch_shape, base_is_scalar_batch, **distribution_kwargs)
        for x_i, ildj_i in zip(x, ildj)]
    return sum(prob_on_fibers)

  def _finish_prob_for_one_fiber(
      self, y, x, ildj, event_ndims, override_event_shape, override_batch_shape,
      base_is_scalar_batch, **distribution_kwargs):
    """Finish computation of prob on one element of the inverse image."""
    x = self._maybe_rotate_dims(
        x, override_event_shape, override_batch_shape, base_is_scalar_batch,
        rotate_right=True)
    prob = self.distribution.prob(x, **distribution_kwargs)
    if self._is_maybe_event_override:
      prob = tf.reduce_prod(
          prob,
          axis=self._reduce_event_indices(
              override_event_shape, override_batch_shape, base_is_scalar_batch))
    prob = prob * tf.exp(tf.cast(ildj, prob.dtype))
    if self._is_maybe_event_override and isinstance(event_ndims, int):
      tensorshape_util.set_shape(
          prob,
          tf.broadcast_static_shape(y.shape[:-event_ndims], self.batch_shape))
    return prob

  def _log_cdf(self, y, **kwargs):
    if self._is_maybe_event_override:
      raise NotImplementedError('`log_cdf` is not implemented when overriding '
                                '`event_shape`.')
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`log_cdf` is not implemented when '
                                '`bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    dist = self.distribution
    # TODO(b/141130733): Check/fix any gradient numerics issues.
    return prefer_static.smart_where(
        self.bijector._internal_is_increasing(**bijector_kwargs),  # pylint: disable=protected-access
        lambda: dist.log_cdf(x, **distribution_kwargs),
        lambda: dist.log_survival_function(x, **distribution_kwargs))

  def _cdf(self, y, **kwargs):
    if self._is_maybe_event_override:
      raise NotImplementedError('`cdf` is not implemented when overriding '
                                '`event_shape`.')
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`cdf` is not implemented when '
                                '`bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    # TODO(b/141130733): Check/fix any gradient numerics issues.
    return prefer_static.smart_where(
        self.bijector._internal_is_increasing(**bijector_kwargs),  # pylint: disable=protected-access
        lambda: self.distribution.cdf(x, **distribution_kwargs),
        lambda: self.distribution.survival_function(x, **distribution_kwargs))

  def _log_survival_function(self, y, **kwargs):
    if self._is_maybe_event_override:
      raise NotImplementedError('`log_survival_function` is not implemented '
                                'when overriding `event_shape`.')
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`log_survival_function` is not implemented '
                                'when `bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    dist = self.distribution
    # TODO(b/141130733): Check/fix any gradient numerics issues.
    return prefer_static.smart_where(
        self.bijector._internal_is_increasing(**bijector_kwargs),  # pylint: disable=protected-access
        lambda: dist.log_survival_function(x, **distribution_kwargs),
        lambda: dist.log_cdf(x, **distribution_kwargs))

  def _survival_function(self, y, **kwargs):
    if self._is_maybe_event_override:
      raise NotImplementedError('`survival_function` is not implemented when '
                                'overriding `event_shape`.')
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`survival_function` is not implemented when '
                                '`bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    # TODO(b/141130733): Check/fix any gradient numerics issues.
    return prefer_static.smart_where(
        self.bijector._internal_is_increasing(**bijector_kwargs),  # pylint: disable=protected-access
        lambda: self.distribution.survival_function(x, **distribution_kwargs),
        lambda: self.distribution.cdf(x, **distribution_kwargs))

  def _quantile(self, value, **kwargs):
    if self._is_maybe_event_override:
      raise NotImplementedError('`quantile` is not implemented when overriding '
                                '`event_shape`.')
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`quantile` is not implemented when '
                                '`bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    value = prefer_static.smart_where(
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
      raise NotImplementedError('`mean` is not implemented for non-affine '
                                '`bijectors`.')

    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = getattr(self.distribution, attr)(**distribution_kwargs)

    if self._is_maybe_batch_override or self._is_maybe_event_override:
      override_event_shape = tf.convert_to_tensor(self._override_event_shape)
      override_batch_shape = tf.convert_to_tensor(self._override_batch_shape)
      base_batch_shape_tensor = self.distribution.batch_shape_tensor()
      base_event_shape_tensor = self.distribution.event_shape_tensor()

      # A batch (respectively event) shape override is only allowed if the batch
      # (event) shape of the base distribution is [], so concatenating all the
      # shapes does the right thing.
      new_shape = prefer_static.concat([
          prefer_static.ones_like(override_batch_shape),
          base_batch_shape_tensor,
          prefer_static.ones_like(override_event_shape),
          base_event_shape_tensor,
      ], 0)
      x = tf.reshape(x, new_shape)
      new_shape = prefer_static.concat(
          [self._batch_shape_tensor(
              override_batch_shape, base_batch_shape_tensor),
           self._event_shape_tensor(
               override_event_shape, base_event_shape_tensor)],
          0)
      x = tf.broadcast_to(x, new_shape)

    y = self.bijector.forward(x, **bijector_kwargs)

    sample_shape = tf.convert_to_tensor([], dtype=tf.int32, name='sample_shape')
    y = self._set_sample_static_shape(y, sample_shape)
    return y

  def _entropy(self, **kwargs):
    if not self.bijector.is_constant_jacobian:
      raise NotImplementedError('`entropy` is not implemented.')
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError('`entropy` is not implemented when '
                                '`bijector` is not injective.')
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    override_event_shape = tf.convert_to_tensor(self._override_event_shape)
    override_batch_shape = tf.convert_to_tensor(self._override_batch_shape)
    base_batch_shape_tensor = self.distribution.batch_shape_tensor()
    base_event_shape_tensor = self.distribution.event_shape_tensor()
    # Suppose Y = g(X) where g is a diffeomorphism and X is a continuous rv. It
    # can be shown that:
    #   H[Y] = H[X] + E_X[(log o abs o det o J o g)(X)].
    # If is_constant_jacobian then:
    #   E_X[(log o abs o det o J o g)(X)] = (log o abs o det o J o g)(c)
    # where c can by anything.
    entropy = self.distribution.entropy(**distribution_kwargs)
    if self._is_maybe_event_override:
      # H[X] = sum_i H[X_i] if X_i are mutually independent.
      # This means that a reduce_sum is a simple rescaling.
      entropy = entropy * tf.cast(
          tf.reduce_prod(override_event_shape),
          dtype=dtype_util.base_dtype(entropy.dtype))
    if self._is_maybe_batch_override:
      new_shape = tf.concat([
          prefer_static.ones_like(override_batch_shape),
          base_batch_shape_tensor
      ], 0)
      entropy = tf.reshape(entropy, new_shape)
      multiples = tf.concat([
          override_batch_shape,
          prefer_static.ones_like(base_batch_shape_tensor)
      ], 0)
      entropy = tf.tile(entropy, multiples)

    # Create a dummy event of zeros to pass to
    # `bijector.inverse_log_det_jacobian` to extract the constant Jacobian.
    event_shape_tensor = self._event_shape_tensor(
        override_event_shape, base_event_shape_tensor)
    event_ndims = tf.nest.map_structure(
        prefer_static.rank_from_shape,
        event_shape_tensor, self.event_shape)
    dummy = tf.nest.map_structure(
        prefer_static.zeros, event_shape_tensor, self.dtype)

    ildj = self.bijector.inverse_log_det_jacobian(
        dummy, event_ndims=event_ndims, **bijector_kwargs)

    entropy = entropy - tf.cast(ildj, entropy.dtype)
    tensorshape_util.set_shape(entropy, self.batch_shape)
    return entropy

  def _maybe_rotate_dims(
      self, x, override_event_shape, override_batch_shape, base_is_scalar_batch,
      rotate_right=False):
    """Helper which rolls left event_dims left or right event_dims right."""
    needs_rotation_const = tf.get_static_value(
        self._needs_rotation(
            override_event_shape, override_batch_shape, base_is_scalar_batch))
    if needs_rotation_const is not None and not needs_rotation_const:
      return x
    ndims = prefer_static.rank(x)
    rotate_ndims = self._get_rotation_ndims(
        override_event_shape, override_batch_shape, base_is_scalar_batch)
    n = (ndims - rotate_ndims) if rotate_right else rotate_ndims
    perm = prefer_static.concat([
        prefer_static.range(n, ndims), prefer_static.range(0, n)], axis=0)
    return tf.transpose(a=x, perm=perm)

  def _maybe_validate_shape_override(
      self, override_shape, base_is_scalar_fn, static_base_shape, is_init):
    """Helper which ensures override batch/event_shape are valid."""

    assertions = []
    concretized_shape = None

    # Check valid dtype
    if is_init:  # No xor check because `dtype` cannot change.
      dtype_ = override_shape.dtype
      if dtype_ is None:
        if concretized_shape is None:
          concretized_shape = tf.convert_to_tensor(override_shape)
        dtype_ = concretized_shape.dtype
      if dtype_util.base_dtype(dtype_) not in {tf.int32, tf.int64}:
        raise TypeError('Shape override must be integer type; '
                        'saw {}.'.format(dtype_util.name(dtype_)))

    # Check non-negative elements
    if is_init != tensor_util.is_ref(override_shape):
      override_shape_ = tf.get_static_value(override_shape)
      msg = 'Shape override must have non-negative elements.'
      if override_shape_ is not None:
        if np.any(np.array(override_shape_) < 0):
          raise ValueError('{} Saw: {}'.format(msg, override_shape_))
      elif self.validate_args:
        if concretized_shape is None:
          concretized_shape = tf.convert_to_tensor(override_shape)
        assertions.append(
            assert_util.assert_non_negative(concretized_shape, message=msg))

    # Check valid shape
    override_ndims_ = tensorshape_util.rank(override_shape.shape)
    if is_init != (override_ndims_ is None):
      msg = 'Shape override must be a vector.'
      if override_ndims_ is not None:
        if override_ndims_ != 1:
          raise ValueError(msg)
      elif self.validate_args:
        if concretized_shape is None:
          concretized_shape = tf.convert_to_tensor(override_shape)
        override_rank = tf.rank(concretized_shape)
        assertions.append(
            assert_util.assert_equal(override_rank, 1, message=msg))

    static_base_rank = tf.nest.map_structure(
        tensorshape_util.rank, static_base_shape)

    # Determine if the override shape is `[]` (static_override_dims == [0]),
    # in which case the base distribution may be nonscalar.
    static_override_dims = tensorshape_util.dims(override_shape.shape)

    if is_init != (static_base_rank is None or static_override_dims is None):
      msg = 'Base distribution is not scalar.'
      if static_base_rank is not None and static_override_dims is not None:
        if static_base_rank != 0 and static_override_dims != [0]:
          raise ValueError(msg)
      elif self.validate_args:
        if concretized_shape is None:
          concretized_shape = tf.convert_to_tensor(override_shape)
        override_is_empty = tf.logical_not(
            self._has_nonzero_rank(concretized_shape))
        assertions.append(
            assert_util.assert_equal(
                tf.logical_or(base_is_scalar_fn(), override_is_empty),
                True, message=msg))
    return assertions

  # pylint: disable=protected-access, not-callable
  def _default_event_space_bijector(self):
    if self.distribution._experimental_default_event_space_bijector() is None:
      return None
    return self.bijector(
        self.distribution._experimental_default_event_space_bijector())
  # pylint: enable=protected-access, not-callable

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    assertions.extend(self._maybe_validate_shape_override(
        self._override_batch_shape,
        self.distribution.is_scalar_batch,
        self.distribution.batch_shape,
        is_init))

    assertions.extend(self._maybe_validate_shape_override(
        self._override_event_shape,
        self.distribution.is_scalar_event,
        self.distribution.event_shape,
        is_init))

    return assertions
