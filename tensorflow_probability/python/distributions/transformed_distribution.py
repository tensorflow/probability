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

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    "ConditionalTransformedDistribution",
    "TransformedDistribution",
]


# The following helper functions attempt to statically perform a TF operation.
# These functions make debugging easier since we can do more validation during
# graph construction.


def _pick_scalar_condition(pred, cond_true, cond_false):
  """Convenience function which chooses the condition based on the predicate."""
  # Note: This function is only valid if all of pred, cond_true, and cond_false
  # are scalars. This means its semantics are arguably more like tf.cond than
  # tf.where even though we use tf.where to implement it.
  pred_ = tf.get_static_value(tf.convert_to_tensor(pred))
  if pred_ is None:
    return tf.where(pred, cond_true, cond_false)
  return cond_true if pred_ else cond_false


def _is_scalar_from_shape_tensor(shape):
  """Returns `True` `Tensor` if `Tensor` shape implies a scalar."""
  return prefer_static.equal(prefer_static.rank_from_shape(shape), 0)


def _default_kwargs_split_fn(kwargs):
  """Default `kwargs` `dict` getter."""
  return (kwargs.get("distribution_kwargs", {}),
          kwargs.get("bijector_kwargs", {}))


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
  ds = tfp.distributions
  log_normal = ds.TransformedDistribution(
    distribution=ds.Normal(loc=0., scale=1.),
    bijector=ds.bijectors.Exp(),
    name="LogNormalTransformedDistribution")
  ```

  A `LogNormal` made from callables:

  ```python
  ds = tfp.distributions
  log_normal = ds.TransformedDistribution(
    distribution=ds.Normal(loc=0., scale=1.),
    bijector=ds.bijectors.Inline(
      forward_fn=tf.exp,
      inverse_fn=tf.log,
      inverse_log_det_jacobian_fn=(
        lambda y: -tf.reduce_sum(tf.log(y), axis=-1)),
    name="LogNormalTransformedDistribution")
  ```

  Another example constructing a Normal from a StandardNormal:

  ```python
  ds = tfp.distributions
  normal = ds.TransformedDistribution(
    distribution=ds.Normal(loc=0., scale=1.),
    bijector=ds.bijectors.Affine(
      shift=-1.,
      scale_identity_multiplier=2.)
    name="NormalTransformedDistribution")
  ```

  A `TransformedDistribution`'s batch- and event-shape are implied by the base
  distribution unless explicitly overridden by `batch_shape` or `event_shape`
  arguments. Specifying an overriding `batch_shape` (`event_shape`) is
  permitted only if the base distribution has scalar batch-shape (event-shape).
  The bijector is applied to the distribution as if the distribution possessed
  the overridden shape(s). The following example demonstrates how to construct a
  multivariate Normal as a `TransformedDistribution`.

  ```python
  ds = tfp.distributions
  # We will create two MVNs with batch_shape = event_shape = 2.
  mean = [[-1., 0],      # batch:0
          [0., 1]]       # batch:1
  chol_cov = [[[1., 0],
               [0, 1]],  # batch:0
              [[1, 0],
               [2, 2]]]  # batch:1
  mvn1 = ds.TransformedDistribution(
      distribution=ds.Normal(loc=0., scale=1.),
      bijector=ds.bijectors.Affine(shift=mean, scale_tril=chol_cov),
      batch_shape=[2],  # Valid because base_distribution.batch_shape == [].
      event_shape=[2])  # Valid because base_distribution.event_shape == [].
  mvn2 = ds.MultivariateNormalTriL(loc=mean, scale_tril=chol_cov)
  # mvn1.log_prob(x) == mvn2.log_prob(x)
  ```

  """

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
    """
    parameters = dict(locals()) if parameters is None else parameters
    name = name or (("" if bijector is None else bijector.name) +
                    (distribution.name or ""))
    with tf.name_scope(name) as name:
      self._kwargs_split_fn = (_default_kwargs_split_fn
                               if kwargs_split_fn is None
                               else kwargs_split_fn)
      # For convenience we define some handy constants.
      self._zero = tf.constant(0, dtype=tf.int32, name="zero")
      self._empty = tf.constant([], dtype=tf.int32, name="empty")

      # We will keep track of a static and dynamic version of
      # self._is_{batch,event}_override. This way we can do more prior to graph
      # execution, including possibly raising Python exceptions.

      self._override_batch_shape = self._maybe_validate_shape_override(
          batch_shape, distribution.is_scalar_batch(), validate_args,
          "batch_shape")
      self._is_batch_override = prefer_static.logical_not(
          prefer_static.equal(
              prefer_static.rank_from_shape(self._override_batch_shape),
              self._zero))
      self._is_maybe_batch_override = bool(
          tf.get_static_value(self._override_batch_shape) is None or
          tf.get_static_value(self._override_batch_shape).size != 0)

      self._override_event_shape = self._maybe_validate_shape_override(
          event_shape, distribution.is_scalar_event(), validate_args,
          "event_shape")
      self._is_event_override = prefer_static.logical_not(
          prefer_static.equal(
              prefer_static.rank_from_shape(self._override_event_shape),
              self._zero))
      self._is_maybe_event_override = bool(
          tf.get_static_value(self._override_event_shape) is None or
          tf.get_static_value(self._override_event_shape).size != 0)

      # To convert a scalar distribution into a multivariate distribution we
      # will draw dims from the sample dims, which are otherwise iid. This is
      # easy to do except in the case that the base distribution has batch dims
      # and we're overriding event shape. When that case happens the event dims
      # will incorrectly be to the left of the batch dims. In this case we'll
      # cyclically permute left the new dims.
      self._needs_rotation = prefer_static.reduce_all([
          self._is_event_override,
          prefer_static.logical_not(self._is_batch_override),
          prefer_static.logical_not(distribution.is_scalar_batch())])
      override_event_ndims = prefer_static.rank_from_shape(
          self._override_event_shape)
      self._rotate_ndims = _pick_scalar_condition(
          self._needs_rotation, override_event_ndims, 0)
      # We'll be reducing the head dims (if at all), i.e., this will be []
      # if we don't need to reduce.
      self._reduce_event_indices = tf.range(
          self._rotate_ndims - override_event_ndims, self._rotate_ndims)

    self._distribution = distribution
    self._bijector = bijector
    super(TransformedDistribution, self).__init__(
        dtype=self._distribution.dtype,
        reparameterization_type=self._distribution.reparameterization_type,
        validate_args=validate_args,
        allow_nan_stats=self._distribution.allow_nan_stats,
        parameters=parameters,
        # We let TransformedDistribution access _graph_parents since this class
        # is more like a baseclass than derived.
        graph_parents=(distribution._graph_parents +  # pylint: disable=protected-access
                       bijector.graph_parents),
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
          "Slicing TransformedDistribution with underlying distribution of "
          "unknown rank is not yet implemented")
    overrides = {}
    if (tensorshape_util.rank(self.distribution.batch_shape) == 0 and
        self.parameters.get("batch_shape", None) is not None):
      overrides["batch_shape"] = tf.shape(
          tf.zeros(self.parameters["batch_shape"])[slices])
    elif self.parameters.get("distribution", None) is not None:
      overrides["distribution"] = self.distribution[slices]
    return self.copy(**overrides)

  def _event_shape_tensor(self):
    return self.bijector.forward_event_shape_tensor(
        distribution_util.pick_vector(
            self._is_event_override,
            self._override_event_shape,
            self.distribution.event_shape_tensor()))

  def _event_shape(self):
    # If there's a chance that the event_shape has been overridden, we return
    # what we statically know about the `event_shape_override`. This works
    # because: `_is_maybe_event_override` means `static_override` is `None` or a
    # non-empty list, i.e., we don't statically know the `event_shape` or we do.
    #
    # Since the `bijector` may change the `event_shape`, we then forward what we
    # know to the bijector. This allows the `bijector` to have final say in the
    # `event_shape`.
    static_override = tensorshape_util.constant_value_as_shape(
        self._override_event_shape)
    return self.bijector.forward_event_shape(
        static_override
        if self._is_maybe_event_override
        else self.distribution.event_shape)

  def _batch_shape_tensor(self):
    return distribution_util.pick_vector(
        self._is_batch_override,
        self._override_batch_shape,
        self.distribution.batch_shape_tensor())

  def _batch_shape(self):
    # If there's a chance that the batch_shape has been overridden, we return
    # what we statically know about the `batch_shape_override`. This works
    # because: `_is_maybe_batch_override` means `static_override` is `None` or a
    # non-empty list, i.e., we don't statically know the `batch_shape` or we do.
    #
    # Notice that this implementation parallels the `_event_shape` except that
    # the `bijector` doesn't get to alter the `batch_shape`. Recall that
    # `batch_shape` is a property of a distribution while `event_shape` is
    # shared between both the `distribution` instance and the `bijector`.
    static_override = tensorshape_util.constant_value_as_shape(
        self._override_batch_shape)
    return (static_override
            if self._is_maybe_batch_override
            else self.distribution.batch_shape)

  def _sample_n(self, n, seed=None, **distribution_kwargs):
    sample_shape = prefer_static.concat([
        distribution_util.pick_vector(self._needs_rotation, self._empty, [n]),
        self._override_batch_shape,
        self._override_event_shape,
        distribution_util.pick_vector(self._needs_rotation, [n], self._empty),
    ], axis=0)
    x = self.distribution.sample(sample_shape=sample_shape, seed=seed,
                                 **distribution_kwargs)
    x = self._maybe_rotate_dims(x)
    # We'll apply the bijector in the `_call_sample_n` function.
    return x

  def _call_sample_n(self, sample_shape, seed, name, **kwargs):
    # We override `_call_sample_n` rather than `_sample_n` so we can ensure that
    # the result of `self.bijector.forward` is not modified (and thus caching
    # works).
    with self._name_and_control_scope(name):
      sample_shape = tf.convert_to_tensor(
          sample_shape, dtype=tf.int32, name="sample_shape")
      sample_shape, n = self._expand_sample_shape_to_vector(
          sample_shape, "sample_shape")

      distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)

      # First, generate samples. We will possibly generate extra samples in the
      # event that we need to reinterpret the samples as part of the
      # event_shape.
      x = self._sample_n(n, seed, **distribution_kwargs)

      # Next, we reshape `x` into its final form. We do this prior to the call
      # to the bijector to ensure that the bijector caching works.
      batch_event_shape = tf.shape(x)[1:]
      final_shape = tf.concat([sample_shape, batch_event_shape], 0)
      x = tf.reshape(x, final_shape)

      # Finally, we apply the bijector's forward transformation. For caching to
      # work, it is imperative that this is the last modification to the
      # returned result.
      y = self.bijector.forward(x, **bijector_kwargs)
      y = self._set_sample_static_shape(y, sample_shape)

      return y

  def _log_prob(self, y, **kwargs):
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)

    # For caching to work, it is imperative that the bijector is the first to
    # modify the input.
    x = self.bijector.inverse(y, **bijector_kwargs)
    event_ndims = self._maybe_get_static_event_ndims()

    ildj = self.bijector.inverse_log_det_jacobian(
        y, event_ndims=event_ndims, **bijector_kwargs)
    if self.bijector._is_injective:  # pylint: disable=protected-access
      return self._finish_log_prob_for_one_fiber(
          y, x, ildj, event_ndims, **distribution_kwargs)

    lp_on_fibers = [
        self._finish_log_prob_for_one_fiber(
            y, x_i, ildj_i, event_ndims, **distribution_kwargs)
        for x_i, ildj_i in zip(x, ildj)]
    return tf.reduce_logsumexp(tf.stack(lp_on_fibers), axis=0)

  def _finish_log_prob_for_one_fiber(self, y, x, ildj, event_ndims,
                                     **distribution_kwargs):
    """Finish computation of log_prob on one element of the inverse image."""
    x = self._maybe_rotate_dims(x, rotate_right=True)
    log_prob = self.distribution.log_prob(x, **distribution_kwargs)
    if self._is_maybe_event_override:
      log_prob = tf.reduce_sum(log_prob, axis=self._reduce_event_indices)
    log_prob += tf.cast(ildj, log_prob.dtype)
    if self._is_maybe_event_override and isinstance(event_ndims, int):
      tensorshape_util.set_shape(
          log_prob,
          tf.broadcast_static_shape(
              tensorshape_util.with_rank_at_least(y.shape, 1)[:-event_ndims],
              self.batch_shape))
    return log_prob

  def _prob(self, y, **kwargs):
    if not hasattr(self.distribution, "_prob"):
      return tf.exp(self.log_prob(y, **kwargs))
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)

    x = self.bijector.inverse(y, **bijector_kwargs)
    event_ndims = self._maybe_get_static_event_ndims()
    ildj = self.bijector.inverse_log_det_jacobian(
        y, event_ndims=event_ndims, **bijector_kwargs)
    if self.bijector._is_injective:  # pylint: disable=protected-access
      return self._finish_prob_for_one_fiber(
          y, x, ildj, event_ndims, **distribution_kwargs)

    prob_on_fibers = [
        self._finish_prob_for_one_fiber(
            y, x_i, ildj_i, event_ndims, **distribution_kwargs)
        for x_i, ildj_i in zip(x, ildj)]
    return sum(prob_on_fibers)

  def _finish_prob_for_one_fiber(self, y, x, ildj, event_ndims,
                                 **distribution_kwargs):
    """Finish computation of prob on one element of the inverse image."""
    x = self._maybe_rotate_dims(x, rotate_right=True)
    prob = self.distribution.prob(x, **distribution_kwargs)
    if self._is_maybe_event_override:
      prob = tf.reduce_prod(prob, axis=self._reduce_event_indices)
    prob *= tf.exp(tf.cast(ildj, prob.dtype))
    if self._is_maybe_event_override and isinstance(event_ndims, int):
      tensorshape_util.set_shape(
          prob,
          tf.broadcast_static_shape(
              tensorshape_util.with_rank_at_least(y.shape, 1)[:-event_ndims],
              self.batch_shape))
    return prob

  def _log_cdf(self, y, **kwargs):
    if self._is_maybe_event_override:
      raise NotImplementedError("log_cdf is not implemented when overriding "
                                "event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("log_cdf is not implemented when "
                                "bijector is not injective.")
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.log_cdf(x, **distribution_kwargs)

  def _cdf(self, y, **kwargs):
    if self._is_maybe_event_override:
      raise NotImplementedError("cdf is not implemented when overriding "
                                "event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("cdf is not implemented when "
                                "bijector is not injective.")
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.cdf(x, **distribution_kwargs)

  def _log_survival_function(self, y, **kwargs):
    if self._is_maybe_event_override:
      raise NotImplementedError("log_survival_function is not implemented when "
                                "overriding event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("log_survival_function is not implemented when "
                                "bijector is not injective.")
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.log_survival_function(x, **distribution_kwargs)

  def _survival_function(self, y, **kwargs):
    if self._is_maybe_event_override:
      raise NotImplementedError("survival_function is not implemented when "
                                "overriding event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("survival_function is not implemented when "
                                "bijector is not injective.")
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.survival_function(x, **distribution_kwargs)

  def _quantile(self, value, **kwargs):
    if self._is_maybe_event_override:
      raise NotImplementedError("quantile is not implemented when overriding "
                                "event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("quantile is not implemented when "
                                "bijector is not injective.")
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    # x_q is the "qth quantile" of X iff q = P[X <= x_q].  Now, since X =
    # g^{-1}(Y), q = P[X <= x_q] = P[g^{-1}(Y) <= x_q] = P[Y <= g(x_q)],
    # implies the qth quantile of Y is g(x_q).
    inv_cdf = self.distribution.quantile(value, **distribution_kwargs)
    return self.bijector.forward(inv_cdf, **bijector_kwargs)

  def _mean(self, **kwargs):
    if not self.bijector.is_constant_jacobian:
      raise NotImplementedError("mean is not implemented for non-affine "
                                "bijectors")

    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
    x = self.distribution.mean(**distribution_kwargs)

    if self._is_maybe_batch_override or self._is_maybe_event_override:
      # A batch (respectively event) shape override is only allowed if the batch
      # (event) shape of the base distribution is [], so concatenating all the
      # shapes does the right thing.
      new_shape = prefer_static.concat([
          prefer_static.ones_like(self._override_batch_shape),
          self.distribution.batch_shape_tensor(),
          prefer_static.ones_like(self._override_event_shape),
          self.distribution.event_shape_tensor(),
      ], 0)
      x = tf.reshape(x, new_shape)
      new_shape = prefer_static.concat(
          [self.batch_shape_tensor(),
           self.event_shape_tensor()], 0)
      x = tf.broadcast_to(x, new_shape)

    y = self.bijector.forward(x, **bijector_kwargs)

    sample_shape = tf.convert_to_tensor([], dtype=tf.int32, name="sample_shape")
    y = self._set_sample_static_shape(y, sample_shape)
    return y

  def _entropy(self, **kwargs):
    if not self.bijector.is_constant_jacobian:
      raise NotImplementedError("entropy is not implemented")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("entropy is not implemented when "
                                "bijector is not injective.")
    distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
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
      entropy *= tf.cast(
          tf.reduce_prod(self._override_event_shape),
          dtype=dtype_util.base_dtype(entropy.dtype))
    if self._is_maybe_batch_override:
      new_shape = tf.concat([
          prefer_static.ones_like(self._override_batch_shape),
          self.distribution.batch_shape_tensor()
      ], 0)
      entropy = tf.reshape(entropy, new_shape)
      multiples = tf.concat([
          self._override_batch_shape,
          prefer_static.ones_like(self.distribution.batch_shape_tensor())
      ], 0)
      entropy = tf.tile(entropy, multiples)
    dummy = prefer_static.zeros(
        shape=tf.concat(
            [self.batch_shape_tensor(), self.event_shape_tensor()],
            0),
        dtype=self.dtype)
    event_ndims = (
        tensorshape_util.rank(self.event_shape)
        if tensorshape_util.rank(self.event_shape) is not None else tf.size(
            self.event_shape_tensor()))
    ildj = self.bijector.inverse_log_det_jacobian(
        dummy, event_ndims=event_ndims, **bijector_kwargs)

    entropy -= tf.cast(ildj, entropy.dtype)
    tensorshape_util.set_shape(entropy, self.batch_shape)
    return entropy

  def _maybe_validate_shape_override(self, override_shape, base_is_scalar,
                                     validate_args, name):
    """Helper to __init__ which ensures override batch/event_shape are valid."""
    if override_shape is None:
      override_shape = []

    override_shape = tf.convert_to_tensor(
        override_shape, dtype=tf.int32, name=name)

    if not dtype_util.is_integer(override_shape.dtype):
      raise TypeError("shape override must be an integer")

    override_is_scalar = _is_scalar_from_shape_tensor(override_shape)
    if tf.get_static_value(override_is_scalar):
      return self._empty

    dynamic_assertions = []

    if tensorshape_util.rank(override_shape.shape) is not None:
      if tensorshape_util.rank(override_shape.shape) != 1:
        raise ValueError("shape override must be a vector")
    elif validate_args:
      dynamic_assertions += [
          assert_util.assert_rank(
              override_shape, 1, message="shape override must be a vector")
      ]

    if tf.get_static_value(override_shape) is not None:
      if any(s < 0 for s in tf.get_static_value(override_shape)):
        raise ValueError("shape override must have non-negative elements")
    elif validate_args:
      dynamic_assertions += [
          assert_util.assert_non_negative(
              override_shape,
              message="shape override must have non-negative elements")
      ]

    is_both_nonscalar = prefer_static.logical_and(
        prefer_static.logical_not(base_is_scalar),
        prefer_static.logical_not(override_is_scalar))
    if tf.get_static_value(is_both_nonscalar) is not None:
      if tf.get_static_value(is_both_nonscalar):
        raise ValueError("base distribution not scalar")
    elif validate_args:
      dynamic_assertions += [
          assert_util.assert_equal(
              is_both_nonscalar, False, message="base distribution not scalar")
      ]

    if not dynamic_assertions:
      return override_shape
    return distribution_util.with_dependencies(
        dynamic_assertions, override_shape)

  def _maybe_rotate_dims(self, x, rotate_right=False):
    """Helper which rolls left event_dims left or right event_dims right."""
    needs_rotation_const = tf.get_static_value(self._needs_rotation)
    if needs_rotation_const is not None and not needs_rotation_const:
      return x
    ndims = prefer_static.rank(x)
    n = (ndims - self._rotate_ndims) if rotate_right else self._rotate_ndims
    perm = prefer_static.concat([
        prefer_static.range(n, ndims), prefer_static.range(0, n)], axis=0)
    return tf.transpose(a=x, perm=perm)

  def _maybe_get_static_event_ndims(self):
    if tensorshape_util.rank(self.event_shape) is not None:
      return tensorshape_util.rank(self.event_shape)

    event_ndims = tf.size(self.event_shape_tensor())
    event_ndims_ = distribution_util.maybe_get_static_value(event_ndims)

    if event_ndims_ is not None:
      return event_ndims_

    return event_ndims


class ConditionalTransformedDistribution(TransformedDistribution):
  """A TransformedDistribution that allows intrinsic conditioning."""

  @deprecation.deprecated(
      "2019-07-01",
      "`ConditionalTransformedDistribution` is no longer required; "
      "`TransformedDistribution` top-level functions now pass-through "
      "`**kwargs`.",
      warn_once=True)
  def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
    return super(ConditionalTransformedDistribution, cls).__new__(cls)
