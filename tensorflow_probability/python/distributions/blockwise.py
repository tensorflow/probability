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
"""The Blockwise distribution."""

import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import joint_distribution_sequential
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util


def _is_iterable(x):
  try:
    _ = iter(x)
  except TypeError:
    return False
  return True


class _NonCompositeTensorCast(distribution_lib.Distribution):
  """Utility distribution to cast inputs/outputs of another distribution."""

  def __init__(self, distribution, dtype):
    parameters = dict(locals())
    name = 'CastTo{}'.format(dtype_util.name(dtype))
    with tf.name_scope(name) as name:
      self._distribution = distribution
      self._dtype = dtype
      super(_NonCompositeTensorCast, self).__init__(
          dtype=dtype,
          validate_args=distribution.validate_args,
          allow_nan_stats=distribution.allow_nan_stats,
          reparameterization_type=distribution.reparameterization_type,
          parameters=parameters,
          name=name)

  def _batch_shape(self):
    return self._distribution.batch_shape

  def _batch_shape_tensor(self):
    return self._distribution.batch_shape_tensor()

  def _event_shape(self):
    return self._distribution.event_shape

  def _event_shape_tensor(self):
    return self._distribution.event_shape_tensor()

  def _sample_n(self, n, seed=None):
    return tf.nest.map_structure(lambda x: tf.cast(x, self._dtype),
                                 self._distribution.sample(n, seed))

  def _log_prob(self, x):
    x = tf.nest.map_structure(tf.cast, x, self._distribution.dtype)
    return tf.cast(self._distribution.log_prob(x), self._dtype)

  def _entropy(self):
    return self._distribution.entropy()

  def _mean(self):
    return tf.nest.map_structure(lambda x: tf.cast(x, self._dtype),
                                 self._distribution.mean())


class _Cast(_NonCompositeTensorCast,
            distribution_lib.AutoCompositeTensorDistribution):
  """Utility distribution to cast inputs/outputs of another distribution."""

  def __new__(cls, *args, **kwargs):
    """Maybe return a `_NonCompositeTensorCast`."""

    if cls is _Cast:
      if args:
        distribution = args[0]
      else:
        distribution = kwargs.get('distribution')

      if not auto_composite_tensor.is_composite_tensor(distribution):
        return _NonCompositeTensorCast(*args, **kwargs)
    return super(_Cast, cls).__new__(cls)


@kullback_leibler.RegisterKL(_NonCompositeTensorCast, _NonCompositeTensorCast)
def _kl_blockwise_cast(d0, d1, name=None):
  return d0._distribution.kl_divergence(d1._distribution, name=name)  # pylint: disable=protected-access


class _Blockwise(distribution_lib.Distribution):
  """Blockwise distribution.

  This distribution converts a distribution or list of distributions into a
  vector-variate distribution by doing a sequence of reshapes and concatenating
  the results. This is particularly useful for converting `JointDistribution`
  instances to vector-variate for downstream uses which can only handle
  single-`Tensor` distributions.

  #### Examples

  Flattening a sequence of distrbutions:

  ```python
  tfd = tfp.distributions

  d = tfd.Blockwise(
      [
          tfd.Independent(
              tfd.Normal(
                  loc=tf.zeros(4, dtype=tf.float64),
                  scale=1),
              reinterpreted_batch_ndims=1),
          tfd.MultivariateNormalTriL(
              scale_tril=tf.eye(2, dtype=tf.float32)),
      ],
      dtype_override=tf.float32,
  )
  x = d.sample([2, 1])
  y = d.log_prob(x)
  x.shape  # ==> (2, 1, 4 + 2)
  x.dtype  # ==> tf.float32
  y.shape  # ==> (2, 1)
  y.dtype  # ==> tf.float32

  d.mean()  # ==> np.zeros((4 + 2,))
  ```

  Flattening a joint distribution:

  ```python
  tfd = tfp.distributions

  Root = tfd.JointDistributionCoroutine.Root  # Convenient alias.
  def model():
    e = yield Root(tfd.Independent(tfd.Exponential(rate=[100, 120]), 1))
    g = yield tfd.Gamma(concentration=e[..., 0], rate=e[..., 1])
    n = yield Root(tfd.Normal(loc=0, scale=2.))
    yield tfd.Normal(loc=n, scale=g)

  joint = tfd.JointDistributionCoroutine(model)
  d = tfd.Blockwise(joint)

  x = d.sample([2, 1])
  y = d.log_prob(x)
  x.shape  # ==> (2, 1, 2 + 1 + 1 + 1)
  x.dtype  # ==> tf.float32
  y.shape  # ==> (2, 1)
  y.dtype  # ==> tf.float32
  ```

  """

  def __init__(self,
               distributions,
               dtype_override=None,
               validate_args=False,
               allow_nan_stats=False,
               name='Blockwise'):
    """Construct the `Blockwise` distribution.

    Args:
      distributions: Python `list` of `tfp.distributions.Distribution`
        instances or a single `tfp.distributions.JointDistribution` instance.
        If `list`, all distribution instances must have the same `batch_shape`
        and all must have `event_ndims==1`, i.e., be vector-variate
        distributions.
      dtype_override: samples of `distributions` will be cast to this `dtype`.
        If unspecified, all `distributions` must have the same `dtype`.
        Default value: `None` (i.e., do not cast).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._distributions = distributions
      if dtype_override is not None:
        distributions = tf.nest.map_structure(
            lambda d: _Cast(d, dtype_override), distributions)
      if _is_iterable(distributions):
        self._distribution = (
            joint_distribution_sequential.JointDistributionSequential(
                list(distributions)))
      else:
        self._distribution = distributions

      # Need to cache these for JointDistributions as the batch shape of that
      # distribution can change after `_sample` calls.
      self._cached_batch_shape_tensor = self._distribution.batch_shape_tensor()
      self._cached_batch_shape = self._distribution.batch_shape

      if dtype_override is not None:
        dtype = dtype_override
      else:
        dtype = set(
            dtype_util.base_dtype(dtype)
            for dtype in tf.nest.flatten(self._distribution.dtype)
            if dtype is not None)
        if len(dtype) == 0:  # pylint: disable=g-explicit-length-test
          dtype = tf.float32
        elif len(dtype) == 1:
          dtype = dtype.pop()
        else:
          raise TypeError(
              'Distributions must have same dtype; found: {}.'.format(
                  self._distribution.dtype))

      reparameterization_type = set(
          tf.nest.flatten(self._distribution.reparameterization_type))
      reparameterization_type = (
          reparameterization_type.pop() if len(reparameterization_type) == 1
          else reparameterization.NOT_REPARAMETERIZED)

      super(_Blockwise, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization_type,
          parameters=parameters,
          name=name)

  @property
  def distributions(self):
    return self._distributions

  @property
  def experimental_is_sharded(self):
    any_is_sharded = any(
        d.experimental_is_sharded for d in self.distributions)
    all_are_sharded = all(
        d.experimental_is_sharded for d in self.distributions)
    if any_is_sharded and not all_are_sharded:
      raise ValueError('`Blockwise.distributions` sharding must match.')
    return all_are_sharded

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        distributions=parameter_properties.BatchedComponentProperties(
            event_ndims=(
                lambda self: [0 for _ in self.distributions])))

  def _batch_shape(self):
    return functools.reduce(tensorshape_util.merge_with,
                            tf.nest.flatten(self._cached_batch_shape),
                            tf.TensorShape(None))

  def _batch_shape_tensor(self):
    # We could get partial static-ness by swapping in values from
    # `self.batch_shape`, however this would require multiple graph ops.
    return tf.nest.flatten(self._cached_batch_shape_tensor)[0]

  def _event_shape(self):
    event_sizes = tf.nest.map_structure(tensorshape_util.num_elements,
                                        self._distribution.event_shape)
    if any(r is None for r in tf.nest.flatten(event_sizes)):
      return tf.TensorShape([None])
    return tf.TensorShape([sum(tf.nest.flatten(event_sizes))])

  def _event_shape_tensor(self):
    event_sizes = tf.nest.map_structure(tensorshape_util.num_elements,
                                        self._distribution.event_shape)

    if any(s is None for s in tf.nest.flatten(event_sizes)):
      event_sizes = tf.nest.map_structure(
          lambda static_size, shape_tensor:  # pylint: disable=g-long-lambda
          (tf.reduce_prod(shape_tensor)
           if static_size is None else static_size),
          event_sizes,
          self._distribution.event_shape_tensor())

    return tf.reduce_sum(tf.nest.flatten(event_sizes))[tf.newaxis]

  def _flatten_and_concat_event(self, x):

    def _reshape_part(part, event_shape):
      part = tf.cast(part, self.dtype)
      static_rank = tf.get_static_value(ps.rank_from_shape(event_shape))
      if static_rank == 1:
        return part
      new_shape = ps.concat([
          ps.shape(part)[:ps.size(ps.shape(part)) - ps.size(event_shape)], [-1]
      ],
                            axis=-1)
      return tf.reshape(part, ps.cast(new_shape, tf.int32))

    if all(
        tensorshape_util.is_fully_defined(s)
        for s in tf.nest.flatten(self._distribution.event_shape)):
      x = tf.nest.map_structure(_reshape_part, x,
                                self._distribution.event_shape)
    else:
      x = tf.nest.map_structure(_reshape_part, x,
                                self._distribution.event_shape_tensor())
    return tf.concat(tf.nest.flatten(x), axis=-1)

  def _split_and_reshape_event(self, x):
    event_tensors = self._distribution.event_shape_tensor()
    splits = [
        ps.maximum(1, ps.reduce_prod(s))
        for s in tf.nest.flatten(event_tensors)
    ]
    x = tf.nest.pack_sequence_as(event_tensors, tf.split(x, splits, axis=-1))

    def _reshape_part(part, dtype, event_shape):
      part = tf.cast(part, dtype)
      static_rank = tf.get_static_value(ps.rank_from_shape(event_shape))
      if static_rank == 1:
        return part
      new_shape = ps.concat([ps.shape(part)[:-1], event_shape], axis=-1)
      return tf.reshape(part, ps.cast(new_shape, tf.int32))

    if all(
        tensorshape_util.is_fully_defined(s)
        for s in tf.nest.flatten(self._distribution.event_shape)):
      x = tf.nest.map_structure(_reshape_part, x, self._distribution.dtype,
                                self._distribution.event_shape)
    else:
      x = tf.nest.map_structure(_reshape_part, x, self._distribution.dtype,
                                self._distribution.event_shape_tensor())
    return x

  def _sample_n(self, n, seed=None):
    return self._flatten_and_concat_event(
        self._distribution.sample(n, seed=seed))

  def _sample_and_log_prob(self, sample_shape, seed):
    x, lp = self._distribution.experimental_sample_and_log_prob(
        sample_shape, seed=seed)
    return self._flatten_and_concat_event(x), lp

  def _log_prob(self, x):
    return self._distribution.log_prob(self._split_and_reshape_event(x))

  def _entropy(self):
    return self._distribution.entropy()

  def _prob(self, x):
    return self._distribution.prob(self._split_and_reshape_event(x))

  def _mean(self):
    return self._flatten_and_concat_event(self._distribution.mean())

  def _default_event_space_bijector(self):
    return self._distribution.experimental_default_event_space_bijector()

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    message = 'Distributions must have the same `batch_shape`'

    if is_init:
      batch_shapes = tf.nest.flatten(self._cached_batch_shape)
      if all(tensorshape_util.is_fully_defined(b) for b in batch_shapes):
        if batch_shapes[1:] != batch_shapes[:-1]:
          raise ValueError('{}; found: {}.'.format(message, batch_shapes))

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if self.validate_args:
      batch_shapes = self._cached_batch_shape
      if not all(
          tensorshape_util.is_fully_defined(s)
          for s in tf.nest.flatten(batch_shapes)):
        batch_shapes = tf.nest.map_structure(
            lambda static_shape, shape_tensor:  # pylint: disable=g-long-lambda
            (static_shape if tensorshape_util.is_fully_defined(static_shape)
             else shape_tensor), batch_shapes, self._cached_batch_shape_tensor)
      batch_shapes = tf.nest.flatten(batch_shapes)
      assertions.extend(
          assert_util.assert_equal(  # pylint: disable=g-complex-comprehension
              b1,
              b2,
              message='{}.'.format(message))
          for b1, b2 in zip(batch_shapes[1:], batch_shapes[:-1]))
      assertions.extend(
          assert_util.assert_equal(  # pylint: disable=g-complex-comprehension
              tf.size(b1),
              tf.size(b2),
              message='{}.'.format(message))
          for b1, b2 in zip(batch_shapes[1:], batch_shapes[:-1]))

    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    message = 'Input must have at least one dimension.'
    if tensorshape_util.rank(x.shape) is not None:
      if tensorshape_util.rank(x.shape) == 0:
        raise ValueError(message)
    elif self.validate_args:
      assertions.append(assert_util.assert_rank_at_least(x, 1, message=message))
    return assertions


class Blockwise(_Blockwise, distribution_lib.AutoCompositeTensorDistribution):

  def __new__(cls, *args, **kwargs):
    """Maybe return a non-`CompositeTensor` `_Blockwise`."""

    if cls is Blockwise:
      if args:
        distributions = args[0]
      else:
        distributions = kwargs.get('distributions')

      if not all(auto_composite_tensor.is_composite_tensor(d)
                 for d in tf.nest.flatten(distributions)):
        return _Blockwise(*args, **kwargs)
    return super(Blockwise, cls).__new__(cls)


Blockwise.__doc__ = _Blockwise.__doc__ + '\n' + (
    'If all members of `distributions` are `CompositeTensor`s, then the '
    'resulting `Blockwise` instance is a `CompositeTensor` as well. Otherwise, '
    'a non-`CompositeTensor` `_Blockwise` instance is created instead. '
    'Distribution subclasses that inherit from `Blockwise` will also inherit '
    'from `CompositeTensor`.')


@kullback_leibler.RegisterKL(_Blockwise, _Blockwise)
def _kl_blockwise_blockwise(b0, b1, name=None):
  """Calculate the batched KL divergence KL(b0 || b1) with b0 and b1 Blockwise distributions.

  Args:
    b0: instance of a Blockwise distribution object.
    b1: instance of a Blockwise distribution object.
    name: (optional) Name to use for created operations. Default is
      "kl_blockwise_blockwise".

  Returns:
    kl_blockwise_blockwise: `Tensor`. The batchwise KL(b0 || b1).
  """
  return b0._distribution.kl_divergence(b1._distribution, name=name)  # pylint: disable=protected-access
