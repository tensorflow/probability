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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import reparameterization


class Blockwise(distribution.Distribution):
  """Blockwise distribution.

  This distribution enables combining different distribution families as one.
  """
  # TODO(b/126733380): Add documentation.

  def __init__(self,
               distributions,
               dtype_override=None,
               validate_args=False,
               allow_nan_stats=False,
               name='Blockwise'):
    """Construct the `Blockwise` distribution.

    Args:
      distributions: Python `list` of `tfp.distributions.Distribution`
        instances. All distribution instances must have the same `batch_shape`
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
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.compat.v1.name_scope(name) as name:
      self._assertions = _maybe_validate_distributions(
          distributions, dtype_override, validate_args)

      if dtype_override is not None:
        dtype = dtype_override
      else:
        dtype = set(d.dtype.base_dtype for d in distributions
                    if d.dtype is not None)
        if len(dtype) == 0:  # pylint: disable=g-explicit-length-test
          dtype = tf.float32
        elif len(dtype) == 1:
          dtype = dtype.pop()
        else:
          # Shouldn't be here: we already threw an exception in
          # `_maybe_validate_distributions`.
          raise ValueError('Internal Error: unable to resolve `dtype`.')

      reparameterization_type = set(d.reparameterization_type
                                    for d in distributions)
      reparameterization_type = (reparameterization_type.pop()
                                 if len(reparameterization_type) == 1
                                 else reparameterization.NOT_REPARAMETERIZED)

      self._distributions = distributions
      super(Blockwise, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization_type,
          parameters=parameters,
          graph_parents=_flatten(d._graph_parents for d in distributions),  # pylint: disable=protected-access
          name=name)

  @property
  def distributions(self):
    return self._distributions

  def _batch_shape(self):
    return functools.reduce(lambda b, d: b.merge_with(d.batch_shape),
                            self.distributions, tf.TensorShape(None))

  def _batch_shape_tensor(self):
    with tf.control_dependencies(self._assertions):
      # We could get partial static-ness by swapping in values from
      # `self.batch_shape`, however this would require multiple graph ops.
      return self.distributions[0].batch_shape_tensor()

  def _event_shape(self):
    event_size = [d.event_shape.num_elements() for d in self.distributions]
    if any(r is None for r in event_size):
      return tf.TensorShape([None])
    return tf.TensorShape([sum(event_size)])

  def _event_shape_tensor(self):
    with tf.control_dependencies(self._assertions):
      event_sizes = [tf.reduce_prod(input_tensor=d.event_shape_tensor())  # pylint: disable=g-complex-comprehension
                     if d.event_shape.num_elements() is None
                     else d.event_shape.num_elements()
                     for d in self._distributions]
      return tf.reduce_sum(input_tensor=event_sizes)[tf.newaxis]

  def _sample_n(self, n, seed=None):
    with tf.control_dependencies(self._assertions):
      seed = seed_stream.SeedStream(seed=seed, salt='BlockwiseDistribution')
      xs = [tf.cast(d.sample(n, seed=seed()), self.dtype)
            for d in self.distributions]
      return tf.concat(xs, axis=-1)

  def _log_prob(self, x):
    additional_assertions = []
    message = 'Input must have at least one dimension.'
    if x.shape.ndims is not None:
      if x.shape.ndims == 0:
        raise ValueError(message)
    elif self.validate_args:
      additional_assertions.append(
          tf.compat.v1.assert_rank_at_least(x, 1, message=message))
    with tf.control_dependencies(self._assertions + additional_assertions):
      event_sizes = [d.event_shape_tensor()[0] for d in self.distributions]
      xs = tf.split(x, event_sizes, axis=-1)
      return sum(tf.cast(d.log_prob(tf.cast(x, d.dtype)), self.dtype)
                 for d, x in zip(self.distributions, xs))

  def _mean(self):
    with tf.control_dependencies(self._assertions):
      means = [tf.cast(d.mean(), self.dtype) for d in self._distributions]
      return tf.concat(means, axis=-1)


def _flatten(list_of_lists):
  return [item for sublist in list_of_lists for item in sublist]  # pylint: disable=g-complex-comprehension


def _is_iterable(x):
  try:
    _ = iter(x)
  except TypeError:
    return False
  return True


def _maybe_validate_distributions(distributions, dtype_override, validate_args):
  """Checks that `distributions` satisfies all assumptions."""
  assertions = []

  if not _is_iterable(distributions) or not distributions:
    raise ValueError('`distributions` must be a list of one or more '
                     'distributions.')

  if dtype_override is None:
    dts = [d.dtype.base_dtype for d in distributions if d.dtype is not None]
    if dts[1:] != dts[:-1]:
      raise TypeError('Distributions must have same dtype; '
                      'found: {}.'.format(set(dt.name for dt in dts)))

  # Validate event_ndims.
  for d in distributions:
    if d.event_shape.ndims is not None:
      if d.event_shape.ndims != 1:
        raise ValueError('`Distribution` must be vector variate, '
                         'found event nimds: {}.'.format(d.event_shape.ndims))
    elif validate_args:
      assertions.append(
          tf.compat.v1.assert_equal(
              1, tf.size(input=d.event_shape_tensor()),
              message='`Distribution` must be vector variate.'))

  batch_shapes = [d.batch_shape for d in distributions]
  if all(b.is_fully_defined() for b in batch_shapes):
    if batch_shapes[1:] != batch_shapes[:-1]:
      raise ValueError('Distributions must have the same `batch_shape`; '
                       'found: {}.'.format(batch_shapes))
  elif validate_args:
    batch_shapes = [d.batch_shape.as_list()  # pylint: disable=g-complex-comprehension
                    if d.batch_shape.is_fully_defined()
                    else d.batch_shape_tensor()
                    for d in distributions]
    assertions.extend(
        tf.compat.v1.assert_equal(  # pylint: disable=g-complex-comprehension
            b1, b2,
            message='Distribution `batch_shape`s must be identical.')
        for b1, b2 in zip(batch_shapes[1:], batch_shapes[:-1]))

  return assertions


@kullback_leibler.RegisterKL(Blockwise, Blockwise)
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
  if len(b0.distributions) != len(b1.distributions):
    raise ValueError(
        'Can only compute KL divergence between Blockwise distributions with '
        'the same number of component distributions.')

  # We also need to check that the event shapes match for each one.
  b0_event_sizes = [_event_size(d) for d in b0.distributions]
  b1_event_sizes = [_event_size(d) for d in b1.distributions]

  assertions = []
  message = ('Can only compute KL divergence between Blockwise distributions '
             'with the same pairwise event shapes.')

  if (all(isinstance(event_size, int) for event_size in b0_event_sizes) and
      all(isinstance(event_size, int) for event_size in b1_event_sizes)):
    if b0_event_sizes != b1_event_sizes:
      raise ValueError(message)
  else:
    if b0.validate_args or b1.validate_args:
      assertions.extend(
          tf.compat.v1.assert_equal(  # pylint: disable=g-complex-comprehension
              e1, e2, message=message)
          for e1, e2 in zip(b0_event_sizes, b1_event_sizes))

  with tf.compat.v1.name_scope(name, 'kl_blockwise_blockwise'):
    with tf.control_dependencies(assertions):
      return sum([
          kullback_leibler.kl_divergence(d1, d2) for d1, d2 in zip(
              b0.distributions, b1.distributions)])


def _event_size(d):
  if d.event_shape.num_elements() is not None:
    return d.event_shape.num_elements()
  return tf.reduce_prod(input_tensor=d.event_shape_tensor())
