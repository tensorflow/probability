# Copyright 2020 The TensorFlow Probability Authors.
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
"""The BatchConcat distribution."""

import functools
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'BatchConcat',
]


class _BatchConcat(distribution_lib.Distribution):
  r"""The Batch-Concatenating distribution.

  This distribution concatenates a list of distributions
  along a given axis of their batch shapes.

  In order to be concatenable along `axis`,
  `dist1`  and `dist2` should meet the following requirements:
  1. `dist1.event_shape == dist2.event_shape`
  2. `len(dist1.batch_shape) == len(dist2.batch_shape)`
  3. `\forall i \neq axis, dist1.batch_shape[i] == 1 or
                           dist1.batch_shape[i] == dist2.batch_shape[i]`
  If `dist1.batch_shape[i] == 1`, the distribution will be
  broadcasted along this axis.

  TODO(b/179916710): The distribution does NOT support sample broadcasting.
  Given a sample `x` and a BatchConcat distribution `dist`, `dist.log_prob(x)`
  requires that `x.shape == sample_shape + dist.batch_shape + dist.event_shape`.

  #### Example

  ```python
  tfd = tfp.distributions

  dtype = np.float32
  dims = 2
  batch_shape_1 = [32, 2]
  batch_shape_2 = [32, 6]

  scale_1 = np.ones(batch_shape_1 + [dims], dtype)
  mvn_1 = tfd.MultivariateNormalDiag(scale_diag=scale_1)
  scale_2 = np.ones(batch_shape_2 + [dims], dtype)
  mvn_2 = tfd.MultivariateNormalDiag(scale_diag=scale_2)
  batched_mvn = tfd.BatchConcat(
      distributions=[mvn_1, mvn_2],
      axis=1,
      validate_args=True)

  batched_mvn.batch_shape
  # ==> [32, 8]

  x = batched_mvn.sample(sample_shape=[4, 5])
  x.shape
  # ==> [4, 5, 32, 8, 2] == sample_shape + batched_mvn.batch_shape + [dims]

  reshape_mvn.log_prob(x).shape
  # ==> [4, 5, 32, 8] == sample_shape + batched_mvn.batch_shape
  ```

  #### Example for broadcasting.
  The distributions need to have the same batch shape size
  to enable broadcasting.

  ```python
  tfd = tfp.distributions

  dtype = np.float32
  dims = 2
  batch_shape_1 = [1, 2, 16]
  batch_shape_2 = [32, 6, 1]

  scale_1 = np.ones(batch_shape_1 + [dims], dtype)
  mvn_1 = tfd.MultivariateNormalDiag(scale_diag=scale_1)
  scale_2 = np.ones(batch_shape_2 + [dims], dtype)
  mvn_2 = tfd.MultivariateNormalDiag(scale_diag=scale_2)
  batched_mvn = tfd.BatchConcat(
      distributions=[mvn_1, mvn_2],
      axis=1,
      validate_args=True)

  batched_mvn.batch_shape
  # ==> [32, 8, 16]

  x = batched_mvn.sample(sample_shape=[4, 5])
  x.shape
  # ==> [4, 5, 32, 8, 16, 2] == sample_shape + batched_mvn.batch_shape + [dims]

  reshape_mvn.log_prob(x).shape
  # ==> [4, 5, 32, 8, 16] == sample_shape + batched_mvn.batch_shape
  ```
  """

  def __init__(self,
               distributions,
               axis,
               validate_args=False,
               allow_nan_stats=True,
               name='BatchConcat'):
    """Construct BatchConcat distribution.

    Args:
      distributions: A sequence of `Distribution` instances to concatenate.
        Instances of the distributions should have the same event_shape and
        compatible batch_shapes that can be concatenated along the specified
        axis.
      axis: Positive `int` specifying the axis along which to concatenate
        the given distributions.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value `NaN` to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: The name to give Ops created by this `Distribution`.
        Default value: "BatchConcat"

    Raises:
      ValueError: if `event_shape` of given distributions are not the same.
      ValueError: if `batch_shape` of given distributions are not concatenable.
      ValueError: if distributions don't have the same type.
      ValueError: if axis is out of range or negative.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._distributions = distributions
      self._axis = axis

      dtype = dtype_util.common_dtype(distributions)

      reparameterizable = all(
          d._reparameterization_type == reparameterization.FULLY_REPARAMETERIZED
          for d in self._distributions)

      reparameterization_type = (reparameterization.FULLY_REPARAMETERIZED
                                 if reparameterizable
                                 else reparameterization.NOT_REPARAMETERIZED)

      super(_BatchConcat, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        distributions=parameter_properties.BatchedComponentProperties(
            event_ndims=lambda self: [0 for _ in self.distributions]),
        axis=parameter_properties.ShapeParameterProperties())

  def _broadcast(self, x, sample_shape):
    """Broadcasts x's batch dims (except self.axis) to match self.batch_shape.

    Specifically, x is broadcasted to have shape
    `sample_shape + target_batch + event_shape`
    where target_batch == self.batch_shape except along the concatenation axis.

    Args:
      x: tf.Tensor with shape sample_shape + batch_shape + ndims.
      sample_shape: sample_shape of the input tensor.

    Returns:
      Broadcasted tensor.
    """
    x_shape = ps.shape(x)

    batch_shape = self._calculate_batch_shape()
    sample_batch_ndims = ps.size(sample_shape) + ps.size(batch_shape)
    _, x_event_shape = ps.split(
        x_shape, [sample_batch_ndims,
                  ps.rank(x) - sample_batch_ndims])
    target_shape = ps.concat([
        sample_shape, batch_shape[:self._axis],
        [x_shape[self._axis + ps.size(sample_shape)]],
        batch_shape[self._axis + 1:], x_event_shape
    ],
                             axis=0)
    return tf.broadcast_to(x, target_shape)

  def _calculate_batch_shape(self):
    """Computes fully defined batch shape for the new distribution."""
    all_batch_shapes = [d.batch_shape.as_list()
                        if tensorshape_util.is_fully_defined(d.batch_shape)
                        else d.batch_shape_tensor() for d in self.distributions]
    original_shape = ps.stack(all_batch_shapes, axis=0)
    index_mask = ps.cast(
        ps.one_hot(self._axis, ps.shape(original_shape)[1]),
        dtype=tf.bool)
    new_concat_dim = ps.cast(
        ps.reduce_sum(original_shape, axis=0)[self._axis], dtype=tf.int32)
    return ps.where(index_mask, new_concat_dim,
                    ps.reduce_max(original_shape, axis=0))

  def _split_sample(self, x):
    result_batch_shape = self._calculate_batch_shape()
    sample_shape_size = (ps.rank(x) -
                         ps.size(result_batch_shape) -
                         ps.size(self.event_shape_tensor()))
    all_batch_shapes = [d.batch_shape.as_list()
                        if tensorshape_util.is_fully_defined(d.batch_shape)
                        else d.batch_shape_tensor() for d in self.distributions]
    original_shapes = ps.stack(all_batch_shapes, axis=0)
    all_compose_shapes = ps.gather(original_shapes, self._axis, axis=1)
    x_split = tf.split(x, all_compose_shapes, axis=sample_shape_size+self._axis)
    return sample_shape_size, x_split

  def _sample_control_dependencies(self, x):
    """Helper which validates sample arg, e.g., input to `log_prob`."""
    x_ndims = (
        tf.rank(x) if tensorshape_util.rank(x.shape) is None else
        tensorshape_util.rank(x.shape))
    event_ndims = (
        tf.size(self.event_shape_tensor())
        if tensorshape_util.rank(self.event_shape) is None else
        tensorshape_util.rank(self.event_shape))
    batch_ndims = (
        tf.size(self.batch_shape_tensor())
        if tensorshape_util.rank(self.batch_shape) is None else
        tensorshape_util.rank(self.batch_shape))
    expected_batch_event_ndims = batch_ndims + event_ndims

    if (isinstance(x_ndims, int) and
        isinstance(expected_batch_event_ndims, int)):
      if x_ndims < expected_batch_event_ndims:
        raise NotImplementedError(
            'Broadcasting is not supported; too few batch and event dims '
            '(expected at least {}, saw {}).'.format(
                expected_batch_event_ndims, x_ndims))
      ndims_assertion = []
    elif self.validate_args:
      ndims_assertion = [
          assert_util.assert_greater_equal(
              x_ndims,
              expected_batch_event_ndims,
              message=('Broadcasting is not supported; too few '
                       'batch and event dims.'),
              name='assert_batch_and_event_ndims_large_enough'),
      ]

    if (tensorshape_util.is_fully_defined(self.batch_shape) and
        tensorshape_util.is_fully_defined(self.event_shape)):
      expected_batch_event_shape = np.int32(
          tensorshape_util.concatenate(self.batch_shape, self.event_shape))
    else:
      expected_batch_event_shape = tf.concat(
          [
              self.batch_shape_tensor(),
              self.event_shape_tensor(),
          ], axis=0)

    sample_ndims = x_ndims - expected_batch_event_ndims
    if isinstance(sample_ndims, int):
      sample_ndims = max(sample_ndims, 0)
    if (isinstance(sample_ndims, int) and
        tensorshape_util.is_fully_defined(x.shape[sample_ndims:])):
      actual_batch_event_shape = np.int32(x.shape[sample_ndims:])
    else:
      sample_ndims = tf.maximum(sample_ndims, 0)
      actual_batch_event_shape = tf.shape(x)[sample_ndims:]

    assertions = []
    if (isinstance(expected_batch_event_shape, np.ndarray) and
        isinstance(actual_batch_event_shape, np.ndarray)):
      if any(expected_batch_event_shape != actual_batch_event_shape):
        raise NotImplementedError('Broadcasting is not supported; '
                                  'unexpected batch and event shape '
                                  '(expected {}, saw {}).'.format(
                                      expected_batch_event_shape,
                                      actual_batch_event_shape))
      assertions.extend(ndims_assertion)
    elif self.validate_args:
      with tf.control_dependencies(ndims_assertion):
        shape_assertion = assert_util.assert_equal(
            expected_batch_event_shape,
            actual_batch_event_shape,
            message=('Broadcasting is not supported; '
                     'unexpected batch and event shape.'),
            name='assert_batch_and_event_shape_same')
      assertions.append(shape_assertion)

    return assertions

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    if is_init:
      axis_ = tf.get_static_value(self._axis)
      if axis_ is not None and axis_ < 0:
        raise ValueError('Axis should be positive, %d was given' % axis_)
      if axis_ is None:
        assertions.append(tf.assert_greater_equal(axis_, 0))

      all_event_shapes = [d.event_shape for d in self._distributions]
      if all(tensorshape_util.is_fully_defined(event_shape)
             for event_shape in all_event_shapes):
        if all_event_shapes[1:] != all_event_shapes[:-1]:
          raise ValueError('Distributions must have the same `event_shape`;'
                           'found: {}' % all_event_shapes)

      all_batch_shapes = [d.batch_shape for d in self._distributions]
      if all(tensorshape_util.is_fully_defined(batch_shape)
             for batch_shape in all_batch_shapes):
        batch_shape = all_batch_shapes[0].as_list()
        batch_shape[self._axis] = 1
        for b in all_batch_shapes[1:]:
          b = b.as_list()
          if len(batch_shape) != len(b):
            raise ValueError('Incompatible batch shape % s with %s' %
                             (batch_shape, b))
          b[self._axis] = 1
          tf.broadcast_static_shape(
              tensorshape_util.constant_value_as_shape(batch_shape),
              tensorshape_util.constant_value_as_shape(b))

    if not self.validate_args:
      return []

    if self.validate_args:
      # Validate that event shapes all match.
      all_event_shapes = [d.event_shape for d in self._distributions]
      if not all(tensorshape_util.is_fully_defined(event_shape)
                 for event_shape in all_event_shapes):
        all_event_shape_tensors = [d.event_shape_tensor() for
                                   d in self._distributions]
        def _get_shapes(static_shape, dynamic_shape):
          if tensorshape_util.is_fully_defined(static_shape):
            return static_shape
          else:
            return dynamic_shape
        event_shapes = tf.nest.map_structure(_get_shapes,
                                             all_event_shapes,
                                             all_event_shape_tensors)
        event_shapes = tf.nest.flatten(event_shapes)
        assertions.extend(
            assert_util.assert_equal(
                e1, e2, message='Distributions should have same event shapes.')
            for e1, e2 in zip(event_shapes[1:], event_shapes[:-1]))

      # Validate that batch shapes are broadcastable and concatenable along
      # the specified axis.
      if not all(tensorshape_util.is_fully_defined(d.batch_shape)
                 for d in self._distributions):
        for i, d in enumerate(self._distributions[:-1]):
          assertions.append(tf.assert_equal(
              tf.size(d.batch_shape_tensor()),
              tf.size(self._distributions[i+1].batch_shape_tensor())))

        batch_shape_tensors = [
            ps.tensor_scatter_nd_update(d.batch_shape_tensor(), updates=1,
                                        indices=[self._axis])
            for d in self._distributions
        ]
        assertions.append(
            functools.reduce(tf.broadcast_dynamic_shape,
                             batch_shape_tensors[1:],
                             batch_shape_tensors[:-1]))
    return assertions

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
      raise ValueError('`BatchConcat.distributions` sharding must match.')
    return all_are_sharded

  def _batch_shape_tensor(self):
    return self._calculate_batch_shape()

  def _batch_shape(self):
    return tensorshape_util.constant_value_as_shape(
        self._calculate_batch_shape())

  def _event_shape_tensor(self):
    return self.distributions[0].event_shape_tensor()

  def _event_shape(self):
    return self.distributions[0].event_shape

  def _sample_n(self, n, seed=None, **kwargs):
    all_seeds = samplers.split_seed(
        seed, len(self._distributions), salt='BatchConcat')
    samples = []
    for d, s in zip(self._distributions, all_seeds):
      samples.append(self._broadcast(d.sample([n], s), [n]))

    return tf.concat(samples, axis=self._axis+1)

  def _sample_and_log_prob(self, sample_shape, seed, **kwargs):
    all_seeds = samplers.split_seed(
        seed, len(self._distributions), salt='BatchConcat')
    samples = []
    log_probs = []
    for d, s in zip(self._distributions, all_seeds):
      x, lp = d.experimental_sample_and_log_prob(sample_shape, s)
      samples.append(self._broadcast(x, sample_shape))
      log_probs.append(self._broadcast(lp, sample_shape))

    sample_shape_size = ps.rank_from_shape(sample_shape)
    return (tf.concat(samples, axis=self._axis + sample_shape_size),
            tf.concat(log_probs, axis=self._axis + sample_shape_size))

  def _call_split_concat(self, fn, x, **kwargs):
    sample_shape_size, split_x = self._split_sample(x)
    result = [
        getattr(d, fn)(i, **kwargs)
        for (i, d) in zip(split_x, self._distributions)
    ]
    return tf.concat(result, axis=self._axis + sample_shape_size)

  def _call_concat(self, fn, **kwargs):
    result = [
        self._broadcast(getattr(d, fn)(**kwargs), [])
        for d in self._distributions
    ]
    return tf.concat(result, axis=self._axis)

  def _log_prob(self, x, **kwargs):
    return self._call_split_concat('log_prob', x, **kwargs)

  def _prob(self, x, **kwargs):
    return self._call_split_concat('prob', x, **kwargs)

  def _log_cdf(self, x, **kwargs):
    return self._call_split_concat('log_cdf', x, **kwargs)

  def _cdf(self, x, **kwargs):
    return self._call_split_concat('cdf', x, **kwargs)

  def _log_survival_function(self, x, **kwargs):
    return self._call_split_concat('log_survival_function', x, **kwargs)

  def _survival_function(self, x, **kwargs):
    return self._call_split_concat('survival_function', x, **kwargs)

  def _entropy(self, **kwargs):
    return self._call_concat('entropy')

  def _mean(self, **kwargs):
    return self._call_concat('mean')

  def _mode(self, **kwargs):
    return self._call_concat('mode')

  def _stddev(self, **kwargs):
    return self._call_concat('stddev')

  def _variance(self, **kwargs):
    return self._call_concat('variance')

  def _covariance(self, **kwargs):
    return self._call_concat('covariance')


class BatchConcat(
    _BatchConcat, distribution_lib.AutoCompositeTensorDistribution):

  def __new__(cls, *args, **kwargs):
    """Maybe return a non-`CompositeTensor` `_BatchConcat`."""

    if cls is BatchConcat:
      if args:
        distributions = args[0]
      else:
        distributions = kwargs.get('distributions')

      if not all(auto_composite_tensor.is_composite_tensor(d)
                 for d in distributions):
        return _BatchConcat(*args, **kwargs)
    return super(BatchConcat, cls).__new__(cls)


BatchConcat.__doc__ = _BatchConcat.__doc__ + '\n' + (
    'If all elements of `distributions` are `CompositeTensor`s, then the '
    'resulting `BatchConcat` instance is a `CompositeTensor` as well. '
    'Otherwise, a non-`CompositeTensor` `_BatchConcat` instance is created '
    'instead. Distribution subclasses that inherit from `BatchConcat` will '
    'also inherit from `CompositeTensor`.')
