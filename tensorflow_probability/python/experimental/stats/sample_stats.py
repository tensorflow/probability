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
"""Functions for computing statistics of samples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps


__all__ = [
    'RunningCovariance',
    'RunningCovarianceState',
    'RunningVariance',
]


RunningCovarianceState = collections.namedtuple(
    'RunningCovarianceState', 'num_samples, mean, sum_squared_residuals')


def _update_running_covariance(
    state, new_sample, event_ndims, dtype, axis
):
  """Updates the streaming `state` with a `new_sample`."""
  new_sample = tf.cast(new_sample, dtype=dtype)
  if axis is not None:
    chunk_n = tf.cast(ps.shape(new_sample)[axis], dtype=dtype)
    chunk_mean = tf.math.reduce_mean(new_sample, axis=axis)
    chunk_delta_mean = new_sample - tf.expand_dims(chunk_mean, axis=axis)
    chunk_sum_squared_residuals = tf.reduce_sum(
        _batch_outer_product(chunk_delta_mean, event_ndims),
        axis=axis
    )
  else:
    chunk_n = tf.ones((), dtype=dtype)
    chunk_mean = new_sample
    chunk_sum_squared_residuals = tf.zeros(
        ps.shape(state.sum_squared_residuals),
        dtype=dtype)

  new_n = state.num_samples + chunk_n
  delta_mean = chunk_mean - state.mean
  new_mean = state.mean + chunk_n * delta_mean / new_n
  all_pairwise_deltas = _batch_outer_product(
      delta_mean, event_ndims)
  adj_factor = state.num_samples * chunk_n / (state.num_samples + chunk_n)
  new_sum_squared_residuals = (state.sum_squared_residuals
                               + chunk_sum_squared_residuals
                               + adj_factor * all_pairwise_deltas)
  return RunningCovarianceState(new_n, new_mean, new_sum_squared_residuals)


def _batch_outer_product(target, event_ndims):
  """Calculates the batch outer product along `target`'s event dimensions.

  More precisely, A `tf.einsum` operation is used to calculate desired
  pairwise products as follows:

  For `event_ndims=0`, the return value is:
    `tf.einsum("...,...->...", target, target)`
  For `event_ndims=1`:
    `tf.einsum("...a,...b->...ab", target, target)`
  For `event_ndims=2`:
    `tf.einsum("...ab,...cd->...abcd", target, target)`
  ...

  Args:
    target: Target `Tensor` for the `tf.einsum` computation.
    event_ndims: Both the number of dimensions that specify the event
      shape and the desired number of dimensions for cross product
      terms.

  Returns:
    outer_product: A `Tensor` with shape B + E + E for all pairwise
      products of `target` in the event dimensions.
  """
  assign_indices = ''.join(list(
      map(chr, range(ord('a'), ord('a') + event_ndims * 2))))
  first_indices = assign_indices[:event_ndims]
  second_indices = assign_indices[event_ndims:]
  einsum_formula = '...{},...{}->...{}'.format(
      first_indices, second_indices, assign_indices)
  return tf.einsum(einsum_formula, target, target)


class RunningCovariance(object):
  """Holds metadata for and facilitates covariance computation.

  `RunningCovariance` objects do not hold state information. That information,
  which includes intermediate calculations, are held in a
  `RunningCovarianceState` as returned via `initialize` and `update` method
  calls.

  `RunningCovariance` is meant to serve general streaming covariance needs.
  For a specialized version that fits streaming over MCMC samples, see
  `CovarianceReducer` in `tfp.experimental.mcmc`.
  """

  def __init__(self, shape, event_ndims=None, dtype=tf.float32):
    """A `RunningCovariance` object holds metadata for covariance computation.

    The running covariance computation supports batching. The `event_ndims`
    parameter indicates the number of trailing dimensions to treat as part of
    the event, and to compute covariance across. The leading dimensions, if
    any, are treated as batch shape, and no cross terms are computed.

    For example, if the incoming samples have shape `[5, 7]`, the `event_ndims`
    selects among three different covariance computations:
    - `event_ndims=0` treats the samples as a `[5, 7]` batch of scalar random
      variables, and computes their variances in batch.  The shape of the result
      is `[5, 7]`.
    - `event_ndims=1` treats the samples as a `[5]` batch of vector random
      variables of shape `[7]`, and computes their covariances in batch.  The
      shape of the result is `[5, 7, 7]`.
    - `event_ndims=2` treats the samples as a single random variable of
      shape `[5, 7]` and computes its covariance.  The shape of the result
      is `[5, 7, 5, 7]`.

    Args:
      shape: Python `Tuple` or `TensorShape` representing the shape of
        incoming samples.
      event_ndims:  Number of dimensions that specify the event shape, from
        the inner-most dimensions.  Specifying `None` returns all cross
        product terms (no batching) and is the default.
      dtype: Dtype of incoming samples and the resulting statistics.
        By default, the dtype is `tf.float32`.

    Raises:
      ValueError: if `event_ndims` is greater than the rank of the intended
        incoming samples (operation is extraneous).
    """
    if event_ndims is None:
      event_ndims = len(shape)
    if event_ndims > len(shape):
      raise ValueError('Cannot calculate cross-products in {} dimensions for '
                       'samples of rank {}'.format(event_ndims, len(shape)))
    if event_ndims > 13:
      raise ValueError('`event_ndims` over 13 not supported')
    self.shape = shape
    self.event_ndims = event_ndims
    self.dtype = dtype

  def initialize(self):
    """Initializes a `RunningCovarianceState` using previously defined metadata.

    Returns:
      state: `RunningCovarianceState` representing a stream of no inputs.
    """
    # we need a secondary `RunningCovarianceState` so that future calls to
    # `update` are compatible with `tf.while_loop`. Namely, we need to
    # somehow store the `event_ndims` and `dtype` without explicitly passing or
    # returning them from a `tf.while_loop`
    if self.event_ndims == 0:
      extra_ndims_shape = ()
    else:
      extra_ndims_shape = self.shape[-self.event_ndims:]  # pylint: disable=invalid-unary-operand-type
    return RunningCovarianceState(
        num_samples=tf.zeros((), dtype=self.dtype),
        mean=tf.zeros(self.shape, dtype=self.dtype),
        sum_squared_residuals=tf.zeros(
            self.shape + extra_ndims_shape, dtype=self.dtype),
    )

  def update(self, state, new_sample, axis=None):
    """Update the `RunningCovarianceState` with a new sample.

    The update formula is from Philippe Pebay (2008) [1]. This implementation
    supports both batched and chunked covariance computation. A "batch" is the
    usual parallel computation, namely a batch of size N implies N independent
    covariance computations, each stepping one sample (or chunk) at a time. A
    "chunk" of size M implies incorporating M samples into a single covariance
    computation at once, which is more efficient than one by one.

    To further illustrate the difference between batching and chunking, consider
    the following example:

    ```python
    # treat as 3 samples from each of 5 independent vector random variables of
    # shape (2,)
    sample = tf.ones((3, 5, 2))
    running_cov = tfp.experimental.stats.RunningCovariance(
        (5, 2), event_ndims=1)
    state = running_cov.initialize()
    state = running_cov.update(state, sample, axis=0)
    final_cov = running_cov.finalize(state)
    final_cov.shape # (5, 2, 2)
    ```

    Args:
      state: `RunningCovarianceState` that represents the current state of
        running statistics.
      new_sample: Incoming sample with shape and dtype compatible with those
        used to form the `RunningCovarianceState`.
      axis: If chunking is desired, this is an integer that specifies the axis
        with chunked samples. For individual samples, set this to `None`. By
        default, samples are not chunked (`axis` is None).

    Returns:
      state: `RunningCovarianceState` with updated calculations.

    #### References
    [1]: Philippe Pebay. Formulas for Robust, One-Pass Parallel Computation of
         Covariances and Arbitrary-Order Statistical Moments. _Technical Report
         SAND2008-6212_, 2008.
         https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
    """
    updated_state = _update_running_covariance(
        state, new_sample, self.event_ndims, self.dtype, axis)
    return state._replace(**updated_state._asdict())

  def finalize(self, state, ddof=0):
    """Finalizes running covariance computation for the `state`.

    Args:
      state: `RunningCovarianceState` that represents the current state of
        running statistics.
      ddof: Requested dynamic degrees of freedom for the covariance calculation.
        For example, use `ddof=0` for population covariance and `ddof=1` for
        sample covariance. Defaults to the population covariance.

    Returns:
      covariance: An estimate of the covariance.
    """
    return state.sum_squared_residuals / (state.num_samples - ddof)


class RunningVariance(RunningCovariance):
  """Holds metadata for and facilitates variance computation.

  `RunningVariance` objects do not hold state information. That information,
  which includes intermediate calculations, are held in a
  `RunningCovarianceState` as returned via `initialize` and `update` method
  calls.

  `RunningVariance` is meant to serve general streaming variance needs.
  For a specialized version that fits streaming over MCMC samples, see
  `VarianceReducer` in `tfp.experimental.mcmc`.
  """

  def __init__(self, shape=(), dtype=tf.float32):
    """A `RunningVariance` object holds metadata for variance computation.

    This is a special case of `RunningCovariance` with `event_ndims=0`,
    provided for convenience.

    Args:
      shape: Python `Tuple` or `TensorShape` representing the shape of
        incoming samples. By default, the shape is assumed to be scalar.
      dtype: Dtype of incoming samples and the resulting statistics.
        By default, the dtype is `tf.float32`.
    """
    super(RunningVariance, self).__init__(shape, event_ndims=0, dtype=dtype)
