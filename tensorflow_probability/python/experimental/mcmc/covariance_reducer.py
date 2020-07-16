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
"""CovarianceReducer for pre-packaged Streaming Covariance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.stats.sample_stats import RunningCovariance
from tensorflow_probability.python.experimental.mcmc import reducer as reducer_base


__all__ = [
    'CovarianceReducer',
    'VarianceReducer'
]


class CovarianceReducer(reducer_base.Reducer):
  """CovarianceReducer is a `Reducer` that computes a running covariance.

  As with all reducers, CovarianceReducer does not hold state information;
  rather, it stores supplied metadata. Intermediate calculations are held in
  a state object, which is returned via `initialize` and `one_step` method
  calls.
  """

  def __init__(self, shape, event_ndims=None, dtype=tf.float32):
    """Instantiates this object.

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
      event_ndims:  Number of inner-most dimensions that represent the event
        shape. Specifying `None` returns all cross product terms (no batching)
        and is the default.
      dtype: Dtype of incoming samples and the resulting statistics.
        By default, the dtype is `tf.float32`.
    """
    self.strm = RunningCovariance(shape, event_ndims, dtype)

  def initialize(self, initial_chain_state=None, initial_kernel_results=None):
    """Initializes a `RunningCovarianceState` using previously defined metadata.

    Args:
      initial_chain_state: `Tensor` or Python `list` of `Tensor`s representing
        the current state(s) of the Markov chain(s). For streaming covariance,
        this argument has no influence on computation. Hence, by default, it
        is `None`.
      initial_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made in a related
        `TransitionKernel`. For streaming covariance, this argument also has no
        influence on computation; hence, it is also `None` by default.

    Returns:
      state: `RunningCovarianceState` representing a stream of no inputs.
    """
    # initial state not included as a sample for consistency with `sample_chain`
    return self.strm.initialize()

  def one_step(
      self,
      sample,
      current_reducer_state,
      previous_kernel_results=None,
      axis=None):
    """Update the `current_reducer_state` with a new sample.

    Args:
      sample: Incoming sample with shape and dtype compatible with those
        used to initialize the `CovarianceReducer`.
      state: `RunningCovarianceState` representing the current state of the
        running covariance.
      previous_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made in a related
        `TransitionKernel`. For streaming covariance, this argument has no
        influence on computation; hence, it is `None` by default.
      axis: If chunking is desired, this is an integer that specifies the axis
        with chunked samples. For individual samples, set this to `None`. By
        default, samples are not chunked (`axis` is None).

    Returns:
      new_state: `RunningCovarianceState` with updated running statistics.
    """
    return self.strm.update(current_reducer_state, sample, axis=axis)

  def finalize(self, final_state, ddof=0):
    """Finalizes covariance calculation from the `final_state`.

    final_state: `RunningCovarianceState` that represents the final state of
        running statistics.
    ddof: Requested dynamic degrees of freedom. For example, use `ddof=0`
      for population covariance and `ddof=1` for sample covariance. Defaults
      to the population covariance.

    Returns:
      covariance: an estimate of the covariance.
    """
    return self.strm.finalize(final_state, ddof=ddof)

  @property
  def shape(self):
    return self.strm.shape

  @property
  def event_ndims(self):
    return self.strm.event_ndims

  @property
  def dtype(self):
    return self.strm.dtype


class VarianceReducer(CovarianceReducer):
  """VarianceReducer is a `Reducer` that computes running variance.

  This object is a direct extension of `CovarianceReducer` but simplified
  in the special case of `event_ndims=0` for convenience. See
  `CovarianceReducer` for more information.
  """

  def __init__(self, shape=(), dtype=tf.float32):
    self.strm = RunningCovariance(shape=shape, event_ndims=0, dtype=dtype)
  