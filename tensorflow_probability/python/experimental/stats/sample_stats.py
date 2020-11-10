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
import functools
import inspect
import math
# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.mcmc import diagnostic
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'RunningCentralMoments',
    'RunningCentralMomentsState',
    'RunningCovariance',
    'RunningMean',
    'RunningMeanState',
    'RunningPotentialScaleReduction',
    'RunningVariance',
]


@auto_composite_tensor.auto_composite_tensor(omit_kwargs='name')
class RunningCovariance(object):
  """A running covariance computation.

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

  `RunningCovariance` is meant to serve general streaming covariance needs.
  For a specialized version that fits streaming over MCMC samples, see
  `CovarianceReducer` in `tfp.experimental.mcmc`.
  """

  def __init__(self, num_samples, mean, sum_squared_residuals, event_ndims,
               name='RunningCovariance'):
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [num_samples, mean, sum_squared_residuals], dtype_hint=tf.float32)
      self.num_samples = tf.convert_to_tensor(num_samples, dtype=dtype)
      self.mean = tf.convert_to_tensor(mean, dtype=dtype)
      self.sum_squared_residuals = tf.convert_to_tensor(
          sum_squared_residuals, dtype=dtype)
      self.event_ndims = event_ndims
      self.name = name

  @classmethod
  def from_shape(cls, shape=(), dtype=tf.float32, event_ndims=None,
                 name='RunningCovariance'):
    """Starts a `RunningCovariance` from shape and dtype metadata.

    Args:
      shape: Python `Tuple` or `TensorShape` representing the shape of incoming
        samples.  This is useful to supply if the `RunningCovariance` will be
        carried by a `tf.while_loop`, so that broadcasting does not change the
        shape across loop iterations.
      dtype: Dtype of incoming samples and the resulting statistics.
        By default, the dtype is `tf.float32`. Any integer dtypes will be
        cast to corresponding floats (i.e. `tf.int32` will be cast to
        `tf.float32`), as intermediate calculations should be performing
        floating-point division.
      event_ndims:  Number of dimensions that specify the event shape, from
        the inner-most dimensions.  Specifying `None` returns all cross
        product terms (no batching) and is the default.
      name: Python `str` name prefixed to Ops created by this class.

    Returns:
      cov: An empty `RunningCovariance`, ready for incoming samples.

    Raises:
      ValueError: if `event_ndims` is greater than the rank of the intended
        incoming samples (operation is extraneous).
    """
    dtype = _float_dtype_like(dtype)
    event_ndims = _default_covariance_event_ndims(event_ndims, shape)
    if event_ndims == 0:
      extra_ndims_shape = ()
    else:
      # event_ndims cannot be None by this point
      extra_ndims_shape = shape[-event_ndims:]  # pylint: disable=invalid-unary-operand-type
    return cls(
        num_samples=tf.zeros((), dtype=dtype),
        mean=tf.zeros(shape, dtype=dtype),
        sum_squared_residuals=tf.zeros(
            ps.concat([shape, extra_ndims_shape], axis=0), dtype=dtype),
        event_ndims=event_ndims,
        name=name,
    )

  def update(self, new_sample, axis=None):
    """Update the `RunningCovariance` with a new sample.

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
    running_cov = tfp.experimental.stats.RunningCovariance.from_shape(
        (5, 2), event_ndims=1)
    running_cov = running_cov.update(sample, axis=0)
    final_cov = running_cov.covariance()
    final_cov.shape # (5, 2, 2)
    ```

    Args:
      new_sample: Incoming sample with shape and dtype compatible with those
        used to form this `RunningCovariance`.
      axis: If chunking is desired, this is an integer that specifies the axis
        with chunked samples. For individual samples, set this to `None`. By
        default, samples are not chunked (`axis` is None).

    Returns:
      cov: Newly allocated `RunningCovariance` updated to include `new_sample`.

    #### References
    [1]: Philippe Pebay. Formulas for Robust, One-Pass Parallel Computation of
         Covariances and Arbitrary-Order Statistical Moments. _Technical Report
         SAND2008-6212_, 2008.
         https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
    """
    with tf.name_scope((self.name or 'RunningCovariance') + '.update'):
      dtype = self.mean.dtype
      new_sample = tf.cast(new_sample, dtype=dtype)
      if axis is not None:
        chunk_n = tf.cast(ps.shape(new_sample)[axis], dtype=dtype)
        chunk_mean = tf.math.reduce_mean(new_sample, axis=axis)
        chunk_delta_mean = new_sample - tf.expand_dims(chunk_mean, axis=axis)
        chunk_sum_squared_residuals = tf.reduce_sum(
            _batch_outer_product(chunk_delta_mean, self.event_ndims),
            axis=axis
        )
      else:
        chunk_n = tf.ones((), dtype=dtype)
        chunk_mean = new_sample
        chunk_sum_squared_residuals = tf.zeros(
            ps.shape(self.sum_squared_residuals),
            dtype=dtype)

      new_n = self.num_samples + chunk_n
      delta_mean = chunk_mean - self.mean
      new_mean = self.mean + chunk_n * delta_mean / new_n
      all_pairwise_deltas = _batch_outer_product(
          delta_mean, self.event_ndims)
      adj_factor = self.num_samples * chunk_n / (self.num_samples + chunk_n)
      new_sum_squared_residuals = (self.sum_squared_residuals
                                   + chunk_sum_squared_residuals
                                   + adj_factor * all_pairwise_deltas)
      return type(self)(
          new_n, new_mean, new_sum_squared_residuals, self.event_ndims)

  def covariance(self, ddof=0):
    """Returns the covariance accumulated so far.

    Args:
      ddof: Requested dynamic degrees of freedom for the covariance calculation.
        For example, use `ddof=0` for population covariance and `ddof=1` for
        sample covariance. Defaults to the population covariance.

    Returns:
      covariance: An estimate of the covariance.
    """
    with tf.name_scope((self.name or 'RunningCovariance') + '.covariance'):
      ddof = tf.convert_to_tensor(ddof, dtype_hint=self.num_samples.dtype)
      return self.sum_squared_residuals / (self.num_samples - ddof)


def _default_covariance_event_ndims(event_ndims, shape):
  """Try to default `event_ndims` to 'full covariance' for the given `shape`."""
  shape = tf.TensorShape(shape)
  if event_ndims is None:
    if shape.rank is None:
      raise ValueError('Cannot default to computing all covariances for '
                       'samples of statically unknown rank.')
    else:
      event_ndims = shape.rank
  if shape.rank is not None and event_ndims > shape.rank:
    raise ValueError('Cannot calculate cross-products in {} dimensions for '
                     'samples of rank {}'.format(event_ndims, len(shape)))
  if event_ndims > 13:
    # This 13 is because we use an einsum to construct the needed outer product
    # in `_batch_outer_product`.  Each event dimension requires two distinct
    # characters to represent, and we don't want to rely on `tf.einsum`
    # supporting a larger alphabet than lower-case English letters.
    raise ValueError('`event_ndims` over 13 not supported')
  return event_ndims


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


def _float_dtype_like(dtype):
  if dtype is tf.int64:
    return tf.float64
  if dtype.is_integer:
    return tf.float32
  return dtype


@auto_composite_tensor.auto_composite_tensor(omit_kwargs='name')
class RunningVariance(RunningCovariance):
  """A running variance computation.

  This is just an alias for `RunningCovariance`, with the `event_ndims` set to 0
  to compute variances.

  `RunningVariance` is meant to serve general streaming variance needs.
  For a specialized version that fits streaming over MCMC samples, see
  `VarianceReducer` in `tfp.experimental.mcmc`.
  """

  @classmethod
  def from_shape(cls, shape=(), dtype=tf.float32):
    """Starts a `RunningVariance` from shape and dtype metadata.

    Args:
      shape: Python `Tuple` or `TensorShape` representing the shape of incoming
        samples.  This is useful to supply if the `RunningVariance` will be
        carried by a `tf.while_loop`, so that broadcasting does not change the
        shape across loop iterations.
      dtype: Dtype of incoming samples and the resulting statistics.
        By default, the dtype is `tf.float32`. Any integer dtypes will be
        cast to corresponding floats (i.e. `tf.int32` will be cast to
        `tf.float32`), as intermediate calculations should be performing
        floating-point division.

    Returns:
      var: An empty `RunningCovariance`, ready for incoming samples.
    """
    # TODO(b/172068479): Get rid of this method resolution order hack.
    mro = inspect.getmro(RunningVariance)
    # This `super` needs to exclude not just the subclass of CompositeTensor
    # that `auto_composite_tensor` generates, but also the base class
    # `RunningVariance`.  That way, we get the from_shape of
    # `RunningCovariance`, which is what we want here.
    # pylint: disable=bad-super-call
    return super(mro[1], cls).from_shape(shape, dtype, event_ndims=0)

  def variance(self, ddof=0):
    """Returns the variance accumulated so far.

    Args:
      ddof: Requested dynamic degrees of freedom for the variance calculation.
        For example, use `ddof=0` for population variance and `ddof=1` for
        sample variance. Defaults to the population variance.

    Returns:
      variance: An estimate of the variance.
    """
    return self.covariance(ddof)


RunningMeanState = collections.namedtuple(
    'RunningMeanState', 'num_samples, mean')


class RunningMean(object):
  """Holds metadata for and computes a running mean.

  In computation, samples can be provided individually or in chunks. A
  "chunk" of size M implies incorporating M samples into a single expectation
  computation at once, which is more efficient than one by one. If more than one
  sample is accepted and chunking is enabled, the chunked `axis` will define
  chunking semantics for all samples.

  `RunningMean` objects do not hold state information. That information,
  which includes intermediate calculations, are held in a
  `RunningMeanState` as returned via `initialize` and `update` method
  calls.

  `RunningMean` is meant to serve general streaming expectations.
  For a specialized version that fits streaming over MCMC samples, see
  `ExpectationsReducer` in `tfp.experimental.mcmc`.
  """

  def __init__(self, shape, dtype=tf.float32):
    """Instantiates this object.

    Args:
      shape: Python `Tuple` or `TensorShape` representing the shape of
        incoming samples.
      dtype: Dtype of incoming samples and the resulting statistics.
        By default, the dtype is `tf.float32`. Any integer dtypes will be
        cast to corresponding floats (i.e. `tf.int32` will be cast to
        `tf.float32`), as intermediate calculations should be performing
        floating-point division.
    """
    self.shape = shape
    if dtype is tf.int64:
      dtype = tf.float64
    elif dtype.is_integer:
      dtype = tf.float32
    self.dtype = dtype

  def initialize(self):
    """Initializes an empty `RunningMeanState`.

    Returns:
      state: `RunningMeanState` representing a stream of no inputs.
    """
    return RunningMeanState(
        num_samples=tf.zeros((), dtype=self.dtype),
        mean=tf.zeros(self.shape, self.dtype))

  def update(self, state, new_sample, axis=None):
    """Update the `RunningMeanState` with a new sample.

    The update formula is from Philippe Pebay (2008) [1] and is identical to
    that used to calculate the intermediate mean in
    `tfp.experimental.stats.RunningCovariance` and
    `tfp.experimental.stats.RunningVariance`.

    Args:
      state: `RunningMeanState` that represents the current state of
        running statistics.
      new_sample: Incoming `Tensor` sample with shape and dtype compatible with
        those used to form the `RunningMeanState`.
      axis: If chunking is desired, this is an integer that specifies the axis
        with chunked samples. For individual samples, set this to `None`. By
        default, samples are not chunked (`axis` is None).

    Returns:
      state: `RunningMeanState` with updated calculations.

    #### References
    [1]: Philippe Pebay. Formulas for Robust, One-Pass Parallel Computation of
         Covariances and Arbitrary-Order Statistical Moments. _Technical Report
         SAND2008-6212_, 2008.
         https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
    """
    new_sample = tf.nest.map_structure(
        lambda new_sample: tf.cast(new_sample, dtype=self.dtype),
        new_sample)
    if axis is None:
      chunk_n = tf.cast(1, dtype=self.dtype)
      chunk_mean = new_sample
    else:
      chunk_n = tf.cast(ps.shape(new_sample)[axis], dtype=self.dtype)
      chunk_mean = tf.math.reduce_mean(new_sample, axis=axis)
    new_n = state.num_samples + chunk_n
    delta_mean = chunk_mean - state.mean
    new_mean = state.mean + chunk_n * delta_mean / new_n
    return RunningMeanState(new_n, new_mean)

  def finalize(self, state):
    """Finalizes expectation computation for the `state`.

    If the `finalized` method is invoked on a running state of no inputs,
    `RunningMean` will return a corresponding structure of `tf.zeros`.

    Args:
      state: `RunningMeanState` that represents the current state of
        running statistics.

    Returns:
      mean: An estimate of the mean.
    """
    return state.mean


RunningCentralMomentsState = collections.namedtuple(
    'RunningCentralMomentsState',
    'mean_state, sum_exponentiated_residuals')


class RunningCentralMoments(object):
  """Holds metadata for and computes running central moments.

  `RunningCentralMoments` will compute arbitrary central moments in
  streaming fashion following the formula proposed by Philippe Pebay
  (2008) [1]. For reference, the formula we refer to is the incremental
  version of arbitrary moments (equation 2.9). Since the algorithm computes
  moments as a function of lower ones, even if not requested, all lower
  moments will be computed as well. The moments that are actually returned
  is specified by the `moment` parameter at initialization. Note, while
  any arbitrarily high central moment is theoretically supported,
  `RunningCentralMoments` cannot guarantee numerical stability for all
  moments.

  `RunningCentralMoments` objects do not hold state information. That
  information, which includes intermediate calculations, are held in a
  `RunningCentralMomentsState` as returned via `initialize` and `update`
  method calls.

  #### References
  [1]: Philippe Pebay. Formulas for Robust, One-Pass Parallel Computation of
        Covariances and Arbitrary-Order Statistical Moments. _Technical Report
        SAND2008-6212_, 2008.
        https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
  """

  def __init__(self, shape, moment, dtype=tf.float32):
    """Instantiates this object.

    Args:
      shape: Python `Tuple` or `TensorShape` representing the shape of
        incoming samples.
      moment: Integer or iterable of integers that represent the
        desired moments to return.
      dtype: Dtype of incoming samples and the resulting statistics.
        By default, the dtype is `tf.float32`. Any integer dtypes will be
        cast to corresponding floats (i.e. `tf.int32` will be cast to
        `tf.float32`), as intermediate calculations should be performing
        floating-point division.
    """
    self.shape = shape
    if isinstance(moment, (tuple, list, np.ndarray)):
      # we want to support numpy arrays too, but must convert to a list to not
      # confuse `tf.nest.map_structure` in `finalize`
      self.moment = list(moment)
      self.max_moment = max(self.moment)
    else:
      self.moment = moment
      self.max_moment = moment
    if dtype is tf.int64:
      dtype = tf.float64
    elif dtype.is_integer:
      dtype = tf.float32
    self.dtype = dtype
    self.mean_stream = RunningMean(
        self.shape, self.dtype
    )

  def initialize(self):
    """Initializes an empty `RunningCentralMomentsState`.

    The `RunningCentralMomentsState` contains a `RunningMeanState` and
    a `Tensor` representing the sum of exponentiated residuals. The sum
    of exponentiated residuals is a `Tensor` of shape
    (`self.max_moment - 1`, `self.shape`), which contains the sum of
    residuals raised to the nth power, for all `2 <= n <= self.max_moment`.

    Returns:
      state: `RunningCentralMomentsState` representing a stream of no
        inputs.
    """
    return RunningCentralMomentsState(
        mean_state=self.mean_stream.initialize(),
        sum_exponentiated_residuals=tf.zeros(
            (self.max_moment - 1,) + self.shape, self.dtype),
    )

  def update(self, state, new_sample):
    """Update the `RunningCentralMomentsState` with a new sample.

    Args:
      state: `RunningCentralMomentsState` that represents the current
        state of running statistics.
      new_sample: Incoming `Tensor` sample with shape and dtype compatible with
        those used to form the `RunningCentralMomentsState`.

    Returns:
      state: `RunningCentralMomentsState` with updated calculations.
    """
    n_2 = 1
    n_1 = state.mean_state.num_samples
    n = tf.cast(n_1 + n_2, dtype=self.dtype)
    delta_mean = new_sample - state.mean_state.mean
    new_mean_state = self.mean_stream.update(state.mean_state, new_sample)
    old_res = tf.concat([
        tf.zeros((1,) + self.shape, self.dtype),
        state.sum_exponentiated_residuals], axis=0)
    # the sum of exponentiated residuals can be thought of as an estimation
    # of the central moment before diving through by the number of samples.
    # Since the first central moment is always 0, it simplifies update
    # logic to prepend an appropriate structure of zeros.
    new_sum_exponentiated_residuals = [tf.zeros(self.shape, self.dtype)]

    # the following two nested for loops calculate equation 2.9 in Pebay's
    # 2008 paper from smallest moment to highest.
    for p in range(2, self.max_moment + 1):
      summation = tf.zeros(self.shape, self.dtype)
      for k in range(1, p - 1):
        adjusted_old_res = ((-delta_mean / n) ** k) * old_res[p - k - 1]
        summation += self._n_choose_k(p, k) * adjusted_old_res
      # the `adj_term` refers to the final term in equation 2.9 and is not
      # transcribed exactly; rather, it's simplified to avoid having a
      # `(n - 1)` denominator.
      adj_term = (((delta_mean / n) ** p) * (n - 1) *
                  ((n - 1) ** (p - 1) + (-1) ** p))
      new_sum_pth_residual = old_res[p - 1] + summation + adj_term
      new_sum_exponentiated_residuals.append(new_sum_pth_residual)

    return RunningCentralMomentsState(
        new_mean_state,
        sum_exponentiated_residuals=tf.convert_to_tensor(
            new_sum_exponentiated_residuals[1:], dtype=self.dtype
        )
    )

  def finalize(self, state):
    """Finalizes streaming computation for all central moments.

    Args:
      state: `RunningCentralMomentsState` that represents the current state
        of running statistics.

    Returns:
      all_moments: A `Tensor` representing estimates of the requested central
        moments. Its leading dimension indexes the moment, in order of those
        requested (i.e. in order of `self.moment`).
    """
    # prepend a structure of zeros for the first moment
    all_unfinalized_moments = tf.concat([
        tf.zeros((1,) + self.shape, self.dtype),
        state.sum_exponentiated_residuals], axis=0)
    all_moments = all_unfinalized_moments / tf.cast(
        state.mean_state.num_samples, self.dtype)
    return tf.convert_to_tensor(tf.nest.map_structure(
        lambda i: all_moments[i - 1],
        self.moment), self.dtype)

  def _n_choose_k(self, n, k):
    """Computes nCk."""
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


@auto_composite_tensor.auto_composite_tensor(omit_kwargs='name')
class RunningPotentialScaleReduction(object):
  """A running R-hat diagnostic.

  `RunningPotentialScaleReduction` uses Gelman and Rubin (1992)'s potential
  scale reduction (also known as R-hat) for chain convergence [1].

  If multiple independent R-hat computations are desired across a latent
  state, one should use a (possibly nested) collection for initialization
  parameters `independent_chain_ndims` and `shape`. Subsequent chain states
  used to update the streaming R-hat should mimic their identical structure.

  `RunningPotentialScaleReduction` also assumes that incoming samples have shape
  `[Ci1, Ci2,...,CiD] + A`. Dimensions `0` through `D - 1` index the
  `Ci1 x ... x CiD` independent chains to be tested for convergence to the same
  target. The remaining dimensions, `A`, represent the event shape and hence,
  can have any shape (even empty, which implies scalar samples). The number of
  independent chain dimensions is defined by the `independent_chain_ndims`
  parameter at initialization.

  `RunningPotentialScaleReduction` is meant to serve general streaming R-hat.
  For a specialized version that fits streaming over MCMC samples, see
  `PotentialScaleReductionReducer` in `tfp.experimental.mcmc`.

  #### References

  [1]: Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation
       Using Multiple Sequences. _Statistical Science_, 7(4):457-472, 1992.
  """

  def __init__(self, chain_variances, independent_chain_ndims):
    """Construct a `RunningPotentialScaleReduction`.

    Args:
      chain_variances: A `RunningVariance` or nested structure of
        `RunningVariance`s, giving the variance estimates for the variables of
        interest.
      independent_chain_ndims: A Python `int` or structure of Python `ints`
        parallel to `chain_variances` giving the number of leading dimensions in
        `chain_variances` that index the independent chains over which the
        potential scale reduction factor should be computed.  Must be at least
        1.
    """
    self.chain_variances = chain_variances
    self.independent_chain_ndims = independent_chain_ndims

  @classmethod
  def from_shape(cls, shape=(), independent_chain_ndims=1, dtype=tf.float32):
    """Starts an empty `RunningPotentialScaleReduction` from metadata.

    Args:
      shape: Python `Tuple` or `TensorShape` representing the shape of incoming
        samples. Using a collection implies that future samples will mimic that
        exact structure. This is useful to supply if the
        `RunningPotentialScaleReduction` will be carried by a `tf.while_loop`,
        so that broadcasting does not change the shape across loop iterations.
      independent_chain_ndims: Integer or Integer type `Tensor` with value
        `>= 1` giving the number of leading dimensions holding independent
        chain results to be tested for convergence. Using a collection
        implies that future samples will mimic that exact structure.
      dtype: Dtype of incoming samples and the resulting statistics.
        By default, the dtype is `tf.float32`. Any integer dtypes will be
        cast to corresponding floats (i.e. `tf.int32` will be cast to
        `tf.float32`), as intermediate calculations should be performing
        floating-point division.

    Returns:
      state: `RunningPotentialScaleReduction` representing a stream
        of no inputs.
    """
    dtype = tf.nest.map_structure(_float_dtype_like, dtype)

    dtype = nest_util.broadcast_structure(independent_chain_ndims, dtype)
    chain_variances = nest.map_structure_up_to(
        independent_chain_ndims,
        RunningVariance.from_shape,
        shape,
        dtype,
        check_types=False)
    return cls(chain_variances, independent_chain_ndims)

  def update(self, new_sample):
    """Update the `RunningPotentialScaleReduction` with a new sample.

    Args:
      new_sample: Incoming `Tensor` sample or (possibly nested) collection of
        `Tensor`s with shape and dtype compatible with those used to form the
        `RunningPotentialScaleReduction`.

    Returns:
      state: `RunningPotentialScaleReduction` updated to include the new sample.
    """
    def _update_for_one_state(chain_variances, new_sample):
      """Updates the running variance for one group of Markov chains."""
      # TODO(axch): chunking could be reasonably added here by accepting and
      # including the chunked axis to the running variance object
      return chain_variances.update(new_sample)
    updated_chain_variancess = tf.nest.map_structure(
        _update_for_one_state,
        self.chain_variances,
        new_sample,
        check_types=False
    )
    return type(self)(updated_chain_variancess, self.independent_chain_ndims)

  def potential_scale_reduction(self):
    """Computes the potential scale reduction for samples accumulated so far.

    Returns:
      rhat: An estimate of the R-hat.
    """
    def _finalize_for_one_state(chain_ndims, chain_variances):
      """Calculates R-hat for one group of Markov chains."""
      # using notation from Brooks and Gelman (1998),
      # n := num samples / chain; m := number of chains
      n = chain_variances.num_samples
      shape = chain_variances.mean.shape
      m = tf.cast(
          functools.reduce((lambda x, y: x * y), (shape[:chain_ndims])),
          n.dtype)

      # b/n is the between-chain variance (the variance of the chain means)
      b_div_n = diagnostic._reduce_variance(  # pylint:disable=protected-access
          tf.convert_to_tensor(chain_variances.mean),
          axis=tf.range(chain_ndims),
          biased=False)

      # W is the within sequence variance (the mean of the chain variances)
      sum_of_chain_squared_residuals = tf.reduce_sum(
          chain_variances.sum_squared_residuals, axis=tf.range(chain_ndims))
      w = sum_of_chain_squared_residuals / (m * (n - 1))

      # the `true_variance_estimate` is denoted as sigma^2_+ in the 1998 paper
      true_variance_estimate = ((n - 1) / n) * w + b_div_n
      return ((m + 1.) / m) * true_variance_estimate / w - (n - 1.) / (m * n)

    return tf.nest.map_structure(
        _finalize_for_one_state,
        self.independent_chain_ndims,
        self.chain_variances,
        check_types=False
    )
