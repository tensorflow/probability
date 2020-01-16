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
"""Functions for computing statistics of samples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.generic import reduce_logmeanexp


__all__ = [
    'auto_correlation',
    'cholesky_covariance',
    'correlation',
    'covariance',
    'log_average_probs',
    'stddev',
    'variance',
]


# TODO(langmore) Write separate versions of this for real/complex dtype, taking
# advantage of optimized real-fft ops.
def auto_correlation(x,
                     axis=-1,
                     max_lags=None,
                     center=True,
                     normalize=True,
                     name='auto_correlation'):
  """Auto correlation along one axis.

  Given a `1-D` wide sense stationary (WSS) sequence `X`, the auto correlation
  `RXX` may be defined as  (with `E` expectation and `Conj` complex conjugate)

  ```
  RXX[m] := E{ W[m] Conj(W[0]) } = E{ W[0] Conj(W[-m]) },
  W[n]   := (X[n] - MU) / S,
  MU     := E{ X[0] },
  S**2   := E{ (X[0] - MU) Conj(X[0] - MU) }.
  ```

  This function takes the viewpoint that `x` is (along one axis) a finite
  sub-sequence of a realization of (WSS) `X`, and then uses `x` to produce an
  estimate of `RXX[m]` as follows:

  After extending `x` from length `L` to `inf` by zero padding, the auto
  correlation estimate `rxx[m]` is computed for `m = 0, 1, ..., max_lags` as

  ```
  rxx[m] := (L - m)**-1 sum_n w[n + m] Conj(w[n]),
  w[n]   := (x[n] - mu) / s,
  mu     := L**-1 sum_n x[n],
  s**2   := L**-1 sum_n (x[n] - mu) Conj(x[n] - mu)
  ```

  The error in this estimate is proportional to `1 / sqrt(len(x) - m)`, so users
  often set `max_lags` small enough so that the entire output is meaningful.

  Note that since `mu` is an imperfect estimate of `E{ X[0] }`, and we divide by
  `len(x) - m` rather than `len(x) - m - 1`, our estimate of auto correlation
  contains a slight bias, which goes to zero as `len(x) - m --> infinity`.

  Args:
    x:  `float32` or `complex64` `Tensor`.
    axis:  Python `int`. The axis number along which to compute correlation.
      Other dimensions index different batch members.
    max_lags:  Positive `int` tensor.  The maximum value of `m` to consider (in
      equation above).  If `max_lags >= x.shape[axis]`, we effectively re-set
      `max_lags` to `x.shape[axis] - 1`.
    center:  Python `bool`.  If `False`, do not subtract the mean estimate `mu`
      from `x[n]` when forming `w[n]`.
    normalize:  Python `bool`.  If `False`, do not divide by the variance
      estimate `s**2` when forming `w[n]`.
    name:  `String` name to prepend to created ops.

  Returns:
    `rxx`: `Tensor` of same `dtype` as `x`.  `rxx.shape[i] = x.shape[i]` for
      `i != axis`, and `rxx.shape[axis] = max_lags + 1`.

  Raises:
    TypeError:  If `x` is not a supported type.
  """
  # Implementation details:
  # Extend length N / 2 1-D array x to length N by zero padding onto the end.
  # Then, set
  #   F[x]_k := sum_n x_n exp{-i 2 pi k n / N }.
  # It is not hard to see that
  #   F[x]_k Conj(F[x]_k) = F[R]_k, where
  #   R_m := sum_n x_n Conj(x_{(n - m) mod N}).
  # One can also check that R_m / (N / 2 - m) is an unbiased estimate of RXX[m].

  # Since F[x] is the DFT of x, this leads us to a zero-padding and FFT/IFFT
  # based version of estimating RXX.
  # Note that this is a special case of the Wiener-Khinchin Theorem.
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name='x')

    # Rotate dimensions of x in order to put axis at the rightmost dim.
    # FFT op requires this.
    rank = prefer_static.rank(x)
    if axis < 0:
      axis = rank + axis
    shift = rank - 1 - axis
    # Suppose x.shape[axis] = T, so there are T 'time' steps.
    #   ==> x_rotated.shape = B + [T],
    # where B is x_rotated's batch shape.
    x_rotated = distribution_util.rotate_transpose(x, shift)

    if center:
      x_rotated -= tf.reduce_mean(x_rotated, axis=-1, keepdims=True)

    # x_len = N / 2 from above explanation.  The length of x along axis.
    # Get a value for x_len that works in all cases.
    x_len = prefer_static.shape(x_rotated)[-1]

    # TODO(langmore) Investigate whether this zero padding helps or hurts.  At
    # the moment is necessary so that all FFT implementations work.
    # Zero pad to the next power of 2 greater than 2 * x_len, which equals
    # 2**(ceil(Log_2(2 * x_len))).  Note: Log_2(X) = Log_e(X) / Log_e(2).
    x_len_float64 = tf.cast(x_len, np.float64)
    target_length = tf.pow(
        np.float64(2.), tf.math.ceil(
            tf.math.log(x_len_float64 * 2) / np.log(2.)))
    pad_length = tf.cast(target_length - x_len_float64, np.int32)

    # We should have:
    # x_rotated_pad.shape = x_rotated.shape[:-1] + [T + pad_length]
    #                     = B + [T + pad_length]
    x_rotated_pad = distribution_util.pad(
        x_rotated, axis=-1, back=True, count=pad_length)

    dtype = x.dtype
    if not dtype_util.is_complex(dtype):
      if not dtype_util.is_floating(dtype):
        raise TypeError('Argument x must have either float or complex dtype'
                        ' found: {}'.format(dtype))
      x_rotated_pad = tf.complex(
          x_rotated_pad,
          dtype_util.as_numpy_dtype(dtype_util.real_dtype(dtype))(0.))

    # Autocorrelation is IFFT of power-spectral density (up to some scaling).
    fft_x_rotated_pad = tf.signal.fft(x_rotated_pad)
    spectral_density = fft_x_rotated_pad * tf.math.conj(fft_x_rotated_pad)
    # shifted_product is R[m] from above detailed explanation.
    # It is the inner product sum_n X[n] * Conj(X[n - m]).
    shifted_product = tf.signal.ifft(spectral_density)

    # Cast back to real-valued if x was real to begin with.
    shifted_product = tf.cast(shifted_product, dtype)

    # Figure out if we can deduce the final static shape, and set max_lags.
    # Use x_rotated as a reference, because it has the time dimension in the far
    # right, and was created before we performed all sorts of crazy shape
    # manipulations.
    know_static_shape = True
    if not tensorshape_util.is_fully_defined(x_rotated.shape):
      know_static_shape = False
    if max_lags is None:
      max_lags = x_len - 1
    else:
      max_lags = tf.convert_to_tensor(max_lags, name='max_lags')
      max_lags_ = tf.get_static_value(max_lags)
      if max_lags_ is None or not know_static_shape:
        know_static_shape = False
        max_lags = tf.minimum(x_len - 1, max_lags)
      else:
        max_lags = min(x_len - 1, max_lags_)

    # Chop off the padding.
    # We allow users to provide a huge max_lags, but cut it off here.
    # shifted_product_chopped.shape = x_rotated.shape[:-1] + [max_lags]
    shifted_product_chopped = shifted_product[..., :max_lags + 1]

    # If possible, set shape.
    if know_static_shape:
      chopped_shape = tensorshape_util.as_list(x_rotated.shape)
      chopped_shape[-1] = min(x_len, max_lags + 1)
      tensorshape_util.set_shape(shifted_product_chopped, chopped_shape)

    # Recall R[m] is a sum of N / 2 - m nonzero terms x[n] Conj(x[n - m]).  The
    # other terms were zeros arising only due to zero padding.
    # `denominator = (N / 2 - m)` (defined below) is the proper term to
    # divide by to make this an unbiased estimate of the expectation
    # E[X[n] Conj(X[n - m])].
    x_len = tf.cast(x_len, dtype_util.real_dtype(dtype))
    max_lags = tf.cast(max_lags, dtype_util.real_dtype(dtype))
    denominator = x_len - tf.range(0., max_lags + 1.)
    denominator = tf.cast(denominator, dtype)
    shifted_product_rotated = shifted_product_chopped / denominator

    if normalize:
      shifted_product_rotated /= shifted_product_rotated[..., :1]

    # Transpose dimensions back to those of x.
    return distribution_util.rotate_transpose(shifted_product_rotated, -shift)


def cholesky_covariance(x, sample_axis=0, keepdims=False, name=None):
  """Cholesky factor of the covariance matrix of vector-variate random samples.

  This function can be use to fit a multivariate normal to data.

  ```python
  tf.enable_eager_execution()
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Assume data.shape = (1000, 2).  1000 samples of a random variable in R^2.
  observed_data = read_data_samples(...)

  # The mean is easy
  mu = tf.reduce_mean(observed_data, axis=0)

  # Get the scale matrix
  L = tfp.stats.cholesky_covariance(observed_data)

  # Make the best fit multivariate normal (under maximum likelihood condition).
  mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)

  # Plot contours of the pdf.
  xs, ys = tf.meshgrid(
      tf.linspace(-5., 5., 50), tf.linspace(-5., 5., 50), indexing='ij')
  xy = tf.stack((tf.reshape(xs, [-1]), tf.reshape(ys, [-1])), axis=-1)
  pdf = tf.reshape(mvn.prob(xy), (50, 50))
  CS = plt.contour(xs, ys, pdf, 10)
  plt.clabel(CS, inline=1, fontsize=10)
  ```

  Why does this work?
  Given vector-variate random variables `X = (X1, ..., Xd)`, one may obtain the
  sample covariance matrix in `R^{d x d}` (see `tfp.stats.covariance`).

  The [Cholesky factor](https://en.wikipedia.org/wiki/Cholesky_decomposition)
  of this matrix is analogous to standard deviation for scalar random variables:
  Suppose `X` has covariance matrix `C`, with Cholesky factorization `C = L L^T`
  Then multiplying a vector of iid random variables which have unit variance by
  `L` produces a vector with covariance `L L^T`, which is the same as `X`.

  ```python
  observed_data = read_data_samples(...)
  L = tfp.stats.cholesky_covariance(observed_data, sample_axis=0)

  # Make fake_data with the same covariance as observed_data.
  uncorrelated_normal = tf.random_normal(shape=(500, 10))
  fake_data = tf.linalg.matvec(L, uncorrelated_normal)
  ```

  Args:
    x:  Numeric `Tensor`.  The rightmost dimension of `x` indexes events. E.g.
      dimensions of a random vector.
    sample_axis: Scalar or vector `Tensor` designating axis holding samples.
      Default value: `0` (leftmost dimension). Cannot be the rightmost dimension
        (since this indexes events).
    keepdims:  Boolean.  Whether to keep the sample axis as singletons.
    name: Python `str` name prefixed to Ops created by this function.
          Default value: `None` (i.e., `'covariance'`).

  Returns:
    chol:  `Tensor` of same `dtype` as `x`.  The last two dimensions hold
      lower triangular matrices (the Cholesky factors).
  """
  with tf.name_scope(name or 'cholesky_covariance'):
    sample_axis = tf.convert_to_tensor(sample_axis, dtype=tf.int32)
    cov = covariance(
        x, sample_axis=sample_axis, event_axis=-1, keepdims=keepdims)
    return tf.linalg.cholesky(cov)


def covariance(x,
               y=None,
               sample_axis=0,
               event_axis=-1,
               keepdims=False,
               name=None):
  """Sample covariance between observations indexed by `event_axis`.

  Given `N` samples of scalar random variables `X` and `Y`, covariance may be
  estimated as

  ```none
  Cov[X, Y] := N^{-1} sum_{n=1}^N (X_n - Xbar) Conj{(Y_n - Ybar)}
  Xbar := N^{-1} sum_{n=1}^N X_n
  Ybar := N^{-1} sum_{n=1}^N Y_n
  ```

  For vector-variate random variables `X = (X1, ..., Xd)`, `Y = (Y1, ..., Yd)`,
  one is often interested in the covariance matrix, `C_{ij} := Cov[Xi, Yj]`.

  ```python
  x = tf.random_normal(shape=(100, 2, 3))
  y = tf.random_normal(shape=(100, 2, 3))

  # cov[i, j] is the sample covariance between x[:, i, j] and y[:, i, j].
  cov = tfp.stats.covariance(x, y, sample_axis=0, event_axis=None)

  # cov_matrix[i, m, n] is the sample covariance of x[:, i, m] and y[:, i, n]
  cov_matrix = tfp.stats.covariance(x, y, sample_axis=0, event_axis=-1)
  ```

  Notice we divide by `N` (the numpy default), which does not create `NaN`
  when `N = 1`, but is slightly biased.

  Args:
    x:  A numeric `Tensor` holding samples.
    y:  Optional `Tensor` with same `dtype` and `shape` as `x`.
      Default value: `None` (`y` is effectively set to `x`).
    sample_axis: Scalar or vector `Tensor` designating axis holding samples, or
      `None` (meaning all axis hold samples).
      Default value: `0` (leftmost dimension).
    event_axis:  Scalar or vector `Tensor`, or `None` (scalar events).
      Axis indexing random events, whose covariance we are interested in.
      If a vector, entries must form a contiguous block of dims. `sample_axis`
      and `event_axis` should not intersect.
      Default value: `-1` (rightmost axis holds events).
    keepdims:  Boolean.  Whether to keep the sample axis as singletons.
    name: Python `str` name prefixed to Ops created by this function.
          Default value: `None` (i.e., `'covariance'`).

  Returns:
    cov: A `Tensor` of same `dtype` as the `x`, and rank equal to
      `rank(x) - len(sample_axis) + 2 * len(event_axis)`.

  Raises:
    AssertionError:  If `x` and `y` are found to have different shape.
    ValueError:  If `sample_axis` and `event_axis` are found to overlap.
    ValueError:  If `event_axis` is found to not be contiguous.
  """

  with tf.name_scope(name or 'covariance'):
    x = tf.convert_to_tensor(x, name='x')
    # Covariance *only* uses the centered versions of x (and y).
    x -= tf.reduce_mean(x, axis=sample_axis, keepdims=True)

    if y is None:
      y = x
    else:
      y = tf.convert_to_tensor(y, name='y', dtype=x.dtype)
      # If x and y have different shape, sample_axis and event_axis will likely
      # be wrong for one of them!
      tensorshape_util.assert_is_compatible_with(x.shape, y.shape)
      y -= tf.reduce_mean(y, axis=sample_axis, keepdims=True)

    if event_axis is None:
      return tf.reduce_mean(
          x * tf.math.conj(y), axis=sample_axis, keepdims=keepdims)

    if sample_axis is None:
      raise ValueError(
          'sample_axis was None, which means all axis hold events, and this '
          'overlaps with event_axis ({})'.format(event_axis))

    event_axis = _make_positive_axis(event_axis, tf.rank(x))
    sample_axis = _make_positive_axis(sample_axis, tf.rank(x))

    # If we get lucky and axis is statically defined, we can do some checks.
    if _is_list_like(event_axis) and _is_list_like(sample_axis):
      event_axis = tuple(map(int, event_axis))
      sample_axis = tuple(map(int, sample_axis))
      if set(event_axis).intersection(sample_axis):
        raise ValueError(
            'sample_axis ({}) and event_axis ({}) overlapped'.format(
                sample_axis, event_axis))
      if (np.diff(np.array(sorted(event_axis))) > 1).any():
        raise ValueError(
            'event_axis must be contiguous. Found: {}'.format(event_axis))
      batch_axis = list(
          sorted(
              set(range(tensorshape_util.rank(
                  x.shape))).difference(sample_axis + event_axis)))
    else:
      batch_axis = prefer_static.setdiff1d(
          tf.range(0, tf.rank(x)), tf.concat((sample_axis, event_axis), 0))

    event_axis = tf.convert_to_tensor(
        event_axis, name='event_axis', dtype=tf.int32)
    sample_axis = tf.convert_to_tensor(
        sample_axis, name='sample_axis', dtype=tf.int32)
    batch_axis = tf.convert_to_tensor(
        batch_axis, name='batch_axis', dtype=tf.int32)

    # Permute x/y until shape = B + E + S
    perm_for_xy = tf.concat((batch_axis, event_axis, sample_axis), 0)
    x_permed = tf.transpose(a=x, perm=perm_for_xy)
    y_permed = tf.transpose(a=y, perm=perm_for_xy)

    batch_ndims = tf.size(batch_axis)
    batch_shape = tf.shape(x_permed)[:batch_ndims]
    event_ndims = tf.size(event_axis)
    event_shape = tf.shape(x_permed)[batch_ndims:batch_ndims + event_ndims]
    sample_shape = tf.shape(x_permed)[batch_ndims + event_ndims:]
    sample_ndims = tf.size(sample_shape)
    n_samples = tf.reduce_prod(sample_shape)
    n_events = tf.reduce_prod(event_shape)

    # Flatten sample_axis into one long dim.
    x_permed_flat = tf.reshape(
        x_permed, tf.concat((batch_shape, event_shape, [n_samples]), 0))
    y_permed_flat = tf.reshape(
        y_permed, tf.concat((batch_shape, event_shape, [n_samples]), 0))
    # Do the same for event_axis.
    x_permed_flat = tf.reshape(
        x_permed, tf.concat((batch_shape, [n_events], [n_samples]), 0))
    y_permed_flat = tf.reshape(
        y_permed, tf.concat((batch_shape, [n_events], [n_samples]), 0))

    # After matmul, cov.shape = batch_shape + [n_events, n_events]
    cov = tf.matmul(
        x_permed_flat, y_permed_flat, adjoint_b=True) / tf.cast(
            n_samples, x.dtype)

    # Insert some singletons to make
    # cov.shape = batch_shape + event_shape**2 + [1,...,1]
    # This is just like x_permed.shape, except the sample_axis is all 1's, and
    # the [n_events] became event_shape**2.
    cov = tf.reshape(
        cov,
        tf.concat(
            (
                batch_shape,
                # event_shape**2 used here because it is the same length as
                # event_shape, and has the same number of elements as one
                # batch of covariance.
                event_shape**2,
                tf.ones([sample_ndims], tf.int32)),
            0))
    # Permuting by the argsort inverts the permutation, making
    # cov.shape have ones in the position where there were samples, and
    # [n_events * n_events] in the event position.
    cov = tf.transpose(a=cov, perm=tf.math.invert_permutation(perm_for_xy))

    # Now expand event_shape**2 into event_shape + event_shape.
    # We here use (for the first time) the fact that we require event_axis to be
    # contiguous.
    e_start = event_axis[0]
    e_len = 1 + event_axis[-1] - event_axis[0]
    cov = tf.reshape(
        cov,
        tf.concat((tf.shape(cov)[:e_start], event_shape, event_shape,
                   tf.shape(cov)[e_start + e_len:]), 0))

    # tf.squeeze requires python ints for axis, not Tensor.  This is enough to
    # require our axis args to be constants.
    if not keepdims:
      squeeze_axis = tf.where(sample_axis < e_start, sample_axis,
                              sample_axis + e_len)
      cov = _squeeze(cov, axis=squeeze_axis)

    return cov


def correlation(x,
                y=None,
                sample_axis=0,
                event_axis=-1,
                keepdims=False,
                name=None):
  """Sample correlation (Pearson) between observations indexed by `event_axis`.

  Given `N` samples of scalar random variables `X` and `Y`, correlation may be
  estimated as

  ```none
  Corr[X, Y] := Cov[X, Y] / Sqrt(Cov[X, X] * Cov[Y, Y]),
  where
  Cov[X, Y] := N^{-1} sum_{n=1}^N (X_n - Xbar) Conj{(Y_n - Ybar)}
  Xbar := N^{-1} sum_{n=1}^N X_n
  Ybar := N^{-1} sum_{n=1}^N Y_n
  ```

  Correlation is always in the interval `[-1, 1]`, and `Corr[X, X] == 1`.

  For vector-variate random variables `X = (X1, ..., Xd)`, `Y = (Y1, ..., Yd)`,
  one is often interested in the correlation matrix, `C_{ij} := Corr[Xi, Yj]`.

  ```python
  x = tf.random_normal(shape=(100, 2, 3))
  y = tf.random_normal(shape=(100, 2, 3))

  # corr[i, j] is the sample correlation between x[:, i, j] and y[:, i, j].
  corr = tfp.stats.correlation(x, y, sample_axis=0, event_axis=None)

  # corr_matrix[i, m, n] is the sample correlation of x[:, i, m] and y[:, i, n]
  corr_matrix = tfp.stats.correlation(x, y, sample_axis=0, event_axis=-1)
  ```

  Notice we divide by `N` (the numpy default), which does not create `NaN`
  when `N = 1`, but is slightly biased.

  Args:
    x:  A numeric `Tensor` holding samples.
    y:  Optional `Tensor` with same `dtype` and `shape` as `x`.
      Default value: `None` (`y` is effectively set to `x`).
    sample_axis: Scalar or vector `Tensor` designating axis holding samples, or
      `None` (meaning all axis hold samples).
      Default value: `0` (leftmost dimension).
    event_axis:  Scalar or vector `Tensor`, or `None` (scalar events).
      Axis indexing random events, whose correlation we are interested in.
      If a vector, entries must form a contiguous block of dims. `sample_axis`
      and `event_axis` should not intersect.
      Default value: `-1` (rightmost axis holds events).
    keepdims:  Boolean.  Whether to keep the sample axis as singletons.
    name: Python `str` name prefixed to Ops created by this function.
          Default value: `None` (i.e., `'correlation'`).

  Returns:
    corr: A `Tensor` of same `dtype` as the `x`, and rank equal to
      `rank(x) - len(sample_axis) + 2 * len(event_axis)`.

  Raises:
    AssertionError:  If `x` and `y` are found to have different shape.
    ValueError:  If `sample_axis` and `event_axis` are found to overlap.
    ValueError:  If `event_axis` is found to not be contiguous.
  """

  with tf.name_scope(name or 'correlation'):
    # Corr[X, Y] = Cov[X, Y] / (Stddev[X] * Stddev[Y])
    #            = Cov[X / Stddev[X], Y / Stddev[Y]]
    # So we could compute covariance first then divide by stddev, or
    # divide by stddev and compute covariance.
    # Dividing by stddev then computing covariance is potentially more stable.
    # But... computing covariance first then dividing involves 2 fewer large
    # broadcasts.  We choose to divide first, largely because it avoids
    # difficulties with the various options for sample/event axis kwargs.

    x /= stddev(x, sample_axis=sample_axis, keepdims=True)
    if y is not None:
      y /= stddev(y, sample_axis=sample_axis, keepdims=True)

    return covariance(
        x=x,
        y=y,
        event_axis=event_axis,
        sample_axis=sample_axis,
        keepdims=keepdims)


def stddev(x, sample_axis=0, keepdims=False, name=None):
  """Estimate standard deviation using samples.

  Given `N` samples of scalar valued random variable `X`, standard deviation may
  be estimated as

  ```none
  Stddev[X] := Sqrt[Var[X]],
  Var[X] := N^{-1} sum_{n=1}^N (X_n - Xbar) Conj{(X_n - Xbar)},
  Xbar := N^{-1} sum_{n=1}^N X_n
  ```

  ```python
  x = tf.random_normal(shape=(100, 2, 3))

  # stddev[i, j] is the sample standard deviation of the (i, j) batch member.
  stddev = tfp.stats.stddev(x, sample_axis=0)
  ```

  Scaling a unit normal by a standard deviation produces normal samples
  with that standard deviation.

  ```python
  observed_data = read_data_samples(...)
  stddev = tfp.stats.stddev(observed_data)

  # Make fake_data with the same standard deviation as observed_data.
  fake_data = stddev * tf.random_normal(shape=(100,))
  ```

  Notice we divide by `N` (the numpy default), which does not create `NaN`
  when `N = 1`, but is slightly biased.

  Args:
    x:  A numeric `Tensor` holding samples.
    sample_axis: Scalar or vector `Tensor` designating axis holding samples, or
      `None` (meaning all axis hold samples).
      Default value: `0` (leftmost dimension).
    keepdims:  Boolean.  Whether to keep the sample axis as singletons.
    name: Python `str` name prefixed to Ops created by this function.
          Default value: `None` (i.e., `'stddev'`).

  Returns:
    stddev: A `Tensor` of same `dtype` as the `x`, and rank equal to
      `rank(x) - len(sample_axis)`
  """
  with tf.name_scope(name or 'stddev'):
    return tf.sqrt(variance(x, sample_axis=sample_axis, keepdims=keepdims))


def variance(x, sample_axis=0, keepdims=False, name=None):
  """Estimate variance using samples.

  Given `N` samples of scalar valued random variable `X`, variance may
  be estimated as

  ```none
  Var[X] := N^{-1} sum_{n=1}^N (X_n - Xbar) Conj{(X_n - Xbar)}
  Xbar := N^{-1} sum_{n=1}^N X_n
  ```

  ```python
  x = tf.random_normal(shape=(100, 2, 3))

  # var[i, j] is the sample variance of the (i, j) batch member of x.
  var = tfp.stats.variance(x, sample_axis=0)
  ```

  Notice we divide by `N` (the numpy default), which does not create `NaN`
  when `N = 1`, but is slightly biased.

  Args:
    x:  A numeric `Tensor` holding samples.
    sample_axis: Scalar or vector `Tensor` designating axis holding samples, or
      `None` (meaning all axis hold samples).
      Default value: `0` (leftmost dimension).
    keepdims:  Boolean.  Whether to keep the sample axis as singletons.
    name: Python `str` name prefixed to Ops created by this function.
          Default value: `None` (i.e., `'variance'`).

  Returns:
    var: A `Tensor` of same `dtype` as the `x`, and rank equal to
      `rank(x) - len(sample_axis)`
  """
  with tf.name_scope(name or 'variance'):
    return covariance(
        x, y=None, sample_axis=sample_axis, event_axis=None, keepdims=keepdims)


def log_average_probs(logits, sample_axis=0, event_axis=None, keepdims=False,
                      validate_args=False, name=None):
  """Computes `log(average(to_probs(logits)))` in a numerically stable manner.

  The meaning of `to_probs` is controlled by the `event_axis` argument. When
  `event_axis` is `None`, `to_probs = tf.math.sigmoid` and otherwise
  `to_probs = lambda x: tf.math.log_softmax(x, axis=event_axis)`.

  `sample_axis` and `event_axis` should have a null intersection. This
  requirement is always verified when `validate_args` is `True`.

  Args:
    logits: A `float` `Tensor` representing logits.
    sample_axis: Scalar or vector `Tensor` designating axis holding samples, or
      `None` (meaning all axis hold samples).
      Default value: `0` (leftmost dimension).
    event_axis: Scalar or vector `Tensor` designating the axis representing
      categorical logits.
      Default value: `None` (i.e., Bernoulli logits).
    keepdims:  Boolean.  Whether to keep the sample axis as singletons.
      Default value: `False` (i.e., squeeze the reduced dimensions).
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
      Default value: `False` (i.e., do not validate args).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'log_average_probs'`).

  Returns:
    log_avg_probs: The natural log of the average of probs computed from logits.
  """
  with tf.name_scope(name or 'average_sigmoid'):
    logits = tf.convert_to_tensor(logits, dtype_hint=tf.float32, name='logits')
    if sample_axis is not None:
      sample_axis = tf.convert_to_tensor(
          sample_axis, dtype_hint=tf.int32, name='sample_axis')
    if event_axis is not None:
      event_axis = tf.convert_to_tensor(
          event_axis, dtype_hint=tf.int32, name='event_axis')
    if event_axis is None:
      # log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log1p(exp(-x)) = -sp(-x)
      log_probs = -tf.math.softplus(-logits)
    else:
      sample_axis, event_axis = _log_average_probs_process_args(
          logits, validate_args, sample_axis, event_axis)
      with tf.control_dependencies(_log_average_probs_maybe_check_args(
          sample_axis, event_axis, validate_args)):
        log_probs = _log_softmax(logits, axis=event_axis)
    return reduce_logmeanexp(log_probs, axis=sample_axis, keepdims=keepdims)


# TODO(b/137873989): Use tf.log_softmax once it correctly supports axis arg.
def _log_softmax(x, axis, name=None):
  """Alternative to `tf.log_softmax` which correctly supports axis arg."""
  with tf.name_scope(name or 'log_softmax'):
    if axis is None:
      return tf.math.log_softmax(x, axis=None, name=name)
    rank = prefer_static.rank(axis)
    if rank == 0:
      return tf.math.log_softmax(x, axis=axis, name=name)
    if rank == 1:
      return tf.math.log_softmax(x, axis=axis[0], name=name)
    # The following handles the case when axis is a vector and which is not
    # currently supported by tf.math.log_softmax.
    x = tf.convert_to_tensor(x, dtype_hint=tf.float32, name='x')
    return x - tf.reduce_logsumexp(x, axis=axis, keepdims=True)


def _log_average_probs_process_args(
    logits, validate_args, sample_axis, event_axis):
  """Processes args for `log_average_probs`."""
  rank = prefer_static.rank(logits)
  if sample_axis is None or validate_args:
    event_axis = prefer_static.reshape(
        prefer_static.non_negative_axis(event_axis, rank),
        shape=[-1])
  if sample_axis is None:
    sample_axis = prefer_static.setdiff1d(
        prefer_static.range(rank), event_axis)
  elif validate_args:
    sample_axis = prefer_static.reshape(
        prefer_static.non_negative_axis(sample_axis, rank),
        shape=[-1])
  return sample_axis, event_axis


def _log_average_probs_maybe_check_args(sample_axis, event_axis, validate_args):
  """Assertions for `log_average_probs`."""
  assertions = []
  msg = 'Arguments `sample_axis` and `event_axis` must be distinct.'
  sample_setdiff = prefer_static.setdiff1d(sample_axis, event_axis)
  if prefer_static.is_numpy(sample_setdiff):
    if not np.array_equal(sample_setdiff, tf.get_static_value(sample_axis)):
      raise ValueError(msg)
  elif validate_args:
    assertions.append(_assert_array_equal(
        sample_setdiff, sample_axis,
        message=msg, name='sample_setdiff_rank_check'))
  event_setdiff = prefer_static.setdiff1d(event_axis, sample_axis)
  if prefer_static.is_numpy(event_setdiff):
    if not np.array_equal(event_setdiff, tf.get_static_value(event_axis)):
      raise ValueError(msg)
  elif validate_args:
    assertions.append(_assert_array_equal(
        event_setdiff, event_axis,
        message=msg, name='event_setdiff_rank_check'))
  return assertions


def _assert_array_equal(x, y, message, name=None):
  """TF assertion similar to checking `np.array_equal`."""
  with tf.name_scope(name or 'array_equal_check'):
    rank_check = assert_util.assert_equal(
        tf.rank(x), tf.rank(y),
        message=message, name='rank_check')
    shape_check = assert_util.assert_equal(
        tf.shape(x), tf.shape(y),
        message=message, name='shape_check')
    with tf.control_dependencies([rank_check]):
      with tf.control_dependencies([shape_check]):
        return assert_util.assert_equal(
            x, y, message=message, name='value_check')


def _is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))


def _make_list_or_1d_tensor(values):
  """Return a list (preferred) or 1d Tensor from values, if values.ndims < 2."""
  values = tf.convert_to_tensor(values, name='values')
  values_ = tf.get_static_value(values)

  # Static didn't work.
  if values_ is None:
    # Cheap way to bring to at least 1d.
    return values + tf.zeros([1], dtype=values.dtype)

  # Static worked!
  if values_.ndim > 1:
    raise ValueError('values had > 1 dim: {}'.format(values_.shape))
  # Cheap way to bring to at least 1d.
  values_ = values_ + np.zeros([1], dtype=values_.dtype)
  return list(values_)


def _make_positive_axis(axis, ndims):
  """Rectify possibly negatively axis. Prefer return Python list."""
  axis = _make_list_or_1d_tensor(axis)

  ndims = tf.convert_to_tensor(ndims, name='ndims', dtype=tf.int32)
  ndims_ = tf.get_static_value(ndims)

  if _is_list_like(axis) and ndims_ is not None:
    # Static case
    positive_axis = []
    for a in axis:
      if a < 0:
        a = ndims_ + a
      positive_axis.append(a)
  else:
    # Dynamic case
    axis = tf.convert_to_tensor(axis, name='axis', dtype=tf.int32)
    positive_axis = tf.where(axis >= 0, axis, axis + ndims)

  return positive_axis


def _squeeze(x, axis):
  """A version of squeeze that works with dynamic axis."""
  x = tf.convert_to_tensor(x, name='x')
  if axis is None:
    return tf.squeeze(x, axis=None)
  axis = tf.convert_to_tensor(axis, name='axis', dtype=tf.int32)
  axis += tf.zeros([1], dtype=axis.dtype)  # Make axis at least 1d.
  keep_axis = prefer_static.setdiff1d(tf.range(0, tf.rank(x)), axis)
  return tf.reshape(x, tf.gather(tf.shape(x), keep_axis))
