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
"""Quasi Monte Carlo support: Halton sequence."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import samplers


__all__ = [
    'sample_halton_sequence',
]


# The maximum dimension we support. This is limited by the number of primes
# in the _PRIMES array.
_MAX_DIMENSION = 1000


def sample_halton_sequence(dim,
                           num_results=None,
                           sequence_indices=None,
                           dtype=tf.float32,
                           randomized=True,
                           seed=None,
                           name=None):
  r"""Returns a sample from the `dim` dimensional Halton sequence.

  Warning: The sequence elements take values only between 0 and 1. Care must be
  taken to appropriately transform the domain of a function if it differs from
  the unit cube before evaluating integrals using Halton samples. It is also
  important to remember that quasi-random numbers without randomization are not
  a replacement for pseudo-random numbers in every context. Quasi random numbers
  are completely deterministic and typically have significant negative
  autocorrelation unless randomization is used.

  Computes the members of the low discrepancy Halton sequence in dimension
  `dim`. The `dim`-dimensional sequence takes values in the unit hypercube in
  `dim` dimensions. Currently, only dimensions up to 1000 are supported. The
  prime base for the k-th axes is the k-th prime starting from 2. For example,
  if `dim` = 3, then the bases will be [2, 3, 5] respectively and the first
  element of the non-randomized sequence will be: [0.5, 0.333, 0.2]. For a more
  complete description of the Halton sequences see
  [here](https://en.wikipedia.org/wiki/Halton_sequence). For low discrepancy
  sequences and their applications see
  [here](https://en.wikipedia.org/wiki/Low-discrepancy_sequence).

  If `randomized` is true, this function produces a scrambled version of the
  Halton sequence introduced by [Owen (2017)][1]. For the advantages of
  randomization of low discrepancy sequences see [here](
  https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method#Randomization_of_quasi-Monte_Carlo).

  The number of samples produced is controlled by the `num_results` and
  `sequence_indices` parameters. The user must supply either `num_results` or
  `sequence_indices` but not both.
  The former is the number of samples to produce starting from the first
  element. If `sequence_indices` is given instead, the specified elements of
  the sequence are generated. For example, sequence_indices=tf.range(10) is
  equivalent to specifying n=10.

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  # Produce the first 1000 members of the Halton sequence in 3 dimensions.
  num_results = 1000
  dim = 3
  sample = tfp.mcmc.sample_halton_sequence(
    dim,
    num_results=num_results,
    seed=127)

  # Evaluate the integral of x_1 * x_2^2 * x_3^3  over the three dimensional
  # hypercube.
  powers = tf.range(1.0, limit=dim + 1)
  integral = tf.reduce_mean(tf.reduce_prod(sample ** powers, axis=-1))
  true_value = 1.0 / tf.reduce_prod(powers + 1.0)
  with tf.Session() as session:
    values = session.run((integral, true_value))

  # Produces a relative absolute error of 1.7%.
  print ("Estimated: %f, True Value: %f" % values)

  # Now skip the first 1000 samples and recompute the integral with the next
  # thousand samples. The sequence_indices argument can be used to do this.


  sequence_indices = tf.range(start=1000, limit=1000 + num_results,
                              dtype=tf.int32)
  sample_leaped = tfp.mcmc.sample_halton_sequence(
      dim,
      sequence_indices=sequence_indices,
      seed=111217)

  integral_leaped = tf.reduce_mean(tf.reduce_prod(sample_leaped ** powers,
                                                  axis=-1))
  with tf.Session() as session:
    values = session.run((integral_leaped, true_value))
  # Now produces a relative absolute error of 0.05%.
  print ("Leaped Estimated: %f, True Value: %f" % values)
  ```

  Args:
    dim: Positive Python `int` representing each sample's `event_size.` Must
      not be greater than 1000.
    num_results: (Optional) Positive scalar `Tensor` of dtype int32. The number
      of samples to generate. Either this parameter or sequence_indices must
      be specified but not both. If this parameter is None, then the behaviour
      is determined by the `sequence_indices`.
      Default value: `None`.
    sequence_indices: (Optional) `Tensor` of dtype int32 and rank 1. The
      elements of the sequence to compute specified by their position in the
      sequence. The entries index into the Halton sequence starting with 0 and
      hence, must be whole numbers. For example, sequence_indices=[0, 5, 6] will
      produce the first, sixth and seventh elements of the sequence. If this
      parameter is None, then the `num_results` parameter must be specified
      which gives the number of desired samples starting from the first sample.
      Default value: `None`.
    dtype: (Optional) The dtype of the sample. One of: `float16`, `float32` or
      `float64`.
      Default value: `tf.float32`.
    randomized: (Optional) bool indicating whether to produce a randomized
      Halton sequence. If True, applies the randomization described in
      [Owen (2017)][1].
      Default value: `True`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details. Only used if
      `randomized` is True. If not supplied and `randomized` is True, no seed is
      set.
      Default value: `None`.
    name:  (Optional) Python `str` describing ops managed by this function. If
      not supplied the name of this function is used.
      Default value: "sample_halton_sequence".

  Returns:
    halton_elements: Elements of the Halton sequence. `Tensor` of supplied dtype
      and `shape` `[num_results, dim]` if `num_results` was specified or shape
      `[s, dim]` where s is the size of `sequence_indices` if `sequence_indices`
      were specified.

  Raises:
    ValueError: if both `sequence_indices` and `num_results` were specified or
      if dimension `dim` is less than 1 or greater than 1000.

  #### References

  [1]: Art B. Owen. A randomized Halton algorithm in R. _arXiv preprint
       arXiv:1706.02808_, 2017. https://arxiv.org/abs/1706.02808
  """
  if dim < 1 or dim > _MAX_DIMENSION:
    raise ValueError(
        'Dimension must be between 1 and {}. Supplied {}'.format(_MAX_DIMENSION,
                                                                 dim))
  if (num_results is None) == (sequence_indices is None):
    raise ValueError('Either `num_results` or `sequence_indices` must be'
                     ' specified but not both.')

  if not dtype_util.is_floating(dtype):
    raise ValueError('dtype must be of `float`-type')

  with tf.name_scope(name or 'sample'):
    # Here and in the following, the shape layout is as follows:
    # [sample dimension, event dimension, coefficient dimension].
    # The coefficient dimension is an intermediate axes which will hold the
    # weights of the starting integer when expressed in the (prime) base for
    # an event dimension.
    if num_results is not None:
      num_results = tf.convert_to_tensor(num_results)
    if sequence_indices is not None:
      sequence_indices = tf.convert_to_tensor(sequence_indices)
    indices = _get_indices(num_results, sequence_indices, dtype)
    radixes = tf.constant(_PRIMES[0:dim], dtype=dtype, shape=[dim, 1])

    max_sizes_by_axes = _base_expansion_size(
        tf.reduce_max(indices), radixes)

    max_size = tf.reduce_max(max_sizes_by_axes)

    # The powers of the radixes that we will need. Note that there is a bit
    # of an excess here. Suppose we need the place value coefficients of 7
    # in base 2 and 3. For 2, we will have 3 digits but we only need 2 digits
    # for base 3. However, we can only create rectangular tensors so we
    # store both expansions in a [2, 3] tensor. This leads to the problem that
    # we might end up attempting to raise large numbers to large powers. For
    # example, base 2 expansion of 1024 has 10 digits. If we were in 10
    # dimensions, then the 10th prime (29) we will end up computing 29^10 even
    # though we don't need it. We avoid this by setting the exponents for each
    # axes to 0 beyond the maximum value needed for that dimension.
    exponents_by_axes = tf.tile([tf.range(max_size)], [dim, 1])

    # The mask is true for those coefficients that are irrelevant.
    weight_mask = exponents_by_axes < max_sizes_by_axes
    capped_exponents = tf.where(weight_mask,
                                exponents_by_axes,
                                tf.constant(0, exponents_by_axes.dtype))
    weights = radixes ** capped_exponents
    # The following computes the base b expansion of the indices. Suppose,
    # x = a0 + a1*b + a2*b^2 + ... Then, performing a floor div of x with
    # the vector (1, b, b^2, b^3, ...) will produce
    # (a0 + s1 * b, a1 + s2 * b, ...) where s_i are coefficients we don't care
    # about. Noting that all a_i < b by definition of place value expansion,
    # we see that taking the elements mod b of the above vector produces the
    # place value expansion coefficients.
    coeffs = tf.math.floordiv(indices, weights)
    coeffs *= tf.cast(weight_mask, dtype)
    coeffs %= radixes
    if not randomized:
      coeffs /= radixes
      return tf.reduce_sum(coeffs / weights, axis=-1)

    shuffle_seed, zero_correction_seed = samplers.split_seed(
        seed, salt='MCMCSampleHaltonSequence')

    coeffs = _randomize(coeffs, radixes, seed=shuffle_seed)
    # Remove the contribution from randomizing the trailing zero for the
    # axes where max_size_by_axes < max_size. This will be accounted
    # for separately below (using zero_correction).
    coeffs *= tf.cast(weight_mask, dtype)
    coeffs /= radixes
    base_values = tf.reduce_sum(coeffs / weights, axis=-1)

    # The randomization used in Owen (2017) does not leave 0 invariant. While
    # we have accounted for the randomization of the first `max_size_by_axes`
    # coefficients, we still need to correct for the trailing zeros. Luckily,
    # this is equivalent to adding a uniform random value scaled so the first
    # `max_size_by_axes` coefficients are zero. The following statements perform
    # this correction.
    zero_correction = samplers.uniform([dim, 1],
                                       seed=zero_correction_seed,
                                       dtype=dtype)
    zero_correction /= radixes ** max_sizes_by_axes
    return base_values + tf.reshape(zero_correction, [-1])


def _randomize(coeffs, radixes, seed=None):
  """Applies the Owen (2017) randomization to the coefficients."""
  given_dtype = coeffs.dtype
  coeffs = tf.cast(coeffs, dtype=tf.int32)
  num_coeffs = tf.shape(coeffs)[-1]
  radixes = tf.reshape(tf.cast(radixes, dtype=tf.int32), shape=[-1])
  perms = _get_permutations(num_coeffs, radixes, seed=seed)
  perms = tf.reshape(perms, shape=[-1])
  radix_sum = tf.reduce_sum(radixes)
  radix_offsets = tf.reshape(tf.cumsum(radixes, exclusive=True),
                             shape=[-1, 1])
  offsets = radix_offsets + tf.range(num_coeffs) * radix_sum
  permuted_coeffs = tf.gather(perms, coeffs + offsets)
  return tf.cast(permuted_coeffs, dtype=given_dtype)


def _get_permutations(num_results, dims, seed=None):
  """Uniform iid sample from the space of permutations.

  Draws a sample of size `num_results` from the group of permutations of degrees
  specified by the `dims` tensor. These are packed together into one tensor
  such that each row is one sample from each of the dimensions in `dims`. For
  example, if dims = [2,3] and num_results = 2, the result is a tensor of shape
  [2, 2 + 3] and the first row of the result might look like:
  [1, 0, 2, 0, 1]. The first two elements are a permutation over 2 elements
  while the next three are a permutation over 3 elements.

  Args:
    num_results: A positive scalar `Tensor` of integral type. The number of
      draws from the discrete uniform distribution over the permutation groups.
    dims: A 1D `Tensor` of the same dtype as `num_results`. The degree of the
      permutation groups from which to sample.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

  Returns:
    permutations: A `Tensor` of shape `[num_results, sum(dims)]` and the same
    dtype as `dims`.
  """
  seeds = samplers.split_seed(seed, n=tf.size(dims))

  def generate_one(dim, seed):
    return tf.argsort(samplers.uniform([num_results, dim], seed=seed), axis=-1)

  return tf.concat([generate_one(dim, seed)
                    for dim, seed in zip(tf.unstack(dims), tf.unstack(seeds))],
                   axis=-1)


def _get_indices(num_results, sequence_indices, dtype, name=None):
  """Generates starting points for the Halton sequence procedure.

  The k'th element of the sequence is generated starting from a positive integer
  which must be distinct for each `k`. It is conventional to choose the starting
  point as `k` itself (or `k+1` if k is zero based). This function generates
  the starting integers for the required elements and reshapes the result for
  later use.

  Args:
    num_results: Positive scalar `Tensor` of dtype int32. The number of samples
      to generate. If this parameter is supplied, then `sequence_indices`
      should be None.
    sequence_indices: `Tensor` of dtype int32 and rank 1. The entries
      index into the Halton sequence starting with 0 and hence, must be whole
      numbers. For example, sequence_indices=[0, 5, 6] will produce the first,
      sixth and seventh elements of the sequence. If this parameter is not None
      then `n` must be None.
    dtype: The dtype of the sample. One of `float32` or `float64`.
      Default is `float32`.
    name: Python `str` name which describes ops created by this function.

  Returns:
    indices: `Tensor` of dtype `dtype` and shape = `[n, 1, 1]`.
  """
  with tf.name_scope(name or 'get_indices'):
    if sequence_indices is None:
      num_results = tf.cast(num_results, dtype=dtype)
      sequence_indices = tf.range(num_results, dtype=dtype)
    else:
      sequence_indices = tf.cast(sequence_indices, dtype)

    # Shift the indices so they are 1 based.
    indices = sequence_indices + 1

    # Reshape to make space for the event dimension and the place value
    # coefficients.
    return tf.reshape(indices, [-1, 1, 1])


def _base_expansion_size(num, bases):
  """Computes the number of terms in the place value expansion.

  Let num = a0 + a1 b + a2 b^2 + ... ak b^k be the place value expansion of
  `num` in base b (ak <> 0). This function computes and returns `k+1` for each
  base `b` specified in `bases`.

  This can be inferred from the base `b` logarithm of `num` as follows:
    $$k = Floor(log_b (num)) + 1  = Floor( log(num) / log(b)) + 1$$

  Args:
    num: Scalar `Tensor` of dtype either `float32` or `float64`. The number to
      compute the base expansion size of.
    bases: `Tensor` of the same dtype as num. The bases to compute the size
      against.

  Returns:
    Tensor of same dtype and shape as `bases` containing the size of num when
    written in that base.
  """
  return tf.floor(tf.math.log(num) / tf.math.log(bases)) + 1


def _primes_less_than(n):
  # Based on
  # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
  """Returns sorted array of primes such that `2 <= prime < n`."""
  small_primes = np.array((2, 3, 5))
  if n <= 6:
    return small_primes[small_primes < n]
  sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool_)
  sieve[0] = False
  m = int(n ** 0.5) // 3 + 1
  for i in range(m):
    if not sieve[i]:
      continue
    k = 3 * i + 1 | 1
    sieve[k ** 2 // 3::2 * k] = False
    sieve[(k ** 2 + 4 * k - 2 * k * (i & 1)) // 3::2 * k] = False
  return np.r_[2, 3, 3 * np.nonzero(sieve)[0] + 1 | 1]

_PRIMES = _primes_less_than(7919 + 1)


assert len(_PRIMES) == _MAX_DIMENSION
