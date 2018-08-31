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
"""Lewandowski-Kurowicka-Joe distribution on correlation matrices.

The sampler follows the "onion" method from
[1] Daniel Lewandowski, Dorota Kurowicka, and Harry Joe,
"Generating random correlation matrices based on vines and extended
onion method," Journal of Multivariate Analysis 100 (2009), pp
1989-2001.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfd

from tensorflow_probability.python.distributions import seed_stream


__all__ = [
    'LKJ',
]


def _uniform_unit_norm(dimension, shape, dtype, seed):
  """Returns a batch of points chosen uniformly from the unit hypersphere."""
  # This works because the Gaussian distribution is spherically symmetric.
  # raw shape: shape + [dimension]
  raw = tfd.Normal(
      loc=dtype.as_numpy_dtype(0.),
      scale=dtype.as_numpy_dtype(1.)).sample(
          tf.concat([shape, [dimension]], axis=0), seed=seed())
  unit_norm = raw / tf.norm(raw, ord=2, axis=-1)[..., tf.newaxis]
  return unit_norm


def _replicate(n, tensor):
  """Replicate the input tensor n times along a new (major) dimension."""
  # TODO(axch) Does this already exist somewhere?  Should it get contributed?
  multiples = tf.concat([[n], tf.ones_like(tensor.shape)], axis=0)
  return tf.tile(tf.expand_dims(tensor, axis=0), multiples)


class LKJ(tfd.Distribution):
  """The LKJ distribution on correlation matrices.

  This is a one-parameter family of distributions on correlation matrices.  The
  probability density is proportional to the determinant raised to the power of
  the parameter: `pdf(X; eta) = Z(eta) * det(X) ** (eta - 1)`, where `Z(eta)` is
  a normalization constant.  The uniform distribution on correlation matrices is
  the special case `eta = 1`.

  The distribution is named after Lewandowski, Kurowicka, and Joe, who gave a
  sampler for the distribution in [(Lewandowski, Kurowicka, Joe, 2009)][1].

  #### Examples

  ```python
  # Initialize a single 3x3 LKJ with concentration parameter 1.5
  dist = tfp.distributions.LKJ(dimension=3, concentration=1.5)

  # Evaluate this at a batch of two observations, each in R^{3x3}.
  x = ...  # Shape is [2, 3, 3].
  dist.prob(x)  # Shape is [2].

  # Draw 6 LKJ-distributed 3x3 correlation matrices
  ans = dist.sample(sample_shape=[2, 3], seed=42)
  # shape of ans is [2, 3, 3, 3]
  ```
  """

  def __init__(self,
               dimension,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name='LKJ'):
    """Construct LKJ distributions.

    Args:
      dimension: Python `int`. The dimension of the correlation matrices
        to sample.
      concentration: `float` or `double` `Tensor`. The positive concentration
        parameter of the LKJ distributions. The pdf of a sample matrix `X` is
        proportional to `det(X) ** (concentration - 1)`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value `NaN` to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: If `dimension` is negative.
    """
    if dimension < 0:
      raise ValueError(
          'There are no negative-dimension correlation matrices.')
    parameters = dict(locals())
    with tf.name_scope(name, values=[dimension, concentration]):
      concentration = tf.convert_to_tensor(concentration, name='concentration')
      with tf.control_dependencies([
          # concentration >= 1
          # TODO(b/111451422) Generalize to concentration > 0.
          tf.assert_non_negative(concentration - 1.),
      ] if validate_args else []):
        self._dimension = dimension
        self._concentration = tf.identity(concentration, name='concentration')
    super(LKJ, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=tfd.NOT_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._concentration],
        name=name)

  @property
  def dimension(self):
    """Dimension of returned correlation matrices."""
    return self._dimension

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  def _batch_shape_tensor(self):
    return tf.shape(self.concentration)

  def _batch_shape(self):
    return self.concentration.get_shape()

  def _event_shape_tensor(self):
    return tf.constant([self.dimension, self.dimension], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([self.dimension, self.dimension])

  def _sample_n(self, num_samples, seed=None, name=None):
    """Returns a Tensor of samples from an LKJ distribution.

    Args:
      num_samples: Python `int`. The number of samples to draw.
      seed: Python integer seed for RNG
      name: Python `str` name prefixed to Ops created by this function.

    Returns:
      samples: A Tensor of correlation matrices with shape `[n, B, D, D]`,
        where `B` is the shape of the `concentration` parameter, and `D`
        is the `dimension`.

    Raises:
      ValueError: If `dimension` is negative.
    """
    if self.dimension < 0:
      raise ValueError(
          'Cannot sample negative-dimension correlation matrices.')
    # Notation below: B is the batch shape, i.e., tf.shape(concentration)
    seed = seed_stream.SeedStream(seed, 'sample_lkj')
    with tf.name_scope('sample_lkj', name, [self.concentration]):
      if not self.concentration.dtype.is_floating:
        raise TypeError('The concentration argument should have floating type,'
                        ' not {}'.format(self.concentration.dtype.name))

      concentration = _replicate(num_samples, self.concentration)
      concentration_shape = tf.shape(concentration)
      if self.dimension <= 1:
        # For any dimension <= 1, there is only one possible correlation matrix.
        shape = tf.concat([
            concentration_shape, [self.dimension, self.dimension]], axis=0)
        return tf.ones(shape=shape, dtype=self.concentration.dtype)
      beta_conc = concentration + (self.dimension - 2.) / 2.
      beta_dist = tfd.Beta(concentration1=beta_conc, concentration0=beta_conc)

      # Note that the sampler below deviates from [1], by doing the sampling in
      # cholesky space. This does not change the fundamental logic of the
      # sampler, but does speed up the sampling.

      # This is the correlation coefficient between the first two dimensions.
      # This is also `r` in reference [1].
      corr12 = 2. * beta_dist.sample(seed=seed()) - 1.

      # Below we construct the Cholesky of the initial 2x2 correlation matrix,
      # which is of the form:
      # [[1, 0], [r, sqrt(1 - r**2)]], where r is the correlation between the
      # first two dimensions.
      # This is the top-left corner of the cholesky of the final sample.
      first_row = tf.concat([
          tf.ones_like(corr12)[..., tf.newaxis],
          tf.zeros_like(corr12)[..., tf.newaxis]], axis=-1)
      second_row = tf.concat([
          corr12[..., tf.newaxis],
          tf.sqrt(1 - corr12**2)[..., tf.newaxis]], axis=-1)

      chol_result = tf.concat([
          first_row[..., tf.newaxis, :],
          second_row[..., tf.newaxis, :]], axis=-2)

      for n in range(2, self.dimension):
        # Loop invariant: on entry, result has shape B + [n, n]
        beta_conc -= 0.5
        # norm is y in reference [1].
        norm = tfd.Beta(concentration1=n/2., concentration0=beta_conc).sample(
            seed=seed())
        # distance shape: B + [1] for broadcast
        distance = tf.sqrt(norm)[..., tf.newaxis]
        # direction is u in reference [1].
        # direction shape: B + [n]
        direction = _uniform_unit_norm(
            n, concentration_shape, self.concentration.dtype, seed)
        # raw_correlation is w in reference [1].
        raw_correlation = distance * direction  # shape: B + [n]

        # This is the next row in the cholesky of the result,
        # which differs from the construction in reference [1].
        # In the reference, the new row `z` = chol_result @ raw_correlation^T
        # = C @ raw_correlation^T (where as short hand we use C = chol_result).
        # We prove that the below equation is the right row to add to the
        # cholesky, by showing equality with reference [1].
        # Let S be the sample constructed so far, and let `z` be as in
        # reference [1]. Then at this iteration, the new sample S' will be
        # [[S z^T]
        #  [z 1]]
        # In our case we have the cholesky decomposition factor C, so
        # we want our new row x (same size as z) to satisfy:
        #  [[S z^T]  [[C 0]    [[C^T  x^T]         [[CC^T  Cx^T]
        #   [z 1]] =  [x k]]    [0     k]]  =       [xC^t   xx^T + k**2]]
        # Since C @ raw_correlation^T = z = C @ x^T, and C is invertible,
        # we have that x = raw_correlation. Also 1 = xx^T + k**2, so k
        # = sqrt(1 - xx^T) = sqrt(1 - |raw_correlation|**2) = sqrt(1 -
        # distance**2).
        new_row = tf.concat(
            [raw_correlation, tf.sqrt(1. - distance**2)], axis=-1)

        # Finally add this new row, by growing the cholesky of the result.
        chol_result = tf.concat([
            chol_result,
            tf.zeros_like(chol_result[..., 0][..., tf.newaxis])], axis=-1)

        chol_result = tf.concat(
            [chol_result, new_row[..., tf.newaxis, :]], axis=-2)

      result = tf.matmul(chol_result, chol_result, transpose_b=True)
      # The diagonal for a correlation matrix should always be ones. Due to
      # numerical instability the matmul might not achieve that, so manually set
      # these to ones.
      result = tf.matrix_set_diag(result, tf.ones(
          shape=tf.shape(result)[:-1], dtype=result.dtype.base_dtype))
      return result

  def _validate_dimension(self, x):
    x = tf.convert_to_tensor(x, name='x')
    if x.shape[-2:].is_fully_defined():
      if x.shape.dims[-2] == x.shape.dims[-1] == self.dimension:
        pass
      else:
        raise ValueError(
            'Input dimension mismatch: expected [..., {}, {}], got {}'.format(
                self.dimension, self.dimension, x.shape.dims))
    elif self.validate_args:
      msg = 'Input dimension mismatch: expected [..., {}, {}], got {}'.format(
          self.dimension, self.dimension, tf.shape(x))
      with tf.control_dependencies(
          [tf.assert_equal(tf.shape(x)[-2], self.dimension, message=msg),
           tf.assert_equal(tf.shape(x)[-1], self.dimension, message=msg)]):
        x = tf.identity(x)
    return x

  def _validate_correlationness(self, x):
    if not self.validate_args:
      return x
    checks = [
        tf.assert_less_equal(-1., x, message='Correlations must be >= -1.'),
        tf.assert_less_equal(x, 1., message='Correlations must be <= 1.'),
        tf.assert_near(
            tf.matrix_diag_part(x), 1.,
            message='Self-correlations must be = 1.'),
        tf.assert_near(
            x, tf.matrix_transpose(x),
            message='Correlation matrices must be symmetric')
    ]
    with tf.control_dependencies(checks):
      return tf.identity(x)

  def _log_prob(self, x):
    # Despite what one might infer from Eq 15 in [1], the formula
    # given for the normalization constant should be read in the sense
    # of division, not multiplication.
    x = self._validate_dimension(x)
    x = self._validate_correlationness(x)
    normalizer = self._log_normalization()
    return self._log_unnorm_prob(x) - normalizer

  def _log_unnorm_prob(self, x, name=None):
    """Returns the unnormalized log density of an LKJ distribution.

    Args:
      x: `float` or `double` `Tensor` of correlation matrices.  The shape of `x`
        must be `B + [D, D]`, where `B` broadcasts with the shape of
        `concentration`.
      name: Python `str` name prefixed to Ops created by this function.

    Returns:
      log_p: A Tensor of the unnormalized log density of each matrix element of
        `x`, with respect to an LKJ distribution with parameter the
        corresponding element of `concentration`.
    """
    with tf.name_scope('log_unnorm_prob_lkj', name, [self.concentration]):
      x = tf.convert_to_tensor(x, name='x')
      # The density is det(matrix) ** (concentration - 1).
      # Computing the determinant with `logdet` is fine here, since correlation
      # matrices are Hermitian and PSD.  If the input is not PSD, will raise an
      # error, which is not unreasonable.
      # An alternative would have been to check allow_nan_stats and use
      #   eigenvalues, _ = linalg_ops.self_adjoint_eig(x)
      #   psd_mask = math_ops.cast(
      #     math_ops.reduce_min(eigenvalues, axis=-1) >= 0, dtype=x.dtype)
      #   tf.where(psd_mask, answer, float('-inf'))
      # to emit probability 0 for inputs that are not PSD, without ever
      # raising an error.
      answer = (self.concentration - 1.) * tf.linalg.logdet(x)
      return answer

  def _log_normalization(self, name='log_normalization'):
    """Returns the log normalization of an LKJ distribution.

    Args:
      name: Python `str` name prefixed to Ops created by this function.

    Returns:
      log_z: A Tensor of the same shape and dtype as `concentration`, containing
        the corresponding log normalizers.
    """
    # The formula is from D. Lewandowski et al [1], p. 1999, from the
    # proof that eqs 16 and 17 are equivalent.
    with tf.name_scope('log_normalization_lkj', name, [self.concentration]):
      logpi = np.log(np.pi)
      ans = tf.zeros_like(self.concentration)
      for k in range(1, self.dimension):
        ans += logpi * (k / 2.)
        ans += tf.lgamma(self.concentration + (self.dimension - 1 - k) / 2.)
        ans -= tf.lgamma(self.concentration + (self.dimension - 1) / 2.)
      return ans

  def _mean(self):
    # The mean of the LKJ distribution (with any concentration parameter) is the
    # identity matrix.  Proof: Imagine a correlation matrix on D variables, and
    # imagine reversing the sense of the kth of those variables.  The
    # off-diagonal entries in row and column k change sign, but LKJ is symmetric
    # with respect to this operation (because the determinant doesn't change).
    # Ergo, the mean must be invariant under it (for any k), and hence all the
    # off-diagonal entries must be 0.
    return self._identity()

  def _identity(self):
    batch = tf.shape(self.concentration)
    answer = tf.eye(
        num_rows=self.dimension, batch_shape=batch,
        dtype=self.concentration.dtype)
    # set_shape only necessary because tf.eye doesn't do it itself: b/111413915
    answer.set_shape(
        answer.shape[:-2].concatenate([self.dimension, self.dimension]))
    return answer
