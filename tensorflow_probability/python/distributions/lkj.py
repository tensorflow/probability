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

The sampler follows the 'onion' method from
[1] Daniel Lewandowski, Dorota Kurowicka, and Harry Joe,
'Generating random correlation matrices based on vines and extended
onion method,' Journal of Multivariate Analysis 100 (2009), pp
1989-2001.
"""

# Dependency imports
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import cholesky_outer_product as cholesky_outer_product_bijector
from tensorflow_probability.python.bijectors import correlation_cholesky as correlation_cholesky_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import beta
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import linalg
from tensorflow_probability.python.math import special
from tensorflow_probability.python.math.numeric import clip_by_value_preserve_gradient
from tensorflow_probability.python.random import random_ops
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'LKJ',
]


class _ClipByValue(bijector_lib.AutoCompositeTensorBijector):
  """A bijector that clips by value.

  This class is intended for minute numerical issues where `|clip(x) - x| <=
  eps`, as it defines the derivative of its application to be exactly 1.
  """

  def __init__(self,
               clip_value_min,
               clip_value_max,
               validate_args=False,
               name='clip_by_value'):
    """Instantiates the `ClipByValue` bijector.

    Args:
      clip_value_min: Floating-point `Tensor`.
      clip_value_max: Floating-point `Tensor`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([clip_value_min, clip_value_max],
                                      dtype_hint=tf.float32)
      self._clip_value_min = tensor_util.convert_nonref_to_tensor(
          clip_value_min, dtype=dtype, name='clip_value_min')
      self._clip_value_max = tensor_util.convert_nonref_to_tensor(
          clip_value_max, dtype=dtype, name='clip_value_max')
      super(_ClipByValue, self).__init__(
          forward_min_event_ndims=0,
          is_constant_jacobian=True,
          dtype=dtype,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _is_increasing(cls):
    return False

  def _forward(self, x):
    return clip_by_value_preserve_gradient(x, self._clip_value_min,
                                           self._clip_value_max)

  def _inverse(self, y):
    return y

  def _forward_log_det_jacobian(self, x):
    # We deliberately ignore the clipping operation.
    return tf.zeros([], dtype=dtype_util.base_dtype(x.dtype))


def _tril_spherical_uniform(dimension, batch_shape, dtype, seed):
  """Returns a `Tensor` of samples of lower triangular matrices.

  Each row of the lower triangular part follows a spherical uniform
  distribution.

  Args:
    dimension: Scalar `int` `Tensor`, representing the dimensionality of the
      output matrices.
    batch_shape: Vector-shaped, `int` `Tensor` representing batch shape of
      output. The output will have shape `batch_shape + [dimension, dimension]`.
    dtype: TF `dtype` representing `dtype` of output.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

  Returns:
    tril_spherical_uniform: `Tensor` with specified `batch_shape` and `dtype`
      consisting of real values drawn row-wise from a spherical uniform
      distribution.
  """
  # Essentially, we will draw lower triangular samples where each lower
  # triangular entry follows a normal distribution, then apply `x / norm(x)`
  # for each row of the samples.
  # To avoid possible NaNs, we will use spherical_uniform directly for
  # the first two rows.
  assert dimension > 0, '`dimension` needs to be positive.'
  num_seeds = min(dimension, 3)
  seeds = list(samplers.split_seed(seed, n=num_seeds, salt='sample_lkj'))
  rows = []
  paddings_prepend = [[0, 0]] * len(batch_shape)
  for n in range(1, min(dimension, 2) + 1):
    rows.append(
        tf.pad(
            random_ops.spherical_uniform(
                shape=batch_shape, dimension=n, dtype=dtype, seed=seeds.pop()),
            paddings_prepend + [[0, dimension - n]],
            constant_values=0.))
  samples = tf.stack(rows, axis=-2)
  if dimension > 2:
    normal_shape = ps.concat(
        [batch_shape, [dimension * (dimension + 1) // 2 - 3]], axis=0)
    normal_samples = samplers.normal(
        shape=normal_shape, dtype=dtype, seed=seeds.pop())
    # We fill the first two rows of the triangular matrix with ones.
    # Note that fill_triangular fills elements in a clockwise spiral.
    normal_samples = tf.concat([
        normal_samples[..., :dimension],
        tf.ones(ps.concat([batch_shape, [1]], axis=0), dtype=dtype),
        normal_samples[..., dimension:(2 * dimension - 1)],
        tf.ones(ps.concat([batch_shape, [2]], axis=0), dtype=dtype),
        normal_samples[..., (2 * dimension - 1):],
    ],
                               axis=-1)
    normal_samples = linalg.fill_triangular(
        normal_samples, upper=False)[..., 2:, :]
    remaining_rows = normal_samples / tf.norm(
        normal_samples, ord=2, axis=-1, keepdims=True)
    samples = tf.concat([samples, remaining_rows], axis=-2)
  return samples


def sample_lkj(
    num_samples,
    dimension,
    concentration,
    cholesky_space=False,
    seed=None,
    name=None):
  """Returns a Tensor of samples from an LKJ distribution.

  Args:
    num_samples: Python `int`. The number of samples to draw.
    dimension: Python `int`. The dimension of correlation matrices.
    concentration: `Tensor` representing the concentration of the LKJ
      distribution.
    cholesky_space: Python `bool`. Whether to take samples from LKJ or
      Chol(LKJ).
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    samples: A Tensor of correlation matrices (or Cholesky factors of
      correlation matrices if `cholesky_space = True`) with shape
      `[n] + B + [D, D]`, where `B` is the shape of the `concentration`
      parameter, and `D` is the `dimension`.

  Raises:
    ValueError: If `dimension` is negative.
  """
  if dimension < 0:
    raise ValueError(
        'Cannot sample negative-dimension correlation matrices.')
  # Notation below: B is the batch shape, i.e., tf.shape(concentration)

  with tf.name_scope('sample_lkj' or name):
    concentration = tf.convert_to_tensor(concentration)
    if not dtype_util.is_floating(concentration.dtype):
      raise TypeError(
          'The concentration argument should have floating type, not '
          '{}'.format(dtype_util.name(concentration.dtype)))

    batch_shape = ps.concat([[num_samples], ps.shape(concentration)], axis=0)
    dtype = concentration.dtype
    if dimension <= 1:
      # For any dimension <= 1, there is only one possible correlation matrix.
      shape = ps.concat([batch_shape, [dimension, dimension]], axis=0)
      return tf.ones(shape=shape, dtype=dtype)

    # We need 1 seed for beta and 1 seed for tril_spherical_uniform.
    beta_seed, tril_spherical_uniform_seed = samplers.split_seed(
        seed, n=2, salt='sample_lkj')

    # Note that the sampler below deviates from [1], by doing the sampling in
    # cholesky space. This does not change the fundamental logic of the
    # sampler, but does speed up the sampling.
    # In addition, we also vectorize the computation to make the sampler
    # more feasible to use in problems where `dimension` is large.

    beta_conc = concentration + (dimension - 2.) / 2.
    dimension_range = np.arange(
        1., dimension, dtype=dtype_util.as_numpy_dtype(dtype))
    beta_conc1 = dimension_range / 2.
    beta_conc0 = beta_conc[..., tf.newaxis] - (dimension_range - 1) / 2.
    beta_dist = beta.Beta(concentration1=beta_conc1, concentration0=beta_conc0)
    # norm is y in reference [1].
    norm = beta_dist.sample(sample_shape=[num_samples], seed=beta_seed)
    # distance shape: B + [dimension - 1, 1] for broadcast
    distance = tf.sqrt(norm)[..., tf.newaxis]

    # direction is u in reference [1].
    # direction follows the spherical uniform distribution and will be stored
    # in a lower triangular matrix, hence it will have shape:
    # B + [dimension - 1, dimension - 1]
    direction = _tril_spherical_uniform(dimension - 1, batch_shape, dtype,
                                        tril_spherical_uniform_seed)

    # raw_correlation is w in reference [1].
    # shape: B + [dimension - 1, dimension - 1]
    raw_correlation = distance * direction

    # This is the rows in the cholesky of the result,
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
    paddings_prepend = [[0, 0]] * len(batch_shape)
    diag = tf.pad(
        tf.sqrt(1. - norm), paddings_prepend + [[1, 0]], constant_values=1.)
    chol_result = tf.pad(
        raw_correlation,
        paddings_prepend + [[1, 0], [0, 1]],
        constant_values=0.)
    chol_result = tf.linalg.set_diag(chol_result, diag)

    if cholesky_space:
      return chol_result

    result = tf.matmul(chol_result, chol_result, transpose_b=True)
    # The diagonal for a correlation matrix should always be ones. Due to
    # numerical instability the matmul might not achieve that, so manually set
    # these to ones.
    result = tf.linalg.set_diag(
        result, tf.ones(shape=ps.shape(result)[:-1], dtype=result.dtype))
    # This sampling algorithm can produce near-PSD matrices on which standard
    # algorithms such as `tf.linalg.cholesky` or
    # `tf.linalg.self_adjoint_eigvals` fail. Specifically, as documented in
    # b/116828694, around 2% of trials of 900,000 5x5 matrices (distributed
    # according to 9 different concentration parameter values) contained at
    # least one matrix on which the Cholesky decomposition failed.
    return result


class LKJ(distribution.AutoCompositeTensorDistribution):
  """The LKJ distribution on correlation matrices.

  This is a one-parameter family of distributions on correlation matrices.  The
  probability density is proportional to the determinant raised to the power of
  the parameter: `pdf(X; eta) = Z(eta) * det(X) ** (eta - 1)`, where `Z(eta)` is
  a normalization constant.  The uniform distribution on correlation matrices is
  the special case `eta = 1`.

  The distribution is named after Lewandowski, Kurowicka, and Joe, who gave a
  sampler for the distribution in [(Lewandowski, Kurowicka, Joe, 2009)][1].

  Note: For better numerical stability, it is recommended that you use
  `CholeskyLKJ` instead.

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
               input_output_cholesky=False,
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
      input_output_cholesky: Python `bool`. If `True`, functions whose input or
        output have the semantics of samples assume inputs are in Cholesky form
        and return outputs in Cholesky form. In particular, if this flag is
        `True`, input to `log_prob` is presumed of Cholesky form and output from
        `sample` is of Cholesky form.  Setting this argument to `True` is purely
        a computational optimization and does not change the underlying
        distribution. Additionally, validation checks which are only defined on
        the multiplied-out form are omitted, even if `validate_args` is `True`.
        Default value: `False` (i.e., input/output does not have Cholesky
        semantics). WARNING: Do not set this boolean to true, when using
        `tfp.mcmc`. The density is not the density of Cholesky factors of
        correlation matrices drawn via LKJ.
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
    if dimension > 65536:
      raise ValueError(
          ('Given dimension ({}) is greater than 65536, and will overflow '
           'int32 array sizes.').format(dimension))
    parameters = dict(locals())
    self._input_output_cholesky = input_output_cholesky
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype([concentration], tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, name='concentration', dtype=dtype)
      self._dimension = dimension
      super(LKJ, self).__init__(
          dtype=self._concentration.dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        concentration=parameter_properties.ParameterProperties(
            shape_fn=lambda sample_shape: sample_shape[:-2],
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(
                    low=tf.convert_to_tensor(
                        1. + dtype_util.eps(dtype), dtype=dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def dimension(self):
    """Dimension of returned correlation matrices."""
    return self._dimension

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def input_output_cholesky(self):
    """Boolean indicating if `Tensor` input/outputs are Cholesky factorized."""
    return self._input_output_cholesky

  def _event_shape_tensor(self):
    return tf.constant([self.dimension, self.dimension], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([self.dimension, self.dimension])

  def _sample_n(self, num_samples, seed=None, name=None):
    """Returns a Tensor of samples from an LKJ distribution.

    Args:
      num_samples: Python `int`. The number of samples to draw.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      name: Python `str` name prefixed to Ops created by this function.

    Returns:
      samples: A Tensor of correlation matrices with shape `[n, B, D, D]`,
        where `B` is the shape of the `concentration` parameter, and `D`
        is the `dimension`.

    Raises:
      ValueError: If `dimension` is negative.
    """
    return sample_lkj(
        num_samples=num_samples,
        dimension=self.dimension,
        concentration=self.concentration,
        cholesky_space=self.input_output_cholesky,
        seed=seed,
        name=name)

  def _log_prob(self, x):
    # Despite what one might infer from Eq 15 in [1], the formula
    # given for the normalization constant should be read in the sense
    # of division, not multiplication.
    concentration = tf.convert_to_tensor(self.concentration)
    normalizer = self._log_normalization(concentration=concentration)
    return self._log_unnorm_prob(x, concentration) - normalizer

  def _log_unnorm_prob(self, x, concentration, name=None):
    """Returns the unnormalized log density of an LKJ distribution.

    Args:
      x: `float` or `double` `Tensor` of correlation matrices.  The shape of `x`
        must be `B + [D, D]`, where `B` broadcasts with the shape of
        `concentration`.
      concentration: `float` or `double` `Tensor`. The positive concentration
        parameter of the LKJ distributions.
      name: Python `str` name prefixed to Ops created by this function.

    Returns:
      log_p: A Tensor of the unnormalized log density of each matrix element of
        `x`, with respect to an LKJ distribution with parameter the
        corresponding element of `concentration`.
    """
    with tf.name_scope(name or 'log_unnorm_prob_lkj'):
      x = tf.convert_to_tensor(x, name='x')
      # The density is det(matrix) ** (concentration - 1).
      # Computing the determinant with `logdet` is usually fine, since
      # correlation matrices are Hermitian and PSD. But in some cases, for a
      # PSD matrix whose eigenvalues are close to zero, `logdet` raises an error
      # complaining that it is not PSD. The root cause is the computation of the
      # cholesky decomposition in `logdet`. Hence, we use the less efficient but
      # more robust `slogdet` which does not use `cholesky`.
      #
      # An alternative would have been to check allow_nan_stats and use
      #   eigenvalues = tf.linalg.self_adjoint_eigvals(x)
      #   psd_mask = tf.cast(
      #     tf.reduce_min(eigenvalues, axis=-1) >= 0, dtype=x.dtype)
      #   tf.where(psd_mask, answer, float('-inf'))
      # to emit probability 0 for inputs that are not PSD, without ever raising
      # an error. More care must be taken, as due to numerical stability issues,
      # self_adjoint_eigvals can return slightly negative eigenvalues even for
      # a PSD matrix.
      if self.input_output_cholesky:
        logdet = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(x)), axis=[-1])
      else:
        # TODO(b/162937268): Remove the hackaround.
        if (not tf.executing_eagerly() and
            control_flow_util.GraphOrParentsInXlaContext(
                tf1.get_default_graph())):
          s = tf.linalg.svd(x, compute_uv=False)
          logdet = tf.math.reduce_sum(tf.math.log(s), -1)
        else:
          logdet = tf.linalg.slogdet(x).log_abs_determinant
      answer = (concentration - 1.) * logdet
      return answer

  def _log_normalization(self, concentration=None, name='log_normalization'):
    """Returns the log normalization of an LKJ distribution.

    Args:
      concentration: `float` or `double` `Tensor`. The positive concentration
        parameter of the LKJ distributions.
      name: Python `str` name prefixed to Ops created by this function.

    Returns:
      log_z: A Tensor of the same shape and dtype as `concentration`, containing
        the corresponding log normalizers.
    """
    # The formula is from D. Lewandowski et al [1], p. 1999, from the
    # proof that eqs 16 and 17 are equivalent.
    # Instead of using a for loop for k from 1 to (dimension - 1), we will
    # vectorize the computation by performing operations on the vector
    # `dimension_range = np.arange(1, dimension)`.
    with tf.name_scope(name or 'log_normalization_lkj'):
      concentration = (
          tf.convert_to_tensor(self.concentration
                               if concentration is None else concentration))
      logpi = float(np.log(np.pi))
      dimension_range = np.arange(
          1.,
          self.dimension,
          dtype=dtype_util.as_numpy_dtype(concentration.dtype))
      effective_concentration = (
          concentration[..., tf.newaxis] +
          (self.dimension - 1 - dimension_range) / 2.)
      ans = tf.reduce_sum(
          special.log_gamma_difference(
              dimension_range / 2., effective_concentration),
          axis=-1)
      # Then we add to `ans` the sum of `logpi / 2 * k` for `k` run from 1 to
      # `dimension - 1`.
      ans = ans + logpi * (self.dimension * (self.dimension - 1) / 4.)
      return ans

  def _mean(self):
    # The mean of the LKJ distribution (with any concentration parameter) is the
    # identity matrix.  Proof: Imagine a correlation matrix on D variables, and
    # imagine reversing the sense of the kth of those variables.  The
    # off-diagonal entries in row and column k change sign, but LKJ is symmetric
    # with respect to this operation (because the determinant doesn't change).
    # Ergo, the mean must be invariant under it (for any k), and hence all the
    # off-diagonal entries must be 0.
    concentration = tf.convert_to_tensor(self.concentration)
    batch = ps.shape(concentration)
    answer = tf.eye(
        num_rows=self.dimension, batch_shape=batch,
        dtype=concentration.dtype)
    return answer

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    cholesky_bijector = correlation_cholesky_bijector.CorrelationCholesky(
        validate_args=self.validate_args)

    if self.input_output_cholesky:
      return cholesky_bijector
    return chain_bijector.Chain([
        # We need to explictly clip the output of this bijector because the
        # other two bijectors sometimes return values that exceed the bounds by
        # an epsilon due to minute numerical errors. Even numerically stable
        # algorithms (which the other two bijectors employ) allow for symmetric
        # errors about the true value, which is inappropriate for a one-sided
        # validity constraint associated with correlation matrices.
        _ClipByValue(-1., tf.ones([], self.dtype)),
        cholesky_outer_product_bijector.CholeskyOuterProduct(
            validate_args=self.validate_args),
        cholesky_bijector
    ], validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    assertions = []
    if not self.validate_args:
      return assertions
    if is_init != tensor_util.is_ref(self.concentration):
      # concentration >= 1
      # TODO(b/111451422, b/115950951) Generalize to concentration > 0.
      assertions.append(assert_util.assert_non_negative(
          self.concentration - 1,
          message='Argument `concentration` must be >= 1.'))
    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if tensorshape_util.is_fully_defined(x.shape[-2:]):
      if not (tensorshape_util.dims(x.shape)[-2] ==
              tensorshape_util.dims(x.shape)[-1] ==
              self.dimension):
        raise ValueError(
            'Input dimension mismatch: expected [..., {}, {}], got {}'.format(
                self.dimension, self.dimension, tensorshape_util.dims(x.shape)))
    elif self.validate_args:
      msg = 'Input dimension mismatch: expected [..., {}, {}], got {}'.format(
          self.dimension, self.dimension, tf.shape(x))
      assertions.append(assert_util.assert_equal(
          tf.shape(x)[-2], self.dimension, message=msg))
      assertions.append(assert_util.assert_equal(
          tf.shape(x)[-1], self.dimension, message=msg))

    if self.validate_args and not self.input_output_cholesky:
      assertions.append(assert_util.assert_less_equal(
          dtype_util.as_numpy_dtype(x.dtype)(-1),
          x,
          message='Correlations must be >= -1.',
          summarize=30))
      assertions.append(assert_util.assert_less_equal(
          x,
          dtype_util.as_numpy_dtype(x.dtype)(1),
          message='Correlations must be <= 1.',
          summarize=30))
      assertions.append(assert_util.assert_near(
          tf.linalg.diag_part(x),
          dtype_util.as_numpy_dtype(x.dtype)(1),
          message='Self-correlations must be = 1.',
          summarize=30))
      assertions.append(assert_util.assert_near(
          x,
          tf.linalg.matrix_transpose(x),
          message='Correlation matrices must be symmetric.',
          summarize=30))
    return assertions
