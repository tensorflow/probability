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
"""Distribution of a vectorized Laplace, with uncorrelated components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import vector_laplace_linear_operator as vllo
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'VectorLaplaceDiag',
]


# Copied from distribution_util, where it is to be removed, and duplicated here
# to support VectorLaplaceDiag until the deprecation window is closed.
def _make_diag_scale(loc=None,
                     scale_diag=None,
                     scale_identity_multiplier=None,
                     shape_hint=None,
                     validate_args=False,
                     assert_positive=False,
                     name=None,
                     dtype=None):
  """Creates a LinearOperator representing a diagonal matrix.

  Args:
    loc: Floating-point `Tensor`. This is used for inferring shape in the case
      where only `scale_identity_multiplier` is set.
    scale_diag: Floating-point `Tensor` representing the diagonal matrix.
      `scale_diag` has shape [N1, N2, ...  k], which represents a k x k diagonal
      matrix. When `None` no diagonal term is added to the LinearOperator.
    scale_identity_multiplier: floating point rank 0 `Tensor` representing a
      scaling done to the identity matrix. When `scale_identity_multiplier =
      scale_diag = scale_tril = None` then `scale += IdentityMatrix`. Otherwise
      no scaled-identity-matrix is added to `scale`.
    shape_hint: scalar integer `Tensor` representing a hint at the dimension of
      the identity matrix when only `scale_identity_multiplier` is set.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness.
    assert_positive: Python `bool` indicating whether LinearOperator should be
      checked for being positive definite.
    name: Python `str` name given to ops managed by this object.
    dtype: TF `DType` to prefer when converting args to `Tensor`s. Else, we fall
      back to a compatible dtype across all of `loc`, `scale_diag`, and
      `scale_identity_multiplier`.

  Returns:
    `LinearOperator` representing a lower triangular matrix.

  Raises:
    ValueError:  If only `scale_identity_multiplier` is set and `loc` and
      `shape_hint` are both None.
  """

  with tf.name_scope(name or 'make_diag_scale'):
    if dtype is None:
      dtype = dtype_util.common_dtype(
          [loc, scale_diag, scale_identity_multiplier],
          dtype_hint=tf.float32)
    loc = tensor_util.convert_nonref_to_tensor(loc, name='loc', dtype=dtype)
    scale_diag = tensor_util.convert_nonref_to_tensor(
        scale_diag, name='scale_diag', dtype=dtype)
    scale_identity_multiplier = tensor_util.convert_nonref_to_tensor(
        scale_identity_multiplier,
        name='scale_identity_multiplier',
        dtype=dtype)

    if scale_diag is not None:
      if scale_identity_multiplier is not None:
        scale_diag = scale_diag + scale_identity_multiplier[..., tf.newaxis]
      return tf.linalg.LinearOperatorDiag(
          diag=scale_diag,
          is_non_singular=True,
          is_self_adjoint=True,
          is_positive_definite=assert_positive)

    if loc is None and shape_hint is None:
      raise ValueError('Cannot infer `event_shape` unless `loc` or '
                       '`shape_hint` is specified.')

    num_rows = shape_hint
    del shape_hint
    if num_rows is None:
      num_rows = tf.compat.dimension_value(loc.shape[-1])
      if num_rows is None:
        num_rows = tf.shape(loc)[-1]

    if scale_identity_multiplier is None:
      return tf.linalg.LinearOperatorIdentity(
          num_rows=num_rows,
          dtype=dtype,
          is_self_adjoint=True,
          is_positive_definite=True,
          assert_proper_shapes=validate_args)

    return tf.linalg.LinearOperatorScaledIdentity(
        num_rows=num_rows,
        multiplier=scale_identity_multiplier,
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=assert_positive,
        assert_proper_shapes=validate_args)


class VectorLaplaceDiag(vllo.VectorLaplaceLinearOperator):
  """The vectorization of the Laplace distribution on `R^k`.

  The vector laplace distribution is defined over `R^k`, and parameterized by
  a (batch of) length-`k` `loc` vector (the means) and a (batch of) `k x k`
  `scale` matrix:  `covariance = 2 * scale @ scale.T`, where `@` denotes
  matrix-multiplication.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, scale) = exp(-||y||_1) / Z,
  y = inv(scale) @ (x - loc),
  Z = 2**k |det(scale)|,
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
  * `Z` denotes the normalization constant, and,
  * `||y||_1` denotes the `l1` norm of `y`, `sum_i |y_i|.

  A (non-batch) `scale` matrix is:

  ```none
  scale = diag(scale_diag + scale_identity_multiplier * ones(k))
  ```

  where:

  * `scale_diag.shape = [k]`, and,
  * `scale_identity_multiplier.shape = []`.

  Additional leading dimensions (if any) will index batches.

  If both `scale_diag` and `scale_identity_multiplier` are `None`, then
  `scale` is the Identity matrix.

  The VectorLaplace distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X = (X_1, ..., X_k), each X_i ~ Laplace(loc=0, scale=1)
  Y = (Y_1, ...,Y_k) = scale @ X + loc
  ```

  #### About `VectorLaplace` and `Vector` distributions in TensorFlow.

  The `VectorLaplace` is a non-standard distribution that has useful properties.

  The marginals `Y_1, ..., Y_k` are *not* Laplace random variables, due to
  the fact that the sum of Laplace random variables is not Laplace.

  Instead, `Y` is a vector whose components are linear combinations of Laplace
  random variables.  Thus, `Y` lives in the vector space generated by `vectors`
  of Laplace distributions.  This allows the user to decide the mean and
  covariance (by setting `loc` and `scale`), while preserving some properties of
  the Laplace distribution.  In particular, the tails of `Y_i` will be (up to
  polynomial factors) exponentially decaying.

  To see this last statement, note that the pdf of `Y_i` is the convolution of
  the pdf of `k` independent Laplace random variables.  One can then show by
  induction that distributions with exponential (up to polynomial factors) tails
  are closed under convolution.

  #### Examples

  ```python
  tfd = tfp.distributions

  # Initialize a single 2-variate VectorLaplace.
  vla = tfd.VectorLaplaceDiag(
      loc=[1., -1],
      scale_diag=[1, 2.])

  vla.mean().eval()
  # ==> [1., -1]

  vla.stddev().eval()
  # ==> [1., 2] * sqrt(2)

  # Evaluate this on an observation in `R^2`, returning a scalar.
  vla.prob([-1., 0]).eval()  # shape: []

  # Initialize a 3-batch, 2-variate scaled-identity VectorLaplace.
  vla = tfd.VectorLaplaceDiag(
      loc=[1., -1],
      scale_identity_multiplier=[1, 2., 3])

  vla.mean().eval()  # shape: [3, 2]
  # ==> [[1., -1]
  #      [1, -1],
  #      [1, -1]]

  vla.stddev().eval()  # shape: [3, 2]
  # ==> sqrt(2) * [[1., 1],
  #                [2, 2],
  #                [3, 3]]

  # Evaluate this on an observation in `R^2`, returning a length-3 vector.
  vla.prob([-1., 0]).eval()  # shape: [3]

  # Initialize a 2-batch of 3-variate VectorLaplace's.
  vla = tfd.VectorLaplaceDiag(
      loc=[[1., 2, 3],
           [11, 22, 33]]           # shape: [2, 3]
      scale_diag=[[1., 2, 3],
                  [0.5, 1, 1.5]])  # shape: [2, 3]

  # Evaluate this on a two observations, each in `R^3`, returning a length-2
  # vector.
  x = [[-1., 0, 1],
       [-11, 0, 11.]]   # shape: [2, 3].
  vla.prob(x).eval()    # shape: [2]
  ```

  """

  @deprecation.deprecated(
      '2020-01-01',
      '`VectorLaplaceDiag` is deprecated. If all you need is the diagonal '
      'scale, you can use '
      '`tfd.Independent(tfd.Laplace(loc=loc, scale=scale_diag), 1)` instead. '
      'You can also directly use `VectorLaplaceLinearOperator` with the '
      'appropriate `LinearOperator` instance.')
  def __init__(self,
               loc=None,
               scale_diag=None,
               scale_identity_multiplier=None,
               validate_args=False,
               allow_nan_stats=True,
               name='VectorLaplaceDiag'):
    """Construct Vector Laplace distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` (if provided) must broadcast with this.

    Recall that `covariance = 2 * scale @ scale.T`.

    ```none
    scale = diag(scale_diag + scale_identity_multiplier * ones(k))
    ```

    where:

    * `scale_diag.shape = [k]`, and,
    * `scale_identity_multiplier.shape = []`.

    Additional leading dimensions (if any) will index batches.

    If both `scale_diag` and `scale_identity_multiplier` are `None`, then
    `scale` is the Identity matrix.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      scale_diag: Non-zero, floating-point `Tensor` representing a diagonal
        matrix added to `scale`. May have shape `[B1, ..., Bb, k]`, `b >= 0`,
        and characterizes `b`-batches of `k x k` diagonal matrices added to
        `scale`. When both `scale_identity_multiplier` and `scale_diag` are
        `None` then `scale` is the `Identity`.
      scale_identity_multiplier: Non-zero, floating-point `Tensor` representing
        a scaled-identity-matrix added to `scale`. May have shape
        `[B1, ..., Bb]`, `b >= 0`, and characterizes `b`-batches of scaled
        `k x k` identity matrices added to `scale`. When both
        `scale_identity_multiplier` and `scale_diag` are `None` then `scale` is
        the `Identity`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: if at most `scale_identity_multiplier` is specified.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      with tf.name_scope('init'):
        dtype = dtype_util.common_dtype(
            [loc, scale_diag, scale_identity_multiplier], tf.float32)
        # No need to validate_args while making diag_scale.  The returned
        # LinearOperatorDiag has an assert_non_singular method that is called by
        # the Bijector.
        scale = _make_diag_scale(
            loc=loc,
            scale_diag=scale_diag,
            scale_identity_multiplier=scale_identity_multiplier,
            validate_args=False,
            assert_positive=False,
            dtype=dtype)
    super(VectorLaplaceDiag, self).__init__(
        loc=loc,
        scale=scale,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)
    self._parameters = parameters

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=1, scale_diag=1, scale_identity_multiplier=0)
