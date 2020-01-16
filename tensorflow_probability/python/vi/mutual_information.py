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
"""Mutual information estimators and helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'lower_bound_barber_agakov',
    'lower_bound_info_nce',
    'lower_bound_jensen_shannon',
    'lower_bound_nguyen_wainwright_jordan',
]


def _check_mask_shape(logu, joint_sample_mask):
  """Validate the mask specifying joint samples of the score matrix."""
  row_sum = tf.reduce_sum(
      tf.cast(joint_sample_mask, logu.dtype),
      axis=[-1])
  assertion_list = []
  assertion_list.append(
      assert_util.assert_equal(
          logu.shape, joint_sample_mask.shape,
          message='logu should be of the same shape as joint_sample_mask.'))
  assertion_list.append(
      assert_util.assert_equal(
          row_sum, tf.ones_like(row_sum),
          message='joint_sample_mask should be one-hot for each row.'))
  return assertion_list


def _check_and_get_mask(logu, joint_sample_mask=None, validate_args=False):
  """Helper function for creating masks for joint/marginal samples.

  The function is designed to do:
    - when `joint_sample_mask` is provided, check and validate the mask.
    - when `joint_sample_mask` is not provided, generate a default mask.

  Variational bounds on mutual information require knowing which
  elements of the score matrix correspond to positive elements
  (sampled from joint distribution `p(x,y)`) and negative elements
  (sampled from marginal distribution `p(x)p(y)`). By default, we assume
  that the diagonal elements of scores contain positaive pairs, and all
  other elements are negatives.

  Args:
    logu: `float`-like `Tensor` representing scores to be masked.
    joint_sample_mask: `bool`-like `Tensor` of the same shape of `logu`
      masking the joint samples by `True`, i.e. samples from joint
      distributions `p(x, y)`.
      Default value: `None`. By default, an identity matrix is constructed as
      the mask.
    validate_args: Python `bool`, default `False`. Whether to validate input
      with asserts. If `validate_args` is `False`, and the inputs are invalid,
      correct behavior is not guaranteed.

  Returns:
    logu: `float`-like `Tensor` based on input `logu`.
    joint_sample_mask: `bool`-like `Tensor` for joint samples.
  """
  with tf.name_scope('get_default_mask'):
    logu = tf.convert_to_tensor(
        logu, dtype_hint=tf.float32, name='logu')

    if joint_sample_mask is None:
      num_rows, num_cols = tf.unstack(tf.shape(logu)[-2:])
      joint_sample_mask = tf.eye(num_rows, num_cols, dtype=tf.bool)
    else:
      joint_sample_mask = tf.convert_to_tensor(
          joint_sample_mask, dtype_hint=tf.bool, name='joint_sample_mask')
      with tf.control_dependencies(
          _check_mask_shape(logu, joint_sample_mask) if validate_args else []):
        joint_sample_mask = tf.identity(joint_sample_mask)
    return logu, joint_sample_mask


def _get_masked_scores(logu, mask):
  """Helper function to mask out selected elements from a logit tensor.

  All the elements corresponding to `False` in the `mask` will
  be set to `-inf`, so that they are excluded from logsumexp calculations.

  Args:
    logu: `float`-like `Tensor` representing scores to be masked.
    mask: `bool`-like `Tensor` of the same shape of `logu` representing
      a mask for selected samples.

  Returns:
    masked_scores: `float`-like `Tensor` contains the masked values.
  """
  with tf.name_scope('get_masked_scores'):
    logu = tf.convert_to_tensor(logu, dtype_hint=tf.float32, name='logu')
    mask = tf.convert_to_tensor(mask, dtype=tf.bool, name='mask')
    return tf.where(mask, logu, tf.constant(-np.inf, dtype=logu.dtype))


def _masked_logmeanexp(input_tensor,
                       mask_tensor,
                       axis=None):
  """Compute log(mean(exp(input_tensor))) on masked elements.

  Args:
    input_tensor: `float`-like `Tensor` to be reduced.
    mask_tensor: `bool`-like `Tensor` of the same shape of `input_tensor`.
      Only the elements from `input_tensor` with the mask of `True` in
      `mask_tensor` will be selected for calculation.
    axis: The dimensions to sum across.
      Default value: `None`, i.e. all dimensions will be reduced.

  Returns:
    reduced_tensor: `float`-like `Tensor` contains the reduced result.
  """
  with tf.name_scope('masked_logmeanexp'):
    input_tensor = tf.convert_to_tensor(
        input_tensor, dtype_hint=tf.float32, name='input_tensor')
    mask_tensor = tf.convert_to_tensor(
        mask_tensor, dtype_hint=tf.bool, name='mask_tensor')
    # To mask out a value from log space, one could push the value to -inf.
    masked_input = _get_masked_scores(input_tensor, mask_tensor)
    log_n = tf.math.log(tf.cast(
        tf.reduce_sum(tf.where(mask_tensor, 1, 0), axis=axis),
        masked_input.dtype))
    return tf.reduce_logsumexp(masked_input, axis=axis) - log_n


def _maybe_assert_float_matrix(logu, validate_args):
  """Assertion check for the scores matrix to be float type."""
  logu = tf.convert_to_tensor(logu, dtype_hint=tf.float32, name='logu')

  if not dtype_util.is_floating(logu.dtype):
    raise TypeError('Input argument must be `float` type.')

  assertions = []
  # Check scores is a matrix.
  msg = 'Input argument must be a (batch of) matrix.'
  rank = tensorshape_util.rank(logu.shape)
  if rank is not None:
    if rank < 2:
      raise ValueError(msg)
  elif validate_args:
    assertions.append(assert_util.assert_rank_at_least(logu, 2, msg))

  # Check scores has the shape [..., N, M], M >= N
  msg = 'Input argument must be a (batch of) matrix of the shape [N, M], M > N.'
  if (rank is not None and
      tensorshape_util.is_fully_defined(logu.shape[-2:])):
    if logu.shape[-2] > logu.shape[-1]:
      raise ValueError(msg)
  elif validate_args:
    n, m = tf.unstack(logu.shape[-2:])
    assertions.append(assert_util.assert_greater_equal(m, n, message=msg))
  return assertions


def lower_bound_barber_agakov(logu, entropy, name=None):
  """Lower bound on mutual information from [Barber and Agakov (2003)][1].

  This method gives a lower bound on the mutual information I(X; Y),
  by replacing the unknown conditional p(x|y) with a variational
  decoder q(x|y), but it requires knowing the entropy of X, h(X).
  The lower bound was introduced in [Barber and Agakov (2003)][1].
  ```none
  I(X; Y) = E_p(x, y)[log( p(x|y) / p(x) )]
          = E_p(x, y)[log( q(x|y) / p(x) )] + E_p(y)[KL[ p(x|y) || q(x|y) ]]
          >= E_p(x, y)[log( q(x|y) )] + h(X) = I_[lower_bound_barbar_agakov]
  ```

  Example:

  `x`, `y` are samples from a joint Gaussian distribution, with correlation
  `0.8` and both of dimension `1`.

  ```python
  batch_size, rho, dim = 10000, 0.8, 1
  y, eps = tf.split(
      value=tf.random.normal(shape=(2 * batch_size, dim), seed=7),
      num_or_size_splits=2, axis=0)
  mean, conditional_stddev = rho * y, tf.sqrt(1. - tf.square(rho))
  x = mean + conditional_stddev * eps

  # Conditional distribution of p(x|y)
  conditional_dist = tfd.MultivariateNormalDiag(
      mean, scale_identity_multiplier=conditional_stddev)

  # Scores/unnormalized likelihood of pairs of joint samples `x[i], y[i]`
  joint_scores = conditional_dist.log_prob(x)

  # Differential entropy of `X` that is `1-D` Normal distributed.
  entropy_x = 0.5 * np.log(2 * np.pi * np.e)


  # Barber and Agakov lower bound on mutual information
  lower_bound_barber_agakov(logu=joint_scores, entropy=entropy_x)
  ```

  Args:
    logu: `float`-like `Tensor` of size [batch_size] representing
      log(q(x_i | y_i)) for each (x_i, y_i) pair.
    entropy: `float`-like `scalar` representing the entropy of X.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'lower_bound_barber_agakov').

  Returns:
    lower_bound: `float`-like `scalar` for lower bound on mutual information.

  #### References

  [1]: David Barber, Felix V. Agakov. The IM algorithm: a variational
       approach to Information Maximization. In _Conference on Neural
       Information Processing Systems_, 2003.
  """

  with tf.name_scope(name or 'lower_bound_barber_agakov'):
    logu = tf.convert_to_tensor(logu, dtype_hint=tf.float32, name='logu')
    # The first term is 1/K * sum(i=1:K, log(q(x_i | y_i)), where K is
    # the `batch_size` and q(x_i | y_i) is the likelihood from a tractable
    # decoder for the samples from the joint distribution.
    # The second term is simply the entropy of p(x), which we assume
    # is tractable.
    return tf.reduce_mean(logu, axis=[-1]) + entropy


def lower_bound_info_nce(logu, joint_sample_mask=None,
                         validate_args=False, name=None):
  """InfoNCE lower bound on mutual information.

  InfoNCE lower bound is proposed in [van den Oord et al. (2018)][1]
  based on noise contrastive estimation (NCE).
  ```none
  I(X; Y) >= 1/K sum(i=1:K, log( p_joint[i] / p_marginal[i])),
  ```
  where the numerator and the denominator are, respectively,
  ```none
  p_joint[i] = p(x[i] | y[i]) = exp( f(x[i], y[i]) ),
  p_marginal[i] = 1/K sum(j=1:K, p(x[i] | y[j]) )
                = 1/K sum(j=1:K, exp( f(x[i], y[j]) ) ),
  ```
  and `(x[i], y[i]), i=1:K` are samples from joint distribution `p(x, y)`.
  Pairs of points (x, y) are scored using a critic function `f`.

  Example:

  `X`, `Y` are samples from a joint Gaussian distribution, with
  correlation `0.8` and both of dimension `1`.

  ```python
  batch_size, rho, dim = 10000, 0.8, 1
  y, eps = tf.split(
      value=tf.random.normal(shape=(2 * batch_size, dim), seed=7),
      num_or_size_splits=2, axis=0)
  mean, conditional_stddev = rho * y, tf.sqrt(1. - tf.square(rho))
  x = mean + conditional_stddev * eps

  # Conditional distribution of p(x|y)
  conditional_dist = tfd.MultivariateNormalDiag(
      mean, scale_identity_multiplier=conditional_stddev)

  # Scores/unnormalized likelihood of pairs of samples `x[i], y[j]`
  # (The scores has its shape [x_batch_size, distibution_batch_size]
  # as the `lower_bound_info_nce` requires `scores[i, j] = f(x[i], y[j])
  # = log p(x[i] | y[j])`.)
  scores = conditional_dist.log_prob(x[:, tf.newaxis, :])

  # Mask for joint samples
  joint_sample_mask = tf.eye(batch_size, dtype=bool)

  # InfoNCE lower bound on mutual information
  lower_bound_info_nce(logu=scores, joint_sample_mask=joint_sample_mask)
  ```

  Args:
    logu: `float`-like `Tensor` of size `[batch_size_1, batch_size_2]`
      representing critic scores (scores) for pairs of points (x, y) with
      `logu[i, j] = f(x[i], y[j])`.
    joint_sample_mask: `bool`-like `Tensor` of the same size as `logu`
      masking the positive samples by `True`, i.e. samples from joint
      distribution `p(x, y)`.
      Default value: `None`. By default, an identity matrix is constructed as
      the mask.
    validate_args: Python `bool`, default `False`. Whether to validate input
      with asserts. If `validate_args` is `False`, and the inputs are invalid,
      correct behavior is not guaranteed.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'lower_bound_info_nce').

  Returns:
    lower_bound: `float`-like `scalar` for lower bound on mutual information.

  #### References

  [1]: Aaron van den Oord, Yazhe Li, Oriol Vinyals. Representation
       Learning with Contrastive Predictive Coding. _arXiv preprint
       arXiv:1807.03748_, 2018. https://arxiv.org/abs/1807.03748.
  """

  with tf.name_scope(name or 'lower_bound_info_nce'):
    # Follow the notation of eq.(12) of Poole et al. (2019)
    # On Variational Bounds of Mutual Information,
    # https://arxiv.org/abs/1905.06922, where the expectation is taken by
    # sampling.
    # The first term is `1/K * sum(i=1:K, f(x[i], y[i])`, where `K` is the
    # `batch_size` and `(x[i], y[i])` is the joint sample.
    # The second term is `1/K * sum(i=1:K, log(sum(j=1:K, exp(f(x[i], y[j]))))`,
    # where the joint samples are when `i=j`, and the marginal ones are `i!=j`.

    with tf.control_dependencies(
        _maybe_assert_float_matrix(logu, validate_args)):
      if joint_sample_mask is None:
        logu = tf.convert_to_tensor(
            logu, dtype_hint=tf.float32, name='logu')
        joint_term = tf.reduce_mean(
            tf.linalg.diag_part(logu), axis=[-1])

      else:
        logu, joint_sample_mask = _check_and_get_mask(
            logu, joint_sample_mask, validate_args=validate_args)
        joint_term = tf.reduce_mean(
            tf.boolean_mask(logu, joint_sample_mask),
            axis=[-1])

      log_n = tf.math.log(tf.cast(logu.shape[-1], logu.dtype))
      marginal_term = (
          tf.reduce_mean(tf.reduce_logsumexp(logu, axis=[-1]), axis=[-1])
          - log_n)
      return joint_term - marginal_term


def lower_bound_jensen_shannon(logu, joint_sample_mask=None,
                               validate_args=False, name=None):
  """Lower bound on Jensen-Shannon (JS) divergence.

  This lower bound on JS divergence is proposed in
  [Goodfellow et al. (2014)][1] and [Nowozin et al. (2016)][2].
  When estimating lower bounds on mutual information, one can also use
  different approaches for training the critic w.r.t. estimating
  mutual information [(Poole et al., 2018)][3]. The JS lower bound is
  used to train the critic with the standard lower bound on the
  Jensen-Shannon divergence as used in GANs, and then evaluates the
  critic using the NWJ lower bound on KL divergence, i.e. mutual information.
  As Eq.7 and Eq.8 of [Nowozin et al. (2016)][2], the bound is given by
  ```none
  I_JS = E_p(x,y)[log( D(x,y) )] + E_p(x)p(y)[log( 1 - D(x,y) )]
  ```
  where the first term is the expectation over the samples from joint
  distribution (positive samples), and the second is for the samples
  from marginal distributions (negative samples), with
  ```none
  D(x, y) = sigmoid(f(x, y)),
  log(D(x, y)) = softplus(-f(x, y)).
  ```
  `f(x, y)` is a critic function that scores all pairs of samples.

  Example:

  `X`, `Y` are samples from a joint Gaussian distribution, with
  correlation `0.8` and both of dimension `1`.

  ```python
  batch_size, rho, dim = 10000, 0.8, 1
  y, eps = tf.split(
      value=tf.random.normal(shape=(2 * batch_size, dim), seed=7),
      num_or_size_splits=2, axis=0)
  mean, conditional_stddev = rho * y, tf.sqrt(1. - tf.square(rho))
  x = mean + conditional_stddev * eps

  # Scores/unnormalized likelihood of pairs of samples `x[i], y[j]`
  # (For JS lower bound, the optimal critic is of the form `f(x, y) = 1 +
  # log(p(x | y) / p(x))` [(Poole et al., 2018)][3].)
  conditional_dist = tfd.MultivariateNormalDiag(
      mean, scale_identity_multiplier=conditional_stddev)
  conditional_scores = conditional_dist.log_prob(y[:, tf.newaxis, :])
  marginal_dist = tfd.MultivariateNormalDiag(tf.zeros(dim), tf.ones(dim))
  marginal_scores = marginal_dist.log_prob(y)[:, tf.newaxis]
  scores = 1 + conditional_scores - marginal_scores

  # Mask for joint samples in the score tensor
  # (The `scores` has its shape [x_batch_size, y_batch_size], i.e.
  # `scores[i, j] = f(x[i], y[j]) = log p(x[i] | y[j])`.)
  joint_sample_mask = tf.eye(batch_size, dtype=bool)

  # Lower bound on Jensen Shannon divergence
  lower_bound_jensen_shannon(logu=scores, joint_sample_mask=joint_sample_mask)
  ```

  Args:
    logu: `float`-like `Tensor` of size `[batch_size_1, batch_size_2]`
      representing critic scores (scores) for pairs of points (x, y) with
      `logu[i, j] = f(x[i], y[j])`.
    joint_sample_mask: `bool`-like `Tensor` of the same size as `logu`
      masking the positive samples by `True`, i.e. samples from joint
      distribution `p(x, y)`.
      Default value: `None`. By default, an identity matrix is constructed as
      the mask.
    validate_args: Python `bool`, default `False`. Whether to validate input
      with asserts. If `validate_args` is `False`, and the inputs are invalid,
      correct behavior is not guaranteed.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'lower_bound_jensen_shannon').

  Returns:
    lower_bound: `float`-like `scalar` for lower bound on JS divergence.

  #### References:

  [1]: Ian J. Goodfellow, et al. Generative Adversarial Nets. In
       _Conference on Neural Information Processing Systems_, 2014.
       https://arxiv.org/abs/1406.2661.
  [2]: Sebastian Nowozin, Botond Cseke, Ryota Tomioka. f-GAN: Training
       Generative Neural Samplers using Variational Divergence Minimization.
       In _Conference on Neural Information Processing Systems_, 2016.
       https://arxiv.org/abs/1606.00709.
  [3]: Ben Poole, Sherjil Ozair, Aaron van den Oord, Alexander A. Alemi,
       George Tucker. On Variational Bounds of Mutual Information. In
       _International Conference on Machine Learning_, 2019.
       https://arxiv.org/abs/1905.06922.
  """

  with tf.name_scope(name or 'lower_bound_jensen_shannon'):
    with tf.control_dependencies(
        _maybe_assert_float_matrix(logu, validate_args)):
      if joint_sample_mask is None:
        logu = tf.convert_to_tensor(
            logu, dtype_hint=tf.float32, name='logu')
        logu_diag = tf.linalg.diag_part(logu)
        joint_samples_nll = -tf.reduce_mean(
            tf.nn.softplus(-logu_diag), axis=[-1])
        n, m = tf.unstack(tf.cast(tf.shape(logu)[-2:], dtype=logu.dtype))
        marginal_samples_nll = (
            (tf.reduce_sum(tf.nn.softplus(logu), axis=[-2, -1])
             - tf.reduce_sum(tf.nn.softplus(logu_diag), axis=[-1]))
            / (n * (m - 1.)))
        return joint_samples_nll - marginal_samples_nll

      logu, joint_sample_mask = _check_and_get_mask(
          logu, joint_sample_mask, validate_args=validate_args)

      joint_samples = tf.boolean_mask(logu, joint_sample_mask)
      lower_bound = -tf.reduce_mean(tf.math.softplus(-joint_samples),
                                    axis=[-1])

      marginal_samples = tf.boolean_mask(
          logu, ~joint_sample_mask)  # pylint: disable=invalid-unary-operand-type
      lower_bound -= tf.reduce_mean(tf.math.softplus(marginal_samples),
                                    axis=[-1])
      return lower_bound


def lower_bound_nguyen_wainwright_jordan(logu, joint_sample_mask=None,
                                         validate_args=False, name=None):
  """Lower bound on Kullback-Leibler (KL) divergence from Nguyen at al.

  The lower bound was introduced by Nguyen, Wainwright, Jordan (NWJ) in
  [Nguyen et al. (2010)][1], and also known as `f-GAN KL` [(Nowozin et al.,
  2016)][2] and `MINE-f` [(Belghazi et al., 2018)][3].
  ```none
  I_NWJ = E_p(x,y)[f(x, y)] - 1/e * E_p(y)[Z(y)],
  ```
  where `f(x, y)` is a critic function that scores pairs of samples `(x, y)`,
  and `Z(y)` is the corresponding partition function:
  ```none
  Z(y) = E_p(x)[ exp(f(x, y)) ].
  ```

  Example:

  `X`, `Y` are samples from a joint Gaussian distribution, with correlation
  `0.8` and both of dimension `1`.

  ```python
  batch_size, rho, dim = 10000, 0.8, 1
  y, eps = tf.split(
      value=tf.random.normal(shape=(2 * batch_size, dim), seed=7),
      num_or_size_splits=2, axis=0)
  mean, conditional_stddev = rho * y, tf.sqrt(1. - tf.square(rho))
  x = mean + conditional_stddev * eps

  # Scores/unnormalized likelihood of pairs of samples `x[i], y[j]`
  # (For NWJ lower bound, the optimal critic is of the form `f(x, y) = 1 +
  # log(p(x | y) / p(x))` [(Poole et al., 2018)][4]. )
  conditional_dist = tfd.MultivariateNormalDiag(
      mean, scale_identity_multiplier=conditional_stddev)
  conditional_scores = conditional_dist.log_prob(y[:, tf.newaxis, :])
  marginal_dist = tfd.MultivariateNormalDiag(tf.zeros(dim), tf.ones(dim))
  marginal_scores = marginal_dist.log_prob(y)[:, tf.newaxis]
  scores = 1 + conditional_scores - marginal_scores

  # Mask for joint samples in score tensor
  # (The `scores` has its shape [x_batch_size, y_batch_size], i.e.
  # `scores[i, j] = f(x[i], y[j]) = log p(x[i] | y[j])`.)
  joint_sample_mask = tf.eye(batch_size, dtype=bool)

  # Lower bound on KL divergence between p(x,y) and p(x)p(y),
  # i.e. the mutual information between `X` and `Y`.
  lower_bound_nguyen_wainwright_jordan(
      logu=scores, joint_sample_mask=joint_sample_mask)
  ```

  Args:
    logu: `float`-like `Tensor` of size `[batch_size_1, batch_size_2]`
      representing critic scores (scores) for pairs of points (x, y) with
      `logu[i, j] = f(x[i], y[j])`.
    joint_sample_mask: `bool`-like `Tensor` of the same size as `logu`
      masking the positive samples by `True`, i.e. samples from joint
      distribution `p(x, y)`.
      Default value: `None`. By default, an identity matrix is constructed as
      the mask.
    validate_args: Python `bool`, default `False`. Whether to validate input
      with asserts. If `validate_args` is `False`, and the inputs are invalid,
      correct behavior is not guaranteed.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'lower_bound_nguyen_wainwright_jordan').

  Returns:
    lower_bound: `float`-like `scalar` for lower bound on KL divergence
      between joint and marginal distrbutions.

  #### References:

  [1]: XuanLong Nguyen, Martin J. Wainwright, Michael I. Jordan.
       Estimating Divergence Functionals and the Likelihood Ratio
       by Convex Risk Minimization. _IEEE Transactions on Information Theory_,
       56(11):5847-5861, 2010. https://arxiv.org/abs/0809.0853.
  [2]: Sebastian Nowozin, Botond Cseke, Ryota Tomioka. f-GAN: Training
       Generative Neural Samplers using Variational Divergence Minimization.
       In _Conference on Neural Information Processing Systems_, 2016.
       https://arxiv.org/abs/1606.00709.
  [3]: Mohamed Ishmael Belghazi, et al. MINE: Mutual Information Neural
       Estimation. In _International Conference on Machine Learning_, 2018.
       https://arxiv.org/abs/1801.04062.
  [4]: Ben Poole, Sherjil Ozair, Aaron van den Oord, Alexander A. Alemi,
       George Tucker. On Variational Bounds of Mutual Information. In
       _International Conference on Machine Learning_, 2019.
       https://arxiv.org/abs/1905.06922.
  """

  with tf.name_scope(name or 'lower_bound_nguyen_wainwright_jordan'):
    with tf.control_dependencies(
        _maybe_assert_float_matrix(logu, validate_args)):
      if joint_sample_mask is None:
        logu = tf.convert_to_tensor(
            logu, dtype_hint=tf.float32, name='logu')
        joint_term = tf.reduce_mean(
            tf.linalg.diag_part(logu), axis=[-1])
        num_rows, num_cols = tf.unstack(tf.shape(logu)[-2:])
        marginal_sample_mask = ~tf.eye(num_rows, num_cols, dtype=tf.bool)
      else:
        logu, joint_sample_mask = _check_and_get_mask(
            logu, joint_sample_mask, validate_args=validate_args)
        joint_term = tf.reduce_mean(
            tf.boolean_mask(logu, joint_sample_mask), axis=[-1])
        marginal_sample_mask = ~joint_sample_mask  # pylint: disable=invalid-unary-operand-type

      marginal_term = _masked_logmeanexp(logu, marginal_sample_mask,
                                         axis=[-2, -1])
      return joint_term - tf.math.exp(marginal_term - 1.)
