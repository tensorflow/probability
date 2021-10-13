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
"""The Normal distribution: conjugate posterior closed form calculations."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import normal

from tensorflow.python.ops.linalg import linear_operator_addition  # pylint: disable=g-direct-tensorflow-import


def mvn_conjugate_linear_update(prior_scale,
                                linear_transformation,
                                likelihood_scale,
                                observation,
                                prior_mean=None,
                                name=None):
  """Computes a conjugate normal posterior for a Bayesian linear regression.

  We assume the following model:

  ```
  latent ~ MVN(loc=prior_mean, scale=prior_scale)
  observation ~ MVN(loc=linear_transformation.matvec(latent),
                    scale=likelihood_scale)
  ```

  For Bayesian linear regression, the `latent` represents the weights, and the
  provided `linear_transformation` is the design matrix.

  This method computes the multivariate normal
  posterior `p(latent | observation)`, using `LinearOperator`s to perform
  perform computations efficiently when the matrices involved have special
  structure.

  Args:
    prior_scale: Instance of `tf.linalg.LinearOperator` of shape
      `[..., num_features, num_features]`, specifying a
      scale matrix (any matrix `L` such that `LL' = Q` where `Q` is the
      covariance) for the prior on regression weights. May optionally be a
      float `Tensor`.
    linear_transformation: Instance of `tf.linalg.LinearOperator` of shape
      `[..., num_outputs, num_features])`, specifying a transformation of the
      latent values. May optionally be a float `Tensor`.
    likelihood_scale: Instance of `tf.linalg.LinearOperator` of shape
      `[..., num_outputs, num_outputs]` specifying a scale matrix (any matrix
      `L` such that `LL' = Q` where `Q` is the covariance) for the likelihood
      of observed targets. May optionally be a float `Tensor`.
    observation: Float `Tensor` of shape `[..., num_outputs]]), specifying the
      observed values or regression targets.
    prior_mean: Optional float `Tensor` of shape `[..., num_features]`,
      specifying the prior mean. If `None`, the prior mean is assumed to be
      zero and some computation is avoided.
      Default value: `None`.
    name: Option Python `str` name given to ops created by this function.
      Default value: 'mvn_conjugate_linear_update'.
  Returns:
    posterior_mean: Float `Tensor` of shape `[..., num_features]`, giving the
      mean of the multivariate normal posterior on the latent value.
    posterior_prec: Instance of `tf.linalg.LinearOperator` of shape
      shape `[..., num_features, num_features]`, giving the
      posterior precision (inverse covariance) matrix.

  #### Mathematical details

  Let the prior precision be denoted by
  `prior_prec = prior_scale.matmul(prior_scale, adjoint_arg=True).inverse()`
  and the likelihood precision by `likelihood_prec = likelihood_scale.matmul(
  likelihood_scale, adjoint_arg=True).inverse()`. Then the posterior
  `p(latent | observation)` is multivariate normal with precision

  ```python
  posterior_prec = (
    linear_transformation.matmul(
      likelihood_prec.matmul(linear_transformation), adjoint=True) +
     prior_prec)
  ```

  and mean

  ```python
  posterior_mean = posterior_prec.solvevec(
    linear_transformation.matvec(
      likelihood_prec.matvec(observation) +
      prior_prec.matvec(prior_mean)))
  ```

  """
  with tf.name_scope(name or 'mvn_conjugate_linear_update'):

    def ensure_is_linop(x):
      return x if hasattr(x, 'solve') else tf.linalg.LinearOperatorFullMatrix(x)
    prior_scale = ensure_is_linop(prior_scale)
    likelihood_scale = ensure_is_linop(likelihood_scale)
    linear_transformation = ensure_is_linop(linear_transformation)

    observation = tf.convert_to_tensor(observation, name='observation')
    if prior_mean is not None:
      prior_mean = tf.convert_to_tensor(prior_mean, name='prior_mean')

    prior_prec_chol = prior_scale.inverse()
    prior_prec = prior_prec_chol.matmul(prior_prec_chol, adjoint=True)

    # Compute `evidence_prec = X.T @ Q^-1 @ X`, with
    #  Q = likelihood covariance (`likelihood_scale @ likelihood_scale.T`)
    #  X = linear transformation.
    scaled_transform = likelihood_scale.solve(linear_transformation)
    evidence_prec = scaled_transform.matmul(scaled_transform, adjoint=True)

    try:  # Attempt to add prior + evidence efficiently by exploiting structure.
      sum_terms = linear_operator_addition.add_operators(
          [prior_prec, evidence_prec])  # Unregistered linops raise a TypeError.
      if len(sum_terms) > 1:
        raise TypeError('LinearOperator addition failed to reduce terms.')
      posterior_prec = sum_terms[0]
    except TypeError:  # We have to do things the hard way.
      posterior_prec = tf.linalg.LinearOperatorFullMatrix(
          prior_prec.to_dense() + evidence_prec.to_dense())

    # Hint to LinearOperator that precision matrices are always PSD.
    # pylint: disable=protected-access
    posterior_prec._is_positive_definite = True
    posterior_prec._is_self_adjoint = True
    posterior_prec._is_square = True
    # pylint: enable=protected-access

    # The posterior mean is a weighted combination of the prior mean and the
    # observed value, scaled by the posterior covariance.
    prior_plus_observed_value = scaled_transform.matvec(
        likelihood_scale.solvevec(observation), adjoint=True)
    if prior_mean is not None:
      prior_plus_observed_value += prior_prec.matvec(prior_mean)
    posterior_mean = posterior_prec.solvevec(prior_plus_observed_value)

    return posterior_mean, posterior_prec


def normal_conjugates_known_scale_posterior(prior, scale, s, n):
  """Posterior Normal distribution with conjugate prior on the mean.

  This model assumes that `n` observations (with sum `s`) come from a
  Normal with unknown mean `loc` (described by the Normal `prior`)
  and known variance `scale**2`. The "known scale posterior" is
  the distribution of the unknown `loc`.

  Accepts a prior Normal distribution object, having parameters
  `loc0` and `scale0`, as well as known `scale` values of the predictive
  distribution(s) (also assumed Normal),
  and statistical estimates `s` (the sum(s) of the observations) and
  `n` (the number(s) of observations).

  Returns a posterior (also Normal) distribution object, with parameters
  `(loc', scale'**2)`, where:

  ```
  mu ~ N(mu', sigma'**2)
  sigma'**2 = 1/(1/sigma0**2 + n/sigma**2),
  mu' = (mu0/sigma0**2 + s/sigma**2) * sigma'**2.
  ```

  Distribution parameters from `prior`, as well as `scale`, `s`, and `n`.
  will broadcast in the case of multidimensional sets of parameters.

  Args:
    prior: `Normal` object of type `dtype`:
      the prior distribution having parameters `(loc0, scale0)`.
    scale: tensor of type `dtype`, taking values `scale > 0`.
      The known stddev parameter(s).
    s: Tensor of type `dtype`. The sum(s) of observations.
    n: Tensor of type `int`. The number(s) of observations.

  Returns:
    A new Normal posterior distribution object for the unknown observation
    mean `loc`.

  Raises:
    TypeError: if dtype of `s` does not match `dtype`, or `prior` is not a
      Normal object.
  """
  if not isinstance(prior, normal.Normal):
    raise TypeError('Expected prior to be an instance of type Normal')

  if s.dtype != prior.dtype:
    raise TypeError(
        'Observation sum s.dtype does not match prior dtype: %s vs. %s'
        % (s.dtype, prior.dtype))

  n = tf.cast(n, prior.dtype)
  scale0_2 = tf.square(prior.scale)
  scale_2 = tf.square(scale)
  scalep_2 = 1.0/(1/scale0_2 + n/scale_2)
  return normal.Normal(
      loc=(prior.loc / scale0_2 + s / scale_2) * scalep_2,
      scale=tf.sqrt(scalep_2))


def normal_conjugates_known_scale_predictive(prior, scale, s, n):
  """Posterior predictive Normal distribution w. conjugate prior on the mean.

  This model assumes that `n` observations (with sum `s`) come from a
  Normal with unknown mean `loc` (described by the Normal `prior`)
  and known variance `scale**2`. The "known scale predictive"
  is the distribution of new observations, conditioned on the existing
  observations and our prior.

  Accepts a prior Normal distribution object, having parameters
  `loc0` and `scale0`, as well as known `scale` values of the predictive
  distribution(s) (also assumed Normal),
  and statistical estimates `s` (the sum(s) of the observations) and
  `n` (the number(s) of observations).

  Calculates the Normal distribution(s) `p(x | sigma**2)`:

  ```
  p(x | sigma**2) = int N(x | mu, sigma**2)N(mu | prior.loc, prior.scale**2) dmu
                  = N(x | prior.loc, 1 / (sigma**2 + prior.scale**2))
  ```

  Returns the predictive posterior distribution object, with parameters
  `(loc', scale'**2)`, where:

  ```
  sigma_n**2 = 1/(1/sigma0**2 + n/sigma**2),
  mu' = (mu0/sigma0**2 + s/sigma**2) * sigma_n**2.
  sigma'**2 = sigma_n**2 + sigma**2,
  ```

  Distribution parameters from `prior`, as well as `scale`, `s`, and `n`.
  will broadcast in the case of multidimensional sets of parameters.

  Args:
    prior: `Normal` object of type `dtype`:
      the prior distribution having parameters `(loc0, scale0)`.
    scale: tensor of type `dtype`, taking values `scale > 0`.
      The known stddev parameter(s).
    s: Tensor of type `dtype`. The sum(s) of observations.
    n: Tensor of type `int`. The number(s) of observations.

  Returns:
    A new Normal predictive distribution object.

  Raises:
    TypeError: if dtype of `s` does not match `dtype`, or `prior` is not a
      Normal object.
  """
  if not isinstance(prior, normal.Normal):
    raise TypeError('Expected prior to be an instance of type Normal')

  if s.dtype != prior.dtype:
    raise TypeError(
        'Observation sum s.dtype does not match prior dtype: %s vs. %s'
        % (s.dtype, prior.dtype))

  n = tf.cast(n, prior.dtype)
  scale0_2 = tf.square(prior.scale)
  scale_2 = tf.square(scale)
  scalep_2 = 1.0/(1/scale0_2 + n/scale_2)
  return normal.Normal(
      loc=(prior.loc / scale0_2 + s / scale_2) * scalep_2,
      scale=tf.sqrt(scalep_2 + scale_2))
