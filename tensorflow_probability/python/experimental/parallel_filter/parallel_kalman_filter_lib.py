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
"""Library for parallel Kalman filtering using `scan_associative`."""

import collections
import functools

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers


__all__ = ['kalman_filter',
           'sample_walk']


# Kalman filter parameters that are time dependent.
TimeDependentParameters = collections.namedtuple(
    'TimeDependentParameters',
    ['transition_matrix',
     'transition_cov',
     'transition_scale_tril',
     'transition_mean',
     'observation_matrix',
     'observation_cov',
     'observation_scale_tril',
     'observation_mean'])


# Kalman filter parameters that are not time dependent.
TimeIndependentParameters = collections.namedtuple(
    'TimeIndependentParameters',
    ['initial_cov',
     'initial_scale_tril',
     'initial_mean'])

Observations = collections.namedtuple('Observations', ['y', 'mask'])

# Results of Kalman filtering.
FilterResults = collections.namedtuple(
    'FilterResults',
    ['log_likelihoods',
     'filtered_means',
     'filtered_covs',
     'predicted_means',
     'predicted_covs',
     'observation_means',
     'observation_covs'])

AffineUpdate = collections.namedtuple('AffineUpdate',
                                      ['transition_matrix',
                                       'mean'])

# The naming of the elements of `FilterElements` are derived from
# p.4 of Temporal Parallelization of Bayesian Smoothers
# https://arxiv.org/abs/1905.13002
FilterElements = collections.namedtuple(
    'FilterElements',
    [   # First element (f): p(x_k | y_k, x_{k-1}) = N(x_k; A * x_{k-1} + b, C).
        'posterior_link_matrix',  # A
        'posterior_mean',  #  b
        'posterior_cov',  # C
        # Second element (g): p(y_k | x_{k-1}) = N(x_{k-1}; eta, J).
        'marginal_likelihood_meanprec',  # eta
        'marginal_likelihood_prec'  # J
    ])


def broadcast_to_full_batch_shape(time_indep,
                                  time_dep,
                                  observation=Observations(None, None)):
  """Ensures that all provided Tensors have full batch shape."""
  time_indep_ranks = TimeIndependentParameters(2, 2, 1)
  time_dep_ranks = TimeDependentParameters(2, 2, 2, 1, 2, 2, 2, 1)
  observation_ranks = Observations(1, 0)

  batch_shape = functools.reduce(
      ps.broadcast_shape,
      tf.nest.flatten(
          tf.nest.map_structure(
              _extract_batch_shape,
              (time_indep, time_dep, observation),
              # Time-dependent tensors have a num_timesteps 'sample' dimension.
              (TimeIndependentParameters(0, 0, 0),
               TimeDependentParameters(1, 1, 1, 1, 1, 1, 1, 1),
               Observations(1, 1)),
              (time_indep_ranks, time_dep_ranks, observation_ranks))))

  time_indep = tf.nest.map_structure(
      lambda x, r: _broadcast_to_full_batch_shape_helper(  # pylint: disable=g-long-lambda
          x, r, batch_shape=batch_shape),
      time_indep, time_indep_ranks)

  time_dep, observation = tf.nest.map_structure(
      lambda x, r: _broadcast_to_full_batch_shape_helper(  # pylint: disable=g-long-lambda
          x, r, batch_shape=batch_shape, sample_ndims=1),
      (time_dep, observation),
      (time_dep_ranks, observation_ranks))

  return time_indep, time_dep, observation


def combine_walk(u0, u1):
  """Combines two elements of a latent-space prior sample."""
  with tf.name_scope('combine_walk'):
    return AffineUpdate(
        transition_matrix=tf.linalg.matmul(u1.transition_matrix,
                                           u0.transition_matrix),
        mean=tf.linalg.matvec(u1.transition_matrix, u0.mean) + u1.mean)


def sample_walk(transition_matrix,
                transition_mean,
                transition_scale_tril,
                observation_matrix,
                observation_mean,
                observation_scale_tril,
                initial_mean,
                initial_scale_tril,
                seed=None):
  """Samples from the joint distribution of a linear Gaussian state-space model.

  This method draws samples from the joint prior distribution on latent and
  observed variables in a linear Gaussian state-space model. The sampling is
  parallelized over timesteps, so that sampling a sequence of length
  `num_timesteps` requires only `O(log(num_timesteps))` sequential steps.

  As with a naive sequential implementation, the total FLOP count scales
  linearly in `num_timesteps` (as `O(T + T/2 + T/4 + ...) = O(T)`), so this
  approach does not require extra resources in an asymptotic sense. However, it
  likely has a somewhat larger constant factor, so a sequential sampler
  may be preferred when throughput rather than latency is the highest priority.

  Args:
    transition_matrix: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, latent_size, latent_size]`.
    transition_mean: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, latent_size]`.
    transition_scale_tril: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, latent_size, latent_size]`.
    observation_matrix: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, observation_size, latent_size]`.
    observation_mean: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, observation_size]`.
    observation_scale_tril: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, observation_size, observation_size]`.
    initial_mean: float `Tensor` of shape
       `[B1, .., BN, latent_size]`.
    initial_scale_tril: float `Tensor` of shape
       `[B1, .., BN, latent_size, latent_size]`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
  Returns:
    x: float `Tensor` of shape `[num_timesteps, B1, .., BN, latent_size]`.
    y: float `Tensor` of shape `[num_timesteps, B1, .., BN, observation_size]`.

  ### Mathematical Details

  The assumed model consists of latent state vectors
  `x[:num_timesteps, :latent_size]` and corresponding observed values
  `y[:num_timesteps, :observation_size]`, governed by the following dynamics:

  ```
  x[0] ~ MultivariateNormal(mean=initial_mean, scale_tril=initial_scale_tril)
  for t in range(num_timesteps - 1):
    x[t + 1] ~ MultivariateNormal(mean=matmul(transition_matrix[t],
                                              x[t]) + transition_mean[t],
                                  scale_tril=transition_scale_tril[t])
  # Observed values `y[:num_timesteps]` defined at all timesteps.
  y ~ MultivariateNormal(mean=matmul(observation_matrix, x) + observation_mean,
                         scale_tril=observation_scale_tril)
  ```

  ### Tensor layout

  `Tensor` arguments are expected to have `num_timesteps` as their *leftmost*
  axis, preceding any batch dimensions. This layout is used
  for internal computations, so providing arguments in this form avoids the
  need for potentially-spurious transposition. The returned `Tensor`s also
  follow this layout, for the same reason. Note that this differs from the
  layout mandated by the `tfd.Distribution`
  API (and exposed by `tfd.LinearGaussianStateSpaceModel`), in which the time
  axis is to the right of any batch dimensions; it is the caller's
  responsibility to perform any needed transpositions.

  Note that this method takes `scale_tril` matrices specifying the Cholesky
  factors of covariance matrices, in contrast to
  `tfp.experimental.parallel_filter.kalman_filter`, which takes the covariance
  matrices directly. This is to avoid redundant factorization, since the
  sampling process uses Cholesky factors natively, while the filtering updates
  we implement require covariance matrices. In addition, taking `scale_tril`
  matrices directly ensures that sampling is well-defined even when one or more
  components of the model are deterministic (`scale_tril=zeros([...])`).

  Tensor arguments may be specified with partial batch shape, i.e., with
  shape prefix `[num_timesteps, Bk, ..., BN]` for `k > 1`. They will be
  internally reshaped and broadcast to the full batch shape prefix
  `[num_timesteps, B1, ..., BN]`.

  """
  with tf.name_scope('sample_walk'):
    time_indep, time_dep, _ = broadcast_to_full_batch_shape(
        time_indep=TimeIndependentParameters(
            initial_cov=None,
            initial_scale_tril=initial_scale_tril,
            initial_mean=initial_mean),
        time_dep=TimeDependentParameters(
            transition_matrix=transition_matrix,
            transition_cov=None,
            transition_scale_tril=transition_scale_tril,
            transition_mean=transition_mean,
            observation_matrix=observation_matrix,
            observation_cov=None,
            observation_scale_tril=observation_scale_tril,
            observation_mean=observation_mean))

    s1, s2, s3 = samplers.split_seed(seed, n=3)
    updates = tfp_math.scan_associative(
        combine_walk,
        AffineUpdate(transition_matrix=time_dep.transition_matrix[:-1],
                     mean=mvn_tril.MultivariateNormalTriL(
                         loc=time_dep.transition_mean[:-1],
                         scale_tril=time_dep.transition_scale_tril[:-1]
                         ).sample(seed=s1)))
    x0 = mvn_tril.MultivariateNormalTriL(
        loc=time_indep.initial_mean,
        scale_tril=time_indep.initial_scale_tril).sample(seed=s2)

    x = tf.concat([[x0],
                   tf.linalg.matvec(
                       updates.transition_matrix, x0) + updates.mean],
                  axis=0)
    y = (tf.linalg.matvec(time_dep.observation_matrix, x) +
         time_dep.observation_mean +
         mvn_tril.MultivariateNormalTriL(
             scale_tril=time_dep.observation_scale_tril).sample(seed=s3))
    return x, y


def combine_filter_elements(fi, fj):
  """Binary operation used to combine partial Kalman filter results."""
  with tf.name_scope('combine_filter_elements'):
    m1t = (
        tf.linalg.solve(
            _add_identity_to_diagonal(
                tf.linalg.matmul(fi.posterior_cov,
                                 fj.marginal_likelihood_prec)),
            # TODO(b/168836494): `solve` should support an implicit transpose.
            tf.linalg.matrix_transpose(fj.posterior_link_matrix),
            adjoint=True))
    m2t = (
        tf.linalg.solve(
            _add_identity_to_diagonal(
                tf.linalg.matmul(fj.marginal_likelihood_prec,
                                 fi.posterior_cov)),
            fi.posterior_link_matrix,
            adjoint=True))

    return FilterElements(
        posterior_link_matrix=tf.linalg.matmul(m1t,
                                               fi.posterior_link_matrix,
                                               transpose_a=True),
        posterior_mean=tf.linalg.matvec(m1t,
                                        fi.posterior_mean + tf.linalg.matvec(
                                            fi.posterior_cov,
                                            fj.marginal_likelihood_meanprec),
                                        transpose_a=True) + fj.posterior_mean,
        posterior_cov=(
            tf.linalg.matmul(
                tf.linalg.matmul(m1t,
                                 fi.posterior_cov,
                                 transpose_a=True),
                fj.posterior_link_matrix,
                transpose_b=True) + fj.posterior_cov),
        marginal_likelihood_meanprec=(
            tf.linalg.matvec(
                m2t,
                (fj.marginal_likelihood_meanprec -
                 tf.linalg.matvec(fj.marginal_likelihood_prec,
                                  fi.posterior_mean)),
                transpose_a=True) + fi.marginal_likelihood_meanprec),
        marginal_likelihood_prec=(
            tf.linalg.matmul(
                tf.linalg.matmul(m2t,
                                 fj.marginal_likelihood_prec,
                                 transpose_a=True),
                fi.posterior_link_matrix) + fi.marginal_likelihood_prec)
        )


# This is slightly modified from the paper because we start
# observations at time 0.
# One way to derive this from the paper is to consider the paper in
# the case where the first step of the dynamical system is the identity
# operation with zero variance.
# n is the dimension of the latent state.
# We call the computations associated to the first step `init_*`
# and those associated to the remainder of the steps `mid_*`.
# We have separate processing for the masked and unmasked steps
# using `tf.where` to combine them.
def init_element(observation_matrix,
                 observation_cov,
                 initial_mean,
                 initial_cov,
                 observation_mean,
                 y):
  """Represents the message from an observed value at the initial timestep."""
  with tf.name_scope('init_element'):
    chol_pushforward_cov = (
        tf.linalg.cholesky(
            _propagate_cov(matrix=observation_matrix,
                           cov=initial_cov,
                           added_cov=observation_cov)))
    obs_matrix_initial_cov = tf.linalg.matmul(observation_matrix, initial_cov)
    k_transpose = tf.linalg.cholesky_solve(chol_pushforward_cov,
                                           obs_matrix_initial_cov)

    return FilterElements(
        posterior_link_matrix=tf.zeros_like(initial_cov),
        posterior_mean=tf.linalg.matvec(k_transpose,
                                        y - _propagate_mean(
                                            matrix=observation_matrix,
                                            mean=initial_mean,
                                            added_mean=observation_mean),
                                        transpose_a=True) + initial_mean,
        posterior_cov=initial_cov - tf.linalg.matmul(obs_matrix_initial_cov,
                                                     k_transpose,
                                                     transpose_a=True),
        marginal_likelihood_meanprec=tf.linalg.matvec(observation_matrix,
                                                      _cholsolve_vec(
                                                          chol_pushforward_cov,
                                                          y - observation_mean),
                                                      transpose_a=True),
        marginal_likelihood_prec=tf.linalg.matmul(observation_matrix,
                                                  tf.linalg.cholesky_solve(
                                                      chol_pushforward_cov,
                                                      observation_matrix),
                                                  transpose_a=True))


def init_element_masked(initial_mean, initial_cov):
  """Represents the message from a masked value at the initial timestep."""
  with tf.name_scope('init_element_masked'):
    return FilterElements(
        posterior_link_matrix=tf.zeros_like(initial_cov),
        posterior_mean=initial_mean,
        posterior_cov=initial_cov,
        marginal_likelihood_meanprec=tf.zeros_like(initial_mean),
        marginal_likelihood_prec=tf.zeros_like(initial_cov))


def mid_elements(transition_matrix, transition_cov,
                 observation_matrix, observation_cov,
                 transition_mean, observation_mean, y):
  """Represents messages from observed values at non-initial timesteps."""
  # The naming of these variables has been chosen for consistency with
  # p.4 of Temporal Parallelization of Bayesian Smoothers
  # https://arxiv.org/abs/1905.13002
  with tf.name_scope('mid_elements'):
    chol_pushforward_cov = tf.linalg.cholesky(
        _propagate_cov(matrix=observation_matrix,
                       cov=transition_cov,
                       added_cov=observation_cov))
    inv_pushforward_cov_yhat = _cholsolve_vec(
        chol_pushforward_cov,
        y - (_propagate_mean(matrix=observation_matrix,
                             mean=transition_mean,
                             added_mean=observation_mean)))
    transition_cov_obs_matrix_transpose = tf.linalg.matmul(transition_cov,
                                                           observation_matrix,
                                                           transpose_b=True)
    tmp = (
        _add_identity_to_diagonal(
            -tf.linalg.matmul(transition_cov_obs_matrix_transpose,
                              tf.linalg.cholesky_solve(chol_pushforward_cov,
                                                       observation_matrix))))
    obs_matrix_trans_matrix = tf.linalg.matmul(observation_matrix,
                                               transition_matrix)

    return FilterElements(
        posterior_link_matrix=tf.linalg.matmul(tmp, transition_matrix),
        posterior_mean=tf.linalg.matvec(
            transition_cov_obs_matrix_transpose,
            inv_pushforward_cov_yhat) + transition_mean,
        posterior_cov=tf.linalg.matmul(tmp, transition_cov),
        marginal_likelihood_meanprec=tf.linalg.matvec(obs_matrix_trans_matrix,
                                                      inv_pushforward_cov_yhat,
                                                      transpose_a=True),
        marginal_likelihood_prec=tf.linalg.matmul(obs_matrix_trans_matrix,
                                                  tf.linalg.cholesky_solve(
                                                      chol_pushforward_cov,
                                                      obs_matrix_trans_matrix),
                                                  transpose_a=True))


def mid_elements_masked(transition_matrix, transition_cov, transition_mean):
  """Represents messages from masked values at non-initial timesteps."""
  with tf.name_scope('mid_elements_masked'):
    return FilterElements(
        posterior_link_matrix=transition_matrix,
        posterior_mean=transition_mean,
        posterior_cov=transition_cov,
        marginal_likelihood_meanprec=tf.zeros_like(transition_mean),
        marginal_likelihood_prec=tf.zeros_like(transition_cov))


def filter_elements(time_indep, time_dep, observation):
  """Convert data into form suitable for filtering with `scan_associative`."""
  with tf.name_scope('filter_elements'):
    f1_unmasked = init_element(time_dep.observation_matrix[0],
                               time_dep.observation_cov[0],
                               time_indep.initial_mean,
                               time_indep.initial_cov,
                               time_dep.observation_mean[0],
                               observation.y[0])
    fk_unmasked = mid_elements(time_dep.transition_matrix[:-1],
                               time_dep.transition_cov[:-1],
                               time_dep.observation_matrix[1:],
                               time_dep.observation_cov[1:],
                               time_dep.transition_mean[:-1],
                               time_dep.observation_mean[1:],
                               observation.y[1:])
    elements = tf.nest.map_structure(lambda m, ns: tf.concat([[m], ns], axis=0),
                                     f1_unmasked,
                                     fk_unmasked)
    if observation.mask is not None:
      f1_masked = init_element_masked(time_indep.initial_mean,
                                      time_indep.initial_cov)
      fk_masked = mid_elements_masked(time_dep.transition_matrix[:-1],
                                      time_dep.transition_cov[:-1],
                                      time_dep.transition_mean[:-1])
      elements_masked = tf.nest.map_structure(
          lambda m, ns: tf.concat([[m], ns], axis=0),
          f1_masked,
          fk_masked)
      masks = FilterElements(
          posterior_link_matrix=observation.mask[..., None, None],
          posterior_mean=observation.mask[..., None],
          posterior_cov=observation.mask[..., None, None],
          marginal_likelihood_meanprec=observation.mask[..., None],
          marginal_likelihood_prec=observation.mask[..., None, None])
      elements = tf.nest.map_structure(tf.where,
                                       masks,
                                       elements_masked,
                                       elements)

  return elements


def kalman_filter(transition_matrix,
                  transition_mean,
                  transition_cov,
                  observation_matrix,
                  observation_mean,
                  observation_cov,
                  initial_mean,
                  initial_cov,
                  y,
                  mask,
                  return_all=True):
  """Infers latent values using a parallel Kalman filter.

  This method computes filtered marginal means and covariances of a linear
  Gaussian state-space model using a parallel message-passing algorithm, as
  described by Sarkka and Garcia-Fernandez [1]. The inference process is
  formulated as a prefix-sum problem that can be efficiently computed by
  `tfp.math.scan_associative`, so that inference for a time series of length
  `num_timesteps` requires only `O(log(num_timesteps))` sequential steps.

  As with a naive sequential implementation, the total FLOP count scales
  linearly in `num_timesteps` (as `O(T + T/2 + T/4 + ...) = O(T)`), so this
  approach does not require extra resources in an asymptotic sense. However, it
  likely has a somewhat larger constant factor, so a sequential filter may be
  preferred when throughput rather than latency is the highest priority.

  Args:
    transition_matrix: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, latent_size, latent_size]`.
    transition_mean: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, latent_size]`.
    transition_cov: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, latent_size, latent_size]`.
    observation_matrix: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, observation_size, latent_size]`.
    observation_mean: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, observation_size]`.
    observation_cov: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, observation_size, observation_size]`.
    initial_mean: float `Tensor` of shape
       `[B1, .., BN, latent_size]`.
    initial_cov: float `Tensor` of shape
       `[B1, .., BN, latent_size, latent_size]`.
    y: float `Tensor` of shape
       `[num_timesteps, B1, .., BN, observation_size]`.
    mask: float `Tensor` of shape `[num_timesteps, B1, .., BN]`.
    return_all: Python `bool`, whether to compute log-likelihoods and
      predictive and observation distributions. If `False`, only
      `filtered_means` and `filtered_covs` are computed, and `None` is returned
      for the remaining values.
  Returns:
    log_likelihoods: float `Tensor` of shape `[num_timesteps, B1, .., BN]`, such
      that `log_likelihoods[t] = log p(y[t] | y[:t])`.
    filtered_means: float `Tensor` of shape
      `[num_timesteps, B1, .., BN, latent_size]`, such that
      `filtered_means[t] == E[x[t] | y[:t + 1]]`.
    filtered_covs: float `Tensor` of shape
      `[num_timesteps, B1, .., BN, latent_size, latent_size]`.
    predictive_means: float `Tensor` of shape
      `[num_timesteps, B1, .., BN, latent_size]`, such that
      `predictive_means[t] = E[x[t + 1] | y[:t + 1]]`.
    predictive_covs: float `Tensor` of shape
      `[num_timesteps, B1, .., BN, latent_size, latent_size]`.
    observation_means: float `Tensor` of shape
      `[num_timesteps, B1, .., BN, observation_size]`, such that
      `observation_means[t] = E[y[t] | y[:t]]`.
    observation_covs:float `Tensor` of shape
      `[num_timesteps, B1, .., BN, observation_size, observation_size]`.

  ### Mathematical Details

  The assumed model consists of latent state vectors
  `x[:num_timesteps, :latent_size]` and corresponding observed values
  `y[:num_timesteps, :observation_size]`, governed by the following dynamics:

  ```
  x[0] ~ MultivariateNormal(mean=initial_mean, cov=initial_cov)
  for t in range(num_timesteps - 1):
    x[t + 1] ~ MultivariateNormal(mean=matmul(transition_matrix[t],
                                              x[t]) + transition_mean[t],
                                  cov=transition_cov[t])
  # Observed values `y[:num_timesteps]` defined at all timesteps.
  y ~ MultivariateNormal(mean=matmul(observation_matrix, x) + observation_mean,
                         cov=observation_cov)
  ```

  ### Tensor layout

  `Tensor` arguments are expected to have `num_timesteps` as their *leftmost*
  axis, preceding any batch dimensions. This layout is used
  for internal computations, so providing arguments in this form avoids the
  need for potentially-spurious transposition. The returned `Tensor`s also
  follow this layout, for the same reason. Note that this differs from the
  layout mandated by the `tfd.Distribution`
  API (and exposed by `tfd.LinearGaussianStateSpaceModel`), in which the time
  axis is to the right of any batch dimensions; it is the caller's
  responsibility to perform any needed transpositions.

  Tensor arguments may be specified with partial batch shape, i.e., with
  shape prefix `[num_timesteps, Bk, ..., BN]` for `k > 1`. They will be
  internally reshaped and broadcast to the full batch shape prefix
  `[num_timesteps, B1, ..., BN]`.

  ### References

  [1] Simo Sarkka and Angel F. Garcia-Fernandez. Temporal Parallelization of
      Bayesian Smoothers. _arXiv preprint arXiv:1905.13002_, 2019.
      https://arxiv.org/abs/1905.13002

  """
  with tf.name_scope('kalman_filter'):
    time_indep, time_dep, observation = broadcast_to_full_batch_shape(
        time_indep=TimeIndependentParameters(initial_cov=initial_cov,
                                             initial_scale_tril=None,
                                             initial_mean=initial_mean),
        time_dep=TimeDependentParameters(transition_matrix=transition_matrix,
                                         transition_cov=transition_cov,
                                         transition_scale_tril=None,
                                         transition_mean=transition_mean,
                                         observation_matrix=observation_matrix,
                                         observation_cov=observation_cov,
                                         observation_scale_tril=None,
                                         observation_mean=observation_mean),
        observation=Observations(y, mask))

    # Prevent any masked NaNs from leaking into gradients.
    if observation.mask is not None:
      observation = Observations(
          y=tf.where(observation.mask[..., None],
                     tf.zeros([], dtype=observation.y.dtype),
                     observation.y),
          mask=observation.mask)

    # Run Kalman filter.
    filtered = tfp_math.scan_associative(combine_filter_elements,
                                         filter_elements(time_indep,
                                                         time_dep,
                                                         observation))
    filtered_means = filtered.posterior_mean
    filtered_covs = filtered.posterior_cov
    log_likelihoods = None
    predicted_means, predicted_covs = None, None
    observation_means, observation_covs = None, None
    # Compute derived quantities (predictive distributions, likelihood, etc.).
    if return_all:
      predicted_means = _propagate_mean(
          matrix=time_dep.transition_matrix,
          mean=filtered_means,
          added_mean=time_dep.transition_mean)
      observation_means = _propagate_mean(
          matrix=time_dep.observation_matrix,
          mean=tf.concat([[time_indep.initial_mean],
                          predicted_means[:-1]], axis=0),
          added_mean=time_dep.observation_mean)
      predicted_covs = _propagate_cov(matrix=time_dep.transition_matrix,
                                      cov=filtered_covs,
                                      added_cov=time_dep.transition_cov)
      observation_covs = _propagate_cov(matrix=time_dep.observation_matrix,
                                        cov=tf.concat([[time_indep.initial_cov],
                                                       predicted_covs[:-1]],
                                                      axis=0),
                                        added_cov=time_dep.observation_cov)

      log_likelihoods = mvn_tril.MultivariateNormalTriL(
          loc=observation_means,
          scale_tril=tf.linalg.cholesky(observation_covs)).log_prob(
              observation.y)
      if observation.mask is not None:
        log_likelihoods = tf.where(observation.mask,
                                   tf.zeros([], dtype=log_likelihoods.dtype),
                                   log_likelihoods)

    return FilterResults(log_likelihoods,
                         filtered_means,
                         filtered_covs,
                         predicted_means,
                         predicted_covs,
                         observation_means,
                         observation_covs)


def _extract_batch_shape(x, sample_ndims, event_ndims):
  """Slice out the batch component of `x`'s shape."""
  if x is None:
    return []
  shape = ps.shape(x)
  nd = ps.rank_from_shape(shape)
  return shape[sample_ndims : nd - event_ndims]


def _broadcast_to_full_batch_shape_helper(data,
                                          event_ndims,
                                          batch_shape,
                                          sample_ndims=0):
  """Broadcasts `[sample, ?, event]` to `[sample, batch, event]`."""
  if data is None:
    return None
  data_shape = ps.shape(data)
  data_rank = ps.rank_from_shape(data_shape)
  batch_ndims = ps.rank_from_shape(batch_shape)

  # Reshape the data to have full batch rank. For example, given
  # `batch_shape==[3, 2]`, this would reshape `data.shape==[S, 2, E]` to
  # `[S, 1, 2, E]`).
  # This reshaping is not necessary when `sample_ndims==0`, since with no sample
  # dimensions the batch shape itself is leftmost and can broadcast. For
  # example, we would not need to reshape `[2, E] -> [1, 2, E]`.
  if sample_ndims != 0:
    padding_ndims = batch_ndims - (data_rank - sample_ndims - event_ndims)
    padded_shape = ps.concat([data_shape[:sample_ndims],
                              ps.ones([padding_ndims], dtype=np.int32),
                              data_shape[sample_ndims:]], axis=0)
    data = tf.reshape(data, padded_shape)
    data_shape = padded_shape
    data_rank = ps.rank_from_shape(data_shape)

  # Broadcast the data to have full batch shape. For example, given
  # `batch_shape==[3, 2]`, this would broadcast `data.shape==[S, 1, 2, E]` to
  # `[S, 3, 2, E]`.
  new_shape = tf.concat([data_shape[:sample_ndims],
                         batch_shape,
                         data_shape[data_rank - event_ndims:]], axis=0)
  return tf.broadcast_to(data, new_shape)


def _add_identity_to_diagonal(x):
  return tf.linalg.set_diag(x, tf.linalg.diag_part(x) + 1.)


def _cholsolve_vec(chol, rhs):
  return tf.linalg.cholesky_solve(chol, rhs[..., tf.newaxis])[..., 0]


def _propagate_mean(matrix, mean, added_mean):
  return tf.linalg.matvec(matrix, mean) + added_mean


def _propagate_cov(matrix, cov, added_cov):
  return (
      tf.linalg.matmul(
          tf.linalg.matmul(matrix, cov),
          matrix,
          transpose_b=True) + added_cov)
