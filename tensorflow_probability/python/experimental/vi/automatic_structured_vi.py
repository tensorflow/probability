# Copyright 2021 The TensorFlow Probability Authors.
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
"""Utilities for constructing structured surrogate posteriors."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import restructure
from tensorflow_probability.python.bijectors import scale as scale_lib
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import sigmoid
from tensorflow_probability.python.distributions import beta
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution
from tensorflow_probability.python.distributions import joint_distribution_auto_batched
from tensorflow_probability.python.distributions import joint_distribution_coroutine
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import truncated_normal
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static as ps

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


Root = joint_distribution_coroutine.JointDistributionCoroutine.Root

_NON_STATISTICAL_PARAMS = [
    'name', 'validate_args', 'allow_nan_stats', 'experimental_use_kahan_sum',
    'reinterpreted_batch_ndims', 'dtype'
]
_NON_TRAINABLE_PARAMS = ['low', 'high']

ASVIParameters = collections.namedtuple(
    'ASVIParameters', ['prior_weight', 'mean_field_parameter'])


def _as_trainable_family(distribution):
  """Substitutes prior distributions with more easily trainable ones."""
  with tf.name_scope('as_trainable_family'):

    if isinstance(distribution, half_normal.HalfNormal):
      return truncated_normal.TruncatedNormal(
          loc=0.,
          scale=distribution.scale,
          low=0.,
          high=distribution.scale * 10.)
    elif isinstance(distribution, uniform.Uniform):
      return shift.Shift(distribution.low)(
          scale_lib.Scale(distribution.high - distribution.low)(beta.Beta(
              concentration0=tf.ones(
                  distribution.event_shape_tensor(), dtype=distribution.dtype),
              concentration1=1.)))
    else:
      return distribution


def _make_asvi_trainable_variables(prior,
                                   mean_field=False,
                                   initial_prior_weight=0.5):
  """Generates parameter dictionaries given a prior distribution and list."""
  with tf.name_scope('make_asvi_trainable_variables'):
    param_dicts = []
    prior_dists = prior._get_single_sample_distributions()  # pylint: disable=protected-access
    for dist in prior_dists:
      original_dist = dist.distribution if isinstance(dist, Root) else dist

      substituted_dist = _as_trainable_family(original_dist)

      # Grab the base distribution if it exists
      try:
        actual_dist = substituted_dist.distribution
      except AttributeError:
        actual_dist = substituted_dist

      new_params_dict = {}

      #  Build trainable ASVI representation for each distribution's parameters.
      parameter_properties = actual_dist.parameter_properties(
          dtype=actual_dist.dtype)

      if isinstance(original_dist, sample.Sample):
        posterior_batch_shape = ps.concat([
            actual_dist.batch_shape_tensor(),
            distribution_util.expand_to_vector(original_dist.sample_shape)
        ], axis=0)
      else:
        posterior_batch_shape = actual_dist.batch_shape_tensor()

      for param, value in actual_dist.parameters.items():

        if param in (_NON_STATISTICAL_PARAMS +
                     _NON_TRAINABLE_PARAMS) or value is None:
          continue

        actual_event_shape = parameter_properties[param].shape_fn(
            actual_dist.event_shape_tensor())
        try:
          bijector = parameter_properties[
              param].default_constraining_bijector_fn()
        except NotImplementedError:
          bijector = identity.Identity()

        if mean_field:
          prior_weight = None
        else:
          unconstrained_ones = tf.ones(
              shape=ps.concat([
                  posterior_batch_shape,
                  bijector.inverse_event_shape_tensor(
                      actual_event_shape)
              ], axis=0),
              dtype=tf.convert_to_tensor(value).dtype)

          prior_weight = tfp_util.TransformedVariable(
              initial_prior_weight * unconstrained_ones,
              bijector=sigmoid.Sigmoid(),
              name='prior_weight/{}/{}'.format(dist.name, param))

        # If the prior distribution was a tfd.Sample wrapping a base
        # distribution, we want to give every single sample in the prior its
        # own lambda and alpha value (rather than having a single lambda and
        # alpha).
        if isinstance(original_dist, sample.Sample):
          value = tf.reshape(
              value,
              ps.concat([
                  actual_dist.batch_shape_tensor(),
                  ps.ones(ps.rank_from_shape(original_dist.sample_shape)),
                  actual_event_shape
              ],
                        axis=0))
          value = tf.broadcast_to(
              value,
              ps.concat([posterior_batch_shape, actual_event_shape], axis=0))
        new_params_dict[param] = ASVIParameters(
            prior_weight=prior_weight,
            mean_field_parameter=tfp_util.TransformedVariable(
                value,
                bijector=bijector,
                name='mean_field_parameter/{}/{}'.format(dist.name, param)))

      param_dicts.append(new_params_dict)
  return param_dicts


# TODO(kateslin): Add support for models with prior+likelihood written as
# a single JointDistribution.
def build_asvi_surrogate_posterior(prior,
                                   mean_field=False,
                                   initial_prior_weight=0.5,
                                   name=None):
  """Builds a structured surrogate posterior inspired by conjugate updating.

  ASVI, or Automatic Structured Variational Inference, was proposed by
  Ambrogioni et al. (2020) [1] as a method of automatically constructing a
  surrogate posterior with the same structure as the prior. It does this by
  reparameterizing the variational family of the surrogate posterior by
  structuring each parameter according to the equation
  ```none
  prior_weight * prior_parameter + (1 - prior_weight) * mean_field_parameter
  ```
  In this equation, `prior_parameter` is a vector of prior parameters and
  `mean_field_parameter` is a vector of trainable parameters with the same
  domain as `prior_parameter`. `prior_weight` is a vector of learnable
  parameters where `0. <= prior_weight <= 1.`. When `prior_weight =
  0`, the surrogate posterior will be a mean-field surrogate, and when
  `prior_weight = 1.`, the surrogate posterior will be the prior. This convex
  combination equation, inspired by conjugacy in exponential families, thus
  allows the surrogate posterior to balance between the structure of the prior
  and the structure of a mean-field approximation.

  Args:
    prior: tfd.JointDistribution instance of the prior.
    mean_field: Optional Python boolean. If `True`, creates a degenerate
      surrogate distribution in which all variables are independent,
      ignoring the prior dependence structure. Default value: `False`.
    initial_prior_weight: Optional float value (either static or tensor value)
      on the interval [0, 1]. A larger value creates an initial surrogate
      distribution with more dependence on the prior structure. Default value:
      `0.5`.
    name: Optional string. Default value: `build_asvi_surrogate_posterior`.

  Returns:
    surrogate_posterior: A `tfd.JointDistributionCoroutineAutoBatched` instance
    whose samples have shape and structure matching that of `prior`.

  Raises:
    TypeError: The `prior` argument cannot be a nested `JointDistribution`.

  ### Examples

  Consider a Brownian motion model expressed as a JointDistribution:

  ```python
  prior_loc = 0.
  innovation_noise = .1

  def model_fn():
    new = yield tfd.Normal(loc=prior_loc, scale=innovation_noise)
    for i in range(4):
      new = yield tfd.Normal(loc=new, scale=innovation_noise)

  prior = tfd.JointDistributionCoroutineAutoBatched(model_fn)
  ```

  Let's use variational inference to approximate the posterior. We'll build a
  surrogate posterior distribution by feeding in the prior distribution.

  ```python
  surrogate_posterior =
    tfp.experimental.vi.build_asvi_surrogate_posterior(prior)
  ```

  This creates a trainable joint distribution, defined by variables in
  `surrogate_posterior.trainable_variables`. We use `fit_surrogate_posterior`
  to fit this distribution by minimizing a divergence to the true posterior.

  ```python
  losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior=surrogate_posterior,
    num_steps=100,
    optimizer=tf.optimizers.Adam(0.1),
    sample_size=10)

  # After optimization, samples from the surrogate will approximate
  # samples from the true posterior.
  samples = surrogate_posterior.sample(100)
  posterior_mean = [tf.reduce_mean(x) for x in samples]
  posterior_std = [tf.math.reduce_std(x) for x in samples]
  ```

  #### References
  [1]: Luca Ambrogioni, Max Hinne, Marcel van Gerven. Automatic structured
        variational inference. _arXiv preprint arXiv:2002.00643_, 2020
        https://arxiv.org/abs/2002.00643

  """

  with tf.name_scope(name or 'build_asvi_surrogate_posterior'):
    param_dicts = _make_asvi_trainable_variables(
        prior=prior,
        mean_field=mean_field,
        initial_prior_weight=initial_prior_weight)
    def posterior_generator():

      prior_gen = prior._model_coroutine()  # pylint: disable=protected-access
      dist = next(prior_gen)

      i = 0
      try:
        while True:
          original_dist = dist.distribution if isinstance(dist, Root) else dist

          if isinstance(original_dist, joint_distribution.JointDistribution):
            # TODO(kateslin): Build inner JD surrogate in
            # _make_asvi_trainable_variables to avoid rebuilding variables.
            raise TypeError(
                'Argument `prior` cannot be a nested `JointDistribution`.')

          else:

            original_dist = _as_trainable_family(original_dist)

            try:
              actual_dist = original_dist.distribution
            except AttributeError:
              actual_dist = original_dist

            dist_params = actual_dist.parameters
            temp_params_dict = {}

            for param, value in dist_params.items():
              if param in (_NON_STATISTICAL_PARAMS +
                           _NON_TRAINABLE_PARAMS) or value is None:
                temp_params_dict[param] = value
              else:
                prior_weight = param_dicts[i][param].prior_weight
                mean_field_parameter = param_dicts[i][
                    param].mean_field_parameter
                if mean_field:
                  temp_params_dict[param] = mean_field_parameter
                else:
                  temp_params_dict[param] = prior_weight * value + (
                      1. - prior_weight) * mean_field_parameter

            if isinstance(original_dist, sample.Sample):
              inner_dist = type(actual_dist)(**temp_params_dict)

              surrogate_dist = independent.Independent(
                  inner_dist,
                  reinterpreted_batch_ndims=ps.rank_from_shape(
                      original_dist.sample_shape))
            else:
              surrogate_dist = type(actual_dist)(**temp_params_dict)

            if isinstance(original_dist,
                          transformed_distribution.TransformedDistribution):
              surrogate_dist = transformed_distribution.TransformedDistribution(
                  surrogate_dist, bijector=original_dist.bijector)

            if isinstance(original_dist, independent.Independent):
              surrogate_dist = independent.Independent(
                  surrogate_dist,
                  reinterpreted_batch_ndims=original_dist
                  .reinterpreted_batch_ndims)

            if isinstance(dist, Root):
              value_out = yield Root(surrogate_dist)
            else:
              value_out = yield surrogate_dist

          dist = prior_gen.send(value_out)
          i += 1
      except StopIteration:
        pass

    surrogate_posterior = (
        joint_distribution_auto_batched.JointDistributionCoroutineAutoBatched(
            posterior_generator))

    # Ensure that the surrogate posterior structure matches that of the prior
    try:
      nest.assert_same_structure(prior.dtype, surrogate_posterior.dtype)
    except TypeError:
      tokenize = lambda jd: jd._model_unflatten(  # pylint: disable=protected-access, g-long-lambda
          range(len(jd._model_flatten(jd.dtype)))  # pylint: disable=protected-access
      )
      surrogate_posterior = restructure.Restructure(
          output_structure=tokenize(prior),
          input_structure=tokenize(surrogate_posterior))(
              surrogate_posterior)

    surrogate_posterior.also_track = param_dicts
    return surrogate_posterior
