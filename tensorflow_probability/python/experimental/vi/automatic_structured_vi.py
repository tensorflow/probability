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

import collections
import copy
import functools
import inspect

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import restructure
from tensorflow_probability.python.bijectors import scale as scale_lib
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import sigmoid
from tensorflow_probability.python.distributions import beta
from tensorflow_probability.python.distributions import chi2
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_auto_batched
from tensorflow_probability.python.distributions import joint_distribution_coroutine
from tensorflow_probability.python.distributions import markov_chain
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import truncated_normal
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers


__all__ = [
    'register_asvi_substitution_rule',
    'build_asvi_surrogate_posterior'
]


Root = joint_distribution_coroutine.JointDistributionCoroutine.Root

_NON_STATISTICAL_PARAMS = [
    'name', 'validate_args', 'allow_nan_stats', 'experimental_use_kahan_sum',
    'reinterpreted_batch_ndims', 'dtype', 'force_probs_to_zero_outside_support',
    'num_probit_terms_approx'
]
_NON_TRAINABLE_PARAMS = ['low', 'high']

ASVIParameters = collections.namedtuple(
    'ASVIParameters', ['prior_weight', 'mean_field_parameter'])

# Registry of transformations that are applied to distributions in the prior
# before defining the surrogate family.
ASVI_SURROGATE_SUBSTITUTIONS = {}


def _as_substituted_distribution(distribution):
  """Applies all substitution rules that match a distribution."""
  for condition, substitution_fn in ASVI_SURROGATE_SUBSTITUTIONS.items():
    if condition(distribution):
      distribution = substitution_fn(distribution)
  return distribution


def register_asvi_substitution_rule(condition, substitution_fn):
  """Registers a rule for substituting distributions in ASVI surrogates.

  Args:
    condition: Python `callable` that takes a Distribution instance and
      returns a Python `bool` indicating whether or not to substitute it.
      May also be a class type such as `tfd.Normal`, in which case the
      condition is interpreted as
      `lambda distribution: isinstance(distribution, class)`.
    substitution_fn: Python `callable` that takes a Distribution
      instance and returns a new Distribution instance used to define
      the ASVI surrogate posterior. Note that this substitution does not modify
      the original model.

  #### Example

  To use a Normal surrogate for all location-scale family distributions, we
  could register the substitution:

  ```python
  tfp.experimental.vi.register_asvi_surrogate_substitution(
    condition=lambda distribution: (
      hasattr(distribution, 'loc') and hasattr(distribution, 'scale'))
    substitution_fn=lambda distribution: (
      # Invoking the event space bijector applies any relevant constraints,
      # e.g., that HalfCauchy samples must be `>= loc`.
      distribution.experimental_default_event_space_bijector()(
        tfd.Normal(loc=distribution.loc, scale=distribution.scale)))
  ```

  This rule will fire when ASVI encounters a location-scale distribution,
  and instructs ASVI to build a surrogate 'as if' the model had just used a
  (possibly constrained) Normal in its place. Note that we could have used a
  more precise condition, e.g., to limit the substitution to distributions with
  a specific `name`, if we had reason to think that a Normal distribution would
  be a good surrogate for some model variables but not others.

  """
  global ASVI_SURROGATE_SUBSTITUTIONS
  if inspect.isclass(condition):
    condition = lambda distribution, cls=condition: isinstance(  # pylint: disable=g-long-lambda
        distribution, cls)
  ASVI_SURROGATE_SUBSTITUTIONS[condition] = substitution_fn

# Default substitutions attempt to express distributions using the most
# flexible available parameterization.
# pylint: disable=g-long-lambda
register_asvi_substitution_rule(
    half_normal.HalfNormal,
    lambda dist: truncated_normal.TruncatedNormal(
        loc=0., scale=dist.scale, low=0., high=dist.scale * 10.))
register_asvi_substitution_rule(
    uniform.Uniform,
    lambda dist: shift.Shift(dist.low)(
        scale_lib.Scale(dist.high - dist.low)(
            beta.Beta(concentration0=tf.ones_like(dist.mean()),
                      concentration1=1.))))
register_asvi_substitution_rule(
    exponential.Exponential,
    lambda dist: gamma.Gamma(concentration=1., rate=dist.rate))
register_asvi_substitution_rule(
    chi2.Chi2,
    lambda dist: gamma.Gamma(concentration=0.5 * dist.df, rate=0.5))
# pylint: enable=g-long-lambda


# TODO(kateslin): Add support for models with prior+likelihood written as
# a single JointDistribution.
def build_asvi_surrogate_posterior(
    prior,
    mean_field=False,
    initial_prior_weight=0.5,
    seed=None,
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
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
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
    surrogate_posterior, variables = _asvi_surrogate_for_distribution(
        dist=prior,
        base_distribution_surrogate_fn=functools.partial(
            _asvi_convex_update_for_base_distribution,
            mean_field=mean_field,
            initial_prior_weight=initial_prior_weight),
        seed=seed)
    surrogate_posterior.also_track = variables
    return surrogate_posterior


def _asvi_surrogate_for_distribution(dist,
                                     base_distribution_surrogate_fn,
                                     sample_shape=None,
                                     variables=None,
                                     seed=None):
  """Recursively creates ASVI surrogates, and creates new variables if needed.

  Args:
    dist: a `tfd.Distribution` instance.
    base_distribution_surrogate_fn: Callable to build a surrogate posterior
      for a 'base' (non-meta and non-joint) distribution, with signature
      `surrogate_posterior, variables = base_distribution_fn(
      dist, sample_shape=None, variables=None, seed=None)`.
    sample_shape: Optional `Tensor` shape of samples drawn from `dist` by
      `tfd.Sample` wrappers. If not `None`, the surrogate's event will include
      independent sample dimensions, i.e., it will have event shape
      `concat([sample_shape, dist.event_shape], axis=0)`.
      Default value: `None`.
    variables: Optional nested structure of `tf.Variable`s returned from a
      previous call to `_asvi_surrogate_for_distribution`. If `None`,
      new variables will be created; otherwise, constructs a surrogate posterior
      backed by the passed-in variables.
      Default value: `None`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
  Returns:
    surrogate_posterior: Instance of `tfd.Distribution` representing a trainable
      surrogate posterior distribution, with the same structure and `name` as
      `dist`.
    variables: Nested structure of `tf.Variable` trainable parameters for the
      surrogate posterior. If `dist` is a base distribution, this is
      a `dict` of `ASVIParameters` instances. If `dist` is a joint
      distribution, this is a `dist.dtype` structure of such `dict`s.
  """
  # Pass args to any nested surrogates.
  build_nested_surrogate = functools.partial(
      _asvi_surrogate_for_distribution,
      base_distribution_surrogate_fn=base_distribution_surrogate_fn,
      sample_shape=sample_shape,
      seed=seed)

  # Apply any substitutions, while attempting to preserve the original name.
  dist = _set_name(_as_substituted_distribution(dist), name=_get_name(dist))

  # Handle wrapper ("meta") distributions.
  if isinstance(dist, markov_chain.MarkovChain):
    return _asvi_surrogate_for_markov_chain(
        dist=dist,
        variables=variables,
        base_distribution_surrogate_fn=base_distribution_surrogate_fn,
        sample_shape=sample_shape,
        seed=seed)
  if isinstance(dist, sample.Sample):
    dist_sample_shape = distribution_util.expand_to_vector(dist.sample_shape)
    nested_surrogate, variables = build_nested_surrogate(  # pylint: disable=redundant-keyword-arg
        dist=dist.distribution,
        variables=variables,
        sample_shape=(
            dist_sample_shape if sample_shape is None
            else ps.concat([sample_shape, dist_sample_shape], axis=0)))
    surrogate_posterior = independent.Independent(
        nested_surrogate,
        reinterpreted_batch_ndims=ps.rank_from_shape(dist_sample_shape),
        name=_get_name(dist))
  # Treat distributions that subclass TransformedDistribution with their own
  # parameters (e.g., Gumbel, Weibull, MultivariateNormal*, etc) as their
  # own type of base distribution, rather than as explicit TDs.
  elif type(dist) == transformed_distribution.TransformedDistribution:  # pylint: disable=unidiomatic-typecheck
    nested_surrogate, variables = build_nested_surrogate(dist.distribution,
                                                         variables=variables)
    surrogate_posterior = transformed_distribution.TransformedDistribution(
        nested_surrogate,
        bijector=dist.bijector,
        name=_get_name(dist))
  elif isinstance(dist, independent.Independent):
    nested_surrogate, variables = build_nested_surrogate(dist.distribution,
                                                         variables=variables)
    surrogate_posterior = independent.Independent(
        nested_surrogate,
        reinterpreted_batch_ndims=dist.reinterpreted_batch_ndims,
        name=_get_name(dist))
  elif hasattr(dist, '_model_coroutine'):
    surrogate_posterior, variables = _asvi_surrogate_for_joint_distribution(
        dist,
        base_distribution_surrogate_fn=base_distribution_surrogate_fn,
        variables=variables,
        seed=seed)
  elif (hasattr(dist, 'distribution') and
        # Transformed dists not handled above are treated as base distributions.
        not isinstance(dist, transformed_distribution.TransformedDistribution)):
    raise ValueError('Meta-distribution `{}` is not yet supported by this '
                     'implementation of ASVI. Contact '
                     '`tfprobability@tensorflow.org` if you need this '
                     'functionality.'.format(type(dist)))
  else:
    surrogate_posterior, variables = base_distribution_surrogate_fn(
        dist=dist, sample_shape=sample_shape, variables=variables, seed=seed)
  return surrogate_posterior, variables


def _asvi_surrogate_for_joint_distribution(
    dist, base_distribution_surrogate_fn, variables=None, seed=None):
  """Builds a structured joint surrogate posterior for a joint model."""

  # Probabilistic program for ASVI surrogate posterior.
  flat_variables = dist._model_flatten(variables) if variables else None  # pylint: disable=protected-access
  prior_coroutine = dist._model_coroutine  # pylint: disable=protected-access

  def posterior_generator(seed=seed):
    prior_gen = prior_coroutine()
    dist = next(prior_gen)
    i = 0
    try:
      while True:
        was_root = isinstance(dist, Root)
        if was_root:
          dist = dist.distribution

        seed, init_seed = samplers.split_seed(seed)
        surrogate_posterior, variables = _asvi_surrogate_for_distribution(
            dist,
            base_distribution_surrogate_fn=base_distribution_surrogate_fn,
            variables=flat_variables[i] if flat_variables else None,
            seed=init_seed)

        if was_root:
          surrogate_posterior = Root(surrogate_posterior)
        # If variables were not given---i.e., we're creating new
        # variables---then yield the new variables along with the surrogate
        # posterior. This assumes an execution context such as
        # `_extract_variables_from_coroutine_model` below that will capture and
        # save the variables.
        value_out = yield (surrogate_posterior if flat_variables
                           else (surrogate_posterior, variables))
        dist = prior_gen.send(value_out)
        i += 1
    except StopIteration:
      pass

  if variables is None:
    # Run the generator to create variables, then call ourselves again
    # to construct the surrogate JD from these variables. Note that we can't
    # just create a JDC from the current `posterior_generator`, because it will
    # try to build new variables on every invocation; the recursive call will
    # define a new `posterior_generator` that knows about the variables we're
    # about to create.
    return _asvi_surrogate_for_joint_distribution(
        dist=dist,
        base_distribution_surrogate_fn=base_distribution_surrogate_fn,
        variables=dist._model_unflatten(  # pylint: disable=protected-access
            _extract_variables_from_coroutine_model(
                posterior_generator, seed=seed)))

  surrogate_posterior = (
      joint_distribution_auto_batched.JointDistributionCoroutineAutoBatched(
          posterior_generator,
          name=_get_name(dist)))

  # Ensure that the surrogate posterior structure matches that of the prior.
  try:
    tf.nest.assert_same_structure(dist.dtype, surrogate_posterior.dtype)
  except TypeError:
    tokenize = lambda jd: jd._model_unflatten(  # pylint: disable=protected-access, g-long-lambda
        range(len(jd._model_flatten(jd.dtype)))  # pylint: disable=protected-access
    )
    surrogate_posterior = restructure.Restructure(
        output_structure=tokenize(dist),
        input_structure=tokenize(surrogate_posterior))(
            surrogate_posterior, name=_get_name(dist))
  return surrogate_posterior, variables


def _asvi_surrogate_for_markov_chain(dist,
                                     base_distribution_surrogate_fn,
                                     sample_shape=None,
                                     variables=None,
                                     seed=None):
  """Builds a structured surrogate posterior for a Markov chain."""
  prior_seed, transition_seed = samplers.split_seed(seed, 2)
  if variables is None:
    prior_variables, transition_variables = None, None
  else:
    prior_variables, transition_variables = variables

  surrogate_prior, prior_variables = _asvi_surrogate_for_distribution(
      dist.initial_state_prior,
      base_distribution_surrogate_fn=base_distribution_surrogate_fn,
      variables=prior_variables,
      seed=prior_seed)

  if transition_variables is None:
    # Construct variables for all chain steps in a single call. These will have
    # an initial dimension of size `num_steps - 1`, which we can gather from
    # as the chain runs.
    all_steps = tf.range(dist.num_steps - 1)
    batch_state = dist.initial_state_prior.sample(dist.num_steps - 1)
    _, transition_variables = _asvi_surrogate_for_distribution(
        dist.transition_fn(all_steps, batch_state),
        base_distribution_surrogate_fn=base_distribution_surrogate_fn,
        variables=None,
        sample_shape=sample_shape,
        seed=transition_seed)

  def surrogate_transition_fn(step, state):
    surrogate_new_dist, _ = _asvi_surrogate_for_distribution(
        dist.transition_fn(step, state),
        base_distribution_surrogate_fn=base_distribution_surrogate_fn,
        variables=tf.nest.map_structure(
            # Gather parameters for this specific step of the chain.
            lambda v: tf.gather(v, step, axis=0), transition_variables),
        sample_shape=sample_shape,
        seed=transition_seed)
    return surrogate_new_dist

  chain_surrogate = markov_chain.MarkovChain(
      initial_state_prior=surrogate_prior,
      transition_fn=surrogate_transition_fn,
      num_steps=dist.num_steps,
      validate_args=dist.validate_args,
      name=_get_name(dist))

  return chain_surrogate, [prior_variables, transition_variables]


# TODO(davmre): consider breaking the mean field case into a separate method.
def _asvi_convex_update_for_base_distribution(dist,
                                              mean_field,
                                              initial_prior_weight,
                                              sample_shape=None,
                                              variables=None,
                                              seed=None):
  """Creates a trainable surrogate for a (non-meta, non-joint) distribution."""
  if variables is None:
    variables = {}

  posterior_batch_shape = dist.batch_shape_tensor()
  if sample_shape is not None:
    posterior_batch_shape = ps.concat([
        posterior_batch_shape,
        distribution_util.expand_to_vector(sample_shape)
    ], axis=0)

  # Create variables backing each parameter, if needed.
  all_parameter_properties = dist.parameter_properties(dtype=dist.dtype)
  for param, prior_value in dist.parameters.items():
    if (param in variables
        or param in (_NON_STATISTICAL_PARAMS + _NON_TRAINABLE_PARAMS)
        or prior_value is None):
      continue

    param_properties = all_parameter_properties[param]
    try:
      bijector = param_properties.default_constraining_bijector_fn()
    except NotImplementedError:
      bijector = identity.Identity()

    param_shape = ps.concat([
        posterior_batch_shape,
        ps.shape(prior_value)[
            ps.rank(prior_value) - param_properties.event_ndims:]
    ], axis=0)

    prior_weight = (None if mean_field  # pylint: disable=g-long-ternary
                    else tfp_util.TransformedVariable(
                        initial_value=tf.fill(
                            dims=param_shape,
                            value=tf.cast(
                                initial_prior_weight,
                                tf.convert_to_tensor(prior_value).dtype)),
                        bijector=sigmoid.Sigmoid(),
                        name='prior_weight/{}/{}'.format(
                            _get_name(dist), param)))

    # Initialize the mean-field parameter as a (constrained) standard
    # normal sample.
    seed, param_seed = samplers.split_seed(seed)
    variables[param] = ASVIParameters(
        prior_weight=prior_weight,
        mean_field_parameter=tfp_util.TransformedVariable(
            initial_value=bijector.forward(
                samplers.normal(
                    shape=bijector.inverse_event_shape(param_shape),
                    seed=param_seed)),
            bijector=bijector,
            name='mean_field_parameter/{}/{}'.format(
                _get_name(dist), param)))

  temp_params_dict = {'name': _get_name(dist)}
  for param, prior_value in dist.parameters.items():
    if param in (_NON_STATISTICAL_PARAMS +
                 _NON_TRAINABLE_PARAMS) or prior_value is None:
      temp_params_dict[param] = prior_value
    else:
      if mean_field:
        temp_params_dict[param] = variables[param].mean_field_parameter
      else:
        temp_params_dict[param] = (
            variables[param].prior_weight * prior_value + (
                (1. - variables[param].prior_weight) *
                variables[param].mean_field_parameter))
  return type(dist)(**temp_params_dict), variables


def _extract_variables_from_coroutine_model(model_fn, seed=None):
  """Extracts variables from a generator that yields (dist, variables) pairs."""
  gen = model_fn()
  try:
    dist, dist_variables = next(gen)
    flat_variables = [dist_variables]
    while True:
      seed, local_seed = samplers.split_seed(seed, n=2)
      sampled_value = (dist.distribution.sample(seed=local_seed)
                       if isinstance(dist, Root)
                       else dist.sample(seed=local_seed))
      dist, dist_variables = gen.send(sampled_value)
      flat_variables.append(dist_variables)
  except StopIteration:
    pass
  return flat_variables


def _set_name(dist, name):
  """Copies a distribution-like object, replacing its name."""
  if hasattr(dist, 'copy'):
    return dist.copy(name=name)
  # Some distribution-like entities such as JointDistributionPinned don't
  # inherit from tfd.Distribution and don't define `self.copy`. We'll try to set
  # the name directly.
  dist = copy.copy(dist)
  dist._name = name  # pylint: disable=protected-access
  return dist


def _get_name(dist):
  """Attempts to get a distribution's short name, excluding the name scope."""
  return getattr(dist, 'parameters', {}).get('name', dist.name)
