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
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import trainable_state_util


__all__ = [
    'ASVI_DEFAULT_PRIOR_SUBSTITUTION_RULES',
    'ASVI_DEFAULT_SURROGATE_RULES',
    'build_asvi_surrogate_posterior',
    'build_asvi_surrogate_posterior_stateless'
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


# Transformations applied to distributions in the prior before creating a
# surrogate. These generally attempt to induce a richer surrogate family by
# reparameterizing distributions in more expressive forms.
# pylint: disable=g-long-lambda
ASVI_DEFAULT_PRIOR_SUBSTITUTION_RULES = (
    (half_normal.HalfNormal,
     lambda dist: truncated_normal.TruncatedNormal(
         loc=0., scale=dist.scale, low=0., high=dist.scale * 10.)),
    (uniform.Uniform,
     lambda dist: shift.Shift(dist.low)(scale_lib.Scale(dist.high - dist.low)(
         beta.Beta(concentration0=tf.ones_like(dist.mean()),
                   concentration1=1.)))),
    (exponential.Exponential,
     lambda dist: gamma.Gamma(concentration=1., rate=dist.rate)),
    (chi2.Chi2,
     lambda dist: gamma.Gamma(concentration=0.5 * dist.df, rate=0.5))
)
# pylint: enable=g-long-lambda


def _satisfies_condition(dist, condition):
  """Checks the condition for a surrogate or substitution rule."""
  if inspect.isclass(condition):
    return isinstance(dist, condition)
  return condition(dist)


def _as_substituted_distribution(dist, prior_substitution_rules):
  """Applies all substitution rules that match a distribution."""
  for condition, substitution_fn in prior_substitution_rules:
    if _satisfies_condition(dist, condition):
      dist = substitution_fn(dist)
  return dist


# Default rules are registered using the `_asvi_surrogate_rule` decorator.
_ASVI_DEFAULT_SURROGATE_RULES_MUTABLE = []


def _asvi_surrogate_rule(condition, pass_sample_shape=False):
  """Registers a decorated function as a surrogate rule."""
  def wrap(f):
    global _ASVI_DEFAULT_SURROGATE_RULES_MUTABLE
    if pass_sample_shape:
      _ASVI_DEFAULT_SURROGATE_RULES_MUTABLE.append((condition, f))
    else:
      _ASVI_DEFAULT_SURROGATE_RULES_MUTABLE.append(
          (condition, lambda *a, sample_shape=None, **kw: f(*a, **kw)))  # pylint: disable=unnecessary-lambda
    return f
  return wrap


def _asvi_surrogate_for_distribution(dist,
                                     base_distribution_surrogate_fn,
                                     prior_substitution_rules,
                                     surrogate_rules,
                                     sample_shape=None):
  """Recursively creates ASVI surrogates, and creates new variables if needed.

  Args:
    dist: a `tfd.Distribution` instance.
    base_distribution_surrogate_fn: Callable to build a surrogate posterior
      for a 'base' (non-meta and non-joint) distribution, with signature
      `surrogate_posterior, variables = base_distribution_fn(
      dist, sample_shape=None, variables=None, seed=None)`.
    prior_substitution_rules: Iterable of substitution rules applied to the
      prior before constructing a surrogate. Each rule is a `(condition,
      substitution_fn)` tuple; these are checked in order and *all* applicable
      substitutions are made. The `condition` may be either a class or a
      callable returning a boolean (for example, `tfd.Normal` or, equivalently,
      `lambda dist: isinstance(dist, tfd.Normal)`). The `substitution_fn` should
      have signature `new_dist = substitution_fn(dist)`.
    surrogate_rules: Iterable of special-purpose rules to create surrogates
      for specific distribution types. Each rule is a `(condition,
      surrogate_fn)` tuple; these are checked in order and the first applicable
      `surrogate_fn` is used. The `condition` may be either a class or a
      callable returning a boolean (for example, `tfd.Normal` or, equivalently,
      `lambda dist: isinstance(dist, tfd.Normal)`). The `surrogate_fn` should
      have signature `surrogate_posterior, variables = surrogate_fn(dist,
      build_nested_surrogate_fn, sample_shape=None, variables=None, seed=None)`.
    sample_shape: Optional `Tensor` shape of samples drawn from `dist` by
      `tfd.Sample` wrappers. If not `None`, the surrogate's event will include
      independent sample dimensions, i.e., it will have event shape
      `concat([sample_shape, dist.event_shape], axis=0)`.
      Default value: `None`.
  Yields:
    *parameters: sequence of `trainable_state_util.Parameter` namedtuples.
      These are intended to be consumed by
      `trainable_state_util.as_stateful_builder` and
      `trainable_state_util.as_stateless_builder` to define stateful and
      stateless variants respectively.
  """
  dist_name = _get_name(dist)   # Attempt to preserve the original name.
  dist = _as_substituted_distribution(dist, prior_substitution_rules)
  # Apply the first surrogate rule that matches this distribution.
  surrogate_posterior = None
  for condition, surrogate_fn in surrogate_rules:
    if _satisfies_condition(dist, condition):
      # The surrogate fn may be a generator (internal interface) or a stateless
      # trainable builder returning `(init_fn, apply_fn)` (external interface).
      maybe_gen = surrogate_fn(
          dist,
          build_nested_surrogate=functools.partial(
              _asvi_surrogate_for_distribution,
              base_distribution_surrogate_fn=base_distribution_surrogate_fn,
              prior_substitution_rules=prior_substitution_rules,
              surrogate_rules=surrogate_rules,
              sample_shape=sample_shape),
          sample_shape=sample_shape)
      if inspect.isgenerator(maybe_gen):
        surrogate_posterior = yield from maybe_gen
      else:
        init_fn, apply_fn = maybe_gen
        params = yield trainable_state_util.Parameter(init_fn)
        surrogate_posterior = apply_fn(params)
      break
  if surrogate_posterior is None:
    if (hasattr(dist, 'distribution') and
        # Transformed dists not handled above are treated as base distributions.
        not isinstance(dist, transformed_distribution.TransformedDistribution)):
      raise ValueError(
          'None of the provided substitution rules matched meta-distribution: '
          '`{}`.'.format(dist))
    else:
      surrogate_posterior = yield from base_distribution_surrogate_fn(
          dist=dist, sample_shape=sample_shape)
  return _set_name(surrogate_posterior, dist_name)


# Check only for explicit TransformedDistributions, as opposed to using
# `isinstance`, to avoid sweeping up subclasses that have their own parameters
# (e.g., Gumbel, Weibull, MultivariateNormal*, etc), which in general should be
# handled as 'base' distribution types.
@_asvi_surrogate_rule(
    lambda dist: type(dist) == transformed_distribution.TransformedDistribution)  # pylint: disable=unidiomatic-typecheck
def _asvi_surrogate_for_transformed_distribution(dist, build_nested_surrogate):
  """Builds the surrogate for a `tfd.TransformedDistribution`."""
  nested_surrogate = yield from build_nested_surrogate(dist.distribution)
  return transformed_distribution.TransformedDistribution(
      nested_surrogate,
      bijector=dist.bijector)


@_asvi_surrogate_rule(sample.Sample, pass_sample_shape=True)
def _asvi_surrogate_for_sample(dist, build_nested_surrogate,
                               sample_shape=None):
  """Builds the surrogate for a `tfd.Sample`-wrapped distribution."""
  dist_sample_shape = distribution_util.expand_to_vector(dist.sample_shape)
  nested_surrogate = yield from build_nested_surrogate(
      dist=dist.distribution,
      sample_shape=(
          dist_sample_shape if sample_shape is None
          else ps.concat([sample_shape, dist_sample_shape], axis=0)))
  return independent.Independent(
      nested_surrogate,
      reinterpreted_batch_ndims=ps.rank_from_shape(dist_sample_shape))


@_asvi_surrogate_rule(independent.Independent)
def _asvi_surrogate_for_independent(dist, build_nested_surrogate):
  """Builds the surrogate for a `tfd.Independent`-wrapped distribution."""
  nested_surrogate = yield from build_nested_surrogate(dist.distribution)
  return independent.Independent(
      nested_surrogate,
      reinterpreted_batch_ndims=dist.reinterpreted_batch_ndims)


@_asvi_surrogate_rule(lambda dist: hasattr(dist, '_model_coroutine'))
def _asvi_surrogate_for_joint_distribution(dist, build_nested_surrogate):
  """Builds a structured joint surrogate posterior for a joint model."""

  # Probabilistic program for ASVI surrogate posterior.
  prior_coroutine = dist._model_coroutine  # pylint: disable=protected-access

  def posterior_generator():
    # Run the prior model coroutine and create a surrogate for each
    # distribution in turn. This coroutine yields *both* the surrogate
    # parameters and the surrogate distributions themselves. It can therefore
    # run in a `trainable_state_util` context to define parameters (using
    # the `_get_coroutine_parameters` wrapper to hide the yielded distributions)
    # *and* as a JDCoroutine model (using the `_set_coroutine_parameters`
    # wrapper to hide the yielded parameters).
    prior_gen = prior_coroutine()
    dist = next(prior_gen)
    i = 0
    try:
      while True:
        surrogate_posterior = yield from build_nested_surrogate(
            dist.distribution if isinstance(dist, Root) else dist)
        if isinstance(dist, Root):
          surrogate_posterior = Root(surrogate_posterior)
        value_out = yield surrogate_posterior
        dist = prior_gen.send(value_out)
        i += 1
    except StopIteration:
      pass

  parameters = yield from _get_coroutine_parameters(posterior_generator,
                                                    seed=samplers.zeros_seed())
  surrogate_posterior = (
      joint_distribution_auto_batched.JointDistributionCoroutineAutoBatched(
          _set_coroutine_parameters(posterior_generator, parameters)))

  # Ensure that the surrogate posterior structure matches that of the prior.
  try:
    tf.nest.assert_same_structure(dist.dtype, surrogate_posterior.dtype)
  except TypeError:
    tokenize = lambda jd: jd._model_unflatten(  # pylint: disable=protected-access, g-long-lambda
        range(len(jd._model_flatten(jd.dtype)))  # pylint: disable=protected-access
    )
    surrogate_posterior = restructure.Restructure(
        output_structure=tokenize(dist),
        input_structure=tokenize(surrogate_posterior))(surrogate_posterior)
  return surrogate_posterior


@_asvi_surrogate_rule(markov_chain.MarkovChain)
def _asvi_surrogate_for_markov_chain(dist, build_nested_surrogate):
  """Builds a structured surrogate posterior for a Markov chain."""
  surrogate_prior = yield from build_nested_surrogate(dist.initial_state_prior)

  (transition_all_steps_init_fn, _) = trainable_state_util.as_stateless_builder(
      lambda: build_nested_surrogate(  # pylint: disable=g-long-lambda
          dist.transition_fn(
              tf.range(dist.num_steps - 1),
              dist.initial_state_prior.sample(dist.num_steps - 1,
                                              seed=samplers.zeros_seed()))))()
  transition_params = yield trainable_state_util.Parameter(
      transition_all_steps_init_fn,
      name='markov_chain_transition_params')

  build_transition_one_step = trainable_state_util.as_stateless_builder(
      lambda step, state: build_nested_surrogate(  # pylint: disable=g-long-lambda
          dist.transition_fn(step, state)))
  def surrogate_transition_fn(step, state):
    _, one_step_apply_fn = build_transition_one_step(step, state)
    return one_step_apply_fn(tf.nest.map_structure(
        # Gather parameters for this specific step of the chain.
        lambda v: tf.gather(v, step, axis=0), transition_params))

  return markov_chain.MarkovChain(initial_state_prior=surrogate_prior,
                                  transition_fn=surrogate_transition_fn,
                                  num_steps=dist.num_steps,
                                  validate_args=dist.validate_args)


# All surrogate registrations must occur above this line.
ASVI_DEFAULT_SURROGATE_RULES = tuple(_ASVI_DEFAULT_SURROGATE_RULES_MUTABLE)


def _asvi_convex_update_for_base_distribution(dist,
                                              mean_field=False,
                                              initial_prior_weight=0.5,
                                              sample_shape=None):
  """Creates a trainable surrogate for a (non-meta, non-joint) distribution."""
  posterior_batch_shape = dist.batch_shape_tensor()
  if sample_shape is not None:
    posterior_batch_shape = ps.concat([
        posterior_batch_shape,
        distribution_util.expand_to_vector(sample_shape)
    ], axis=0)

  temp_params_dict = {'name': _get_name(dist)}
  all_parameter_properties = dist.parameter_properties(dtype=dist.dtype)
  for param, prior_value in dist.parameters.items():
    if (param in (_NON_STATISTICAL_PARAMS + _NON_TRAINABLE_PARAMS)
        or prior_value is None):
      temp_params_dict[param] = prior_value
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
    # We need to pick a dtype that is compatible with the other parameters that
    # this parameter interacts with. We don't have direct access to this, so we
    # do some heuristics here.
    converted_prior_value = getattr(dist, param, None)
    if converted_prior_value is None:
      converted_prior_value = getattr(dist, '_' + param, None)

    if converted_prior_value is not None:
      # This is the safest option, if we can grab its converted value, then that
      # value is guaranteed to be good. Small chance that the implementor added
      # a property/attribute with the same name, but unrelated semantics...
      param_dtype = converted_prior_value.dtype
    elif bijector.dtype is not None:
      # For many bijectors, parameter_properties is implemented to specify a
      # good dtype value.
      param_dtype = bijector.dtype
    elif dtype_util.is_floating(dist.dtype):
      # If the distribution returns floats, all floats are probably of the same
      # type. This doesn't work for discrete distributions.
      param_dtype = dist.dtype
    else:
      # The greedy local conversion is unlikely to work, but we've run out of
      # places to look.
      param_dtype = tf.convert_to_tensor(prior_value).dtype

    # Initialize the mean-field parameter as a (constrained) standard
    # normal sample.
    # pylint: disable=cell-var-from-loop
    # Safe because the state utils guarantee to either call `init_fn`
    # immediately upon yielding, or not at all.
    mean_field_parameter = yield trainable_state_util.Parameter(
        init_fn=lambda seed: (  # pylint: disable=g-long-lambda
            bijector.forward(
                samplers.normal(
                    shape=bijector.inverse_event_shape(param_shape),
                    dtype=bijector.inverse_dtype(param_dtype),
                    seed=seed))),
        name='mean_field_parameter_{}_{}'.format(_get_name(dist), param),
        constraining_bijector=bijector)
    if mean_field:
      temp_params_dict[param] = mean_field_parameter
    else:
      prior_weight = yield trainable_state_util.Parameter(
          init_fn=lambda: tf.fill(  # pylint: disable=g-long-lambda
              dims=param_shape,
              value=tf.cast(initial_prior_weight,
                            param_dtype)),
          name='prior_weight_{}_{}'.format(_get_name(dist), param),
          constraining_bijector=sigmoid.Sigmoid())
      temp_params_dict[param] = prior_weight * prior_value + (
          (1. - prior_weight) * mean_field_parameter)
  # pylint: enable=cell-var-from-loop

  return type(dist)(**temp_params_dict)


def _build_asvi_surrogate_posterior(prior,
                                    mean_field=False,
                                    initial_prior_weight=0.5,
                                    prior_substitution_rules=(
                                        ASVI_DEFAULT_PRIOR_SUBSTITUTION_RULES),
                                    surrogate_rules=(
                                        ASVI_DEFAULT_SURROGATE_RULES),
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
    prior_substitution_rules: Iterable of substitution rules applied to the
      prior before constructing a surrogate. Each rule is a `(condition,
      substitution_fn)` tuple; these are checked in order and *all* applicable
      substitutions are made. The `condition` may be either a class or a
      callable returning a boolean (for example, `tfd.Normal` or, equivalently,
      `lambda dist: isinstance(dist, tfd.Normal)`). The `substitution_fn` should
      have signature `new_dist = substitution_fn(dist)`.
    surrogate_rules: Iterable of special-purpose rules to create surrogates
      for specific distribution types. Each rule is a `(condition,
      surrogate_fn)` tuple; these are checked in order and the first applicable
      `surrogate_fn` is used. The `condition` may be either a class or a
      callable returning a boolean (for example, `tfd.Normal` or, equivalently,
      `lambda dist: isinstance(dist, tfd.Normal)`). The `surrogate_fn` should
      have signature `init_fn, apply_fn = surrogate_fn(dist,
      build_nested_surrogate_fn, sample_shape=None)`.
    name: Optional string. Default value: `build_asvi_surrogate_posterior`.
  Yields:
    *parameters: sequence of `trainable_state_util.Parameter` namedtuples.
      These are intended to be consumed by
      `trainable_state_util.as_stateful_builder` and
      `trainable_state_util.as_stateless_builder` to define stateful and
      stateless variants respectively.

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
    optimizer=tf_keras.optimizers.Adam(0.1),
    sample_size=10)

  # After optimization, samples from the surrogate will approximate
  # samples from the true posterior.
  samples = surrogate_posterior.sample(100)
  posterior_mean = [tf.reduce_mean(x) for x in samples]
  posterior_std = [tf.math.reduce_std(x) for x in samples]
  ```

  #### References

  [1]: Luca Ambrogioni, Kate Lin, Emily Fertig, Sharad Vikram, Max Hinne,
        Dave Moore, Marcel van Gerven. Automatic structured variational
        inference. _arXiv preprint arXiv:2002.00643_, 2020
        https://arxiv.org/abs/2002.00643

  """
  with tf.name_scope(name or 'build_asvi_surrogate_posterior'):
    surrogate_posterior = yield from _asvi_surrogate_for_distribution(
        dist=prior,
        base_distribution_surrogate_fn=functools.partial(
            _asvi_convex_update_for_base_distribution,
            mean_field=mean_field,
            initial_prior_weight=initial_prior_weight),
        prior_substitution_rules=prior_substitution_rules,
        surrogate_rules=surrogate_rules)
    return surrogate_posterior


build_asvi_surrogate_posterior = trainable_state_util.as_stateful_builder(
    _build_asvi_surrogate_posterior)
# TODO(davmre): replace stateful example code in the stateless docstring.
build_asvi_surrogate_posterior_stateless = (
    trainable_state_util.as_stateless_builder(_build_asvi_surrogate_posterior))


def _get_coroutine_parameters(jdc_model, seed):
  """Runs a coroutine and intercepts yielded dists, yielding params only."""
  gen = jdc_model()
  to_send = None
  raw_parameters = []
  try:
    while True:
      val = gen.send(to_send)
      if isinstance(val, trainable_state_util.Parameter):
        to_send = yield val
        raw_parameters.append(to_send)
      else:  # Random variable.
        seed, local_seed = samplers.split_seed(seed, n=2)
        to_send = (val.distribution.sample(seed=local_seed)
                   if isinstance(val, Root)
                   else val.sample(seed=local_seed))
  except StopIteration:
    pass
  return raw_parameters


def _set_coroutine_parameters(jdc_model, raw_parameters):
  """Wraps a coroutine to intercept yielded params, yielding dists only."""
  def handled_coroutine():
    gen = jdc_model()
    to_send = None
    parameter_idx = 0
    try:
      while True:
        val = gen.send(to_send)
        if isinstance(val, trainable_state_util.Parameter):
          to_send = raw_parameters[parameter_idx]
          parameter_idx += 1
        else:
          to_send = yield val
    except StopIteration:
      pass

  return handled_coroutine


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
