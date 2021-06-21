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

import copy
import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import restructure
from tensorflow_probability.python.bijectors import split
from tensorflow_probability.python.distributions import batch_broadcast
from tensorflow_probability.python.distributions import blockwise
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import \
    joint_distribution_auto_batched
from tensorflow_probability.python.distributions import \
    joint_distribution_coroutine
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.experimental.bijectors import \
    build_trainable_highway_flow
from tensorflow_probability.python.internal import samplers

__all__ = [
  'build_cascading_flow_surrogate_posterior'
]

Root = joint_distribution_coroutine.JointDistributionCoroutine.Root


def build_cascading_flow_surrogate_posterior(
    prior,
    num_auxiliary_variables=0,
    initial_prior_weight=0.98,
    num_layers=3,
    seed=None,
    name=None):
  """Builds a structured surrogate posterior with cascading flows.

  Cascading Flows (CF) [1] is a method that automatically construct a
  variational approximation given an input probabilistic program. CF combines
  ASVI [2] with the flexibility of normalizing flows, by transforming the
  conditional distributions of the prior program with HighwayFlow architectures,
  to steer the prior towards the observed data. More details on the HighwayFlow
  architecture can be found in [1] and in the tfp bijector `HighwayFlow`.
  It is possible to add auxiliary variables to the prior program to further
  increase the flexibility of cascading flows, useful especially in the
  cases where the input program has low dimensionality. The auxiliary variables
  are sampled from a global linear flow, to account for statistical dependencies
  among variables, and then transformed with local HighwayFlows together with
  samples form the prior. Note that when using auxiliary variables it is
  necessary to modify the variational lower bound [3].

  Args:
    prior: tfd.JointDistribution instance of the prior.
    num_auxiliary_variables: The number of auxiliary variables to use for each
      variable in the input program. Default value: `0`.
    initial_prior_weight: Optional float value (either static or tensor value)
      on the interval [0, 1]. A larger value creates an initial surrogate
      distribution with more dependence on the prior structure. Default value:
      `0.98`.
    num_layers: Number of layers to use in each Highway Flow architecture. All
    the layers will have `softplus` activation function, apart from the last one
    which will have linear activation. Default value: `3`.
    seed: Python `int` seed for random initialization.
    name: Optional string. Default value: `build_cascading_flow_surrogate_posterior`.

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
    tfp.experimental.vi.build_cascading_flow_surrogate_posterior(prior)
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

  When using auxiliary variables, we need some modifications for loss and
  samples, as samples will return also the global variables and transformed
  auxiliary variables

  ```python
  num_aux_vars=10
  event_len = len(prior.event_shape_tensor())
  target_dist = tfd.Independent(
  tfd.Normal(loc=tf.Variable(tf.random.normal((event_len,num_aux_vars))),
             scale=tfp.util.TransformedVariable(
             tf.random.uniform((event_len,num_aux_vars), minval=0.01, maxval=1.)
             , bijector=tfb.Softplus())), 2)

  def target_log_prob_aux_vars(z_and_eps):
    z = [x[0] for x in z_and_eps[1:]]
    eps = [x[1] for x in z_and_eps[1:]]
    lp_z = target_log_prob_fn(z)
    lp_eps = tf.reshape(tf.reduce_sum(target_dist.log_prob(eps), 0), lp_z.shape)
    return lp_z + lp_eps

  target_log_prob = lambda *values: target_log_prob_aux_vars(values)
  cascading_flow_surrogate_posterior = build_cascading_flow_surrogate_posterior(prior,
                                          num_auxiliary_variables=num_aux_vars)
  trainable_variables = list(cascading_flow_surrogate_posterior.trainable_variables)
  trainable_variables.extend(list(target_dist.trainable_variables))
  cascading_flow_losses = tfp.vi.fit_surrogate_posterior(target_log_prob,
                                        cascading_flow_surrogate_posterior,
                                        optimizer=tf.optimizers.Adam(0.01),
                                        num_steps=8000,
                                        sample_size=50,
                                        trainable_variables=trainable_variables)

  cascading_flow_posterior_samples = cascading_flow_surrogate_posterior.sample(num_samples)
  cascading_flow_posterior_samples = tf.convert_to_tensor(
                                       [s[0] for s in cascading_flow_posterior_samples[1:]])
  ```

  #### References
  [1]: Ambrogioni, Luca, Gianluigi Silvestri, and Marcel van Gerven. "Automatic
  variational inference with cascading flows." arXiv preprint arXiv:2102.04801
  (2021).

  [2]: Ambrogioni, Luca, et al. "Automatic structured variational inference."
  International Conference on Artificial Intelligence and Statistics. PMLR,
  2021.

  [3]: Ranganath, Rajesh, Dustin Tran, and David Blei. "Hierarchical variational
  models." International Conference on Machine Learning. PMLR, 2016.

  """
  with tf.name_scope(name or 'build_cascading_flow_surrogate_posterior'):
    surrogate_posterior, variables = _cascading_flow_surrogate_for_distribution(
      dist=prior,
      base_distribution_surrogate_fn=functools.partial(
        _cascading_flow_update_for_base_distribution,
        initial_prior_weight=initial_prior_weight,
        num_auxiliary_variables=num_auxiliary_variables,
        num_layers=num_layers),
      num_auxiliary_variables=num_auxiliary_variables,
      num_layers=num_layers,
      seed=seed)
    surrogate_posterior.also_track = variables
    return surrogate_posterior


def _cascading_flow_surrogate_for_distribution(dist,
                                   base_distribution_surrogate_fn,
                                   num_auxiliary_variables,
                                   num_layers,
                                   global_auxiliary_variables=None,
                                   variables=None,
                                   seed=None):
  """Recursively creates CF surrogates, and creates new variables if needed.

  Args:
    dist: a `tfd.Distribution` instance.
    base_distribution_surrogate_fn: Callable to build a surrogate posterior
      for a 'base' (non-meta and non-joint) distribution, with signature
      `surrogate_posterior, variables = base_distribution_fn(
      dist, variables=None, seed=None)`.
    num_auxiliary_variables: The number of auxiliary variables to use for each
      variable in the input program.
    num_layers: Number of layers to use in each Highway Flow architecture.
    global_auxiliary_variables: The sampled global auxiliary variables
      (available only if using auxiliary variables). Default value: None.
    variables: Optional nested structure containing `tf.Variable`s returned from a
      previous call to `_cascading_flow_surrogate_for_distribution`. If `None`,
      new variables will be created; otherwise, constructs a surrogate posterior
      backed by the passed-in variables.
      Default value: `None`.
    seed: Python `int` seed for random initialization.
  Returns:
    surrogate_posterior: Instance of `tfd.Distribution` representing a trainable
      surrogate posterior distribution, with the same structure and `name` as
      `dist`, and with addition of global and local auxiliary variables if
      `num_auxiliary_variables > 0`.
    variables: Nested structure containing `tf.Variable` trainable parameters for the
      surrogate posterior. If `dist` is a base distribution, this is
      a `tfb.Chain` of bijectors containing HighwayFlow blocks and `Reshape`
      bijectors. If `dist` is a joint distribution, this is a `dist.dtype`
      structure of such `tfb.Chain`s.
  """

  if hasattr(dist, '_model_coroutine'):
    surrogate_posterior, variables = _cascading_flow_surrogate_for_joint_distribution(
      dist,
      base_distribution_surrogate_fn=base_distribution_surrogate_fn,
      variables=variables,
      num_auxiliary_variables=num_auxiliary_variables,
      num_layers=num_layers,
      global_auxiliary_variables=global_auxiliary_variables,
      seed=seed)
  else:
    surrogate_posterior, variables = base_distribution_surrogate_fn(
      dist=dist, variables=variables,
      global_auxiliary_variables=global_auxiliary_variables,
      num_layers=num_layers,
      seed=seed)
  return surrogate_posterior, variables


def _build_highway_flow_block(num_layers, width,
                              residual_fraction_initial_value, gate_first_n,
                              seed):
  bijectors = []

  for _ in range(0, num_layers - 1):
    bijectors.append(
      build_trainable_highway_flow(width,
                                   residual_fraction_initial_value=residual_fraction_initial_value,
                                   activation_fn=tf.nn.softplus,
                                   gate_first_n=gate_first_n, seed=seed))
  bijectors.append(
    build_trainable_highway_flow(width,
                                 residual_fraction_initial_value=residual_fraction_initial_value,
                                 activation_fn=None,
                                 gate_first_n=gate_first_n, seed=seed))

  return bijectors


def _cascading_flow_surrogate_for_joint_distribution(
    dist, base_distribution_surrogate_fn, variables,
    num_auxiliary_variables, num_layers, global_auxiliary_variables,
    seed=None):
  """Builds a structured joint surrogate posterior for a joint model."""

  # Probabilistic program for CF surrogate posterior.
  flat_variables = dist._model_flatten(
    variables) if variables else None  # pylint: disable=protected-access
  prior_coroutine = dist._model_coroutine  # pylint: disable=protected-access
  prior_batch_shape = dist.batch_shape_tensor()
  if tf.nest.is_nested(prior_batch_shape):
    prior_batch_shape = functools.reduce(tf.broadcast_static_shape,
                                         dist._model_flatten(prior_batch_shape))

  def posterior_generator(seed=seed):
    prior_gen = prior_coroutine()
    dist = next(prior_gen)

    if num_auxiliary_variables > 0:
      i = 1

      if flat_variables:
        variables = flat_variables[0]

      else:

        bijectors = _build_highway_flow_block(
          num_layers,
          width=num_auxiliary_variables,
          residual_fraction_initial_value=0,  # not used
          gate_first_n=0, seed=seed)
        variables = chain.Chain(bijectors=list(reversed(bijectors)))

      eps = transformed_distribution.TransformedDistribution(
        distribution=batch_broadcast.BatchBroadcast(sample.Sample(normal.Normal(0., 1.),
                                   num_auxiliary_variables), prior_batch_shape),
        bijector=variables)

      eps = Root(eps)

      value_out = yield (eps if flat_variables
                         else (eps, variables))

      global_auxiliary_variables = value_out

    else:
      global_auxiliary_variables = None
      i = 0

    try:
      while True:
        was_root = isinstance(dist, Root)
        if was_root:
          dist = dist.distribution

        seed, init_seed = samplers.split_seed(seed)
        surrogate_posterior, variables = _cascading_flow_surrogate_for_distribution(
          dist,
          base_distribution_surrogate_fn=base_distribution_surrogate_fn,
          num_auxiliary_variables=num_auxiliary_variables,
          num_layers=num_layers,
          variables=flat_variables[i] if flat_variables else None,
          global_auxiliary_variables=global_auxiliary_variables,
          seed=init_seed)

        if was_root and num_auxiliary_variables == 0:
          surrogate_posterior = Root(surrogate_posterior)
        # If variables were not given---i.e., we're creating new
        # variables---then yield the new variables along with the surrogate
        # posterior. This assumes an execution context such as
        # `_extract_variables_from_coroutine_model` below that will capture and
        # save the variables.
        value_out = yield (surrogate_posterior if flat_variables
                           else (surrogate_posterior, variables))

        # When using auxiliary variables, value out is a list containing
        # [latent_variables, auxiliary_variables].
        if num_auxiliary_variables>0:
          dist = prior_gen.send(value_out[0])
        else:
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
    return _cascading_flow_surrogate_for_joint_distribution(
      dist=dist,
      base_distribution_surrogate_fn=base_distribution_surrogate_fn,
      num_auxiliary_variables=num_auxiliary_variables,
      num_layers=num_layers,
      global_auxiliary_variables=global_auxiliary_variables,
      variables=dist._model_unflatten(
        # pylint: disable=protected-access
        _extract_variables_from_coroutine_model(
          posterior_generator, seed=seed)))

  surrogate_posterior = joint_distribution_auto_batched.JointDistributionCoroutineAutoBatched(
    posterior_generator,
    use_vectorized_map=dist.use_vectorized_map,
    name=_get_name(dist))

  tokenize = lambda jd: tf.nest.pack_sequence_as(
    jd.dtype,
    range(len(tf.nest.flatten(jd.dtype))))

  dist_tokens = tokenize(dist)

  if num_auxiliary_variables == 0:
    try:
      tf.nest.assert_same_structure(dist.dtype, surrogate_posterior.dtype)
    except TypeError:
      surrogate_posterior = restructure.Restructure(
        output_structure=dist_tokens,
        input_structure=tokenize(surrogate_posterior))(
        surrogate_posterior, name=_get_name(dist))

  else:
    surrogate_posterior = restructure.Restructure(
      output_structure=(
        tf.nest.map_structure(lambda k: 2 * k + 1, dist_tokens),
        [0] + [2 * k + 2 for k in tf.nest.flatten(dist_tokens)]),
      input_structure=tokenize(surrogate_posterior))(
      surrogate_posterior, name=_get_name(dist))

  return surrogate_posterior, variables


def _cascading_flow_update_for_base_distribution(dist,
                                            initial_prior_weight,
                                            num_auxiliary_variables,
                                            num_layers,
                                            global_auxiliary_variables,
                                            variables,
                                            seed=None):
  """Creates a trainable surrogate for a (non-meta, non-joint) distribution."""
  event_shape = dist.event_shape_tensor()
  flat_event_shape = tf.nest.flatten(event_shape)
  flat_event_size = tf.nest.map_structure(tf.reduce_prod, flat_event_shape)
  ndims = int(tf.reduce_sum(flat_event_size))
  constraining_bijector = dist.experimental_default_event_space_bijector()
  flatten_bijector = reshape.Reshape(
    event_shape_out=[-1],
    event_shape_in=dist.event_shape_tensor())

  constraining_and_flattening_bijector = chain.Chain([flatten_bijector, constraining_bijector])
  processed_dist = transformed_distribution.TransformedDistribution(distribution=dist,
                                                                    bijector=constraining_and_flattening_bijector)
  if variables is None:

    bijectors = []

    bijectors.extend(
      _build_highway_flow_block(
        num_layers,
        width=tf.reduce_prod(
          ndims + num_auxiliary_variables),
        residual_fraction_initial_value=initial_prior_weight,
        gate_first_n=ndims, seed=seed))

    variables = chain.Chain(bijectors=list(reversed(bijectors)))

  if num_auxiliary_variables > 0:
    batch_shape = global_auxiliary_variables.shape[:-1] if len(
      global_auxiliary_variables.shape) > 1 else []
    cascading_flows = split.Split(
      [-1, num_auxiliary_variables])(
      transformed_distribution.TransformedDistribution(
        distribution=blockwise.Blockwise([
          batch_broadcast.BatchBroadcast(processed_dist, to_shape=batch_shape),
          independent.Independent(
            deterministic.Deterministic(global_auxiliary_variables, ),
            reinterpreted_batch_ndims=1)]),
        bijector=variables))

    cascading_flows = joint_map.JointMap(
      [invert.Invert(constraining_and_flattening_bijector), identity.Identity()])(cascading_flows)

  else:
    cascading_flows = transformed_distribution.TransformedDistribution(
      distribution=processed_dist,
      bijector=variables)

    cascading_flows = invert.Invert(constraining_and_flattening_bijector)(cascading_flows)

  return cascading_flows, variables


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
      dist, dist_variables = gen.send(
        sampled_value)  # tf.concat(sampled_value, axis=0)
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
