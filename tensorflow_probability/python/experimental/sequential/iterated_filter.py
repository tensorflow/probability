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
"""Parameter estimation by iterated filtering."""

import collections
import contextlib
import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_named
from tensorflow_probability.python.distributions import joint_distribution_sequential
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution

from tensorflow_probability.python.experimental.mcmc import infer_trajectories

from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization

from tensorflow_probability.python.util import SeedStream


__all__ = [
    'geometric_cooling_schedule',
    'IteratedFilter'
]


# Utility to avoid breakage when passed-in structures are mutated externally.
_copy_structure = lambda struct: tf.nest.map_structure(lambda x: x, struct)


ParametersAndState = collections.namedtuple('ParametersAndState',
                                            ['unconstrained_parameters',
                                             'state'])


def geometric_cooling_schedule(cooling_fraction_per_k_iterations, k=1.):
  """Defines a cooling schedule following a geometric sequence.

  This returns a function `f` such that

  ```python
  f(iteration) = cooling_fraction_per_k_iterations**(iteration / k)
  ```

  Args:
    cooling_fraction_per_k_iterations: float `Tensor` ratio by which the
      original value should be scaled once `k` iterations have been completed.
    k: int `Tensor` number of iterations used to define the schedule.
  Returns:
    f: Python `callable` representing the cooling schedule.
  """
  cooling_fraction_per_k_iterations = tf.convert_to_tensor(
      cooling_fraction_per_k_iterations,
      dtype_hint=tf.float32,
      name='cooling_fraction_per_k_iterations')
  dtype = cooling_fraction_per_k_iterations.dtype
  k = tf.cast(k, dtype=dtype, name='k')

  def f(iteration):
    iteration = tf.cast(iteration, dtype=dtype, name='iteration')
    return cooling_fraction_per_k_iterations ** (iteration / k)
  return f


class DeterministicEmpirical(distribution.Distribution):
  """Dummy 'proposal' distribution that just returns samples we pass in."""

  def __init__(self, values_with_sample_dim, batch_ndims=0, validate_args=False,
               name=None):
    """Initializes an empirical distribution with a list of samples.

    Args:
      values_with_sample_dim: nested structure of `Tensor`s, each of shape
        prefixed by `[num_samples, B1, ..., Bn]`, where `num_samples` as well as
        `B1, ..., Bn` are batch dimensions shared across all `Tensor`s.
      batch_ndims: optional scalar int `Tensor`, or structure matching
        `values_with_sample_dim` of scalar int `Tensor`s, specifying the number
        of batch dimensions. Used to determine the batch and event shapes of the
        distribution.
        Default value: `0`.
      validate_args: Python `bool` indicating whether to perform runtime checks
        that may have performance cost.
        Default value: `False`.
      name: Python `str` name for ops created by this distribution.
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'DeterministicEmpirical') as name:

      # Ensure we don't break if the passed-in structures are externally
      # mutated.
      values_with_sample_dim = _copy_structure(values_with_sample_dim)
      batch_ndims = _copy_structure(batch_ndims)

      # Prevent tf.Module from wrapping passed-in values, because the
      # wrapper breaks JointDistributionNamed (and maybe other JDs). Instead, we
      # save a separate ref to the input that is used only by tf.Module
      # tracking.
      self._values_for_tracking = values_with_sample_dim
      self._values_with_sample_dim = self._no_dependency(values_with_sample_dim)

      if not tf.nest.is_nested(batch_ndims):
        batch_ndims = tf.nest.map_structure(
            lambda _: batch_ndims, values_with_sample_dim)
      self._batch_ndims = batch_ndims

      self._max_num_samples = prefer_static.reduce_min(
          [prefer_static.size0(x)
           for x in tf.nest.flatten(values_with_sample_dim)])

      super(DeterministicEmpirical, self).__init__(
          dtype=tf.nest.map_structure(
              lambda x: x.dtype, self.values_with_sample_dim),
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=True,
          name=name)
      self._parameters = self._no_dependency(parameters)

  @property
  def batch_ndims(self):
    return _copy_structure(self._batch_ndims)

  @property
  def max_num_samples(self):
    return self._max_num_samples

  @property
  def values_with_sample_dim(self):
    return _copy_structure(self._values_with_sample_dim)

  def _event_shape(self):
    return tf.nest.map_structure(
        lambda x, nd: x.shape[1 + nd:],
        self.values_with_sample_dim,
        self.batch_ndims)

  def _event_shape_tensor(self):
    return tf.nest.map_structure(
        lambda x, nd: tf.shape(x)[1 + nd:],
        self.values_with_sample_dim,
        self.batch_ndims)

  def _batch_shape(self):
    return tf.nest.map_structure(
        lambda x, nd: x.shape[1 : 1 + nd],
        self.values_with_sample_dim,
        self.batch_ndims)

  def _batch_shape_tensor(self):
    return tf.nest.map_structure(
        lambda x, nd: tf.shape(x)[1 : 1 + nd],
        self.values_with_sample_dim,
        self.batch_ndims)

  # TODO(b/152797117): Override _sample_n, once it supports joint distributions.
  def sample(self, sample_shape=(), seed=None, name=None):
    with tf.name_scope(name or 'sample'):
      # Grab the required number of values from the provided tensors.
      sample_shape = dist_util.expand_to_vector(sample_shape)
      n = tf.cast(tf.reduce_prod(sample_shape), dtype=tf.int32)

      # Check that we're not trying to draw too many samples.
      assertions = []
      will_overflow_ = tf.get_static_value(n > self.max_num_samples)
      if will_overflow_:
        raise ValueError('Trying to draw {} samples from a '
                         '`DeterministicEmpirical` instance for which only {} '
                         'samples were provided.'.format(
                             tf.get_static_value(n),
                             tf.get_static_value(self.max_num_samples)))
      elif (will_overflow_ is None  # Couldn't determine statically.
            and self.validate_args):
        assertions.append(
            tf.debugging.assert_less_equal(
                n, self.max_num_samples, message='Number of samples to draw '
                'from a `DeterministicEmpirical` instance must not exceed the '
                'number provided at construction.'))

      # Extract the appropriate number of sampled values.
      with tf.control_dependencies(assertions):
        sampled = tf.nest.map_structure(
            lambda x: x[:n, ...], self.values_with_sample_dim)

      # Reshape the values to the appropriate sample shape.
      return tf.nest.map_structure(
          lambda x: tf.reshape(x,  # pylint: disable=g-long-lambda
                               tf.concat([tf.cast(sample_shape, tf.int32),
                                          tf.cast(tf.shape(x)[1:], tf.int32)],
                                         axis=0)),
          sampled)

  def _prob(self, x):
    flat_values = tf.nest.flatten(self.values_with_sample_dim)
    return tf.cast(
        tf.reduce_all([
            tf.equal(a, b[:prefer_static.size0(a)])
            for (a, b) in zip(tf.nest.flatten(x), flat_values)]),
        dtype=flat_values[0].dtype)


def _maybe_build_joint_distribution(structure_of_distributions):
  """Turns a (potentially nested) structure of dists into a single dist."""
  # Base case: if we already have a Distribution, return it.
  if dist_util.is_distribution_instance(structure_of_distributions):
    return structure_of_distributions

  # Otherwise, recursively convert all interior nested structures into JDs.
  outer_structure = tf.nest.map_structure(
      _maybe_build_joint_distribution,
      structure_of_distributions)
  if (hasattr(outer_structure, '_asdict') or
      isinstance(outer_structure, collections.Mapping)):
    return joint_distribution_named.JointDistributionNamed(outer_structure)
  else:
    return joint_distribution_sequential.JointDistributionSequential(
        outer_structure)


def augment_transition_fn_with_parameters(parameter_prior,
                                          parameterized_transition_fn,
                                          parameter_constraining_bijector):
  """Wraps a transition fn on states to act on `ParametersAndState` tuples."""

  def params_and_state_transition_fn(step,
                                     params_and_state,
                                     perturbation_scale,
                                     **kwargs):
    """Transition function operating on a `ParamsAndState` namedtuple."""
    # Extract the state, to pass through to the observation fn.
    unconstrained_params, state = params_and_state
    if 'state_history' in kwargs:
      kwargs['state_history'] = kwargs['state_history'].state

    # Perturb each (unconstrained) parameter with normally-distributed noise.
    if not tf.nest.is_nested(perturbation_scale):
      perturbation_scale = tf.nest.map_structure(
          lambda x: tf.convert_to_tensor(perturbation_scale,  # pylint: disable=g-long-lambda
                                         name='perturbation_scale',
                                         dtype=x.dtype),
          unconstrained_params)
    perturbed_unconstrained_parameter_dists = tf.nest.map_structure(
        lambda x, p, s: independent.Independent(  # pylint: disable=g-long-lambda
            normal.Normal(loc=x, scale=p),
            reinterpreted_batch_ndims=prefer_static.rank_from_shape(s)),
        unconstrained_params,
        perturbation_scale,
        parameter_prior.event_shape_tensor())

    # For the joint transition, pass the perturbed parameters
    # into the original transition fn (after pushing them into constrained
    # space).
    return joint_distribution_named.JointDistributionNamed(
        ParametersAndState(
            unconstrained_parameters=_maybe_build_joint_distribution(
                perturbed_unconstrained_parameter_dists),
            state=lambda unconstrained_parameters: (  # pylint: disable=g-long-lambda
                parameterized_transition_fn(
                    step,
                    state,
                    parameters=parameter_constraining_bijector.forward(
                        unconstrained_parameters),
                    **kwargs))))

  return params_and_state_transition_fn


def augment_observation_fn_with_parameters(parameterized_observation_fn,
                                           parameter_constraining_bijector):
  """Augments an observation fn to take `ParametersAndState` namedtuples."""

  def observation_from_params_and_state_fn(step,
                                           params_and_state,
                                           **kwargs):
    # Extract the state, to pass through to the observation fn.
    unconstrained_parameters, state = params_and_state
    if 'state_history' in kwargs:
      _, kwargs['state_history'] = kwargs['state_history']

    return parameterized_observation_fn(
        step,
        state,
        parameters=parameter_constraining_bijector.forward(
            unconstrained_parameters),
        **kwargs)

  return observation_from_params_and_state_fn


def joint_prior_on_parameters_and_state(parameter_prior,
                                        parameterized_initial_state_prior_fn,
                                        parameter_constraining_bijector,
                                        prior_is_constrained=True):
  """Constructs a joint dist. from p(parameters) and p(state | parameters)."""
  if prior_is_constrained:
    parameter_prior = transformed_distribution.TransformedDistribution(
        parameter_prior,
        invert.Invert(parameter_constraining_bijector),
        name='unconstrained_parameter_prior')

  return joint_distribution_named.JointDistributionNamed(
      ParametersAndState(
          unconstrained_parameters=parameter_prior,
          state=lambda unconstrained_parameters: (  # pylint: disable=g-long-lambda
              parameterized_initial_state_prior_fn(
                  parameter_constraining_bijector.forward(
                      unconstrained_parameters)))))


class IteratedFilter(object):
  """A model augmented with parameter perturbations for iterated filtering."""

  def __init__(self,
               parameter_prior,
               parameterized_initial_state_prior_fn,
               parameterized_transition_fn,
               parameterized_observation_fn,
               parameterized_initial_state_proposal_fn=None,
               parameterized_proposal_fn=None,
               parameter_constraining_bijector=None,
               name=None):
    """Builds an iterated filter for parameter estimation in sequential models.

    Iterated filtering is a parameter estimation method in which parameters
    are included in an augmented state space, with dynamics that introduce
    parameter perturbations, and a filtering
    algorithm such as particle filtering is run several times with perturbations
    of decreasing size. This class implements the IF2 algorithm of
    [Ionides et al., 2015][1], for which, under appropriate conditions
    (including a uniform prior) the final parameter distribution approaches a
    point mass at the maximum likelihood estimate. If a non-uniform prior is
    provided, the final parameter distribution will (under appropriate
    conditions) approach a point mass at the maximum a posteriori (MAP) value.

    This class augments the state space of a sequential model to include
    parameter perturbations, and provides utilities to run particle filtering
    on that augmented model. Alternately, the augmented components may be passed
    directly into a filtering algorithm of the user's choice.

    Args:
      parameter_prior: prior `tfd.Distribution` over parameters (may be a joint
        distribution).
      parameterized_initial_state_prior_fn: `callable` with signature
        `initial_state_prior = parameterized_initial_state_prior_fn(parameters)`
        where `parameters` has the form of a sample from `parameter_prior`,
        and `initial_state_prior` is a distribution over the initial state.
      parameterized_transition_fn: `callable` with signature
        `next_state_dist = parameterized_transition_fn(
        step, state, parameters, **kwargs)`.
      parameterized_observation_fn: `callable` with signature
        `observation_dist = parameterized_observation_fn(
        step, state, parameters, **kwargs)`.
      parameterized_initial_state_proposal_fn: optional `callable` with
        signature `initial_state_proposal =
        parameterized_initial_state_proposal_fn(parameters)` where `parameters`
        has the form of a sample from `parameter_prior`, and
        `initial_state_proposal` is a distribution over the initial state.
      parameterized_proposal_fn: optional `callable` with signature
        `next_state_dist = parameterized_transition_fn(
        step, state, parameters, **kwargs)`.
        Default value: `None`.
      parameter_constraining_bijector: optional `tfb.Bijector` instance
        such that `parameter_constraining_bijector.forward(x)` returns valid
        parameters for any real-valued `x` of the same structure and shape
        as `parameters`. If `None`, the default bijector of the provided
        `parameter_prior` will be used.
        Default value: `None`.
      name: `str` name for ops constructed by this object.
        Default value: `iterated_filter`.

    #### Example

    We'll walk through applying iterated filtering to a toy
    Susceptible-Infected-Recovered (SIR) model, a [compartmental model](
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model)
    of infectious disease. Note that the model we use here is extremely
    simplified and is intended as a pedagogical example; it should not be
    interpreted to describe disease spread in the real world.

    We begin by specifying a prior distribution over the parameters to be
    inferred, thus defining the structure of the parameter space and the support
    of the parameters (which will imply a default constraining bijector). Here
    we'll use uniform priors over ranges that we expect to contain the
    parameters:

    ```python
    parameter_prior = tfd.JointDistributionNamed({
        'infection_rate': tfd.Uniform(low=0., high=3.),
        'recovery_rate': tfd.Uniform(low=0., high=3.),
    })
    ```

    The model specification itself is identical to that used by
    `tfp.experimental.mcmc.infer_trajectories`, except that each component
    accepts an additional `parameters` keyword argument. We start by specifying
    a parameterized prior on initial states. In this case, our state
    includes the current number of susceptible and infected individuals
    (the third compartment, recovered individuals, is implicitly defined
    to include the remaining population). We'll also include, as auxiliary
    variables, the daily counts of new infections and new recoveries; these
    will help ensure that people shift consistently across compartments.

    ```python
    population_size = 1000
    initial_state_prior_fn = lambda parameters: tfd.JointDistributionNamed({
        'new_infections': tfd.Poisson(parameters['infection_rate']),
        'new_recoveries': tfd.Deterministic(
            tf.broadcast_to(0., tf.shape(parameters['recovery_rate']))),
        'susceptible': (lambda new_infections:
                        tfd.Deterministic(population_size - new_infections)),
        'infected': (lambda new_infections:
                     tfd.Deterministic(new_infections))})
    ```

    **Note**: the state prior must have the same batch shape as the
    passed-in parameters; equivalently, it must sample a full state for each
    parameter particle. If any part of the state prior does not depend
    on the parameters, you must manually ensure that it has the appropriate
    batch shape. For example, in the definition of `new_recoveries` above,
    applying `broadcast_to` with the shape of a parameter ensures that
    the batch shape is maintained.

    Next, we specify a transition model. This takes the state at the
    previous day, along with parameters, and returns a distribution
    over the state for the current day.

    ```python
    def parameterized_infection_dynamics(_, previous_state, parameters):
      new_infections = tfd.Poisson(
          parameters['infection_rate'] * previous_state['infected'] *
          previous_state['susceptible'] / population_size)
      new_recoveries = tfd.Poisson(
          previous_state['infected'] * parameters['recovery_rate'])
      return tfd.JointDistributionNamed({
          'new_infections': new_infections,
          'new_recoveries': new_recoveries,
          'susceptible': lambda new_infections: tfd.Deterministic(
            tf.maximum(0., previous_state['susceptible'] - new_infections)),
          'infected': lambda new_infections, new_recoveries: tfd.Deterministic(
            tf.maximum(0.,
                       (previous_state['infected'] +
                        new_infections - new_recoveries)))})
    ```

    Finally, assume that every day we get to observe noisy counts of new
    infections and recoveries.

    ```python
    def parameterized_infection_observations(_, state, parameters):
      del parameters  # Not used.
      return tfd.JointDistributionNamed({
          'new_infections': tfd.Poisson(state['new_infections'] + 0.1),
          'new_recoveries': tfd.Poisson(state['new_recoveries'] + 0.1)})
    ```

    Combining these components, an `IteratedFilter` augments
    the state space to include parameters that may change over time.

    ```python
    iterated_filter = tfp.experimental.sequential.IteratedFilter(
      parameter_prior=parameter_prior,
      parameterized_initial_state_prior_fn=initial_state_prior_fn,
      parameterized_transition_fn=parameterized_infection_dynamics,
      parameterized_observation_fn=parameterized_infection_observations)
    ```

    We may then run the filter to estimate parameters from a series
    of observations:

    ```python
     # Simulated with `infection_rate=1.2` and `recovery_rate=0.1`.
     observed_values = {
       'new_infections': tf.convert_to_tensor([
          2., 7., 14., 24., 45., 93., 160., 228., 252., 158.,  17.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
       'new_recoveries': tf.convert_to_tensor([
          0., 0., 3., 4., 3., 8., 12., 31., 49., 73., 85., 65., 71.,
          58., 42., 65., 36., 31., 32., 27., 31., 20., 19., 19., 14., 27.])
     }
     parameter_particles = iterated_filter.estimate_parameters(
         observations=observed_values,
         num_iterations=20,
         num_particles=4096,
         initial_perturbation_scale=1.0,
         cooling_schedule=(
             tfp.experimental.sequential.geometric_cooling_schedule(
                 0.001, k=20)),
         seed=test_util.test_seed())
     print('Mean of parameter particles from final iteration: {}'.format(
       tf.nest.map_structure(lambda x: tf.reduce_mean(x[-1], axis=0),
                             parameter_particles)))
     print('Standard deviation of parameter particles from '
           'final iteration: {}'.format(
           tf.nest.map_structure(lambda x: tf.math.reduce_std(x[-1], axis=0),
                                 parameter_particles)))
    ```

    For more control, we could alternately choose to run filtering iterations
    on the augmented model manually, using the filter of our choice.
    For example, manually invoking `infer_trajectories` would allow us
    to inspect the parameter and state values at all timesteps, and their
    corresponding log-probabilities:

    ```python
    trajectories, lps = tfp.experimental.mcmc.infer_trajectories(
      observations=observations,
      initial_state_prior=iterated_filter.joint_initial_state_prior,
      transition_fn=functools.partial(
          iterated_filter.joint_transition_fn,
          perturbation_scale=perturbation_scale),
      observation_fn=iterated_filter.joint_observation_fn,
      proposal_fn=iterated_filter.joint_proposal_fn,
      initial_state_proposal=iterated_filter.joint_initial_state_proposal(
          initial_unconstrained_parameters),
      num_particles=4096)
    ```

    #### References:

    [1] Edward L. Ionides, Dao Nguyen, Yves Atchade, Stilian Stoev, and Aaron A.
    King. Inference for dynamic and latent variable models via iterated,
    perturbed Bayes maps. _Proceedings of the National Academy of Sciences_
    112, no. 3: 719-724, 2015.
    https://www.pnas.org/content/pnas/112/3/719.full.pdf
    """
    name = name or 'IteratedFilter'
    with tf.name_scope(name):
      self._parameter_prior = parameter_prior
      self._parameterized_initial_state_prior_fn = (
          parameterized_initial_state_prior_fn)

      if parameter_constraining_bijector is None:
        parameter_constraining_bijector = (
            parameter_prior.experimental_default_event_space_bijector())
      self._parameter_constraining_bijector = parameter_constraining_bijector

      # Augment the prior to include both parameters and states.
      self._joint_initial_state_prior = joint_prior_on_parameters_and_state(
          parameter_prior,
          parameterized_initial_state_prior_fn,
          parameter_constraining_bijector,
          prior_is_constrained=True)

      # Check that prior samples have a consistent number of particles.
      # TODO(davmre): remove the need for dummy shape dependencies,
      # and this check, by using `JointDistributionNamedAutoBatched` with
      # auto-vectorization enabled in `joint_prior_on_parameters_and_state`.
      num_particles_canary = 13
      prior_static_sample_shapes = tf.function(
          lambda: self._joint_initial_state_prior.sample(num_particles_canary),
          autograph=False).get_concrete_function().output_shapes
      if not all([s[:1].is_compatible_with([num_particles_canary])
                  for s in tf.nest.flatten(prior_static_sample_shapes)]):
        raise ValueError('The specified prior does not generate consistent '
                         'shapes when sampled. Please verify that all parts of '
                         '`initial_state_prior_fn` have batch shape matching '
                         'that of the parameters. This may require creating '
                         '"dummy" dependencies on parameters; for example: '
                         '`tf.broadcast_to(value, tf.shape(parameter))`. (in a '
                         'test sample with {} particles, we expected all) '
                         'values to have shape compatible with [{}, ...]; '
                         'saw shapes {})'.format(num_particles_canary,
                                                 num_particles_canary,
                                                 prior_static_sample_shapes))

      # Augment the transition and observation fns to cover both
      # parameters and states.
      self._joint_transition_fn = augment_transition_fn_with_parameters(
          parameter_prior,
          parameterized_transition_fn,
          parameter_constraining_bijector)
      self._joint_observation_fn = augment_observation_fn_with_parameters(
          parameterized_observation_fn,
          parameter_constraining_bijector)

      # If given a proposal for the initial state, augment it into a joint
      # proposal over parameters and states.
      joint_initial_state_proposal = None
      if parameterized_initial_state_proposal_fn:
        joint_initial_state_proposal = joint_prior_on_parameters_and_state(
            parameter_prior,
            parameterized_initial_state_proposal_fn,
            parameter_constraining_bijector)
      else:
        parameterized_initial_state_proposal_fn = (
            parameterized_initial_state_prior_fn)
      self._joint_initial_state_proposal = joint_initial_state_proposal
      self._parameterized_initial_state_proposal_fn = (
          parameterized_initial_state_proposal_fn)

      # If given a conditional proposal fn (for non-initial states), augment
      # it to be joint over states and parameters.
      self._joint_proposal_fn = None
      if parameterized_proposal_fn:
        self._joint_proposal_fn = augment_transition_fn_with_parameters(
            parameter_prior,
            parameterized_proposal_fn,
            parameter_constraining_bijector)

      self._batch_ndims = tf.nest.map_structure(
          prefer_static.rank_from_shape,
          parameter_prior.batch_shape_tensor())
      self._name = name

  @property
  def batch_ndims(self):
    return _copy_structure(self._batch_ndims)

  @property
  def joint_initial_state_prior(self):
    """Initial state prior for the joint (augmented) model."""
    return self._joint_initial_state_prior

  def joint_initial_state_proposal(self, initial_unconstrained_parameters=None):
    """Proposal to initialize the model with given parameter particles."""
    if initial_unconstrained_parameters is None:
      joint_initial_state_proposal = self._joint_initial_state_proposal
    else:
      # Hack: DeterministicEmpirical is a fake distribution whose `sample`
      # just proposes *exactly* the parameters we pass in.
      unconstrained_parameter_proposal = DeterministicEmpirical(
          initial_unconstrained_parameters,
          batch_ndims=self.batch_ndims)

      # Propose initial state conditioned on the parameters.
      joint_initial_state_proposal = joint_prior_on_parameters_and_state(
          unconstrained_parameter_proposal,
          self.parameterized_initial_state_proposal_fn,
          parameter_constraining_bijector=(
              self.parameter_constraining_bijector),
          prior_is_constrained=False)

    # May return `None` if no initial proposal or params were specified.
    return joint_initial_state_proposal

  @property
  def joint_transition_fn(self):
    """Transition function for the joint (augmented) model."""
    return self._joint_transition_fn

  @property
  def joint_observation_fn(self):
    """Observation function for the joint (augmented) model."""
    return self._joint_observation_fn

  @property
  def joint_proposal_fn(self):
    """Proposal function for the joint (augmented) model."""
    return self._joint_proposal_fn

  @property
  def name(self):
    return self._name

  @property
  def parameter_constraining_bijector(self):
    """Bijector mapping unconstrained real values into the parameter space."""
    return self._parameter_constraining_bijector

  @property
  def parameterized_initial_state_prior_fn(self):
    """Prior function that was passed in at construction."""
    return self._parameterized_initial_state_prior_fn

  @property
  def parameterized_initial_state_proposal_fn(self):
    """Initial proposal function passed in at construction."""
    return self._parameterized_initial_state_proposal_fn

  @property
  def parameter_prior(self):
    """Prior distribution on parameters passed in at construction."""
    return self._parameter_prior

  def one_step(self,
               observations,
               perturbation_scale,
               num_particles,
               initial_unconstrained_parameters=None,
               seed=None,
               name=None,
               **kwargs):
    """Runs one step of filtering to sharpen parameter estimates.

    Args:
      observations: observed `Tensor` value(s) on which to condition the
        parameter estimate.
      perturbation_scale: scalar float `Tensor`, or any structure of float
        `Tensor`s broadcasting to the same shape as the unconstrained
        parameters, specifying the scale (standard deviation) of Gaussian
        perturbations to each parameter at each timestep.
      num_particles: scalar int `Tensor` number of particles to use. Must match
        the batch dimension of `initial_unconstrained_parameters`, if specified.
      initial_unconstrained_parameters: optional structure of `Tensor`s, of
        shape matching
        `self.joint_initial_state_prior.sample([
        num_particles]).unconstrained_parameters`,
        used to initialize the filter.
        Default value: `None`.
      seed: int `Tensor` seed for random ops.
      name: `str` name for ops constructed by this method.
      **kwargs: additional keyword arguments passed to
        `tfp.experimental.mcmc.infer_trajectories`.
    Returns:
      final_unconstrained_parameters: structure of `Tensor`s matching
        `initial_unconstrained_parameters`, containing samples of
        unconstrained parameters at the final timestep, as computed by
        `self.filter_fn`.
    """
    with self._name_scope(name or 'one_step'):
      # Run the particle filter.
      (unconstrained_parameter_trajectories, _), _ = (
          infer_trajectories(
              observations=observations,
              initial_state_prior=self.joint_initial_state_prior,
              transition_fn=functools.partial(
                  self.joint_transition_fn,
                  perturbation_scale=perturbation_scale),
              observation_fn=self.joint_observation_fn,
              proposal_fn=self.joint_proposal_fn,
              initial_state_proposal=self.joint_initial_state_proposal(
                  initial_unconstrained_parameters),
              num_particles=num_particles,
              seed=seed,
              **kwargs))
      # Return the parameter estimates from the final step of the trajectory.
      return tf.nest.map_structure(
          lambda part: part[-1],
          unconstrained_parameter_trajectories)

  def estimate_parameters(self,
                          observations,
                          num_iterations,
                          num_particles,
                          initial_perturbation_scale,
                          cooling_schedule,
                          seed=None,
                          name=None,
                          **kwargs):
    """Runs multiple iterations of filtering following a cooling schedule.

    Args:
      observations: observed `Tensor` value(s) on which to condition the
        parameter estimate.
      num_iterations: `int `Tensor` number of filtering iterations to run.
      num_particles: scalar int `Tensor` number of particles to use.
      initial_perturbation_scale: scalar float `Tensor`, or any structure of
        float `Tensor`s broadcasting to the same shape as the (unconstrained)
        parameters, specifying the scale (standard deviation) of Gaussian
        perturbations to each parameter at the first timestep.
      cooling_schedule: callable with signature
        `cooling_factor = cooling_schedule(iteration)` for `iteration` in
        `[0, ..., num_iterations - 1]`. The filter is
        invoked with perturbations of scale
        `initial_perturbation_scale * cooling_schedule(iteration)`.
      seed: int `Tensor` seed for random ops.
      name: `str` name for ops constructed by this method.
      **kwargs: additional keyword arguments passed to
        `tfp.experimental.mcmc.infer_trajectories`.
    Returns:
      final_parameter_particles: structure of `Tensor`s matching
        `self.parameter_prior`, each with batch shape
        `[num_iterations, num_particles]`. These are the populations
        of particles representing the parameter estimate after each iteration
        of filtering.
    """
    seed = SeedStream(seed, 'iterated_filter_estimate_parameters')
    with self._name_scope(name or 'estimate_parameters'):

      initial_perturbation_scale = tf.convert_to_tensor(
          initial_perturbation_scale, name='initial_perturbation_scale')

      # Get initial parameter particles from the first filtering iteration.
      initial_unconstrained_parameters = self.one_step(
          observations=observations,
          num_particles=num_particles,
          perturbation_scale=initial_perturbation_scale,
          seed=seed,
          **kwargs)

      # Run the remaining iterations and accumulate the results.
      @tf.function(autograph=False)
      def loop_body(unconstrained_parameters, cooling_fraction):
        return self.one_step(
            observations=observations,
            num_particles=num_particles,
            perturbation_scale=tf.nest.map_structure(
                lambda s: cooling_fraction * s, initial_perturbation_scale),
            initial_unconstrained_parameters=unconstrained_parameters,
            seed=seed,
            **kwargs)
      estimated_unconstrained_parameters = tf.scan(
          fn=loop_body,
          elems=cooling_schedule(tf.range(1, num_iterations)),
          initializer=initial_unconstrained_parameters)

      return self.parameter_constraining_bijector.forward(
          estimated_unconstrained_parameters)

  @contextlib.contextmanager
  def _name_scope(self, name):
    with tf.name_scope(self.name):
      with tf.name_scope(name) as name_scope:
        yield name_scope
