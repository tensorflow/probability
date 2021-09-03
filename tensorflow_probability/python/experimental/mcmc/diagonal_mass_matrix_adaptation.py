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
"""DiagonalMassMatrixAdaptation TransitionKernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf


from tensorflow_probability.python.experimental.mcmc import preconditioning_utils
from tensorflow_probability.python.experimental.stats import sample_stats
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = [
    'DiagonalMassMatrixAdaptation',
]


def hmc_like_momentum_distribution_setter_fn(kernel_results, new_distribution):
  """Setter for `momentum_distribution` so it can be adapted."""
  # Note that unnest.replace_innermost has a special path for going into
  # `accepted_results` preferentially, so this will set
  # `accepted_results.momentum_distribution`.
  return unnest.replace_innermost(
      kernel_results, momentum_distribution=new_distribution)


def hmc_like_momentum_distribution_getter_fn(kernel_results):
  """Getter for `momentum_distribution` so it can be updated."""
  return unnest.get_innermost(kernel_results, 'momentum_distribution')


class DiagonalMassMatrixAdaptationResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('DiagonalMassMatrixAdaptationResults', [
        'inner_results',
        'running_variance',
        'step',
        'num_estimation_steps',
    ])):
  """Results of the DiagonalMassMatrixAdaptation TransitionKernel.

  Attributes:
    inner_results: Results of the inner kernel.
    running_variance: (List of) instance(s) of
      `tfp.experimental.stats.RunningVariance`, used to set
      the diagonal covariance of the momentum distribution.
    step: Int32 scalar `Tensor`. The current step number as perceived by this
      kernel. Increases by 1 for every call to `one_step`.
    num_estimation_steps: Scalar `int` `Tensor` number of initial steps to
      during which to adjust the running variance. When
      ``step == num_estimation_steps``, the momentum distribution will be
      updated and pushed into the `inner_results`.
  """
  __slots__ = ()


class DiagonalMassMatrixAdaptation(kernel_base.TransitionKernel):
  """Adapts the inner kernel's `momentum_distribution` to estimated variance.

  This kernel uses an online variance estimate to adjust a diagonal covariance
  matrix for each of the state parts. More specifically, the
  `momentum_distribution` of the innermost kernel is set to a diagonal
  multivariate normal distribution whose variance is the *inverse* of the
  online estimate. The inverse of the covariance of the momentum is often called
  the "mass matrix" in the context of Hamiltonian Monte Carlo.

  This preconditioning scheme works well when the covariance is diagonally
  dominant, and may give reasonable results even when the number of draws is
  less than the dimension. In particular, it should generally do a better job
  than no preconditioning, which implicitly uses an identity mass matrix.

  Note that this kernel does not implement a calibrated sampler; rather, it is
  intended to be used as one step of an iterative adaptation process. It
  should not be used when drawing actual samples.
  """

  def __init__(
      self,
      inner_kernel,
      initial_running_variance,
      num_estimation_steps=None,
      momentum_distribution_setter_fn=hmc_like_momentum_distribution_setter_fn,
      momentum_distribution_getter_fn=hmc_like_momentum_distribution_getter_fn,
      validate_args=False,
      experimental_shard_axis_names=None,
      name=None):
    """Creates the diagonal mass matrix adaptation kernel.

    Users must provide an `initial_running_variance`, either from a previous
    `DiagonalMassMatrixAdaptation`, or some other source. See
    `RunningCovariance.from_stats` for a convenient way to construct these.


    Args:
      inner_kernel: `TransitionKernel`-like object.
      initial_running_variance:
        `tfp.experimental.stats.RunningVariance`-like object, or list of them,
        for a batch of momentum distributions. These use `update` on the state
        to maintain an estimate of the variance, and so space, and so must have
        a structure compatible with the state space.
      num_estimation_steps: An optional scalar `int` `Tensor` number of initial
        steps to during which to adjust the running variance. This may be
        greater, less than, or equal to the number of burnin steps. If this
        argument is None, the mass matrix will be updated at each `one_step`
        call. Otherwise, the mass matrix will be updated when the current step
        is equal to `num_estimation_steps`.
      momentum_distribution_setter_fn: A callable with the signature
        `(kernel_results, new_momentum_distribution) -> new_kernel_results`
        where `kernel_results` are the results of the `inner_kernel`,
        `new_momentum_distribution` is a `CompositeTensor` or a nested
        collection of `CompositeTensor`s, and `new_kernel_results` are a
        possibly-modified copy of `kernel_results`. The default,
        `hmc_like_momentum_distribution_setter_fn`, presumes HMC-style
        `kernel_results`, and sets the `momentum_distribution` only under the
        `accepted_results` field.
      momentum_distribution_getter_fn: A callable with the signature
        `kernel_results -> momentum_distribution`
        where `kernel_results` are the results of the `inner_kernel` and
        `momentum_distribution` is a `CompositeTensor` or a nested
        collection of `CompositeTensor`s. The default,
        `hmc_like_momentum_distribution_getter_fn`, presumes HMC-style
        `kernel_results`, and gets the `momentum_distribution` only under the
        `accepted_results` field.
      validate_args: Python `bool`. When `True` kernel parameters are checked
        for validity. When `False` invalid inputs may silently render incorrect
        outputs.
      experimental_shard_axis_names: An optional structure of string names
        indicating how members of the state are sharded.
      name: Python `str` name prefixed to Ops created by this class. Default:
        'diagonal_mass_matrix_adaptation'.
    """
    inner_kernel = mcmc_util.enable_store_parameters_in_results(
        inner_kernel).experimental_with_shard_axes(
            experimental_shard_axis_names)
    self._parameters = dict(
        inner_kernel=inner_kernel,
        initial_running_variance=initial_running_variance,
        num_estimation_steps=num_estimation_steps,
        momentum_distribution_setter_fn=momentum_distribution_setter_fn,
        momentum_distribution_getter_fn=momentum_distribution_getter_fn,
        experimental_shard_axis_names=experimental_shard_axis_names,
        name=name,
    )

  @property
  def inner_kernel(self):
    return self._parameters['inner_kernel']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def initial_running_variance(self):
    return self._parameters['initial_running_variance']

  @property
  def num_estimation_steps(self):
    return self._parameters['num_estimation_steps']

  def momentum_distribution_getter_fn(self, kernel_results):
    return self._parameters['momentum_distribution_getter_fn'](kernel_results)

  def momentum_distribution_setter_fn(self, kernel_results,
                                      new_momentum_distribution):
    return self._parameters['momentum_distribution_setter_fn'](
        kernel_results, new_momentum_distribution)

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'diagonal_mass_matrix_adaptation',
                            'one_step')):
      variance_parts = previous_kernel_results.running_variance
      inner_results = previous_kernel_results.inner_results

      # Step the inner kernel.
      inner_kwargs = {} if seed is None else dict(seed=seed)
      new_state, new_inner_results = self.inner_kernel.one_step(
          current_state, inner_results, **inner_kwargs)

      def update_running_variance():
        diags = [variance_part.variance() for variance_part in variance_parts]
        new_state_parts = tf.nest.flatten(new_state)
        new_variance_parts = []
        for variance_part, diag, state_part in zip(variance_parts, diags,
                                                   new_state_parts):
          # Compute new variance for each variance part, accounting for partial
          # batching of the variance calculation across chains (ie, some, all,
          # or none of the chains may share the estimated mass matrix).
          #
          # For example, say
          #
          # state_part has shape       [2, 3, 4] + [5, 6]  (batch + event)
          # variance_part has shape          [4] + [5, 6]
          # log_prob has shape         [2, 3, 4]
          #
          # i.e., we have a batch of chains of shape [2, 3, 4], and 4 mass
          # matrices, each being shared across a [2, 3]-batch of chains. Note
          # this division is inferred from the shapes of the state part, the
          # log_prob, and the user-provided initial running variances.
          #
          # Until RunningVariance supports rank > 1 chunking, we need to flatten
          # the states that go into updating the variance estimates. In the
          # above example, `state_part` will be reshaped to `[6, 4, 5, 6]`, and
          # fed to `RunningVariance.update(state_part, axis=0)`, recording
          # 6 new observations in the running variance calculation.
          # `RunningVariance.variance()` will then be of shape `[4, 5, 6]`, and
          # the resulting momentum distribution will have batch shape of
          # `[2, 3, 4]` and event_shape of `[5, 6]`, matching the state_part.
          state_rank = ps.rank(state_part)
          variance_rank = ps.rank(diag)
          num_reduce_dims = state_rank - variance_rank

          state_part_shape = ps.shape(state_part)
          # This reshape adds a 1 when reduce_dims==0, and collapses all the
          # lead dimensions to a single one otherwise.
          reshaped_state = ps.reshape(
              state_part,
              ps.concat([[ps.reduce_prod(state_part_shape[:num_reduce_dims])],
                         state_part_shape[num_reduce_dims:]],
                        axis=0))

          # The `axis=0` here removes the leading dimension we got from the
          # reshape above, so the new_variance_parts have the correct shape
          # again.
          new_variance_parts.append(
              variance_part.update(reshaped_state, axis=0))
        return new_variance_parts

      def update_momentum():
        diags = [
            variance_part.variance() for variance_part in new_variance_parts
        ]
        # Update the momentum.
        prev_momentum_distribution = self.momentum_distribution_getter_fn(
            new_inner_results)
        new_momentum_distribution = (
            preconditioning_utils.update_momentum_distribution(
                prev_momentum_distribution, diags))
        updated_new_inner_results = self.momentum_distribution_setter_fn(
            new_inner_results, new_momentum_distribution)
        return updated_new_inner_results

      step = previous_kernel_results.step + 1
      if self.num_estimation_steps is None:
        new_variance_parts = update_running_variance()
        new_inner_results = update_momentum()
      else:
        new_variance_parts = mcmc_util.choose(
            step <= previous_kernel_results.num_estimation_steps,
            update_running_variance(), variance_parts)
        new_inner_results = mcmc_util.choose(
            tf.equal(step, previous_kernel_results.num_estimation_steps),
            update_momentum(), new_inner_results)
      new_kernel_results = previous_kernel_results._replace(
          inner_results=new_inner_results,
          running_variance=new_variance_parts,
          step=step)

      return new_state, new_kernel_results

  def bootstrap_results(self, init_state):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'diagonal_mass_matrix_adaptation',
                            'bootstrap_results')):
      # Step inner results.
      inner_results = self.inner_kernel.bootstrap_results(init_state)

      # Bootstrap the results.
      results = self._bootstrap_from_inner_results(init_state, inner_results)
      if self.num_estimation_steps is not None:
        # We only update the momentum at the end of adaptation phase,
        # so we do not need to set the momentum here.
        return results

      # Set the momentum.
      diags = [
          variance_part.variance() for variance_part in results.running_variance
      ]
      inner_results = results.inner_results
      batch_shape = ps.shape(unnest.get_innermost(inner_results,
                                                  'target_log_prob'))
      init_state_parts = tf.nest.flatten(init_state)
      momentum_distribution = preconditioning_utils.make_momentum_distribution(
          init_state_parts, batch_shape, diags,
          shard_axis_names=self.experimental_shard_axis_names)
      inner_results = self.momentum_distribution_setter_fn(
          inner_results, momentum_distribution)
      proposed = unnest.get_innermost(inner_results, 'proposed_results',
                                      default=None)
      if proposed is not None:
        proposed = proposed._replace(
            momentum_distribution=momentum_distribution)
        inner_results = unnest.replace_innermost(inner_results,
                                                 proposed_results=proposed)
      results = results._replace(inner_results=inner_results)
      return results

  def _bootstrap_from_inner_results(self, init_state, inner_results):
    if isinstance(self.initial_running_variance, sample_stats.RunningVariance):
      variance_parts = [self.initial_running_variance]
    else:
      variance_parts = list(self.initial_running_variance)

    if self.num_estimation_steps is None:
      num_estimation_steps = tf.constant(-1, dtype=tf.int32)
    else:
      num_estimation_steps = tf.cast(self.num_estimation_steps, dtype=tf.int32)
    return DiagonalMassMatrixAdaptationResults(
        inner_results=inner_results,
        running_variance=variance_parts,
        step=tf.constant(0, dtype=tf.int32),
        num_estimation_steps=num_estimation_steps)

  @property
  def is_calibrated(self):
    if self.num_estimation_steps is None:
      return False
    return self.inner_kernel.is_calibrated

  @property
  def experimental_shard_axis_names(self):
    return self._parameters['experimental_shard_axis_names']

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(experimental_shard_axis_names=shard_axis_names)
