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
"""The HiddenMarkovModel distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'HiddenMarkovModel',
]


class HiddenMarkovModel(distribution.Distribution):
  """Hidden Markov model distribution.

  The `HiddenMarkovModel` distribution implements a (batch of) discrete hidden
  Markov models where the initial states, transition probabilities
  and observed states are all given by user-provided distributions.

  In this model, there is a sequence of integer-valued hidden states:
  `z[0], z[1], ..., z[num_steps - 1]` and a sequence of observed states:
  `x[0], ..., x[num_steps - 1]`.
  The distribution of `z[0]` is given by `initial_distribution`.
  The conditional probability of `z[i  +  1]` given `z[i]` is described by
  the batch of distributions in `transition_distribution`.
  For a batch of hidden Markov models, the coordinates before the rightmost one
  of the `transition_distribution` batch correspond to indices into the hidden
  Markov model batch. The rightmost coordinate of the batch is used to select
  which distribution `z[i + 1]` is drawn from.  The distributions corresponding
  to the probability of `z[i + 1]` conditional on `z[i] == k` is given by the
  elements of the batch whose rightmost coordinate is `k`.
  Similarly, the conditional distribution of `z[i]` given `x[i]` is given by
  the batch of `observation_distribution`.
  When the rightmost coordinate of `observation_distribution` is `k` it
  gives the conditional probabilities of `x[i]` given `z[i] == k`.
  The probability distribution associated with the `HiddenMarkovModel`
  distribution is the marginal distribution of `x[0],...,x[num_steps - 1]`.

  #### Examples

  ```python
  tfd = tfp.distributions

  # A simple weather model.

  # Represent a cold day with 0 and a hot day with 1.
  # Suppose the first day of a sequence has a 0.8 chance of being cold.
  # We can model this using the categorical distribution:

  initial_distribution = tfd.Categorical(probs=[0.8, 0.2])

  # Suppose a cold day has a 30% chance of being followed by a hot day
  # and a hot day has a 20% chance of being followed by a cold day.
  # We can model this as:

  transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                   [0.2, 0.8]])

  # Suppose additionally that on each day the temperature is
  # normally distributed with mean and standard deviation 0 and 5 on
  # a cold day and mean and standard deviation 15 and 10 on a hot day.
  # We can model this with:

  observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

  # We can combine these distributions into a single week long
  # hidden Markov model with:

  model = tfd.HiddenMarkovModel(
      initial_distribution=initial_distribution,
      transition_distribution=transition_distribution,
      observation_distribution=observation_distribution,
      num_steps=7)

  # The expected temperatures for each day are given by:

  model.mean()  # shape [7], elements approach 9.0

  # The log pdf of a week of temperature 0 is:

  model.log_prob(tf.zeros(shape=[7]))
  ```

  #### References
  [1] https://en.wikipedia.org/wiki/Hidden_Markov_model
  """

  def __init__(self,
               initial_distribution,
               transition_distribution,
               observation_distribution,
               num_steps,
               validate_args=False,
               allow_nan_stats=True,
               time_varying_transition_distribution=False,
               time_varying_observation_distribution=False,
               name='HiddenMarkovModel'):
    """Initialize hidden Markov model.

    Args:
      initial_distribution: A `Categorical`-like instance.
        Determines probability of first hidden state in Markov chain.
        The number of categories must match the number of categories of
        `transition_distribution` as well as both the rightmost batch
        dimension of `transition_distribution` and the rightmost batch
        dimension of `observation_distribution`.
      transition_distribution: A `Categorical`-like instance.
        The rightmost batch dimension indexes the probability distribution
        of each hidden state conditioned on the previous hidden state.
      observation_distribution: A `tfp.distributions.Distribution`-like
        instance.  The rightmost batch dimension indexes the distribution
        of each observation conditioned on the corresponding hidden state.
      num_steps: The number of steps taken in Markov chain. An integer valued
        tensor. The number of transitions is `num_steps - 1`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        Default value: `True`.
      time_varying_transition_distribution: Python `bool`, default `False`.
        When `True`, the transition_distribution has an additional batch
        dimension that indexes the distribution of each observation conditioned
        on the corresponding timestep. This dimension size should always match
        `num_steps -1` and is the second-to-last batch axis in the batch
        dimensions (just to the left of the dimension for the number of states).
        Because transitions only happen between steps, the number of transitions
        is one less than num_steps.
      time_varying_observation_distribution: Python `bool`, default `False`.
        When `True`, the observation_distribution has an additional batch
        dimension that indexes the distribution of each observation conditioned
        on the corresponding timestep. This dimension size should always match
        num_steps and is the second-to-last batch axis in the batch dimensions
        (just to the left of the dimension for the number of states).
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "HiddenMarkovModel".

    Raises:
      ValueError: if `num_steps` is not at least 1.
      ValueError: if `initial_distribution` does not have scalar `event_shape`.
      ValueError: if `transition_distribution` does not have scalar
        `event_shape.`
      ValueError: if `transition_distribution` and `observation_distribution`
        are fully defined but don't have matching rightmost dimension.
    """

    parameters = dict(locals())

    # pylint: disable=protected-access
    with tf.name_scope(name) as name:

      self._num_steps = tensor_util.convert_nonref_to_tensor(
          num_steps, as_shape_tensor=True)
      self._initial_distribution = initial_distribution
      self._observation_distribution = observation_distribution
      self._transition_distribution = transition_distribution
      self._time_varying_observation_distribution = (
          time_varying_observation_distribution)
      self._time_varying_transition_distribution = (
          time_varying_transition_distribution)

      num_steps_ = tf.get_static_value(num_steps)
      if num_steps_ is not None:
        if np.ndim(num_steps_) != 0:
          raise ValueError(
              '`num_steps` must be a scalar but it has rank {}'.format(
                  np.ndim(num_steps_)))
        else:
          self._static_event_shape = tf.TensorShape(
              [num_steps_]).concatenate(
                  self._observation_distribution.event_shape)
      else:
        self._static_event_shape = tf.TensorShape(
            [None]).concatenate(
                self._observation_distribution.event_shape)

      observation_batch_shape = (
          self._observation_distribution.batch_shape[:-2]
          if self._time_varying_observation_distribution
          else self._observation_distribution.batch_shape[:-1])
      self._static_batch_shape = tf.broadcast_static_shape(
          self._initial_distribution.batch_shape,
          tf.broadcast_static_shape(
              self._transition_distribution.batch_shape[:-2]
              if self._time_varying_transition_distribution else
              self._transition_distribution.batch_shape[:-1],
              observation_batch_shape))

      # pylint: disable=protected-access
      super(HiddenMarkovModel, self).__init__(
          dtype=self._observation_distribution.dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)
      # pylint: enable=protected-access

      self._parameters = parameters

  def _batch_shape_tensor(self):
    observation_batch_shape = (
        self._observation_distribution.batch_shape_tensor()[:-2]
        if self._time_varying_observation_distribution
        else self._observation_distribution.batch_shape_tensor()[:-1])
    return ps.broadcast_shape(
        self._initial_distribution.batch_shape_tensor(),
        ps.broadcast_shape(
            self._transition_distribution.batch_shape_tensor()[:-2]
            if self._time_varying_transition_distribution else
            self._transition_distribution.batch_shape_tensor()[:-1],
            observation_batch_shape))

  def _batch_shape(self):
    return self._static_batch_shape

  def _event_shape_tensor(self):
    return ps.concat([[self._num_steps],
                      self.observation_distribution.event_shape_tensor()],
                     axis=0)

  def _event_shape(self):
    return self._static_event_shape

  @property
  def initial_distribution(self):
    return self._initial_distribution

  @property
  def transition_distribution(self):
    return self._transition_distribution

  @property
  def observation_distribution(self):
    return self._observation_distribution

  @property
  def num_steps(self):
    return self._num_steps

  @property
  def num_states_static(self):
    """The number of hidden states in the hidden Markov model.

    Returns:
      A value of integer type if the number of states can be computed
      statically and `None` otherwise.
    """

    return tf.get_static_value(self.transition_distribution.batch_shape[-1])

  def num_states_tensor(self):
    """The number of hidden states in the hidden Markov model."""

    return self.transition_distribution.batch_shape_tensor()[-1]

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda,protected-access
    return dict(
        initial_distribution=(
            parameter_properties.BatchedComponentProperties()),
        transition_distribution=(
            parameter_properties.BatchedComponentProperties(
                event_ndims=(
                    lambda self: (2
                                  if self._time_varying_transition_distribution
                                  else 1)))),
        observation_distribution=(
            parameter_properties.BatchedComponentProperties(
                event_ndims=(
                    lambda self: (2
                                  if self._time_varying_observation_distribution
                                  else 1)))))
    # pylint: enable=g-long-lambda,protected-access

  def _sample_n(self, n, seed=None):
    init_seed, scan_seed, observation_seed = samplers.split_seed(
        seed, n=3, salt='HiddenMarkovModel')

    transition_batch_shape = self.transition_distribution.batch_shape_tensor()
    num_states = transition_batch_shape[-1]

    batch_shape = self.batch_shape_tensor()
    batch_size = ps.reduce_prod(batch_shape)
    # The batch sizes of the underlying initial distributions and
    # transition distributions might not match the batch size of
    # the HMM distribution.
    # As a result we need to ask for more samples from the
    # underlying distributions and then reshape the results into
    # the correct batch size for the HMM.
    init_repeat = (
        ps.reduce_prod(batch_shape) //
        ps.reduce_prod(self._initial_distribution.batch_shape_tensor()))
    init_state = self._initial_distribution.sample(n * init_repeat,
                                                   seed=init_seed)
    init_state = tf.reshape(init_state, [n, batch_size])
    # init_state :: n batch_size

    transition_repeat = (
        ps.reduce_prod(batch_shape) // ps.reduce_prod(
            transition_batch_shape[:-2]
            if self._time_varying_transition_distribution else
            transition_batch_shape[:-1]))

    init_shape = init_state.shape

    def generate_step(state_and_seed, step):
      """Take a single step in Markov chain."""
      state, seed = state_and_seed
      sample_seed, next_seed = samplers.split_seed(seed)

      if self._time_varying_transition_distribution:
        gen = self._transition_distribution[..., step, :].sample(
            n * transition_repeat, seed=sample_seed)
      else:
        gen = self._transition_distribution.sample(n * transition_repeat,
                                                   seed=sample_seed)
      # gen :: (n * transition_repeat) transition_batch

      new_states = tf.reshape(gen,
                              [n, batch_size, num_states])

      # new_states :: n batch_size num_states

      old_states_one_hot = tf.one_hot(state, num_states, dtype=tf.int32)

      # old_states :: n batch_size num_states

      result = tf.reduce_sum(old_states_one_hot * new_states, axis=-1)
      # We know that `generate_step` must preserve the shape of the
      # tensor of states of each state. This is because
      # the transition matrix must be square. But TensorFlow might
      # not know this so we explicitly tell it that the result has the
      # same shape.
      tensorshape_util.set_shape(result, init_shape)
      return result, next_seed

    def _scan_multiple_steps():
      """Take multiple steps with tf.scan."""
      step = ps.range(self._num_steps - 1, dtype=tf.int32)
      hidden_states, _ = tf.scan(generate_step, step,
                                 initializer=(init_state, scan_seed))

      # TODO(b/115618503): add/use prepend_initializer to tf.scan
      return tf.concat([[init_state],
                        hidden_states], axis=0)

    def _scan_one_step():
      multiples = ps.concat([[self.num_steps],
                             ps.ones(ps.rank(init_state), tf.int32)],
                            axis=0)
      return tf.tile(init_state[tf.newaxis, ...], multiples)

    hidden_states = ps.cond(
        self._num_steps > 1,
        _scan_multiple_steps,
        _scan_one_step)

    hidden_one_hot = tf.one_hot(hidden_states, num_states,
                                dtype=self._observation_distribution.dtype)
    # hidden_one_hot :: num_steps n batch_size num_states

    # The observation distribution batch size might not match
    # the required batch size so as with the initial and
    # transition distributions we generate more samples and
    # reshape.
    observation_repeat = ps.maximum(
        batch_size // ps.reduce_prod(
            self._observation_distribution.batch_shape_tensor()[:-1]),
        1)

    if self._time_varying_observation_distribution:
      possible_observations = self._observation_distribution.sample(
          [observation_repeat * n], seed=observation_seed)
      # possible observations needs to have num_steps moved to the beginning.
      possible_observations = distribution_util.move_dimension(
          possible_observations,
          -(tf.size(self._observation_distribution.event_shape_tensor()) + 2),
          0)
    else:
      possible_observations = self._observation_distribution.sample(
          [self._num_steps, observation_repeat * n], seed=observation_seed)

    inner_shape = self._observation_distribution.event_shape_tensor()

    # possible_observations :: num_steps (observation_repeat * n)
    #                          observation_batch[:-1] num_states inner_shape

    possible_observations = tf.reshape(
        possible_observations,
        ps.concat([[self._num_steps, n],
                   batch_shape,
                   [num_states],
                   inner_shape], axis=0))

    # possible_observations :: steps n batch_size num_states inner_shape

    hidden_one_hot = tf.reshape(hidden_one_hot,
                                ps.concat([[self._num_steps, n],
                                           batch_shape,
                                           [num_states],
                                           ps.ones_like(inner_shape)],
                                          axis=0))

    # hidden_one_hot :: steps n batch_size num_states "inner_shape"

    observations = tf.reduce_sum(
        hidden_one_hot * possible_observations,
        axis=-1 - ps.size(inner_shape))
    # observations :: steps n batch_size inner_shape

    observations = distribution_util.move_dimension(observations, 0,
                                                    1 + ps.size(batch_shape))
    # returned :: n batch_shape steps inner_shape

    return observations

  def _log_prob(self, value):
    # The argument `value` is a tensor of sequences of observations.
    # `observation_batch_shape` is the shape of that tensor with the
    # sequence part removed.
    # `observation_batch_shape` is then broadcast to the full batch shape
    # to give the `batch_shape` that defines the shape of the result.
    observation_tensor_shape = ps.shape(value)
    observation_distribution = self.observation_distribution
    underlying_event_rank = ps.size(
        observation_distribution.event_shape_tensor())
    observation_batch_shape = observation_tensor_shape[
        :-1 - underlying_event_rank]
    # value :: observation_batch_shape num_steps observation_event_shape
    batch_shape = ps.broadcast_shape(observation_batch_shape,
                                     self.batch_shape_tensor())
    num_states = self.transition_distribution.batch_shape_tensor()[-1]
    log_init = _extract_log_probs(num_states,
                                  self.initial_distribution,
                                  move_time_dimension=False)
    # log_init :: batch_shape num_states
    log_init = tf.broadcast_to(log_init,
                               ps.concat([batch_shape,
                                          [num_states]], axis=0))
    log_transition = _extract_log_probs(num_states,
                                        self.transition_distribution,
                                        move_time_dimension=False)

    # `observation_event_shape` is the shape of each sequence of observations
    # emitted by the model.
    observation_event_shape = observation_tensor_shape[
        -1 - underlying_event_rank:]
    working_obs = tf.broadcast_to(value,
                                  ps.concat([batch_shape,
                                             observation_event_shape],
                                            axis=0))
    # working_obs :: batch_shape observation_event_shape
    r = underlying_event_rank

    # Move index into sequence of observations to front so we can apply
    # tf.foldl
    if self._time_varying_observation_distribution:
      working_obs = tf.expand_dims(working_obs, -1 - r)
      # working_obs :: batch_shape num_steps 1 underlying_event_shape
      observation_probs = observation_distribution.log_prob(working_obs)
      # observation_probs :: batch_shape num_steps num_states
      observation_probs = distribution_util.move_dimension(
          observation_probs, -2, 0)
      # observation_probs :: num_steps batch_shape num_states
    else:
      working_obs = distribution_util.move_dimension(working_obs, -1 - r, 0)
      # working_obs :: num_steps batch_shape underlying_event_shape
      working_obs = tf.expand_dims(working_obs, -1 - r)
      # working_obs :: num_steps batch_shape 1 underlying_event_shape

      observation_probs = observation_distribution.log_prob(working_obs)
      # observation_probs :: num_steps batch_shape num_states

    if self._time_varying_transition_distribution:
      log_transition = distribution_util.move_dimension(log_transition, -3, 0)
      def dynamic_forward_step(log_prev_step, log_transition_and_observation):
        log_transition, log_prob_observation = log_transition_and_observation
        return _log_vector_matrix(log_prev_step,
                                  log_transition) + log_prob_observation
      fwd_prob = tf.foldl(dynamic_forward_step,
                          (log_transition, observation_probs[1:]),
                          initializer=log_init + observation_probs[0])
    else:
      def forward_step(log_prev_step, log_prob_observation):
        return _log_vector_matrix(log_prev_step,
                                  log_transition) + log_prob_observation

      fwd_prob = tf.foldl(forward_step, observation_probs[1:],
                          initializer=log_init + observation_probs[0])
    # fwd_prob :: batch_shape num_states

    log_prob = tf.reduce_logsumexp(fwd_prob, axis=-1)
    # log_prob :: batch_shape

    return log_prob

  def _marginal_hidden_probs(self):
    """Compute marginal pdf for each individual observable."""

    forward_log_probs, _ = self._forward_pass(observations=None, mask=None)
    return tf.exp(forward_log_probs)

  def _mean(self):
    observation_distribution = self.observation_distribution
    batch_shape = self.batch_shape_tensor()
    num_states = self.transition_distribution.batch_shape_tensor()[-1]
    probs = self._marginal_hidden_probs()
    # probs :: num_steps batch_shape num_states
    means = observation_distribution.mean()
    # means :: observation_batch_shape[:-2] num_steps num_states
    #                                       observation_event_shape
    # or
    # means :: observation_batch_shape[:-1] num_states
    #                                       observation_event_shape
    # the latter case happens for static observations distributions but it
    # broadcasts successfully unless the distribution has unnecessary batch
    # dimensions.
    means_shape = ps.concat(
        [batch_shape,
         [self._num_steps, num_states],
         observation_distribution.event_shape_tensor()],
        axis=0)
    means = tf.broadcast_to(means, means_shape)
    # means :: batch_shape num_steps num_states observation_event_shape
    observation_event_shape = (
        observation_distribution.event_shape_tensor())
    batch_size = tf.reduce_prod(batch_shape)
    flat_probs_shape = [self._num_steps, batch_size, num_states]
    flat_means_shape = [
        batch_size, self._num_steps, num_states,
        tf.reduce_prod(observation_event_shape)
    ]

    flat_probs = tf.reshape(probs, flat_probs_shape)
    # flat_probs :: num_steps batch_size num_states
    flat_means = tf.reshape(means, flat_means_shape)
    # flat_means :: batch_size num_steps num_states observation_event_size
    flat_mean = tf.einsum('ijk,jikl->jil', flat_probs, flat_means)
    # flat_mean :: batch_size num_steps observation_event_size
    unflat_mean_shape = ps.concat(
        [batch_shape,
         [self._num_steps],
         observation_event_shape],
        axis=0)
    # returns :: batch_shape num_steps observation_event_shape
    return tf.reshape(flat_mean, unflat_mean_shape)

  def _variance(self):
    num_states = self.transition_distribution.batch_shape_tensor()[-1]
    batch_shape = self.batch_shape_tensor()
    probs = self._marginal_hidden_probs()
    # probs :: num_steps batch_shape num_states
    observation_distribution = self.observation_distribution
    means = observation_distribution.mean()
    # means :: observation_batch_shape[:-2] num_steps num_states
    #          observation_event_shape
    # or
    # means :: observation_batch_shape[:-1] num_states
    #                                       observation_event_shape
    # the latter case happens for static observations distributions but it
    # broadcasts successfully unless the distribution has unnecessary batch
    # dimensions.
    means_shape = ps.concat(
        [batch_shape,
         [self._num_steps, num_states],
         observation_distribution.event_shape_tensor()],
        axis=0)
    means = tf.broadcast_to(means, means_shape)
    # means :: batch_shape num_steps num_states observation_event_shape
    observation_event_shape = (
        observation_distribution.event_shape_tensor())
    batch_size = tf.reduce_prod(batch_shape)
    flat_probs_shape = [self._num_steps, batch_size, num_states]
    flat_means_shape = [
        batch_size, self._num_steps, num_states,
        tf.reduce_prod(observation_event_shape)
    ]

    flat_probs = tf.reshape(probs, flat_probs_shape)
    # flat_probs :: num_steps batch_size num_states
    flat_means = tf.reshape(means, flat_means_shape)
    # flat_means :: batch_size num_steps num_states observation_event_size
    flat_mean = tf.einsum('ijk,jikl->jil', flat_probs, flat_means)
    flat_mean = tf.expand_dims(flat_mean, 2)
    # flat_mean :: batch_size num_steps 1 observation_event_size

    variances = observation_distribution.variance()
    variances = tf.broadcast_to(variances, means_shape)
    # variances :: batch_shape num_steps num_states observation_event_shape
    flat_variances = tf.reshape(variances, flat_means_shape)
    # flat_variances :: batch_size num_steps num_states observation_event_size

    # For a mixture of n distributions with mixture probabilities
    # p[i], and where the individual distributions have means and
    # variances given by mean[i] and var[i], the variance of
    # the mixture is given by:
    #
    # var = sum i=1..n p[i] * ((mean[i] - mean)**2 + var[i]**2)

    flat_variance = tf.einsum('ijk,jikl->jil',
                              flat_probs,
                              (flat_means - flat_mean)**2 + flat_variances)
    # flat_variance :: batch_size num_steps observation_event_size
    unflat_mean_shape = ps.concat(
        [batch_shape,
         [self._num_steps],
         observation_event_shape],
        axis=0)

    # returns :: batch_shape num_steps observation_event_shape
    return tf.reshape(flat_variance, unflat_mean_shape)

  def _observation_mask_shape_preconditions(self,
                                            observation_tensor_shape,
                                            mask_tensor_shape,
                                            underlying_event_rank):
    shape_condition = [assert_util.assert_equal(
        observation_tensor_shape[-1 - underlying_event_rank],
        self._num_steps,
        message='The tensor `observations` must consist of sequences '
                'of observations from `HiddenMarkovModel` of length'
                '`num_steps`.')]
    if mask_tensor_shape is not None:
      shape_condition.append(assert_util.assert_equal(
          mask_tensor_shape[-1],
          self._num_steps,
          message='The tensor `mask` must consist of sequences'
                  'of length `num_steps`.'))
    return tf.control_dependencies(shape_condition)

  def _observation_log_probs(self, observations, mask):
    """Compute and shape tensor of log probs associated with observations."""

    # Let E be the underlying event shape
    #     M the number of steps in the HMM
    #     N the number of states of the HMM
    #
    # Then the incoming observations have shape
    #
    # observations : batch_o [M] E
    #
    # and the mask (if present) has shape
    #
    # mask : batch_m [M]
    #
    # Let this HMM distribution have batch shape batch_d
    # We need to broadcast all three of these batch shapes together
    # into the shape batch.
    #
    # We need to move the step dimension to the first dimension to make
    # them suitable for folding or scanning over.
    #
    # When we call `log_prob` for our observations we need to
    # do this for each state the observation could correspond to.
    # We do this by expanding the dimensions by 1 so we end up with:
    #
    # observations : [M] batch [1] [E]
    #
    # After calling `log_prob` we get
    #
    # observation_log_probs : [M] batch [N]
    #
    # We wish to use `mask` to select from this so we also
    # reshape and broadcast it up to shape
    #
    # mask : [M] batch [N]

    observation_distribution = self.observation_distribution
    underlying_event_rank = ps.size(
        observation_distribution.event_shape_tensor())
    observation_tensor_shape = ps.shape(observations)
    observation_batch_shape = observation_tensor_shape[
        :-1 - underlying_event_rank]
    observation_event_shape = observation_tensor_shape[
        -1 - underlying_event_rank:]

    if mask is not None:
      mask_tensor_shape = ps.shape(mask)
      mask_batch_shape = mask_tensor_shape[:-1]

    batch_shape = ps.broadcast_shape(observation_batch_shape,
                                     self.batch_shape_tensor())

    if mask is not None:
      batch_shape = ps.broadcast_shape(batch_shape,
                                       mask_batch_shape)
    observations = tf.broadcast_to(observations,
                                   ps.concat([batch_shape,
                                              observation_event_shape],
                                             axis=0))
    observation_rank = ps.rank(observations)
    if self._time_varying_observation_distribution:
      observations = tf.expand_dims(
          observations,
          observation_rank - underlying_event_rank)
      observation_log_probs = observation_distribution.log_prob(
          observations)
      observation_log_probs = distribution_util.move_dimension(
          observation_log_probs, -2, 0)
    else:
      observations = distribution_util.move_dimension(
          observations, observation_rank - underlying_event_rank - 1, 0)
      observations = tf.expand_dims(
          observations,
          observation_rank - underlying_event_rank)
      observation_log_probs = observation_distribution.log_prob(
          observations)

    if mask is not None:
      mask = tf.broadcast_to(mask,
                             ps.concat([batch_shape, [self._num_steps]],
                                       axis=0))
      mask = distribution_util.move_dimension(mask, -1, 0)
      observation_log_probs = tf.where(mask[..., tf.newaxis],
                                       tf.zeros_like(observation_log_probs),
                                       observation_log_probs)

    return observation_log_probs

  def prior_marginals(self, name='prior'):
    """Compute prior marginal distribution for each state.

    This function computes, for each time step, the
    prior probability that the hidden Markov model is at a given state.
    In other words this function computes:
    `P(z[i])` for all `i` from `0` to `num_steps - 1`.

    Args:
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "priors".

    Returns:
      prior: A `Categorical` distribution object representing the
        prior probability of the hidden Markov model being in each state at
        each step. The rightmost dimension of the `Categorical` distributions
        batch will equal the `num_steps` parameter providing one prior
        distribution for each step.
    """
    with self._name_and_control_scope(name):
      forward_log_probs, _ = self._forward_pass(observations=None, mask=None)
      forward_log_probs = distribution_util.move_dimension(
          forward_log_probs, 0, -2)
    return categorical.Categorical(logits=forward_log_probs)

  def posterior_marginals(self, observations, mask=None,
                          name='posterior_marginals'):
    """Compute marginal posterior distribution for each state.

    This function computes, for each time step, the marginal conditional
    probability that the hidden Markov model was in each possible state given
    the observations that were made at each time step.

    So if the hidden states are `z[0],...,z[num_steps - 1]` and
    the observations are `x[0], ..., x[num_steps - 1]`, then
    this function computes `P(z[i] | x[0], ..., x[num_steps - 1])`
    for all `i` from `0` to `num_steps - 1`.

    This operation is sometimes called smoothing. It uses a form
    of the forward-backward algorithm.

    Note: the behavior of this function is undefined if the
    `observations` argument represents impossible observations
    from the model.

    Args:
      observations: A tensor representing a batch of observations made on the
        hidden Markov model.  The rightmost dimensions of this tensor correspond
        to the dimensions of the observation distributions of the underlying
        Markov chain, if the observations are non-scalar.  The next dimension
        from the right indexes the steps in a sequence of observations from a
        single sample from the hidden Markov model.  The size of this dimension
        should match the `num_steps` parameter of the hidden Markov model
        object.  The other dimensions are the dimensions of the batch and these
        are broadcast with the hidden Markov model's parameters.
      mask: optional bool-type `tensor` with rightmost dimension matching
        `num_steps` indicating which observations the result of this
        function should be conditioned on. When the mask has value
        `True` the corresponding observations aren't used.
        if `mask` is `None` then all of the observations are used.
        the `mask` dimensions left of the last are broadcast with the
        hmm batch as well as with the observations.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "HiddenMarkovModel".

    Returns:
      posterior_marginal: A `Categorical` distribution object representing the
        marginal probability of the hidden Markov model being in each state at
        each step. The rightmost dimension of the `Categorical` distributions
        batch will equal the `num_steps` parameter providing one marginal
        distribution for each step. The other dimensions are the dimensions
        corresponding to the batch of observations.

    Raises:
      ValueError: if rightmost dimension of `observations` does not
      have size `num_steps`.
    """

    with self._name_and_control_scope(name):
      num_states = self.transition_distribution.batch_shape_tensor()[-1]
      log_transition = _extract_log_probs(
          num_states,
          self.transition_distribution,
          move_time_dimension=self._time_varying_transition_distribution)

      forward_log_probs, observation_log_probs = self._forward_pass(
          observations, mask, log_transition)
      total_log_prob = tf.reduce_logsumexp(forward_log_probs[-1], axis=-1)
      backward_log_adjoint_probs = self._backward_pass(observation_log_probs,
                                                       log_transition)
      log_likelihoods = forward_log_probs + backward_log_adjoint_probs

      marginal_log_probs = distribution_util.move_dimension(
          log_likelihoods - total_log_prob[..., tf.newaxis], 0, -2)

      return categorical.Categorical(logits=marginal_log_probs)

  def _forward_pass(self, observations=None, mask=None, log_transition=None):
    """Computes forward pass.

    When observations `x[:T]` are given, returns the unnormalized filtering
      distribution `C(t) * p(z[t]| x[:t])` where `C(t)=p(x[:t])`.
    When observations are not given, returns prior probabilities `p(z[t]).`

    Args:
      observations: optional Tensor of observations.
      mask: optional Tensor of mask.
      log_transition: optional Tensor of log probabilities
        from the transition matrix with shape
        `[num_steps] + batch_shape + [num_states, num_states]`.
        If not specified, will be recomputed from self.transition_distribution.

    Returns:
      forward_log_probs: a Tensor of log probabilities with shape
        `[num_steps] + batch_shape + [num_states]`.
      observation_log_probs: a Tensor containing `log p(x[t])` with shape
        `[num_steps] + batch_shape`.
    """
    num_states = self.transition_distribution.batch_shape_tensor()[-1]
    log_init = _extract_log_probs(num_states,
                                  self.initial_distribution,
                                  move_time_dimension=False)
    if log_transition is None:
      log_transition = _extract_log_probs(
          num_states,
          self.transition_distribution,
          move_time_dimension=self._time_varying_transition_distribution)

    if observations is None:
      observation_log_probs = tf.zeros(
          ps.concat(
              [[self.num_steps], self.batch_shape_tensor(), [num_states]],
              axis=0),
          dtype=log_init.dtype)
    else:
      observation_tensor_shape = ps.shape(observations)
      observation_distribution = self.observation_distribution
      underlying_event_rank = ps.size(
          observation_distribution.event_shape_tensor())
      mask_tensor_shape = ps.shape(mask) if mask is not None else None
      with self._observation_mask_shape_preconditions(observation_tensor_shape,
                                                      mask_tensor_shape,
                                                      underlying_event_rank):
        observation_log_probs = self._observation_log_probs(observations, mask)

    log_prob = log_init + observation_log_probs[0]

    def _scan_multiple_steps_forwards():
      if self._time_varying_transition_distribution:
        def dynamic_forward_step(log_prev_step,
                                 log_transition_and_observation):
          log_transition, log_prob_observation = (
              log_transition_and_observation)
          result = _log_vector_matrix(log_prev_step,
                                      log_transition) + log_prob_observation
          tensorshape_util.set_shape(result, log_prob_observation.shape)
          return result
        forward_log_probs = tf.scan(
            dynamic_forward_step,
            (log_transition, observation_log_probs[1:]),
            initializer=log_prob)
      else:
        def forward_step(log_previous_step, log_prob_observation):
          result = _log_vector_matrix(log_previous_step,
                                      log_transition) + log_prob_observation
          tensorshape_util.set_shape(result, log_prob_observation.shape)
          return result

        forward_log_probs = tf.scan(forward_step, observation_log_probs[1:],
                                    initializer=log_prob,
                                    name='forward_log_probs')
      return ps.concat([[log_prob], forward_log_probs], axis=0)

    def _scan_one_step_forwards():
      multiples = ps.concat([[self.num_steps],
                             ps.ones(ps.rank(log_prob), tf.int32)],
                            axis=0)
      return tf.tile(log_prob[tf.newaxis], multiples)

    forward_log_probs = ps.cond(
        self._num_steps > 1,
        _scan_multiple_steps_forwards,
        _scan_one_step_forwards)
    return forward_log_probs, observation_log_probs

  def _backward_pass(self, observation_log_probs, log_transition=None):
    """Computes backward pass.

    Given observations `x[:T]` and hidden states `z[:T]`,
    returns `p(x[t+1:T]|z[t])`.

    Args:
      observation_log_probs: Tensor representing observation `log p(x[t])` with
        shape `[num_steps] + batch_shape`.
      log_transition: optional Tensor of log probabilities
        from the transition matrix with shape:
        `[num_steps] + batch_shape + [num_states, num_states]`.
        If not specified, will be recomputed from self.transition_distribution.

    Returns:
      Tensor representing the backward log probability `p(x[t+1:T]|z[t])` with
      shape `[num_steps] + batch_shape + [num_states]`.
    """
    log_adjoint_prob = tf.zeros_like(observation_log_probs[0])

    if log_transition is None:
      num_states = self.transition_distribution.batch_shape_tensor()[-1]
      log_transition = _extract_log_probs(
          num_states,
          self.transition_distribution,
          move_time_dimension=self._time_varying_transition_distribution)

    def _scan_multiple_steps_backwards():
      """Perform `scan` operation when `num_steps` > 1."""

      if self._time_varying_transition_distribution:
        def dynamic_backward_step(log_previous_step,
                                  log_transition_and_observation):
          log_transition, log_prob_observation = (
              log_transition_and_observation)
          return _log_matrix_vector(
              log_transition,
              log_prob_observation + log_previous_step)

        backward_log_adjoint_probs = tf.scan(
            dynamic_backward_step,
            (log_transition, observation_log_probs[1:]),
            initializer=log_adjoint_prob,
            reverse=True,
            name='backward_log_adjoint_probs')
      else:
        def backward_step(log_previous_step, log_prob_observation):
          return _log_matrix_vector(
              log_transition,
              log_prob_observation + log_previous_step)

        backward_log_adjoint_probs = tf.scan(
            backward_step,
            observation_log_probs[1:],
            initializer=log_adjoint_prob,
            reverse=True,
            name='backward_log_adjoint_probs')
      return tf.concat([backward_log_adjoint_probs,
                        [log_adjoint_prob]], axis=0)

    def _scan_one_step_backwards():
      multiples = ps.concat([[self.num_steps],
                             ps.ones(ps.rank(log_adjoint_prob), tf.int32)],
                            axis=0)
      return tf.tile(log_adjoint_prob[tf.newaxis], multiples)

    backward_log_adjoint_probs = ps.cond(
        self._num_steps > 1,
        _scan_multiple_steps_backwards,
        _scan_one_step_backwards)
    return backward_log_adjoint_probs

  def posterior_mode(self, observations, mask=None, name='posterior_mode'):
    """Compute maximum likelihood sequence of hidden states.

    When this function is provided with a sequence of observations
    `x[0], ..., x[num_steps - 1]`, it returns the sequence of hidden
    states `z[0], ..., z[num_steps - 1]`, drawn from the underlying
    Markov chain, that is most likely to yield those observations.

    It uses the [Viterbi algorithm](
    https://en.wikipedia.org/wiki/Viterbi_algorithm).

    Note: the behavior of this function is undefined if the
    `observations` argument represents impossible observations
    from the model.

    Note: if there isn't a unique most likely sequence then one
    of the equally most likely sequences is chosen.

    Args:
      observations: A tensor representing a batch of observations made on the
        hidden Markov model.  The rightmost dimensions of this tensor correspond
        to the dimensions of the observation distributions of the underlying
        Markov chain, if the observations are non-scalar.  The next dimension
        from the right indexes the steps in a sequence of observations from a
        single sample from the hidden Markov model.  The size of this dimension
        should match the `num_steps` parameter of the hidden Markov model
        object.  The other dimensions are the dimensions of the batch and these
        are broadcast with the hidden Markov model's parameters.
      mask: optional bool-type `tensor` with rightmost dimension matching
        `num_steps` indicating which observations the result of this
        function should be conditioned on. When the mask has value
        `True` the corresponding observations aren't used.
        if `mask` is `None` then all of the observations are used.
        the `mask` dimensions left of the last are broadcast with the
        hmm batch as well as with the observations.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "HiddenMarkovModel".

    Returns:
      posterior_mode: A `Tensor` representing the most likely sequence of hidden
        states. The rightmost dimension of this tensor will equal the
        `num_steps` parameter providing one hidden state for each step. The
        other dimensions are those of the batch.

    Raises:
      ValueError: if the `observations` tensor does not consist of
      sequences of `num_steps` observations.

    #### Examples

    ```python
    tfd = tfp.distributions

    # A simple weather model.

    # Represent a cold day with 0 and a hot day with 1.
    # Suppose the first day of a sequence has a 0.8 chance of being cold.

    initial_distribution = tfd.Categorical(probs=[0.8, 0.2])

    # Suppose a cold day has a 30% chance of being followed by a hot day
    # and a hot day has a 20% chance of being followed by a cold day.

    transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                     [0.2, 0.8]])

    # Suppose additionally that on each day the temperature is
    # normally distributed with mean and standard deviation 0 and 5 on
    # a cold day and mean and standard deviation 15 and 10 on a hot day.

    observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

    # This gives the hidden Markov model:

    model = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=7)

    # Suppose we observe gradually rising temperatures over a week:
    temps = [-2., 0., 2., 4., 6., 8., 10.]

    # We can now compute the most probable sequence of hidden states:

    model.posterior_mode(temps)

    # The result is [0 0 0 0 0 1 1] telling us that the transition
    # from "cold" to "hot" most likely happened between the
    # 5th and 6th days.
    ```
    """

    with self._name_and_control_scope(name):
      observations = tf.convert_to_tensor(observations, name='observations')
      if mask is not None:
        mask = tf.convert_to_tensor(mask, name='mask', dtype_hint=tf.bool)
      num_states = self.transition_distribution.batch_shape_tensor()[-1]
      observation_distribution = self.observation_distribution
      underlying_event_rank = ps.size(
          observation_distribution.event_shape_tensor())
      observation_tensor_shape = ps.shape(observations)
      mask_tensor_shape = ps.shape(mask) if mask is not None else None
      with self._observation_mask_shape_preconditions(observation_tensor_shape,
                                                      mask_tensor_shape,
                                                      underlying_event_rank):
        observation_log_probs = self._observation_log_probs(observations, mask)
        log_init = _extract_log_probs(
            num_states, self.initial_distribution, move_time_dimension=False)
        log_trans = _extract_log_probs(
            num_states, self.transition_distribution,
            move_time_dimension=self._time_varying_transition_distribution)

        log_prob = log_init + observation_log_probs[0]

        def _reduce_multiple_steps():
          """Perform `reduce_max` operation when `num_steps` > 1."""

          if self._time_varying_transition_distribution:
            def dynamic_forward_step(previous_step_pair, log_trans_and_obs):
              log_trans, log_prob_observation = log_trans_and_obs
              log_prob_previous = previous_step_pair[0]
              log_prob = (log_prob_previous[..., tf.newaxis] +
                          log_trans +
                          log_prob_observation[..., tf.newaxis, :])
              most_likely_given_successor = tf.argmax(log_prob, axis=-2)
              max_log_p_given_successor = tf.reduce_max(log_prob,
                                                        axis=-2)
              return (max_log_p_given_successor, most_likely_given_successor)

            forward_log_probs, all_most_likely_given_successor = tf.scan(
                dynamic_forward_step,
                (log_trans, observation_log_probs[1:]),
                initializer=(log_prob,
                             tf.zeros(ps.shape(log_prob),
                                      dtype=tf.int64)),
                name='forward_log_probs')
          else:
            def forward_step(previous_step_pair, log_prob_observation):
              log_prob_previous = previous_step_pair[0]
              log_prob = (log_prob_previous[..., tf.newaxis] +
                          log_trans +
                          log_prob_observation[..., tf.newaxis, :])
              most_likely_given_successor = tf.argmax(log_prob, axis=-2)
              max_log_p_given_successor = tf.reduce_max(log_prob,
                                                        axis=-2)
              return (max_log_p_given_successor, most_likely_given_successor)

            forward_log_probs, all_most_likely_given_successor = tf.scan(
                forward_step,
                observation_log_probs[1:],
                initializer=(log_prob,
                             tf.zeros(ps.shape(log_prob),
                                      dtype=tf.int64)),
                name='forward_log_probs')

          most_likely_end = tf.argmax(forward_log_probs[-1], axis=-1)

          # We require the operation that gives C from A and B where
          # C[i...j] = A[i...j, B[i...j]]
          # and A = most_likely_given_successor
          #     B = most_likely_successor.
          # tf.gather requires indices of known shape so instead we use
          # reduction with tf.one_hot(B) to pick out elements from B
          def backward_step(most_likely_successor,
                            most_likely_given_successor):
            return tf.reduce_sum(
                (most_likely_given_successor *
                 tf.one_hot(most_likely_successor,
                            num_states,
                            dtype=tf.int64)),
                axis=-1)

          backward_scan = tf.scan(
              backward_step,
              all_most_likely_given_successor,
              most_likely_end,
              reverse=True)
          most_likely_sequences = tf.concat([backward_scan,
                                             [most_likely_end]],
                                            axis=0)
          return distribution_util.move_dimension(
              most_likely_sequences, 0, -1)

        def _reduce_one_step():
          multiples = ps.concat(
              [ps.ones(ps.rank(log_prob) - 1, tf.int32), [self.num_steps]],
              axis=0)
          return tf.tile(
              tf.argmax(log_prob, axis=-1)[..., tf.newaxis], multiples)

        return ps.cond(self.num_steps > 1, _reduce_multiple_steps,
                       _reduce_one_step)

  # pylint: disable=protected-access
  def _default_event_space_bijector(self):
    return (self._observation_distribution.
            _experimental_default_event_space_bijector())
  # pylint: enable=protected-access

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    # Check num_steps is a scalar that's at least 1.
    if is_init != tensor_util.is_ref(self.num_steps):
      num_steps = tf.convert_to_tensor(self.num_steps)
      num_steps_ = tf.get_static_value(num_steps)
      if num_steps_ is not None:
        if np.ndim(num_steps_) != 0:
          raise ValueError(
              '`num_steps` must be a scalar but it has rank {}'.format(
                  np.ndim(num_steps_)))
        if num_steps_ < 1:
          raise ValueError('`num_steps` must be at least 1.')
      elif self.validate_args:
        message = '`num_steps` must be a scalar'
        assertions.append(
            assert_util.assert_rank_at_most(self.num_steps, 0, message=message))
        assertions.append(
            assert_util.assert_greater_equal(
                num_steps, 1,
                message='`num_steps` must be at least 1.'))

    # Check that the initial distribution has scalar events over the
    # integers.
    if is_init and not dtype_util.is_integer(self.initial_distribution.dtype):
      raise ValueError(
          '`initial_distribution.dtype` ({}) is not over integers'.format(
              dtype_util.name(self.initial_distribution.dtype)))

    if tensorshape_util.rank(self.initial_distribution.event_shape) is not None:
      if tensorshape_util.rank(self.initial_distribution.event_shape) != 0:
        raise ValueError('`initial_distribution` must have scalar `event_dim`s')
    elif self.validate_args:
      assertions += [
          assert_util.assert_equal(
              ps.size(self.initial_distribution.event_shape_tensor()),
              0,
              message='`initial_distribution` must have scalar `event_dim`s'),
      ]

    # Check that the transition distribution is over the integers.
    if (is_init and
        not dtype_util.is_integer(self.transition_distribution.dtype)):
      raise ValueError(
          '`transition_distribution.dtype` ({}) is not over integers'.format(
              dtype_util.name(self.transition_distribution.dtype)))

    # Check observations have non-scalar batches.
    # The graph version of this assertion is incorporated as
    # a control dependency of the transition/observation
    # compatibility test.
    if tensorshape_util.rank(self.observation_distribution.batch_shape) == 0:
      raise ValueError(
          "`observation_distribution` can't have scalar batches")

    # Check transitions have non-scalar batches.
    # The graph version of this assertion is incorporated as
    # a control dependency of the transition/observation
    # compatibility test.
    if tensorshape_util.rank(self.transition_distribution.batch_shape) == 0:
      raise ValueError(
          "`transition_distribution` can't have scalar batches")

    # Check compatibility of transition distribution and observation
    # distribution.
    tdbs = self.transition_distribution.batch_shape
    odbs = self.observation_distribution.batch_shape
    if (tensorshape_util.dims(tdbs) is not None and
        tf.compat.dimension_value(odbs[-1]) is not None):
      if (tf.compat.dimension_value(tdbs[-1]) !=
          tf.compat.dimension_value(odbs[-1])):
        raise ValueError(
            '`transition_distribution` and `observation_distribution` '
            'must agree on last dimension of batch size')
      static_num_steps = tf.compat.dimension_value(self._static_event_shape[0])
      if (self._time_varying_observation_distribution and
          static_num_steps is not None and
          tf.compat.dimension_value(odbs[-2]) is not None and
          tf.compat.dimension_value(odbs[-2]) != static_num_steps):
        raise ValueError(
            'A time-varying `observation_distribution` must have a first '
            'batch dimension that matches num_steps.')
      if (self._time_varying_transition_distribution and
          static_num_steps is not None and
          tf.compat.dimension_value(tdbs[-2]) is not None and
          tf.compat.dimension_value(tdbs[-2]) != (static_num_steps - 1)):
        raise ValueError(
            'A time-varying `transition_distribution` must have a first '
            'batch dimension that matches num_steps - 1.')
    elif self.validate_args:
      tdbs = self.transition_distribution.batch_shape_tensor()
      odbs = self.observation_distribution.batch_shape_tensor()
      preconditions = []
      preconditions.append(assert_util.assert_greater(
          ps.size(tdbs), 0,
          message=('`transition_distribution` can\'t have scalar '
                   'batches')))
      preconditions.append(assert_util.assert_greater(
          ps.size(odbs), 0,
          message=('`observation_distribution` can\'t have scalar '
                   'batches')))
      if self._time_varying_transition_distribution:
        preconditions.append(assert_util.assert_equal(
            tdbs[-2], tf.convert_to_tensor(self.num_steps - 1),
            message=('A time-varying `transition_distribution` must have a '
                     'first batch dimension that matches num_steps - 1.')))
      if self._time_varying_observation_distribution:
        preconditions.append(assert_util.assert_equal(
            odbs[-2], tf.convert_to_tensor(self.num_steps),
            message=('A time-varying `observation_distribution` must have a '
                     'first batch dimension that matches num_steps.')))
      with tf.control_dependencies(preconditions):
        assertions += [
            assert_util.assert_equal(
                tdbs[-1],
                odbs[-1],
                message=('`transition_distribution` and '
                         '`observation_distribution` '
                         'must agree on last dimension of batch size'))]

    return assertions


def _log_vector_matrix(vs, ms):
  """Multiply tensor of vectors by matrices assuming values stored are logs."""

  return tf.reduce_logsumexp(vs[..., tf.newaxis] + ms, axis=-2)


def _log_matrix_vector(ms, vs):
  """Multiply tensor of matrices by vectors assuming values stored are logs."""

  return tf.reduce_logsumexp(ms + vs[..., tf.newaxis, :], axis=-1)


def _vector_matrix(vs, ms):
  """Multiply tensor of vectors by matrices."""

  return tf.reduce_sum(vs[..., tf.newaxis] * ms, axis=-2)


def _extract_log_probs(num_states, dist, move_time_dimension=False):
  """Tabulate log probabilities from a batch of distributions.

  Args:
    num_states: Number of states for the hidden variable.
    dist: distribution to get log probabilities from.
    move_time_dimension: optional boolean.
      if True, moves the time dimension to the first dimension.
      Note that the function assumes that the second rightmost dimension
      in dist.batch_shape is the time dimension.

  Returns:
    log_probabilities tf.Tensor with shape:
      dist.batch_shape + [num_states] if move_time_dimension is False.
      [num_steps] + dist.batch_shape + [num_states] otherwise.
  """

  states = tf.reshape(tf.range(num_states),
                      ps.concat([[num_states],
                                 ps.ones_like(dist.batch_shape_tensor())],
                                axis=0))
  log_probabilities = distribution_util.move_dimension(
      dist.log_prob(states), 0, -1)

  if move_time_dimension:
    log_probabilities = distribution_util.move_dimension(
        log_probabilities, -3, 0)
  return log_probabilities
