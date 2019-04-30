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

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    "HiddenMarkovModel",
]


class HiddenMarkovModel(distribution.Distribution):
  """Hidden Markov model distribution.

  The `HiddenMarkovModel` distribution implements a (batch of) hidden
  Markov models where the initial states, transition probabilities
  and observed states are all given by user-provided distributions.
  This model assumes that the transition matrices are fixed over time.

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
               name="HiddenMarkovModel"):
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
      num_steps: The number of steps taken in Markov chain. A python `int`.
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
      self._runtime_assertions = []  # pylint: enable=protected-access

      if num_steps < 1:
        raise ValueError("num_steps ({}) must be at least 1.".format(num_steps))

      self._initial_distribution = initial_distribution
      self._observation_distribution = observation_distribution
      self._transition_distribution = transition_distribution

      if (initial_distribution.event_shape is not None and
          tensorshape_util.rank(initial_distribution.event_shape) != 0):
        raise ValueError(
            "`initial_distribution` must have scalar `event_dim`s")
      elif validate_args:
        self._runtime_assertions += [
            assert_util.assert_equal(
                tf.shape(input=initial_distribution.event_shape_tensor())[0],
                0,
                message="`initial_distribution` must have scalar"
                "`event_dim`s")
        ]

      if (transition_distribution.event_shape is not None and
          tensorshape_util.rank(transition_distribution.event_shape) != 0):
        raise ValueError(
            "`transition_distribution` must have scalar `event_dim`s")
      elif validate_args:
        self._runtime_assertions += [
            assert_util.assert_equal(
                tf.shape(input=transition_distribution.event_shape_tensor())[0],
                0,
                message="`transition_distribution` must have scalar"
                "`event_dim`s")
        ]

      if (transition_distribution.batch_shape is not None and
          tensorshape_util.rank(transition_distribution.batch_shape) == 0):
        raise ValueError(
            "`transition_distribution` can't have scalar batches")
      elif validate_args:
        self._runtime_assertions += [
            assert_util.assert_greater(
                tf.size(input=transition_distribution.batch_shape_tensor()),
                0,
                message="`transition_distribution` can't have scalar "
                "batches")
        ]

      if (observation_distribution.batch_shape is not None and
          tensorshape_util.rank(observation_distribution.batch_shape) == 0):
        raise ValueError(
            "`observation_distribution` can't have scalar batches")
      elif validate_args:
        self._runtime_assertions += [
            assert_util.assert_greater(
                tf.size(input=observation_distribution.batch_shape_tensor()),
                0,
                message="`observation_distribution` can't have scalar "
                "batches")
        ]

      # Infer number of hidden states and check consistency
      # between transitions and observations
      with tf.control_dependencies(self._runtime_assertions):
        self._num_states = ((transition_distribution.batch_shape and
                             transition_distribution.batch_shape[-1]) or
                            transition_distribution.batch_shape_tensor()[-1])

        observation_states = ((observation_distribution.batch_shape and
                               observation_distribution.batch_shape[-1]) or
                              observation_distribution.batch_shape_tensor()[-1])

      if (tf.is_tensor(self._num_states) or tf.is_tensor(observation_states)):
        if validate_args:
          self._runtime_assertions += [
              assert_util.assert_equal(
                  self._num_states,
                  observation_states,
                  message="`transition_distribution` and "
                  "`observation_distribution` must agree on "
                  "last dimension of batch size")
          ]
      elif self._num_states != observation_states:
        raise ValueError("`transition_distribution` and "
                         "`observation_distribution` must agree on "
                         "last dimension of batch size")

      self._log_init = _extract_log_probs(self._num_states,
                                          initial_distribution)
      self._log_trans = _extract_log_probs(self._num_states,
                                           transition_distribution)

      self._num_steps = num_steps
      self._num_states = tf.shape(input=self._log_init)[-1]

      self._underlying_event_rank = tf.size(
          input=self._observation_distribution.event_shape_tensor())

      self.static_event_shape = tf.TensorShape(
          [num_steps]).concatenate(self._observation_distribution.event_shape)

      with tf.control_dependencies(self._runtime_assertions):
        self.static_batch_shape = tf.broadcast_static_shape(
            self._initial_distribution.batch_shape,
            tf.broadcast_static_shape(
                self._transition_distribution.batch_shape[:-1],
                self._observation_distribution.batch_shape[:-1]))

      # pylint: disable=protected-access
      super(HiddenMarkovModel, self).__init__(
          dtype=self._observation_distribution.dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=(self._initial_distribution._graph_parents +
                         self._transition_distribution._graph_parents +
                         self._observation_distribution._graph_parents),
          name=name)
      # pylint: enable=protected-access

      self._parameters = parameters

  def _batch_shape_tensor(self):
    with tf.control_dependencies(self._runtime_assertions):
      return tf.broadcast_dynamic_shape(
          self._initial_distribution.batch_shape_tensor(),
          tf.broadcast_dynamic_shape(
              self._transition_distribution.batch_shape_tensor()[:-1],
              self._observation_distribution.batch_shape_tensor()[:-1]))

  def _batch_shape(self):
    return self.static_batch_shape

  def _event_shape_tensor(self):
    with tf.control_dependencies(self._runtime_assertions):
      return tf.concat([[self._num_steps],
                        self.observation_distribution.event_shape_tensor()],
                       axis=0)

  def _event_shape(self):
    return self.static_event_shape

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
  def num_states(self):
    return self._num_states

  def _sample_n(self, n, seed=None):
    with tf.control_dependencies(self._runtime_assertions):
      seed = seed_stream.SeedStream(seed, salt="HiddenMarkovModel")

      num_states = self._num_states

      batch_shape = self.batch_shape_tensor()
      batch_size = tf.reduce_prod(input_tensor=batch_shape)

      # The batch sizes of the underlying initial distributions and
      # transition distributions might not match the batch size of
      # the HMM distribution.
      # As a result we need to ask for more samples from the
      # underlying distributions and then reshape the results into
      # the correct batch size for the HMM.
      init_repeat = (
          tf.reduce_prod(input_tensor=self.batch_shape_tensor()) //
          tf.reduce_prod(
              input_tensor=self._initial_distribution.batch_shape_tensor()))
      init_state = self._initial_distribution.sample(n * init_repeat,
                                                     seed=seed())
      init_state = tf.reshape(init_state, [n, batch_size])
      # init_state :: n batch_size

      transition_repeat = (
          tf.reduce_prod(input_tensor=self.batch_shape_tensor()) //
          tf.reduce_prod(input_tensor=self._transition_distribution
                         .batch_shape_tensor()[:-1]))

      def generate_step(state, _):
        """Take a single step in Markov chain."""

        gen = self._transition_distribution.sample(n * transition_repeat,
                                                   seed=seed())
        # gen :: (n * transition_repeat) transition_batch

        new_states = tf.reshape(gen,
                                [n, batch_size, num_states])

        # new_states :: n batch_size num_states

        old_states_one_hot = tf.one_hot(state, num_states, dtype=tf.int32)

        # old_states :: n batch_size num_states

        return tf.reduce_sum(
            input_tensor=old_states_one_hot * new_states, axis=-1)

      if self._num_steps > 1:
        dummy_index = tf.zeros(self._num_steps - 1, dtype=tf.float32)
        hidden_states = tf.scan(generate_step, dummy_index,
                                initializer=init_state)

        # TODO(b/115618503): add/use prepend_initializer to tf.scan
        hidden_states = tf.concat([[init_state],
                                   hidden_states], axis=0)
      else:
        hidden_states = init_state[tf.newaxis, ...]

      # hidden_states :: num_steps n batch_size num_states

      hidden_one_hot = tf.one_hot(hidden_states, num_states,
                                  dtype=self._observation_distribution.dtype)
      # hidden_one_hot :: num_steps n batch_size num_states

      # The observation distribution batch size might not match
      # the required batch size so as with the initial and
      # transition distributions we generate more samples and
      # reshape.
      observation_repeat = (
          batch_size //
          tf.reduce_prod(input_tensor=self._observation_distribution
                         .batch_shape_tensor()[:-1]))

      possible_observations = self._observation_distribution.sample(
          [self._num_steps, observation_repeat * n])

      inner_shape = self._observation_distribution.event_shape

      # possible_observations :: num_steps (observation_repeat * n)
      #                          observation_batch[:-1] num_states inner_shape

      possible_observations = tf.reshape(
          possible_observations,
          tf.concat([[self._num_steps, n],
                     batch_shape,
                     [num_states],
                     inner_shape], axis=0))

      # possible_observations :: steps n batch_size num_states inner_shape

      hidden_one_hot = tf.reshape(hidden_one_hot,
                                  tf.concat([[self._num_steps, n],
                                             batch_shape,
                                             [num_states],
                                             tf.ones_like(inner_shape)],
                                            axis=0))

      # hidden_one_hot :: steps n batch_size num_states "inner_shape"

      observations = tf.reduce_sum(
          input_tensor=hidden_one_hot * possible_observations,
          axis=-1 - tf.size(input=inner_shape))

      # observations :: steps n batch_size inner_shape

      observations = distribution_util.move_dimension(
          observations, 0, 1 + tf.size(input=batch_shape))

      # returned :: n batch_shape steps inner_shape

      return observations

  def _log_prob(self, value):
    with tf.control_dependencies(self._runtime_assertions):
      # The argument `value` is a tensor of sequences of observations.
      # `observation_batch_shape` is the shape of that tensor with the
      # sequence part removed.
      # `observation_batch_shape` is then broadcast to the full batch shape
      # to give the `batch_shape` that defines the shape of the result.

      observation_tensor_shape = tf.shape(input=value)
      observation_batch_shape = observation_tensor_shape[
          :-1 - self._underlying_event_rank]
      # value :: observation_batch_shape num_steps observation_event_shape
      batch_shape = tf.broadcast_dynamic_shape(observation_batch_shape,
                                               self.batch_shape_tensor())
      log_init = tf.broadcast_to(self._log_init,
                                 tf.concat([batch_shape,
                                            [self._num_states]], axis=0))
      # log_init :: batch_shape num_states
      log_transition = self._log_trans

      # `observation_event_shape` is the shape of each sequence of observations
      # emitted by the model.
      observation_event_shape = observation_tensor_shape[
          -1 - self._underlying_event_rank:]
      working_obs = tf.broadcast_to(value,
                                    tf.concat([batch_shape,
                                               observation_event_shape],
                                              axis=0))
      # working_obs :: batch_shape observation_event_shape
      r = self._underlying_event_rank

      # Move index into sequence of observations to front so we can apply
      # tf.foldl
      working_obs = distribution_util.move_dimension(working_obs, -1 - r,
                                                     0)[..., tf.newaxis]
      # working_obs :: num_steps batch_shape underlying_event_shape
      observation_probs = (
          self._observation_distribution.log_prob(working_obs))

      def forward_step(log_prev_step, log_prob_observation):
        return _log_vector_matrix(log_prev_step,
                                  log_transition) + log_prob_observation

      fwd_prob = tf.foldl(forward_step, observation_probs, initializer=log_init)
      # fwd_prob :: batch_shape num_states

      log_prob = tf.reduce_logsumexp(input_tensor=fwd_prob, axis=-1)
      # log_prob :: batch_shape

      return log_prob

  def _marginal_hidden_probs(self):
    """Compute marginal pdf for each individual observable."""

    initial_log_probs = tf.broadcast_to(self._log_init,
                                        tf.concat([self.batch_shape_tensor(),
                                                   [self._num_states]],
                                                  axis=0))
    # initial_log_probs :: batch_shape num_states

    if self._num_steps > 1:
      transition_log_probs = self._log_trans

      def forward_step(log_probs, _):
        return _log_vector_matrix(log_probs, transition_log_probs)

      dummy_index = tf.zeros(self._num_steps - 1, dtype=tf.float32)

      forward_log_probs = tf.scan(forward_step, dummy_index,
                                  initializer=initial_log_probs,
                                  name="forward_log_probs")

      forward_log_probs = tf.concat([[initial_log_probs], forward_log_probs],
                                    axis=0)
    else:
      forward_log_probs = initial_log_probs[tf.newaxis, ...]

    # returns :: num_steps batch_shape num_states

    return tf.exp(forward_log_probs)

  def _mean(self):
    with tf.control_dependencies(self._runtime_assertions):
      probs = self._marginal_hidden_probs()
      # probs :: num_steps batch_shape num_states
      means = self._observation_distribution.mean()
      # means :: observation_batch_shape[:-1] num_states
      #          observation_event_shape
      means_shape = tf.concat(
          [self.batch_shape_tensor(),
           [self._num_states],
           self._observation_distribution.event_shape_tensor()],
          axis=0)
      means = tf.broadcast_to(means, means_shape)
      # means :: batch_shape num_states observation_event_shape

      observation_event_shape = (
          self._observation_distribution.event_shape_tensor())
      batch_size = tf.reduce_prod(input_tensor=self.batch_shape_tensor())
      flat_probs_shape = [self._num_steps, batch_size, self._num_states]
      flat_means_shape = [
          batch_size, self._num_states,
          tf.reduce_prod(input_tensor=observation_event_shape)
      ]

      flat_probs = tf.reshape(probs, flat_probs_shape)
      # flat_probs :: num_steps batch_size num_states
      flat_means = tf.reshape(means, flat_means_shape)
      # flat_means :: batch_size num_states observation_event_size
      flat_mean = tf.einsum("ijk,jkl->jil", flat_probs, flat_means)
      # flat_mean :: batch_size num_steps observation_event_size
      unflat_mean_shape = tf.concat(
          [self.batch_shape_tensor(),
           [self._num_steps],
           observation_event_shape],
          axis=0)
      # returns :: batch_shape num_steps observation_event_shape
      return tf.reshape(flat_mean, unflat_mean_shape)

  def _variance(self):
    with tf.control_dependencies(self._runtime_assertions):
      probs = self._marginal_hidden_probs()
      # probs :: num_steps batch_shape num_states
      means = self._observation_distribution.mean()
      # means :: observation_batch_shape[:-1] num_states
      #          observation_event_shape
      means_shape = tf.concat(
          [self.batch_shape_tensor(),
           [self._num_states],
           self._observation_distribution.event_shape_tensor()],
          axis=0)
      means = tf.broadcast_to(means, means_shape)
      # means :: batch_shape num_states observation_event_shape

      observation_event_shape = (
          self._observation_distribution.event_shape_tensor())
      batch_size = tf.reduce_prod(input_tensor=self.batch_shape_tensor())
      flat_probs_shape = [self._num_steps, batch_size, self._num_states]
      flat_means_shape = [
          batch_size, 1, self._num_states,
          tf.reduce_prod(input_tensor=observation_event_shape)
      ]

      flat_probs = tf.reshape(probs, flat_probs_shape)
      # flat_probs :: num_steps batch_size num_states
      flat_means = tf.reshape(means, flat_means_shape)
      # flat_means :: batch_size 1 num_states observation_event_size
      flat_mean = tf.einsum("ijk,jmkl->jiml", flat_probs, flat_means)
      # flat_mean :: batch_size num_steps 1 observation_event_size

      variances = self._observation_distribution.variance()
      variances = tf.broadcast_to(variances, means_shape)
      # variances :: batch_shape num_states observation_event_shape
      flat_variances = tf.reshape(variances, flat_means_shape)
      # flat_variances :: batch_size 1 num_states observation_event_size

      # For a mixture of n distributions with mixture probabilities
      # p[i], and where the individual distributions have means and
      # variances given by mean[i] and var[i], the variance of
      # the mixture is given by:
      #
      # var = sum i=1..n p[i] * ((mean[i] - mean)**2 + var[i]**2)

      flat_variance = tf.einsum("ijk,jikl->jil",
                                flat_probs,
                                (flat_means - flat_mean)**2 + flat_variances)
      # flat_variance :: batch_size num_steps observation_event_size

      unflat_mean_shape = tf.concat(
          [self.batch_shape_tensor(),
           [self._num_steps],
           observation_event_shape],
          axis=0)

      # returns :: batch_shape num_steps observation_event_shape
      return tf.reshape(flat_variance, unflat_mean_shape)

  def _observation_shape_preconditions(self, observation_tensor_shape):
    return tf.control_dependencies([assert_util.assert_equal(
        observation_tensor_shape[-1 - self._underlying_event_rank],
        self._num_steps,
        message="The tensor `observations` must consist of sequences"
                "of observations from `HiddenMarkovModel` of length"
                "`num_steps`.")])

  def posterior_marginals(self, observations, name=None):
    """Compute marginal posterior distribution for each state.

    This function computes, for each time step, the marginal
    conditional probability that the hidden Markov model was in
    each possible state given the observations that were made
    at each time step.
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
      observations: A tensor representing a batch of observations
        made on the hidden Markov model.  The rightmost dimension of this tensor
        gives the steps in a sequence of observations from a single sample from
        the hidden Markov model. The size of this dimension should match the
        `num_steps` parameter of the hidden Markov model object. The other
        dimensions are the dimensions of the batch and these are broadcast with
        the hidden Markov model's parameters.
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

    with tf.name_scope(name or "posterior_marginals"):
      with tf.control_dependencies(self._runtime_assertions):
        observation_tensor_shape = tf.shape(input=observations)

        with self._observation_shape_preconditions(observation_tensor_shape):
          observation_batch_shape = observation_tensor_shape[
              :-1 - self._underlying_event_rank]
          observation_event_shape = observation_tensor_shape[
              -1 - self._underlying_event_rank:]

          batch_shape = tf.broadcast_dynamic_shape(observation_batch_shape,
                                                   self.batch_shape_tensor())
          log_init = tf.broadcast_to(self._log_init,
                                     tf.concat([batch_shape,
                                                [self._num_states]],
                                               axis=0))
          log_transition = self._log_trans

          observations = tf.broadcast_to(observations,
                                         tf.concat([batch_shape,
                                                    observation_event_shape],
                                                   axis=0))
          observation_rank = tf.rank(observations)
          underlying_event_rank = self._underlying_event_rank
          observations = distribution_util.move_dimension(
              observations, observation_rank - underlying_event_rank - 1, 0)
          observations = tf.expand_dims(
              observations,
              observation_rank - underlying_event_rank)
          observation_log_probs = self._observation_distribution.log_prob(
              observations)

          log_adjoint_prob = tf.zeros_like(log_init)

          def forward_step(log_previous_step, log_prob_observation):
            return _log_vector_matrix(log_previous_step,
                                      log_transition) + log_prob_observation

          log_prob = log_init + observation_log_probs[0]

          forward_log_probs = tf.scan(forward_step, observation_log_probs[1:],
                                      initializer=log_prob,
                                      name="forward_log_probs")

          forward_log_probs = tf.concat([[log_prob], forward_log_probs], axis=0)

          def backward_step(log_previous_step, log_prob_observation):
            return _log_matrix_vector(log_transition,
                                      log_prob_observation + log_previous_step)

          backward_log_adjoint_probs = tf.scan(
              backward_step,
              observation_log_probs[1:],
              initializer=log_adjoint_prob,
              reverse=True,
              name="backward_log_adjoint_probs")

          total_log_prob = tf.reduce_logsumexp(
              input_tensor=forward_log_probs[-1], axis=-1)

          backward_log_adjoint_probs = tf.concat([backward_log_adjoint_probs,
                                                  [log_adjoint_prob]], axis=0)

          log_likelihoods = forward_log_probs + backward_log_adjoint_probs

          marginal_log_probs = distribution_util.move_dimension(
              log_likelihoods - total_log_prob[..., tf.newaxis], 0, -2)

          return categorical.Categorical(logits=marginal_log_probs)

  def posterior_mode(self, observations, name=None):
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
        Markov chain.  The next dimension from the right indexes the steps in a
        sequence of observations from a single sample from the hidden Markov
        model.  The size of this dimension should match the `num_steps`
        parameter of the hidden Markov model object.  The other dimensions are
        the dimensions of the batch and these are broadcast with the hidden
        Markov model's parameters.
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

    with tf.name_scope(name or "posterior_mode"):
      with tf.control_dependencies(self._runtime_assertions):
        observation_tensor_shape = tf.shape(input=observations)

        with self._observation_shape_preconditions(observation_tensor_shape):
          observation_batch_shape = observation_tensor_shape[
              :-1 - self._underlying_event_rank]
          observation_event_shape = observation_tensor_shape[
              -1 - self._underlying_event_rank:]

          batch_shape = tf.broadcast_dynamic_shape(observation_batch_shape,
                                                   self.batch_shape_tensor())
          log_init = tf.broadcast_to(self._log_init,
                                     tf.concat([batch_shape,
                                                [self._num_states]],
                                               axis=0))

          observations = tf.broadcast_to(observations,
                                         tf.concat([batch_shape,
                                                    observation_event_shape],
                                                   axis=0))
          observation_rank = tf.rank(observations)
          underlying_event_rank = self._underlying_event_rank
          observations = distribution_util.move_dimension(
              observations, observation_rank - underlying_event_rank - 1, 0)

          # We need to compute the probability of each observation for
          # each possible state.
          # This requires inserting an extra index just before the
          # observation event indices that will be broadcast with the
          # last batch index in `observation_distribution`.
          observations = tf.expand_dims(
              observations,
              observation_rank - underlying_event_rank)
          observation_log_probs = self._observation_distribution.log_prob(
              observations)

          log_prob = log_init + observation_log_probs[0]

          if self._num_steps == 1:
            most_likely_end = tf.argmax(input=log_prob, axis=-1)
            return most_likely_end[..., tf.newaxis]

          def forward_step(previous_step_pair, log_prob_observation):
            log_prob_previous = previous_step_pair[0]
            log_prob = (log_prob_previous[..., tf.newaxis] +
                        self._log_trans +
                        log_prob_observation[..., tf.newaxis, :])
            most_likely_given_successor = tf.argmax(input=log_prob, axis=-2)
            max_log_p_given_successor = tf.reduce_max(input_tensor=log_prob,
                                                      axis=-2)
            return (max_log_p_given_successor, most_likely_given_successor)

          forward_log_probs, all_most_likely_given_successor = tf.scan(
              forward_step,
              observation_log_probs[1:],
              initializer=(log_prob,
                           tf.zeros(tf.shape(input=log_init), dtype=tf.int64)),
              name="forward_log_probs")

          most_likely_end = tf.argmax(input=forward_log_probs[-1], axis=-1)

          # We require the operation that gives C from A and B where
          # C[i...j] = A[i...j, B[i...j]]
          # and A = most_likely_given_successor
          #     B = most_likely_successor.
          # tf.gather requires indices of known shape so instead we use
          # reduction with tf.one_hot(B) to pick out elements from B
          def backward_step(most_likely_successor, most_likely_given_successor):
            return tf.reduce_sum(
                input_tensor=(most_likely_given_successor *
                              tf.one_hot(most_likely_successor,
                                         self._num_states,
                                         dtype=tf.int64)),
                axis=-1)

          backward_scan = tf.scan(
              backward_step,
              all_most_likely_given_successor,
              most_likely_end,
              reverse=True)
          most_likely_sequences = tf.concat([backward_scan, [most_likely_end]],
                                            axis=0)
          return distribution_util.move_dimension(most_likely_sequences, 0, -1)


def _log_vector_matrix(vs, ms):
  """Multiply tensor of vectors by matrices assuming values stored are logs."""

  return tf.reduce_logsumexp(input_tensor=vs[..., tf.newaxis] + ms, axis=-2)


def _log_matrix_vector(ms, vs):
  """Multiply tensor of matrices by vectors assuming values stored are logs."""

  return tf.reduce_logsumexp(input_tensor=ms + vs[..., tf.newaxis, :], axis=-1)


def _vector_matrix(vs, ms):
  """Multiply tensor of vectors by matrices."""

  return tf.reduce_sum(input_tensor=vs[..., tf.newaxis] * ms, axis=-2)


def _extract_log_probs(num_states, dist):
  """Tabulate log probabilities from a batch of distributions."""

  states = tf.reshape(tf.range(num_states),
                      tf.concat([[num_states],
                                 tf.ones_like(dist.batch_shape_tensor())],
                                axis=0))
  return distribution_util.move_dimension(dist.log_prob(states), 0, -1)
