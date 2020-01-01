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
"""Implementation of the NeuTra Kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import dtype_util

tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = ['NeuTra', 'make_iaf_stack']


def make_iaf_stack(total_event_size,
                   num_hidden_layers=2,
                   seed=None,
                   dtype=tf.float32):
  """Creates an stacked IAF bijector.

  This bijector operates on vector-valued events.

  Args:
    total_event_size: Number of dimensions to operate over.
    num_hidden_layers: How many hidden layers to use in each IAF.
    seed: Random seed for the initializers.
    dtype: DType for the variables.

  Returns:
    bijector: The created bijector.
  """

  seed = tfp.util.SeedStream(seed, 'make_iaf_stack')

  def make_iaf():
    """Create an IAF."""
    initializer = tf.compat.v2.keras.initializers.VarianceScaling(
        2 * 0.01, seed=seed() % (2**31 - 1))

    made = tfb.AutoregressiveNetwork(
        params=2,
        event_shape=[total_event_size],
        hidden_units=[total_event_size] * num_hidden_layers,
        activation=tf.nn.elu,
        kernel_initializer=initializer,
        dtype=dtype)

    def shift_and_scale(x):
      # TODO(siege): Something is losing the static shape.
      x.set_shape(
          x.shape.merge_with([None] * (x.shape.ndims - 1) + [total_event_size]))
      return tf.unstack(made(x), num=2, axis=-1)

    return tfb.Invert(tfb.MaskedAutoregressiveFlow(shift_and_scale))

  def make_swap():
    """Create an swap."""
    permutation = list(reversed(range(total_event_size)))
    return tfb.Permute(permutation)

  bijector = make_iaf()
  bijector = make_swap()(bijector)
  bijector = make_iaf()(bijector)
  bijector = make_swap()(bijector)
  bijector = make_iaf()(bijector)
  bijector = make_swap()(bijector)

  return bijector


class NeuTra(tfp.mcmc.TransitionKernel):
  """The experimental NeuTra kernel.

  Warning: This kernel is experimental. Default arguments and their
  interpreations will very likely change.

  This kernel does not work in TF1 graph mode.

  This implements a transition kernel that implements the NeuTra MCMC
  [(Hoffman et al., 2019)][1]. It operates by learning a neural pre-conditioner
  (bijector) inside the `bootstrap_results` method, and then using it inside
  `one_step` to accelerate HMC. The same bijector is used to initialize the
  state: i.e. the `current_state` as passed to `tfp.mcmc.sample_chain` is
  ignored except to extract the number of chains to sample in parallel.

  This kernel performs step-size adaptation and picks an automatic trajectory
  length for its internal HMC kernel.

  If your problem has constrained support, specify this support via the
  `unconstraining_bijector` argument. This argument is interpreted such that the
  image of the forward transformation of that bijector matches the support of
  your problem. E.g. if one of your random variables is positive, you could use
  the `tfb.Softplus` bijector.

  Since this is still experimental, we provide some facilities to debug the
  bijector training via the `train_debug_fn`, where you can monitor the training
  progress. In practice, it may be prudent to run the training process multiple
  times to verify that the variational approximation is stable. If it is not,
  you can attempt to increase the expressiveness of the bijector via the
  `trainable_bijector_fn` argument. Currently the bijector operates by
  flattening the entire problem state into one vector. This is seamless to the
  user, but is important to remember when designing a bijector. The flattening
  will likely be removed in the future.

  `_flattened_variational_distribution` returns the (flattened) distribution
  resulting from a standard normal being pushed through the bijector. This is
  useful for debugging, as this can be compared to the MCMC chain and your prior
  preconception of what the target distribution should look like.

  Additionally, you can examine the `inner_results.transformed_state` inside the
  kernel results. By construction, for a well-fitted bijector this should
  resemble a standard normal. This kernel assumes that you have a well-fitted
  bijector, so if samples in the transformed space do not look normal, then this
  kernel is not operating efficiently.

  ### Examples

  Sampling from a multivariate log-normal distribution.

  ```python

  target_dist = tfd.MultivariateNormalTriL(scale_tril=[[1., 0.], [2, 1.]])
  target_dist = tfb.Exp()(target_dist)

  num_chains = 64
  state_shape = 2
  num_burnin_steps = 1000

  kernel = NeuTra(
      target_log_prob_fn=target_dist.log_prob,
      state_shape=state_shape,
      num_step_size_adaptation_steps=int(0.8 * num_burnin_steps),
      unconstraining_bijector=tfb.Exp())

  chain = tfp.mcmc.sample_chain(
      num_results=1000,
      num_burnin_steps=1000,
      current_state=tf.ones([num_chains, 2]),
      kernel=kernel,
      trace_fn=None)
  ```

  #### References

  [1]: Hoffman, M., Sountsov, P., Dillon, J. V., Langmore, I., Tran, D., &
       Vasudevan, S. (2019). NeuTra-lizing Bad Geometry in Hamiltonian Monte
       Carlo Using Neural Transport. http://arxiv.org/abs/1903.03704
  """

  def __init__(self,
               target_log_prob_fn,
               state_shape,
               num_step_size_adaptation_steps,
               unconstraining_bijector=None,
               trainable_bijector_fn=make_iaf_stack,
               learning_rate=1e-2,
               train_batch_size=4096,
               num_train_steps=5000,
               train_debug_fn=None,
               seed=None):
    """Creates the kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      state_shape: A Python list, or list of Python lists (or `TensorShape`s)
        that describes the shape of the problem state. Must be the same
        structure as `current_state`. All shapes must be fully defined.
      num_step_size_adaptation_steps: Number of steps to use for step size
        adaptation. See `tfp.mcmc.SimpleStepSizeAdaptation`.
      unconstraining_bijector: A bijector or list of bijectors that go from
        unconstrained space to the support of the corresponding random variable.
        Must be the same structure as `current_state`.
      trainable_bijector_fn: Creates a trainable, vector-valued event bijector.
        Must be a callable with signature: `(total_event_size, seed) ->
          bijector`, where `total_event_size` is the size of the event.
      learning_rate: Base learning rate to use for training the bijector.
        Internally, learning rate decay is used to stabilize learning.
      train_batch_size: Batch size to use for training the bijector.
      num_train_steps: Number of training steps to train the bijector.
      train_debug_fn: A callable with signature `(NeuTra, step, loss)` called
        for every training step. The first argument is this instance, and `step`
        is the current training step.
      seed: A seed for reproducibility.
    """
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn,
        state_shape=state_shape,
        num_step_size_adaptation_steps=num_step_size_adaptation_steps,
        unconstraining_bijector=unconstraining_bijector,
        trainable_bijector_fn=trainable_bijector_fn,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        train_debug_fn=train_debug_fn,
        seed=seed)

    self._state_shape = tf.nest.map_structure(tf.TensorShape, state_shape)

    if unconstraining_bijector is None:
      unconstraining_bijector = tf.nest.map_structure(lambda _: tfb.Identity(),
                                                      self.state_shape)
    self._unconstraining_bijector = unconstraining_bijector

    def _make_reshaped_bijector(b, s):
      return tfb.Reshape(
          event_shape_in=s, event_shape_out=[s.num_elements()])(b)(
              tfb.Reshape(event_shape_out=b.inverse_event_shape(s)))

    # This converts the `unconstraining_bijector` to work on flattened state
    # parts.
    reshaped_bijector = tf.nest.map_structure(_make_reshaped_bijector,
                                              unconstraining_bijector,
                                              self.state_shape)

    blockwise_bijector = tfb.Blockwise(
        bijectors=tf.nest.flatten(reshaped_bijector),
        block_sizes=tf.nest.flatten(
            tf.nest.map_structure(lambda s: s.num_elements(),
                                  self.state_shape)))

    dtype = dtype_util.common_dtype([learning_rate], dtype_hint=tf.float32)
    self._dtype = dtype

    trainable_bijector = trainable_bijector_fn(
        self._total_event_size, seed=seed, dtype=dtype)
    self._trainable_bijector = trainable_bijector

    self._bijector = blockwise_bijector(trainable_bijector)

    def flattened_target_log_prob(flat_state):
      state = self._unflatten_state(flat_state)
      if isinstance(state, (list, tuple)):
        return target_log_prob_fn(*state)
      else:
        return target_log_prob_fn(state)

    self._flattened_target_log_prob_val = flattened_target_log_prob

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=flattened_target_log_prob,
        step_size=1.,
        num_leapfrog_steps=self._num_leapfrog_steps(1.),
        seed=seed if not tf.executing_eagerly() else None)
    kernel = tfp.mcmc.TransformedTransitionKernel(kernel, self._bijector)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=num_step_size_adaptation_steps,
        target_accept_prob=np.array(0.9, dtype.as_numpy_dtype))
    self._kernel = kernel

  @property
  def trainable_bijector(self):
    return self._trainable_bijector

  @property
  def unconstraining_bijector(self):
    return self._unconstraining_bijector

  @property
  def state_shape(self):
    return self._state_shape

  @property
  def train_batch_size(self):
    return self._parameters['train_batch_size']

  @property
  def num_train_steps(self):
    return self._parameters['num_train_steps']

  @property
  def learning_rate(self):
    return self._parameters['learning_rate']

  @property
  def train_debug_fn(self):
    return self._parameters['train_debug_fn']

  @property
  def seed(self):
    return self._parameters['seed']

  @property
  def _total_event_size(self):
    return sum(
        tf.nest.flatten(
            tf.nest.map_structure(
                lambda b, s: b.inverse_event_shape(s).num_elements(),
                self.unconstraining_bijector, self.state_shape)))

  def _num_leapfrog_steps(self, step_size):
    step_size = tf.convert_to_tensor(value=step_size)
    trajectory_length = np.float32(self._total_event_size)**0.25
    return tf.cast(tf.math.ceil(trajectory_length / step_size), dtype=tf.int32)

  def _flatten_state(self, state):
    state_parts = tf.nest.flatten(state)
    flat_state_shapes = tf.nest.flatten(self.state_shape)
    batch_shape = tf.shape(input=state_parts[0])[:-flat_state_shapes[0].ndims]
    flat_shape = tf.concat([batch_shape, [-1]], -1)
    flat_state_parts = tf.nest.map_structure(
        lambda s: tf.reshape(s, flat_shape), state_parts)
    return tf.concat(flat_state_parts, -1)

  def _unflatten_state(self, flat_state):
    state_parts = tf.split(
        flat_state,
        [s.num_elements() for s in tf.nest.flatten(self.state_shape)],
        axis=-1)
    batch_shape = tf.shape(input=flat_state)[:-1]

    state = tf.nest.pack_sequence_as(self.state_shape, state_parts)
    return tf.nest.map_structure(
        lambda part, s: tf.reshape(part, tf.concat([batch_shape, s], 0)), state,
        self.state_shape)

  def is_calibrated(self):
    return True

  def _flattened_variational_distribution(self):
    base = tfd.MultivariateNormalDiag(
        loc=tf.zeros(self._total_event_size, dtype=self._dtype))
    return self._bijector(base)

  @property
  def _flattened_target_log_prob(self):
    return self._flattened_target_log_prob_val

  @tf.function(autograph=False)
  def one_step(self, current_state, previous_kernel_results):
    """Runs one iteration of NeuTra.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s). The first `r` dimensions index
        independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.
    """

    @tfp.mcmc.internal.util.make_innermost_setter
    def set_num_leapfrog_steps(kernel_results, num_leapfrog_steps):
      return kernel_results._replace(
          accepted_results=kernel_results.accepted_results._replace(
              num_leapfrog_steps=num_leapfrog_steps))

    step_size = previous_kernel_results.new_step_size
    previous_kernel_results = set_num_leapfrog_steps(
        previous_kernel_results, self._num_leapfrog_steps(step_size))

    new_state, kernel_results = self._kernel.one_step(
        self._flatten_state(current_state), previous_kernel_results)
    return self._unflatten_state(new_state), kernel_results

  def bootstrap_results(self, state):
    """Trains the bijector and creates initial `previous_kernel_results`.

    The supplied `state` is only used to determine the number of chains to run
    in parallel_iterations

    Args:
      state: `Tensor` or Python `list` of `Tensor`s representing the initial
        state(s) of the Markov chain(s). The first `r` dimensions index
        independent chains, `r = tf.rank(target_log_prob_fn(*state))`.

    Returns:
      kernel_results: Instance of
        `UncalibratedHamiltonianMonteCarloKernelResults` inside
        `MetropolisHastingsResults` inside `TransformedTransitionKernelResults`
        inside `SimpleStepSizeAdaptationResults`.
    """

    def loss():
      q = self._flattened_variational_distribution()
      # TODO(siege): How to seed this?
      samples = q.sample(self.train_batch_size)
      return tf.reduce_mean(
          input_tensor=q.log_prob(samples) -
          self._flattened_target_log_prob(samples),
          axis=-1)

    lr = tf.convert_to_tensor(value=self.learning_rate, dtype=self._dtype)
    dtype = lr.dtype

    learning_rate = tf.compat.v2.optimizers.schedules.PiecewiseConstantDecay(
        list(self.num_train_steps *
             np.array([0.2, 0.8]).astype(dtype.as_numpy_dtype)),
        [lr, lr * 0.1, lr * 0.01])

    opt = tf.compat.v2.optimizers.Adam(learning_rate)

    @tf.function(autograph=False)
    def train_step():
      with tf.GradientTape() as tape:
        loss_val = loss()
      vals = tape.watched_variables()
      grads = tape.gradient(loss_val, vals)
      grads_and_vals = list(zip(grads, vals))
      opt.apply_gradients(grads_and_vals)
      return loss_val

    for step in range(self.num_train_steps):
      loss_val = train_step()
      tf.debugging.assert_all_finite(
          loss_val, 'NeuTra loss is NaN at step {}'.format(step))
      if self.train_debug_fn:
        # pylint: disable=not-callable
        self.train_debug_fn(self, step, loss_val)

    state_parts = tf.nest.flatten(state)
    flat_state_shapes = tf.nest.flatten(self.state_shape)
    batch_shape = tf.shape(input=state_parts[0])[:-flat_state_shapes[0].ndims]

    return self._kernel.bootstrap_results(
        self._flattened_variational_distribution().sample(
            batch_shape, seed=self.seed))
