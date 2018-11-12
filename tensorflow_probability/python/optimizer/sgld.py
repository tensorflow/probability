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
"""An optimizer module for stochastic gradient Langevin dynamics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_probability.python.math import diag_jacobian
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import training_ops


__all__ = [
    'StochasticGradientLangevinDynamics',
]


class StochasticGradientLangevinDynamics(tf.train.Optimizer):
  """An optimizer module for stochastic gradient Langevin dynamics.

  This implements the preconditioned Stochastic Gradient Langevin Dynamics
  optimizer [(Li et al., 2016)][1]. The optimization variable is regarded as a
  sample from the posterior under Stochastic Gradient Langevin Dynamics with
  noise rescaled in each dimension according to [RMSProp](
  http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

  Note: If a prior is included in the loss, it should be scaled by
  `1/data_size`, where `data_size` is the number of points in the data set.
  I.e., it should be divided by the `data_size` term described below.

  #### Examples

  ##### Optimizing energy of a 3D-Gaussian distribution

  This example demonstrates that for a fixed step size SGLD works as an
  approximate version of MALA (tfp.mcmc.MetropolisAdjustedLangevinAlgorithm).

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  import numpy as np

  tfd = tfp.distributions
  dtype = np.float32

  with tf.Session(graph=tf.Graph()) as sess:
    # Set up random seed for the optimizer
    tf.set_random_seed(42)
    true_mean = dtype([0, 0, 0])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
    # Loss is defined through the Cholesky decomposition
    chol = tf.linalg.cholesky(true_cov)
    var_1 = tf.get_variable(
        'var_1', initializer=[1., 1.])
    var_2 = tf.get_variable(
        'var_2', initializer=[1.])

    var = tf.concat([var_1, var_2], axis=-1)
    # Partially defined loss function
    loss_part = tf.cholesky_solve(chol, tf.expand_dims(var, -1))
    # Loss function
    loss = 0.5 * tf.linalg.matvec(loss_part, var, transpose_a=True)

    # Set up the learning rate with a polynomial decay
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = .3
    end_learning_rate = 1e-4
    decay_steps = 1e4
    learning_rate = tf.train.polynomial_decay(starter_learning_rate,
                                              global_step, decay_steps,
                                              end_learning_rate, power=1.)

    # Set up the optimizer
    optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(
        learning_rate=learning_rate, preconditioner_decay_rate=0.99)

    optimizer = optimizer_kernel.minimize(loss)

    init = tf.global_variables_initializer()
    # Number of training steps
    training_steps = 5000
    # Record the steps as and treat them as samples
    samples = [np.zeros([training_steps, 2]), np.zeros([training_steps, 1])]
    sess.run(init)
    for step in range(training_steps):
      sess.run([optimizer, loss])
      sample = [sess.run(var_1), sess.run(var_2)]
      samples[0][step, :] = sample[0]
      samples[1][step, :] = sample[1]

    samples_ = np.concatenate(samples, axis=-1)
    sample_mean = np.mean(samples_, 0)
    print('sample mean', sample_mean)
  ```
  Args:
    learning_rate: Scalar `float`-like `Tensor`. The base learning rate for the
      optimizer. Must be tuned to the specific function being minimized.
    preconditioner_decay_rate: Scalar `float`-like `Tensor`. The exponential
      decay rate of the rescaling of the preconditioner (RMSprop). (This is
      "alpha" in Li et al. (2016)). Should be smaller than but nearly `1` to
      approximate sampling from the posterior. (Default: `0.95`)
    data_size: Scalar `int`-like `Tensor`. The effective number of
      points in the data set. Assumes that the loss is taken as the mean over a
      minibatch. Otherwise if the sum was taken, divide this number by the
      batch size. If a prior is included in the loss function, it should be
      normalized by `data_size`. Default value: `1`.
    burnin: Scalar `int`-like `Tensor`. The number of iterations to collect
      gradient statistics to update the preconditioner before starting to draw
      noisy samples. (Default: `25`)
    diagonal_bias: Scalar `float`-like `Tensor`. Term added to the diagonal of
      the preconditioner to prevent the preconditioner from degenerating.
      (Default: `1e-8`)
    name: Python `str` describing ops managed by this function.
      (Default: `"StochasticGradientLangevinDynamics"`)
    parallel_iterations: the number of coordinates for which the gradients of
        the preconditioning matrix can be computed in parallel. Must be a
        positive integer.
    variable_scope: Variable scope used for calls to `tf.get_variable`.
      If `None`, a new variable scope is created using name
      `tf.get_default_graph().unique_name(name or default_name)`.

  Raises:
    InvalidArgumentError: If preconditioner_decay_rate is a `Tensor` not in
      `(0,1]`.
    NotImplementedError: If eager execution is enabled.

  #### References

  [1]: Chunyuan Li, Changyou Chen, David Carlson, and Lawrence Carin.
       Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural
       Networks. In _Association for the Advancement of Artificial
       Intelligence_, 2016. https://arxiv.org/abs/1512.07666
  """

  def __init__(self,
               learning_rate,
               preconditioner_decay_rate=0.95,
               data_size=1,
               burnin=25,
               diagonal_bias=1e-8,
               name=None,
               parallel_iterations=10,
               variable_scope=None):
    default_name = 'StochasticGradientLangevinDynamics'
    with tf.name_scope(name, default_name, [
        learning_rate, preconditioner_decay_rate, data_size, burnin,
        diagonal_bias
    ]):
      if tf.executing_eagerly():
        raise NotImplementedError('Eager execution currently not supported for '
                                  ' SGLD optimizer.')
      if variable_scope is None:
        var_scope_name = tf.get_default_graph().unique_name(
            name or default_name)
        with tf.variable_scope(var_scope_name) as scope:
          self._variable_scope = scope
      else:
        self._variable_scope = variable_scope

      self._preconditioner_decay_rate = tf.convert_to_tensor(
          preconditioner_decay_rate, name='preconditioner_decay_rate')
      self._data_size = tf.convert_to_tensor(
          data_size, name='data_size')
      self._burnin = tf.convert_to_tensor(burnin, name='burnin')
      self._diagonal_bias = tf.convert_to_tensor(
          diagonal_bias, name='diagonal_bias')
      self._learning_rate = tf.convert_to_tensor(
          learning_rate, name='learning_rate')
      self._parallel_iterations = parallel_iterations

      with tf.variable_scope(self._variable_scope):
        self._counter = tf.get_variable(
            'counter', initializer=0, trainable=False)

      self._preconditioner_decay_rate = control_flow_ops.with_dependencies([
          tf.assert_non_negative(
              self._preconditioner_decay_rate,
              message='`preconditioner_decay_rate` must be non-negative'),
          tf.assert_less_equal(
              self._preconditioner_decay_rate,
              1.,
              message='`preconditioner_decay_rate` must be at most 1.'),
      ], self._preconditioner_decay_rate)

      self._data_size = control_flow_ops.with_dependencies([
          tf.assert_greater(
              self._data_size,
              0,
              message='`data_size` must be greater than zero')
      ], self._data_size)

      self._burnin = control_flow_ops.with_dependencies([
          tf.assert_non_negative(
              self._burnin, message='`burnin` must be non-negative'),
          tf.assert_integer(
              self._burnin, message='`burnin` must be an integer')
      ], self._burnin)

      self._diagonal_bias = control_flow_ops.with_dependencies([
          tf.assert_non_negative(
              self._diagonal_bias,
              message='`diagonal_bias` must be non-negative')
      ], self._diagonal_bias)

      super(StochasticGradientLangevinDynamics, self).__init__(
          use_locking=False, name=name or default_name)

  def _create_slots(self, var_list):
    for v in var_list:
      init_rms = tf.ones_initializer(dtype=v.dtype)
      self._get_or_make_slot_with_initializer(v, init_rms, v.shape,
                                              v.dtype, 'rms', self._name)

  def _prepare(self):
    # We need to put the conversion and check here because a user will likely
    # want to decay the learning rate dynamically.
    self._learning_rate_tensor = control_flow_ops.with_dependencies([
        tf.assert_non_negative(
            self._learning_rate, message='`learning_rate` must be non-negative')
    ], tf.convert_to_tensor(self._learning_rate, name='learning_rate_tensor'))
    self._decay_tensor = tf.convert_to_tensor(
        self._preconditioner_decay_rate, name='preconditioner_decay_rate')

    super(StochasticGradientLangevinDynamics, self)._prepare()

  def _apply_dense(self, grad, var):
    rms = self.get_slot(var, 'rms')
    new_grad = self._apply_noisy_update(rms, grad, var)
    return training_ops.apply_gradient_descent(
        var,
        tf.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        new_grad,
        use_locking=self._use_locking).op

  def _apply_sparse(self, grad, var):
    rms = self.get_slot(var, 'rms')
    new_grad = self._apply_noisy_update(rms, grad, var)
    return training_ops.apply_gradient_descent(
        var,
        tf.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        new_grad,
        use_locking=self._use_locking).op

  def _finish(self, update_ops, name_scope):
    update_ops.append([self._counter.assign_add(1)])
    return tf.group(*update_ops, name=name_scope)

  @property
  def variable_scope(self):
    """Variable scope of all calls to `tf.get_variable`."""
    return self._variable_scope

  def _apply_noisy_update(self, mom, grad, var):
    # Compute and apply the gradient update following
    # preconditioned Langevin dynamics
    stddev = tf.where(
        tf.squeeze(self._counter > self._burnin),
        tf.cast(tf.rsqrt(self._learning_rate), grad.dtype),
        tf.zeros([], grad.dtype))
    # Keep an exponentially weighted moving average of squared gradients.
    # Not thread safe
    decay_tensor = tf.cast(self._decay_tensor, grad.dtype)
    new_mom = decay_tensor * mom + (1. - decay_tensor) * tf.square(grad)
    preconditioner = tf.rsqrt(
        new_mom + tf.cast(self._diagonal_bias, grad.dtype))

    # Compute gradients of the preconsitionaer
    _, preconditioner_grads = diag_jacobian(
        xs=var,
        ys=preconditioner,
        parallel_iterations=self._parallel_iterations)

    mean = 0.5 * (preconditioner * grad *
                  tf.cast(self._data_size, grad.dtype)
                  - preconditioner_grads[0])
    stddev *= tf.sqrt(preconditioner)
    result_shape = tf.broadcast_dynamic_shape(tf.shape(mean),
                                              tf.shape(stddev))
    with tf.control_dependencies([tf.assign(mom, new_mom)]):
      return tf.random_normal(shape=result_shape,
                              mean=mean,
                              stddev=stddev,
                              dtype=grad.dtype)
