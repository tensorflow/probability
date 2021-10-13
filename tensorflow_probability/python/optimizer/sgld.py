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

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.math import diag_jacobian
from tensorflow.python.training import training_ops


__all__ = [
    'StochasticGradientLangevinDynamics',
]


class StochasticGradientLangevinDynamics(tf.optimizers.Optimizer):
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
    tf.random.set_seed(42)
    true_mean = dtype([0, 0, 0])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
    # Loss is defined through the Cholesky decomposition
    chol = tf.linalg.cholesky(true_cov)

    var_1 = tf.Variable(name='var_1', initial_value=[1., 1.])
    var_2 = tf.Variable(name='var_2', initial_value=[1.])

    def loss_fn():
      var = tf.concat([var_1, var_2], axis=-1)
      loss_part = tf.linalg.cholesky_solve(chol, var[..., tf.newaxis])
      return tf.linalg.matvec(loss_part, var, transpose_a=True)

    # Set up the learning rate with a polynomial decay
    step = tf.Variable(0, dtype=tf.int64)
    starter_learning_rate = .3
    end_learning_rate = 1e-4
    decay_steps = 1e4
    learning_rate = tf.compat.v1.train.polynomial_decay(
        starter_learning_rate,
        step,
        decay_steps,
        end_learning_rate,
        power=1.)

    # Set up the optimizer
    optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(
        learning_rate=learning_rate, preconditioner_decay_rate=0.99)
    optimizer_kernel.iterations = step
    optimizer = optimizer_kernel.minimize(loss_fn, var_list=[var_1, var_2])

    # Number of training steps
    training_steps = 5000
    # Record the steps as and treat them as samples
    samples = [np.zeros([training_steps, 2]), np.zeros([training_steps, 1])]
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(training_steps):
      sess.run(optimizer)
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
               parallel_iterations=10):
    default_name = 'StochasticGradientLangevinDynamics'
    with tf.name_scope(name or default_name):
      if tf.executing_eagerly():
        raise NotImplementedError('Eager execution currently not supported for '
                                  ' SGLD optimizer.')

      self._preconditioner_decay_rate = tf.convert_to_tensor(
          preconditioner_decay_rate, name='preconditioner_decay_rate')
      self._data_size = tf.convert_to_tensor(data_size, name='data_size')
      self._burnin = tf.convert_to_tensor(
          burnin,
          name='burnin',
          dtype=dtype_util.common_dtype([burnin], dtype_hint=tf.int64))
      self._diagonal_bias = tf.convert_to_tensor(
          diagonal_bias, name='diagonal_bias')
      # TODO(b/124800185): Consider migrating `learning_rate` to be a
      # hyperparameter handled by the base Optimizer class. This would allow
      # users to plug in a `tf.keras.optimizers.schedules.LearningRateSchedule`
      # object in addition to Tensors.
      self._learning_rate = tf.convert_to_tensor(
          learning_rate, name='learning_rate')
      self._parallel_iterations = parallel_iterations

      self._preconditioner_decay_rate = distribution_util.with_dependencies([
          assert_util.assert_non_negative(
              self._preconditioner_decay_rate,
              message='`preconditioner_decay_rate` must be non-negative'),
          assert_util.assert_less_equal(
              self._preconditioner_decay_rate,
              1.,
              message='`preconditioner_decay_rate` must be at most 1.'),
      ], self._preconditioner_decay_rate)

      self._data_size = distribution_util.with_dependencies([
          assert_util.assert_greater(
              self._data_size,
              0,
              message='`data_size` must be greater than zero')
      ], self._data_size)

      self._burnin = distribution_util.with_dependencies([
          assert_util.assert_non_negative(
              self._burnin, message='`burnin` must be non-negative'),
          assert_util.assert_integer(
              self._burnin, message='`burnin` must be an integer')
      ], self._burnin)

      self._diagonal_bias = distribution_util.with_dependencies([
          assert_util.assert_non_negative(
              self._diagonal_bias,
              message='`diagonal_bias` must be non-negative')
      ], self._diagonal_bias)

      super(StochasticGradientLangevinDynamics,
            self).__init__(name=name or default_name)

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'rms', 'ones')

  def get_config(self):
    # TODO(b/124800185): Consider making `learning_rate`, `data_size`, `burnin`,
    # `preconditioner_decay_rate` and `diagonal_bias` hyperparameters.
    pass

  def _prepare(self, var_list):
    # We need to put the conversion and check here because a user will likely
    # want to decay the learning rate dynamically.
    self._learning_rate_tensor = distribution_util.with_dependencies(
        [
            assert_util.assert_non_negative(
                self._learning_rate,
                message='`learning_rate` must be non-negative')
        ],
        tf.convert_to_tensor(
            self._learning_rate, name='learning_rate_tensor'))
    self._decay_tensor = tf.convert_to_tensor(
        self._preconditioner_decay_rate, name='preconditioner_decay_rate')

    super(StochasticGradientLangevinDynamics, self)._prepare(var_list)

  def _resource_apply_dense(self, grad, var):
    rms = self.get_slot(var, 'rms')
    new_grad = self._apply_noisy_update(rms, grad, var)
    return training_ops.resource_apply_gradient_descent(
        var.handle,
        tf.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        new_grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    rms = self.get_slot(var, 'rms')
    new_grad = self._apply_noisy_update(rms, grad, var, indices)
    return self._resource_scatter_add(
        var, indices,
        -new_grad * tf.cast(self._learning_rate_tensor, var.dtype.base_dtype))

  @property
  def variable_scope(self):
    """Variable scope of all calls to `tf.get_variable`."""
    return self._variable_scope

  def _apply_noisy_update(self, mom, grad, var, indices=None):
    # Compute and apply the gradient update following
    # preconditioned Langevin dynamics
    stddev = tf.where(
        tf.squeeze(self.iterations > tf.cast(self._burnin, tf.int64)),
        tf.cast(tf.math.rsqrt(self._learning_rate), grad.dtype),
        tf.zeros([], grad.dtype))
    # Keep an exponentially weighted moving average of squared gradients.
    # Not thread safe
    decay_tensor = tf.cast(self._decay_tensor, grad.dtype)
    new_mom = decay_tensor * mom + (1. - decay_tensor) * tf.square(grad)
    preconditioner = tf.math.rsqrt(new_mom +
                                   tf.cast(self._diagonal_bias, grad.dtype))

    # Compute gradients of the preconditioner.
    # Note: Since the preconditioner depends indirectly on `var` through `grad`,
    # in Eager mode, `diag_jacobian` would need access to the loss function.
    # This is the only blocker to supporting Eager mode for the SGLD optimizer.
    _, preconditioner_grads = diag_jacobian(
        xs=var,
        ys=preconditioner,
        parallel_iterations=self._parallel_iterations)

    mean = 0.5 * (preconditioner * grad *
                  tf.cast(self._data_size, grad.dtype)
                  - preconditioner_grads[0])
    stddev *= tf.sqrt(preconditioner)
    result_shape = tf.broadcast_dynamic_shape(
        tf.shape(mean), tf.shape(stddev))

    update_ops = []
    if indices is None:
      update_ops.append(mom.assign(new_mom))
    else:
      update_ops.append(self._resource_scatter_update(mom, indices, new_mom))

    with tf.control_dependencies(update_ops):
      return tf.random.normal(
          shape=result_shape, mean=mean, stddev=stddev, dtype=grad.dtype)
