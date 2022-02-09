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
"""MultiTask kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel


__all__ = ['MultiTaskKernel', 'Independent', 'Separable']


class MultiTaskKernel(psd_kernel.AutoCompositeTensorPsdKernel):
  """Abstract base class for Multi-Task Kernels.

  A `MultiTaskKernel` is defined as a positive-semidefinite kernel over tuples
  `(o, x) x (o', x')`, where `o` is an integer representing a "task" while
  `x` lies in some ambient space. Typically, `o` is from a finite set of
  integers `{1, ..., N}` where `N` represents the number of tasks. Another
  way to phrase this, is a `MultiTaskKernel` represents a finite collection
  of positive-semidefinite kernels one for each "task" in `{1, ..., N}`.

  `MultiTaskKernel` offers a public method `matrix_over_all_tasks`.

  `matrix_over_all_tasks` computes the value of the kernel *pairwise* on two
  batches of inputs, for every pair of tasks.

  An instance of this class is required to pass in `num_tasks` (which is `N`).
  """

  def __init__(
      self,
      num_tasks,
      dtype,
      feature_ndims=1,
      name=None,
      validate_args=False,
      parameters=None):
    """Constructs a MultiTaskKernel instance.

    Args:
      num_tasks: Python `integer` indicating the number of tasks.
      dtype: `DType` on which this kernel operates.
      feature_ndims: Python `integer` indicating the number of dims (the rank)
        of the feature space this kernel acts on.
      name: Python `str` name prefixed to Ops created by this class. Default:
        subclass name.
      validate_args: Python `bool`, default `False`. When `True` kernel
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      parameters: Python `dict` of constructor arguments.

    Raises:
      ValueError: if `num_tasks` is not an integer greater than 0.
    """

    if not (isinstance(num_tasks, int) and num_tasks > 0):
      raise ValueError(
          '`num_tasks` must be a Python `integer` greater than zero. ' +
          f'Got: {num_tasks}')
    self._num_tasks = num_tasks

    super(MultiTaskKernel, self).__init__(
        feature_ndims,
        dtype=dtype,
        name=name,
        validate_args=validate_args,
        parameters=parameters)

  @property
  def num_tasks(self):
    """Number of tasks for this kernel."""
    return self._num_tasks

  def matrix_over_all_tasks(self, x1, x2, name='matrix_over_all_tasks'):
    """Returns matrix computation assuming all tasks are observed.

    Given `x1` and `x2` of shape `B1 + [E1, M]` and
    `B2 + [E2, M]`, returns a `LinearOperator` `S` of shape
    `broadcast(B1, B2) + [E1 * N, E2 * N]`, where `N` is `num_tasks`.
    Each `N x N` block of this `LinearOperator` represents the action of this
    kernel at a fixed pair of inputs, over all tasks.

    Args:
      x1: `Tensor` input to the first positional parameter of the kernel, of
        shape `B1 + [e1] + F`, where `B1` may be empty (ie, no batch dims,
        resp.), `e1` is a single integer (ie, `x1` has example ndims exactly 1),
        and `F` (the feature shape) must have rank equal to the kernel's
        `feature_ndims` property. Batch shape must broadcast with the batch
        shape of `x2` and with the kernel's batch shape.
      x2: `Tensor` input to the second positional parameter of the kernel,
        shape `B2 + [e2] + F`, where `B2` may be empty (ie, no batch dims,
        resp.), `e2` is a single integer (ie, `x2` has example ndims exactly 1),
        and `F` (the feature shape) must have rank equal to the kernel's
        `feature_ndims` property. Batch shape must broadcast with the batch
        shape of `x1` and with the kernel's batch shape.
      name: name to give to the op

    Returns:
      `Tensor` containing the matrix (possibly batched) of kernel applications
      to pairs from inputs `x1` and `x2`, over all tasks. If the kernel
      parameters' batch shape is `Bk` then the shape of the `LinearOperator`
      resulting from this method call is
      `broadcast(Bk, B1, B2) + [N * e1, N * e2]`.

    #### Examples

    ```python
    import tensorflow_probability as tfp

    base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(1., 1.)
    kernel = tfs.psd_kernels.SomeVectorKernel(num_tasks=2, base_kernel)
    kernel.batch_shape
    # ==> []

    # Our inputs are two lists of 3-D vectors
    x = np.ones([3, 3], np.float32)
    y = np.ones([3, 3], np.float32)
    kernel.matrix_over_all_tasks(x, y).shape
    # ==> [6, 6]
    """
    with tf.name_scope(name):
      x1 = tf.convert_to_tensor(x1, name='x1', dtype_hint=self.dtype)
      x2 = tf.convert_to_tensor(x2, name='x2', dtype_hint=self.dtype)
      return self._matrix_over_all_tasks(x1, x2)

  def _matrix_over_all_tasks(self, x1, x2):
    raise NotImplementedError(
        'Subclasses must provide `_matrix_over_all_tasks` implementation.')


class Independent(MultiTaskKernel):
  """Represents a multi-task kernel whose computations are independent of task."""

  def __init__(self,
               num_tasks,
               base_kernel,
               validate_args=False,
               name='Independent'):
    parameters = dict(locals())
    with tf.name_scope(name):
      self._base_kernel = base_kernel
      super(Independent, self).__init__(
          num_tasks=num_tasks,
          dtype=base_kernel.dtype,
          feature_ndims=base_kernel.feature_ndims,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @property
  def base_kernel(self):
    return self._base_kernel

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(base_kernel=parameter_properties.BatchedComponentProperties())

  def _matrix_over_all_tasks(self, x1, x2):
    # Because the kernel computations are independent of task,
    # we can use a Kronecker product of an identity matrix.
    task_kernel_matrix = tf.linalg.LinearOperatorIdentity(
        num_rows=self.num_tasks,
        dtype=self.dtype)
    base_kernel_matrix = tf.linalg.LinearOperatorFullMatrix(
        self.base_kernel.matrix(x1, x2))
    return tf.linalg.LinearOperatorKronecker(
        [base_kernel_matrix, task_kernel_matrix])


class Separable(MultiTaskKernel):
  """Represents a multi-task kernel whose kernel can be separated as a product."""

  def __init__(self,
               num_tasks,
               base_kernel,
               task_kernel_matrix_linop,
               name='Separable',
               validate_args=False):

    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype(
          [task_kernel_matrix_linop, base_kernel], tf.float32)
      self._base_kernel = base_kernel
      self._task_kernel_matrix_linop = tensor_util.convert_nonref_to_tensor(
          task_kernel_matrix_linop, dtype, name='task_kernel_matrix_linop')
      super(Separable, self).__init__(
          num_tasks=num_tasks,
          dtype=dtype,
          feature_ndims=base_kernel.feature_ndims,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @property
  def base_kernel(self):
    return self._base_kernel

  @property
  def task_kernel_matrix_linop(self):
    return self._task_kernel_matrix_linop

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        base_kernel=parameter_properties.BatchedComponentProperties(),
        task_kernel_matrix_linop=(
            parameter_properties.BatchedComponentProperties()))

  def _matrix_over_all_tasks(self, x1, x2):
    # Because the kernel computations are independent of task,
    # we can use a Kronecker product of an identity matrix.
    base_kernel_matrix = tf.linalg.LinearOperatorFullMatrix(
        self.base_kernel.matrix(x1, x2))
    return tf.linalg.LinearOperatorKronecker(
        [base_kernel_matrix, self._task_kernel_matrix_linop])
