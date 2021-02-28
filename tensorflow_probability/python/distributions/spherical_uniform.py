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
"""The uniform spherical distribution over vectors on the unit hypersphere."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.bijectors import softmax_centered as softmax_centered_bijector
from tensorflow_probability.python.bijectors import square as square_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.random import random_ops


__all__ = ['SphericalUniform']


class SphericalUniform(distribution.Distribution):
  r"""The uniform distribution over unit vectors on `S^{n-1}`.

  The uniform distribution on the unit hypersphere `S^{n-1}` embedded in
  `n` dimensions (`R^n`).

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; n) = 1. / A(n)
  where,
  A(n) = 2 * pi^{n / 2} / Gamma(n / 2),
  Gamma being the Gamma function.
  ```

  where:
  * `n = dimension`; corresponds to `S^{n-1}` embedded in `R^n`.

  #### Examples

  A SphericalUniform distribution is defined in 3 dimensions.

  ```python
  tfd = tfp.distributions

  # Initialize a single 3-dimension SphericalUniform distribution.
  su = tfd.SphericalUniform(dimension=3, batch_shape=[])

  # Evaluate this on an observation in S^2 (in R^3), returning a scalar.
  su.prob([1., 0, 0])

  # Initialize a batch of 3 4-dimensional SphericalUniform distributions.
  su = tfd.SphericalUniform(dimension=4, batch_shape=[3])
  su.sample(5)  # Shape [5, 3, 4]
  ```
  """

  def __init__(self,
               dimension,
               batch_shape=tuple(),
               dtype=tf.float32,
               validate_args=False,
               allow_nan_stats=True,
               name='SphericalUniform'):
    """Creates a new `SphericalUniform` instance.

    Args:
      dimension: Python `int`. The dimension of the embedded space where the
        sphere resides.
      batch_shape: Positive `int`-like vector-shaped `Tensor` representing
        the new shape of the batch dimensions.
        Default value: [].
      dtype: DType of the generated samples.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: For known-bad arguments, i.e. unsupported event dimension.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      if dimension < 0:
        raise ValueError(
            'Cannot sample negative-dimension unit vectors.')
      shape_dtype = dtype_util.common_dtype([batch_shape], dtype_hint=tf.int32)
      self._dimension = dimension
      self._batch_shape_parameter = tensor_util.convert_nonref_to_tensor(
          batch_shape, dtype=shape_dtype, name='batch_shape',
          as_shape_tensor=True)
      self._batch_shape_static = tensorshape_util.constant_value_as_shape(
          self._batch_shape_parameter)

      super(SphericalUniform, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict()

  @property
  def dimension(self):
    """Dimension of returned unit vectors."""
    return self._dimension

  def __getitem__(self, slices):
    # The generic slicing machinery doesn't work, because the batch shape is not
    # determined by the shape of a parameter.
    new_batch_shape = tf.shape(tf.zeros(self.parameters['batch_shape'])[slices])
    return self.copy(batch_shape=new_batch_shape)

  def _batch_shape_tensor(self):
    return self._batch_shape_parameter

  def _batch_shape(self):
    return self._batch_shape_static

  def _event_shape_tensor(self):
    return tf.constant([self.dimension], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([self.dimension])

  def _log_prob(self, x):
    log_nsphere_surface_area = (
        np.log(2.) + (self.dimension / 2) * np.log(np.pi) -
        tf.math.lgamma(tf.cast(self.dimension / 2., x.dtype)))
    batch_shape = ps.broadcast_shape(
        ps.shape(x)[:-1], self.batch_shape)
    return tf.fill(batch_shape, -log_nsphere_surface_area)

  def _sample_n(self, n, seed=None):
    return random_ops.spherical_uniform(
        shape=ps.concat([[n], self.batch_shape], axis=0),
        dimension=self.dimension,
        dtype=self.dtype,
        seed=seed)

  def _entropy(self):
    log_nsphere_surface_area = (
        np.log(2.) + (self.dimension / 2) * np.log(np.pi) -
        tf.math.lgamma(tf.cast(self.dimension / 2., self.dtype)))
    return tf.fill(self.batch_shape_tensor(), log_nsphere_surface_area)

  def _mean(self):
    # This can be seen due to symmetry. If (X_1, ... X_k, ..., X_n) is
    # uniformly distributed on the sphere, then so is
    # (X_1, ..., -X_k, ..., X_n). Thus E[X_k] = E[-X_k] = 0.
    return tf.fill(
        tf.concat([self.batch_shape_tensor(), [self.dimension]], axis=0),
        tf.cast(0., self.dtype))

  def _covariance(self):
    # By a similar argument for the mean case, we can show that
    # E[X_i X_j] = -E[X_i X_j], so the covariance terms are zero.
    return tf.eye(
        num_rows=self.dimension,
        batch_shape=self.batch_shape_tensor(),
        dtype=self.dtype) / self.dimension

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    return chain_bijector.Chain([
        invert_bijector.Invert(
            square_bijector.Square(validate_args=self.validate_args),
            validate_args=self.validate_args),
        softmax_centered_bijector.SoftmaxCentered(
            validate_args=self.validate_args)
    ], validate_args=self.validate_args)

  def _sample_control_dependencies(self, samples):
    inner_sample_dim = samples.shape[-1]
    shape_msg = ('Samples must have innermost dimension matching that of '
                 '`self.dimension`. Found {}, expected {}'.format(
                     inner_sample_dim, self.dimension))
    if inner_sample_dim is not None:
      if self.dimension != inner_sample_dim:
        raise ValueError(shape_msg)

    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_near(
        tf.cast(1., dtype=self.dtype),
        tf.linalg.norm(samples, axis=-1),
        message='Samples must be unit length.'))
    assertions.append(assert_util.assert_equal(
        tf.shape(samples)[-1:],
        self.dimension,
        message=shape_msg))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._batch_shape_parameter):
      assertions.append(assert_util.assert_rank(
          self._batch_shape_parameter, 1,
          message='Batch shape must be a vector.'))
      assertions.append(assert_util.assert_non_negative(
          self._batch_shape_parameter,
          message='Shape elements must be >-1.'))
    return assertions
