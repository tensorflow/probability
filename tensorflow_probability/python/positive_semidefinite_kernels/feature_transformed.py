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
"""FeatureTransformed kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel

__all__ = ['FeatureTransformed']


class FeatureTransformed(psd_kernel.PositiveSemidefiniteKernel):
  """Input transformed kernel.

  Given a kernel `k` and function `f`, compute `k_{new}(x, y) = k(f(x), f(y))`.


  ### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  from tensorflow_probability.positive_semidefinite_kernel.internal import util
  tfpk = tfp.positive_semidefinite_kernels

  base_kernel = tfpk.ExponentiatedQuadratic(amplitude=2., length_scale=1.)
  ```

  - Identity function.

  ```python
  # This is the same as base_kernel
  same_kernel = tfpk.FeatureTransformed(
      base_kernel,
      transformation_fn=lambda x, _, _: x)
  ```

  - Exponential transformation.

  ```python
  exp_kernel = tfpk.FeatureTransformed(
      base_kernel,
      transformation_fn=lambda x, _, _: tf.exp(x))
  ```

  - Transformation with broadcasting parameters.

  ```python
  # Exponentiate inputs

  p = np.random.uniform(low=2., high=3., size=[10, 2])
  def inputs_to_power(x, feature_ndims, param_expansion_ndims):
    # Make sure we account for extra feature dimensions for
    # broadcasting purposes.
    power = util.pad_shape_with_ones(
        p,
        ndims=feature_ndims + param_expansion_ndims,
        start=-(feature_ndims + 1))
    return x ** power

  power_kernel = tfpk.FeatureTransformed(
    base_kernel, transformation_fn=inputs_to_power)
  """

  def __init__(
      self,
      kernel,
      transformation_fn,
      validate_args=False,
      name='FeatureTransformed'):
    """Construct an FeatureTransformed kernel instance.

    Args:
      kernel: `PositiveSemidefiniteKernel` instance. Inputs are transformed and
        passed in to this kernel. Parameters to `kernel` must be broadcastable
        with parameters of `transformation_fn`.
      transformation_fn: Callable. `transformation_fn` takes in an input
        `Tensor`, a Python integer representing the number of feature
        dimensions, and a Python integer representing the
        `param_expansion_ndims` arg of `_apply`. Computations in
        `transformation_fn` must be broadcastable with parameters of `kernel`.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    self._kernel = kernel
    self._transformation_fn = transformation_fn
    super(FeatureTransformed, self).__init__(
        feature_ndims=kernel.feature_ndims,
        dtype=kernel.dtype,
        name=name)

  def _apply(self, x1, x2, example_ndims=0):
    return self._kernel.apply(
        self._transformation_fn(
            x1, self.feature_ndims, example_ndims),
        self._transformation_fn(
            x2, self.feature_ndims, example_ndims),
        example_ndims)

  @property
  def kernel(self):
    """Base kernel to pass transformed inputs."""
    return self._kernel

  @property
  def transformation_fn(self):
    """Function that preprocesses inputs before handing them to `kernel`."""
    return self._transformation_fn
