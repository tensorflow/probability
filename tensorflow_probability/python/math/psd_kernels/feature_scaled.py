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
"""Feature scaled kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels.feature_transformed import FeatureTransformed
from tensorflow_probability.python.math.psd_kernels.internal import util

__all__ = ['FeatureScaled']


# TODO(b/132103412): Support more general scaling via LinearOperator, along with
# scaling all feature dimensions.
class FeatureScaled(FeatureTransformed):
  """Kernel that first rescales all feature dimensions.

  Given a kernel `k` and `scale_diag` and inputs `x` and `y`, this kernel
  first rescales the input by computing `x / scale_diag` and
  `y / scale_diag`, and passing this to `k`.

  With 1 feature dimension, this is also called Automatic Relevance
  Determination (ARD) [1].

  #### References
  [1]: Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian
       Processes for Machine Learning. Section 5.1 2006.
       http://www.gaussianprocess.org/gpml/chapters/RW5.pdf
  """

  def __init__(
      self, kernel, scale_diag, validate_args=False, name='FeatureScaled'):
    """Construct an FeatureScaled kernel instance.

    Args:
      kernel: `PositiveSemidefiniteKernel` instance. Inputs are rescaled and
        passed in to this kernel. Parameters to `kernel` must be broadcastable
        with `scale_diag`.
      scale_diag: Floating point `Tensor` that controls how sharp or wide the
        kernel shape is. `scale_diag` must have at least `kernel.feature_ndims`
        dimensions, and extra dimensions must be broadcastable with parameters
        of `kernel`. This is a "diagonal" in the sense that if all the feature
        dimensions were flattened, `scale_diag` acts as the inverse of a
        diagonal matrix.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      self._scale_diag = tensor_util.convert_nonref_to_tensor(
          scale_diag, name='scale_diag')

      def rescale_input(x, feature_ndims, example_ndims):
        """Computes `x / scale_diag`."""
        scale_diag = tf.convert_to_tensor(self.scale_diag)
        scale_diag = util.pad_shape_with_ones(
            scale_diag,
            example_ndims,
            # Start before the first feature dimension. We assume scale_diag has
            # at least as many dimensions as feature_ndims.
            start=-(feature_ndims + 1))
        return x / scale_diag

      super(FeatureScaled, self).__init__(
          kernel,
          transformation_fn=rescale_input,
          validate_args=validate_args,
          name=name,
          parameters=parameters)

  @property
  def scale_diag(self):
    return self._scale_diag

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.kernel.batch_shape,
        self.scale_diag.shape[:-self.kernel.feature_ndims])

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        self.kernel.batch_shape_tensor(),
        tf.shape(self.scale_diag)[:-self.kernel.feature_ndims])

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    scale_diag = self.scale_diag
    if scale_diag is not None and is_init != tensor_util.is_ref(scale_diag):
      assertions.append(assert_util.assert_positive(
          scale_diag,
          message='scale_diag must be positive.'))
    return assertions
