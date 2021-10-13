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

import collections

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import feature_transformed
from tensorflow_probability.python.math.psd_kernels.internal import util

__all__ = ['FeatureScaled']


# TODO(b/132103412): Support more general scaling via LinearOperator, along with
# scaling all feature dimensions.
class FeatureScaled(feature_transformed.FeatureTransformed):
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
      self,
      kernel,
      scale_diag=None,
      inverse_scale_diag=None,
      validate_args=False,
      name='FeatureScaled'):
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
      inverse_scale_diag: Non-negative floating point `Tensor` that is treated
        as `1 / scale_diag`. Only one of `scale_diag` or `inverse_scale_diag`
        should be provided.
        Default value: None
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    if (scale_diag is None) == (inverse_scale_diag is None):
      raise ValueError(
          'Must specify exactly one of `scale_diag` and `inverse_scale_diag`.')
    with tf.name_scope(name):
      self._scale_diag = tensor_util.convert_nonref_to_tensor(
          scale_diag, name='scale_diag')
      self._inverse_scale_diag = tensor_util.convert_nonref_to_tensor(
          inverse_scale_diag, name='inverse_scale_diag')

      def rescale_input(x, feature_ndims, example_ndims):
        """Computes `x / scale_diag`."""
        inverse_scale_diag = self.inverse_scale_diag
        if inverse_scale_diag is None:
          inverse_scale_diag = tf.math.reciprocal(self.scale_diag)
        inverse_scale_diag = tf.convert_to_tensor(inverse_scale_diag)
        inverse_scale_diag = util.pad_shape_with_ones(
            inverse_scale_diag,
            example_ndims,
            # Start before the first feature dimension. We assume scale_diag has
            # at least as many dimensions as feature_ndims.
            start=-(feature_ndims + 1))
        return x * inverse_scale_diag

      super(FeatureScaled, self).__init__(
          kernel,
          transformation_fn=rescale_input,
          validate_args=validate_args,
          name=name,
          parameters=parameters)

  @property
  def scale_diag(self):
    return self._scale_diag

  @property
  def inverse_scale_diag(self):
    return self._inverse_scale_diag

  def __getitem__(self, slices):
    overrides = {}
    if self.parameters.get('kernel', None) is not None:
      overrides['kernel'] = self.kernel[slices]

    diag_slices = (list(slices) if isinstance(
        slices, collections.Sequence) else [slices])
    diag_slices += [slice(None)] * self.kernel.feature_ndims

    if self.parameters.get('scale_diag', None) is not None:
      overrides['scale_diag'] = self.scale_diag[diag_slices]
    if self.parameters.get('inverse_scale_diag', None) is not None:
      overrides['inverse_scale_diag'] = self.inverse_scale_diag[diag_slices]
    return self.copy(**overrides)

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        kernel=parameter_properties.BatchedComponentProperties(),
        scale_diag=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims,
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        inverse_scale_diag=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims,
            default_constraining_bijector_fn=softplus.Softplus))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if (self._inverse_scale_diag is not None and
        is_init != tensor_util.is_ref(self._inverse_scale_diag)):
      assertions.append(assert_util.assert_non_negative(
          self._inverse_scale_diag,
          message='`inverse_scale_diag` must be non-negative.'))
    if (self._scale_diag is not None and
        is_init != tensor_util.is_ref(self._scale_diag)):
      assertions.append(assert_util.assert_positive(
          self._scale_diag,
          message='`scale_diag` must be positive.'))
    return assertions
