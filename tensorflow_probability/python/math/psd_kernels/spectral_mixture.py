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
"""The SpectralMixture kernel."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import generic as tfp_math
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util


__all__ = ['SpectralMixture']


class SpectralMixture(psd_kernel.AutoCompositeTensorPsdKernel):
  """The SpectralMixture kernel.

  This kernel is derived from parameterizing the spectral density of a
  stationary kernel by a mixture of `m` diagonal multivariate normal
  distributions [1].

  This in turn parameterizes the following kernel:

    ```none
    k(x, y) = sum_j w[j] (prod_i
        exp(-2 * (pi * (x[i] - y[i]) * s[j][i])**2) *
        cos(2 * pi * (x[i] - y[i]) * m[j][i]))
    ```

  where:
    * `j` is the number of mixtures (as mentioned above).
    * `w[j]` are the mixture weights.
    * `m[j]` and `s[j]` parameterize a `MultivariateNormalDiag(m[j], s[j])`.
      In other words, they are the mean and diagonal scale for each mixture
      component.

  NOTE: This kernel can result in negative off-diagonal entries.

  #### References
  [1]: A. Wilson, R. P. Adams.
       Gaussian Process Kernels for Pattern Discovery and Extrapolation.
       https://arxiv.org/abs/1302.4245
  """

  def __init__(self,
               logits,
               locs,
               scales,
               feature_ndims=1,
               validate_args=False,
               name='SpectralMixture'):
    """Construct a SpectralMixture kernel instance.

    Args:
      logits: Floating-point `Tensor` of shape `[..., M]`, whose softmax
        represents the mixture weights for the spectral density. Must
        be broadcastable with `locs` and `scales`.
      locs: Floating-point `Tensor` of shape `[..., M, F1, F2, ... FN]`, which
        represents the location parameter of each of the `M` mixture components.
        `N` is `feature_ndims`. Must be broadcastable with `logits` and
        `scales`.
      scales: Positive Floating-point `Tensor` of shape
        `[..., M, F1, F2, ..., FN]`, which represents the scale parameter of
        each of the `M` mixture components. `N` is `feature_ndims`. Must be
        broadcastable with `locs` and `logits`. These parameters act like
        inverse length scale parameters.
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = util.maybe_get_common_dtype([logits, locs, scales])
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, name='logits', dtype=dtype)
      self._locs = tensor_util.convert_nonref_to_tensor(
          locs, name='locs', dtype=dtype)
      self._scales = tensor_util.convert_nonref_to_tensor(
          scales, name='scales', dtype=dtype)
      super(SpectralMixture, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @property
  def logits(self):
    """Logits parameter."""
    return self._logits

  @property
  def locs(self):
    """Location parameter."""
    return self._locs

  @property
  def scales(self):
    """Scale parameter."""
    return self._scales

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        logits=parameter_properties.ParameterProperties(event_ndims=1),
        locs=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.feature_ndims + 1),
        scales=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.feature_ndims + 1,
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))))

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    exponent = -2. * pairwise_square_distance
    locs = util.pad_shape_with_ones(
        self.locs, ndims=example_ndims, start=-(self.feature_ndims + 1))
    cos_coeffs = tf.math.cos(2 * np.pi * (x1 - x2) * locs)
    feature_ndims = ps.cast(self.feature_ndims, ps.rank(cos_coeffs).dtype)
    reduction_axes = ps.range(
        ps.rank(cos_coeffs) - feature_ndims, ps.rank(cos_coeffs))
    coeff_sign = tf.math.reduce_prod(
        tf.math.sign(cos_coeffs), axis=reduction_axes)
    log_cos_coeffs = tf.math.reduce_sum(
        tf.math.log(tf.math.abs(cos_coeffs)), axis=reduction_axes)

    logits = util.pad_shape_with_ones(
        self.logits, ndims=example_ndims, start=-1)

    log_result, sign = tfp_math.reduce_weighted_logsumexp(
        exponent + log_cos_coeffs + logits,
        coeff_sign, return_sign=True, axis=-(example_ndims + 1))

    return sign * tf.math.exp(log_result)

  def _apply(self, x1, x2, example_ndims=0):
    # Add an extra dimension to x1 and x2 so it broadcasts with scales.
    # [B1, ...., E1, ...., E2, M, F1, ..., F2]
    x1 = util.pad_shape_with_ones(
        x1, ndims=1, start=-(self.feature_ndims + example_ndims + 1))
    x2 = util.pad_shape_with_ones(
        x2, ndims=1, start=-(self.feature_ndims + example_ndims + 1))
    scales = util.pad_shape_with_ones(
        self.scales, ndims=example_ndims, start=-(self.feature_ndims + 1))
    pairwise_square_distance = util.sum_rightmost_ndims_preserving_shape(
        tf.math.square(np.pi * (x1 - x2) * scales), ndims=self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=example_ndims)

  def _matrix(self, x1, x2):
    # Add an extra dimension to x1 and x2 so it broadcasts with scales.
    x1 = util.pad_shape_with_ones(x1, ndims=1, start=-(self.feature_ndims + 2))
    x2 = util.pad_shape_with_ones(x2, ndims=1, start=-(self.feature_ndims + 2))
    scales = util.pad_shape_with_ones(
        self.scales, ndims=1, start=-(self.feature_ndims + 1))
    pairwise_square_distance = util.pairwise_square_distance_matrix(
        np.pi * x1 * scales, np.pi * x2 * scales, self.feature_ndims)
    x1 = util.pad_shape_with_ones(x1, ndims=1, start=-(self.feature_ndims + 1))
    x2 = util.pad_shape_with_ones(x2, ndims=1, start=-(self.feature_ndims + 2))
    # Expand `x1` and `x2` so that the broadcast against each other.
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=2)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._scales):
      assertions.append(assert_util.assert_positive(
          self._scales,
          message='`scales` must be positive.'))
    return assertions
