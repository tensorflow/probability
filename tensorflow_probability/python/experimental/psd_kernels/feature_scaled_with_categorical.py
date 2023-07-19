# Copyright 2023 The TensorFlow Probability Authors.
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
"""FeatureScaled kernel over continuous and categorical data."""

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util


ContinuousAndCategoricalValues = collections.namedtuple(
    'ContinuousAndCategoricalValues', ['continuous', 'categorical'])


class FeatureScaledWithCategorical(psd_kernel.AutoCompositeTensorPsdKernel):
  """`FeatureScaled` kernel for continuous and categorical data.

  This kernel is an extension of `FeatureScaled` that handles categorical data
  (encoded as integers, not one-hot) in addition to continuous (float) data.
  `ContinuousAndCategoricalValues` structures, containing arrays of continuous
  and categorical data, are passed to the `apply`, `matrix` and `tensor`
  methods. The continuous inputs are scaled and then passed to the distance
  function, like in `FeatureScaled`. Categorical data is compared for equality,
  and features in the same category are assigned distance 0 and features in
  different categories are assigned distance 1. The scaling factors are then
  applied to the categorical distances.

  #### Examples

  Compute the kernel matrix on synthetic data.

  ```python
  import numpy as np

  continuous_dim = 3
  categorical_dim = 5

  # Define an ARD kernel that takes a structure of continuous and categorical
  # data as inputs, with randomly-sampled length scales. The `categorical`
  # field of `scale_diag` contains a float Tensor because it represents a
  # scaling multiplier for the categorical distance, not the categorical data
  # itself.
  base_kernel = tfpk.MaternFiveHalves()
  kernel = tfpks.FeatureScaledWithCategorical(
      base_kernel,
      scale_diag=tfpks.ContinuousAndCategoricalValues(
          continuous=np.random.uniform(size=[continuous_dim]),
          categorical=np.random.uniform(size=[categorical_dim])),
      validate_args=True)

  # Create `num_points` examples in the continuous/categorical feature space.
  num_points = 12
  num_categories = 10
  x1 = tfpks.ContinuousAndCategoricalValues(
      continuous=np.random.normal(size=(num_points, continuous_dim)),
      categorical=np.random.randint(
          num_categories, size=(num_points, categorical_dim)))
  x2 = tfpks.ContinuousAndCategoricalValues(
      continuous=np.random.normal(size=(num_points, continuous_dim)),
      categorical=np.random.randint(
          num_categories, size=(num_points, categorical_dim)))

  # Evaluate the kernel matrix for `x1` and `x2`.
  kernel.matrix(x1, x2)  # has shape `[num_points, num_points]`

  ```
  """

  def __init__(
      self,
      kernel,
      scale_diag=None,
      inverse_scale_diag=None,
      feature_ndims=None,
      validate_args=False,
      name='FeatureScaledWithCategorical'):
    """Construct an `FeatureScaledWithCategorical` kernel instance.

    Args:
      kernel: `PositiveSemidefiniteKernel` instance. Parameters to `kernel` must
        be broadcastable with `scale_diag`. `kernel` must be isotropic and
        implement an `_apply_with_distance` method.
      scale_diag: `ContinuousAndCategoricalValues` instance containing floating
        point arrays that control the sharpness/width of the kernel shape. Each
        component of `scale_diag` must have dimensionality of at least
        the corresponding (continuous or categorical) element of
        `kernel.feature_ndims`, and extra dimensions must be broadcastable with
        parameters of `kernel`. This is a "diagonal" in the sense that if all
        the feature dimensions were flattened, `scale_diag` acts as the inverse
        of a diagonal matrix.
        Default value: None.
      inverse_scale_diag: `ContinuousAndCategoricalValues` instance containing
        non-negative floating point arrays that are treated as the reciprocals
        of the corresponding components of `scale_diag`.  Only one of
        `scale_diag` or `inverse_scale_diag` should be provided.
        Default value: None
      feature_ndims: `ContinuousAndCategoricalValues` instance containing
        integers indicating the rank of the continuous and categorical feature
        space. Default value: None, i.e. `kernel.feature_ndims` for both
        components of the feature space.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    if (scale_diag is None) == (inverse_scale_diag is None):
      raise ValueError(
          'Must specify exactly one of `scale_diag` and `inverse_scale_diag`.')
    with tf.name_scope(name):
      float_dtype = dtype_util.common_dtype(
          [kernel, scale_diag, inverse_scale_diag], dtype_hint=tf.float32)
      dtype = ContinuousAndCategoricalValues(float_dtype, None)
      if scale_diag is None:
        self._scale_diag = scale_diag
        self._inverse_scale_diag = nest_util.convert_to_nested_tensor(
            inverse_scale_diag, dtype_hint=dtype, name='inverse_scale_diag',
            convert_ref=False)
      else:
        self._scale_diag = nest_util.convert_to_nested_tensor(
            scale_diag, dtype_hint=dtype, name='scale_diag', convert_ref=False)
        self._inverse_scale_diag = inverse_scale_diag
      self._kernel = kernel

      if feature_ndims is None:
        feature_ndims = ContinuousAndCategoricalValues(
            kernel.feature_ndims, kernel.feature_ndims)

      super(FeatureScaledWithCategorical, self).__init__(
          feature_ndims=feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @property
  def kernel(self):
    return self._kernel

  @property
  def scale_diag(self):
    return self._scale_diag

  @property
  def inverse_scale_diag(self):
    return self._inverse_scale_diag

  def inverse_scale_diag_parameters(self):
    inverse_scale_diag = self.inverse_scale_diag
    if inverse_scale_diag is None:
      inverse_scale_diag = tf.nest.map_structure(
          tf.math.reciprocal, self.scale_diag)
    return tf.nest.map_structure(tf.convert_to_tensor, inverse_scale_diag)

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import joint_map  # pylint:disable=g-import-not-at-top
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top

    def _constraining_bijector_fn(low=None):
      softplus_bij = softplus.Softplus(low=low)
      return joint_map.JointMap(
          ContinuousAndCategoricalValues(softplus_bij, softplus_bij))

    return dict(
        kernel=parameter_properties.BatchedComponentProperties(),
        scale_diag=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.feature_ndims,
            default_constraining_bijector_fn=(
                lambda: _constraining_bijector_fn(low=dtype_util.eps(dtype))),
            is_preferred=False),
        inverse_scale_diag=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.feature_ndims,
            default_constraining_bijector_fn=_constraining_bijector_fn))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if self._inverse_scale_diag is not None:
      for t in tf.nest.flatten(self._inverse_scale_diag):
        if is_init != tensor_util.is_ref(t):
          assertions.append(assert_util.assert_non_negative(
              t, message='`inverse_scale_diag` must be non-negative.'))
    if self._scale_diag is not None:
      for t in tf.nest.flatten(self._scale_diag):
        if is_init != tensor_util.is_ref(t):
          assertions.append(assert_util.assert_positive(
              t, message='`scale_diag` must be positive.'))
    return assertions

  def _apply(self, x1, x2, example_ndims=0):
    isd = self.inverse_scale_diag_parameters()
    # TODO(emilyaf): Make `pad_shape_with_ones` handle nested structures.
    isd_cont_padded = util.pad_shape_with_ones(
        isd.continuous,
        ndims=example_ndims,
        start=-(self.feature_ndims.continuous + 1))
    isd_cat_padded = util.pad_shape_with_ones(
        isd.categorical,
        ndims=example_ndims,
        start=-(self.feature_ndims.categorical + 1))
    # TODO(emilyaf): Make `sum_rightmost_ndims_preserving_shape` handle nested
    # structures.
    pairwise_square_distance_cont = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(
            x1.continuous * isd_cont_padded,
            x2.continuous * isd_cont_padded),
        self.feature_ndims.continuous)
    different_categories = tf.cast(
        tf.not_equal(x1.categorical, x2.categorical), self.dtype.continuous)
    pairwise_square_distance_cat = util.sum_rightmost_ndims_preserving_shape(
        tf.square(isd_cat_padded) * different_categories,
        self.feature_ndims.categorical)
    return self.kernel._apply_with_distance(  # pylint: disable=protected-access
        x1, x2,
        pairwise_square_distance_cont + pairwise_square_distance_cat,
        example_ndims=example_ndims)

  def _matrix(self, x1, x2):
    isd = self.inverse_scale_diag_parameters()
    isd_cont_padded = util.pad_shape_with_ones(
        isd.continuous,
        ndims=1,
        start=-(self.feature_ndims.continuous + 1))
    pairwise_square_distance_cont = util.pairwise_square_distance_matrix(
        x1.continuous * isd_cont_padded,
        x2.continuous * isd_cont_padded,
        feature_ndims=self.feature_ndims.continuous)
    isd_cat_padded = util.pad_shape_with_ones(
        isd.categorical,
        ndims=2,
        start=-(self.feature_ndims.categorical + 1))
    pairwise_square_distance_cat = self._cat_pairwise_square_distance_tensor(
        x1.categorical, x2.categorical, x1_example_ndims=1, x2_example_ndims=1,
        feature_ndims=self.feature_ndims.categorical,
        inverse_scale_diag=isd_cat_padded)
    return self.kernel._apply_with_distance(  # pylint: disable=protected-access
        x1, x2,
        pairwise_square_distance_cont + pairwise_square_distance_cat,
        example_ndims=2)

  def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    isd = self.inverse_scale_diag_parameters()
    isd_cont_x1 = util.pad_shape_with_ones(
        isd.continuous,
        ndims=x1_example_ndims,
        start=-(self.feature_ndims.continuous + 1))
    isd_cont_x2 = util.pad_shape_with_ones(
        isd.continuous,
        ndims=x2_example_ndims,
        start=-(self.feature_ndims.continuous + 1))
    isd_cat = util.pad_shape_with_ones(
        isd.categorical,
        ndims=x1_example_ndims+x2_example_ndims,
        start=-(self.feature_ndims.categorical + 1))
    pairwise_square_distance_cont = util.pairwise_square_distance_tensor(
        x1.continuous * isd_cont_x1,
        x2.continuous * isd_cont_x2,
        self.feature_ndims.continuous,
        x1_example_ndims,
        x2_example_ndims)
    pairwise_square_distance_cat = self._cat_pairwise_square_distance_tensor(
        x1.categorical, x2.categorical,
        x1_example_ndims=x1_example_ndims, x2_example_ndims=x2_example_ndims,
        feature_ndims=self.feature_ndims.categorical,
        inverse_scale_diag=isd_cat)
    return self.kernel._apply_with_distance(  # pylint: disable=protected-access
        x1, x2,
        pairwise_square_distance_cont + pairwise_square_distance_cat,
        example_ndims=x1_example_ndims+x2_example_ndims)

  def _cat_pairwise_square_distance_tensor(
      self, x1, x2, x1_example_ndims, x2_example_ndims, feature_ndims,
      inverse_scale_diag):
    x1 = util.pad_shape_with_ones(
        x1,
        ndims=x2_example_ndims,
        start=-(feature_ndims + 1))
    x2 = util.pad_shape_with_ones(
        x2,
        ndims=x1_example_ndims,
        start=-(feature_ndims + 1 + x2_example_ndims))
    different_categories = tf.cast(tf.not_equal(x1, x2), self.dtype.continuous)
    return util.sum_rightmost_ndims_preserving_shape(
        different_categories * tf.square(inverse_scale_diag),
        ndims=feature_ndims)
