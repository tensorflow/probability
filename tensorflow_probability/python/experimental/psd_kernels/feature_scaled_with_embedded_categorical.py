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
"""FeatureScaled kernel over continuous and embedded categorical data."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.psd_kernels import feature_scaled_with_categorical as fswc
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util


class FeatureScaledWithEmbeddedCategorical(
    psd_kernel.AutoCompositeTensorPsdKernel):
  """`FeatureScaled` kernel for continuous and embedded categorical data.

  This kernel is an extension of `FeatureScaled` that handles categorical data
  (encoded as integers, not one-hot) in addition to continuous (float) data.
  `ContinuousAndCategoricalValues` structures, containing arrays of continuous
  and categorical data, are passed to the `apply`, `matrix` and `tensor`
  methods. The continuous inputs are scaled and then passed to the distance
  function, like in `FeatureScaled`. Categorical data, encoded as integers,
  is continuously embedded using `LinearOperator`s. When all `LinearOperator`s
  are either `LinearOperatorIdentity` or `LinearOperatorScaledIdentity`
  instances, this kernel is the same as `FeatureScaledWithCategorical`, though
  in that case the latter should be used since it will be more efficient.

  #### Examples

  Compute the kernel matrix on synthetic data.

  ```python
  import numpy as np

  continuous_dim = 3
  categorical_dim = 2

  # Define an ARD kernel that takes a structure of continuous and categorical
  # data as inputs, with randomly-sampled `continuous_scale_diag` values and
  # diagonal embeddings of categorical data.
  base_kernel = tfpk.MaternFiveHalves()
  continuous_scale_diag = np.random.uniform(size=[continuous_dim])

  # Categorical `scale_diag`s are passed as an iterable of `LinearOperator`s,
  # where each `LinearOperator` applies to a categorical feature and has number
  # of rows equivalent to the cardinality of that feature. Categorical data is
  # assumed to be represented as integers between 0 and `n - 1` inclusive, which
  # are used to index into the `inverse_scale_diag` vectors.
  num_categories = [5, 4]
  categorical_embedding_operators = [
      tf.linalg.LinearOperatorDiag(np.random.uniform(size=[n]))
      for n in num_categories]

  kernel = tfpke.FeatureScaledWithEmbeddedCategorical(
      base_kernel,
      categorical_embedding_operators=categorical_embedding_operators,
      continuous_scale_diag=continuous_scale_diag,
      validate_args=True)

  # Create `num_points` examples in the continuous/categorical feature space.
  num_points = 12
  categorical_data_1 = np.stack(
      [np.random.randint(n, size=(num_points,)) for n in num_categories])
  categorical_data_2 = np.stack(
      [np.random.randint(n, size=(num_points,)) for n in num_categories])
  x1 = tfpke.ContinuousAndCategoricalValues(
      continuous=np.random.normal(size=(num_points, continuous_dim)),
      categorical=categorical_data_1)
  x2 = tfpke.ContinuousAndCategoricalValues(
      continuous=np.random.normal(size=(num_points, continuous_dim)),
      categorical=categorical_data_2)

  # Evaluate the kernel matrix for `x1` and `x2`.
  kernel.matrix(x1, x2)  # has shape `[num_points, num_points]`

  ```
  """

  def __init__(
      self,
      kernel,
      categorical_embedding_operators,
      continuous_scale_diag=None,
      continuous_inverse_scale_diag=None,
      feature_ndims=None,
      validate_args=False,
      name='FeatureScaledWithCategorical'):
    """Construct an `FeatureScaledWithCategorical` kernel instance.

    Args:
      kernel: `PositiveSemidefiniteKernel` instance. Parameters to `kernel` must
        be broadcastable with `scale_diag`. `kernel` must be isotropic and
        implement an `_apply_with_distance` method.
      categorical_embedding_operators: Iterable of `LinearOperator` instances
        used to embed the categorical features. If the input categorical data
        has shape `[..., d]` and a single feature dimension, the iterable has
        length `d`.  Each `LinearOperator` has number of rows equal to the
        number of categories, and embeddings are equivalent to one-hot encoded
        categorical vectors multiplied by the densified `LinearOperator`.
        Euclidean distances are computed between the emeddings. If there are 0
        feature dimensions, the iterable should have length 1.
      continuous_scale_diag: Floating point array that control the
        sharpness/width of the kernel shape. Each `continuous_scale_diag` must
        have dimensionality of at least `kernel.feature_ndims.continuous`, and
        extra dimensions must be broadcastable with parameters of `kernel`.
        Default value: None.
      continuous_inverse_scale_diag: Non-negative floating point vectors that
        are treated as the reciprocals of the corresponding components of
        `continuous_scale_diag`.  Only one of `continuous_scale_diag` or
        `continuous_inverse_scale_diag` should be provided.
        Default value: None
      feature_ndims: `ContinuousAndCategoricalValues` instance containing
        integers indicating the rank of the continuous and categorical feature
        space. Default value: None, i.e. `kernel.feature_ndims` for both
        components of the feature space. Categorical `feature_ndims` > 1 is not
        supported.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    if ((continuous_scale_diag is None) ==
        (continuous_inverse_scale_diag is None)):
      raise ValueError(
          'Must specify exactly one of `continuous_scale_diag` and '
          '`continuous_inverse_scale_diag`.')
    with tf.name_scope(name):
      float_dtype = dtype_util.common_dtype(
          [kernel, continuous_scale_diag, continuous_inverse_scale_diag,
           categorical_embedding_operators],
          dtype_hint=tf.float32)
      if continuous_scale_diag is None:
        self._continuous_scale_diag = continuous_scale_diag
        self._continuous_inverse_scale_diag = (
            tensor_util.convert_nonref_to_tensor(
                continuous_inverse_scale_diag,
                dtype_hint=float_dtype,
                name='continuous_inverse_scale_diag'))
      else:
        self._continuous_inverse_scale_diag = continuous_inverse_scale_diag
        self._continuous_scale_diag = (
            tensor_util.convert_nonref_to_tensor(
                continuous_scale_diag,
                dtype_hint=float_dtype,
                name='continuous_scale_diag'))
      self._categorical_embedding_operators = categorical_embedding_operators
      self._kernel = kernel

      if feature_ndims is None:
        feature_ndims = fswc.ContinuousAndCategoricalValues(
            kernel.feature_ndims, kernel.feature_ndims)
      if feature_ndims.categorical > 1:
        raise ValueError('Categorical `feature_ndims` must be 0 or 1.')

      dtype = fswc.ContinuousAndCategoricalValues(float_dtype, None)
      super(FeatureScaledWithEmbeddedCategorical, self).__init__(
          feature_ndims=feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @property
  def kernel(self):
    return self._kernel

  @property
  def continuous_scale_diag(self):
    return self._continuous_scale_diag

  @property
  def continuous_inverse_scale_diag(self):
    return self._continuous_inverse_scale_diag

  @property
  def categorical_embedding_operators(self):
    return self._categorical_embedding_operators

  def continuous_inverse_scale_diag_parameters(self):
    inverse_scale_diag = self.continuous_inverse_scale_diag
    if inverse_scale_diag is None:
      inverse_scale_diag = tf.nest.map_structure(
          tf.math.reciprocal, self.continuous_scale_diag)
    return tf.nest.map_structure(tf.convert_to_tensor, inverse_scale_diag)

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top

    return dict(
        kernel=parameter_properties.BatchedComponentProperties(),
        continuous_scale_diag=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.feature_ndims.continuous,
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype))),
            is_preferred=False),
        continuous_inverse_scale_diag=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.feature_ndims.continuous,
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        categorical_embedding_operators=(
            parameter_properties.BatchedComponentProperties(
                event_ndims=(
                    lambda self: [0] * len(self.categorical_embedding_operators)
                ),
            )))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if self._continuous_inverse_scale_diag is not None:
      if is_init != tensor_util.is_ref(self._continuous_inverse_scale_diag):
        assertions.append(assert_util.assert_non_negative(
            self._continuous_inverse_scale_diag,
            message='`continuous_inverse_scale_diag` must be non-negative.'))
    if self._continuous_scale_diag is not None:
      if is_init != tensor_util.is_ref(self._continuous_scale_diag):
        assertions.append(assert_util.assert_positive(
            self._continuous_scale_diag,
            message='`continuous_scale_diag` must be positive.'))
    return assertions

  def _apply(self, x1, x2, example_ndims=0):
    isd = self.continuous_inverse_scale_diag_parameters()
    isd_cont_padded = util.pad_shape_with_ones(
        isd,
        ndims=example_ndims,
        start=-(self.feature_ndims.continuous + 1))
    pairwise_square_distance_cont = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(
            x1.continuous * isd_cont_padded,
            x2.continuous * isd_cont_padded),
        self.feature_ndims.continuous)
    pairwise_square_distance_cat = 0.
    if self.categorical_embedding_operators:
      pairwise_square_distance_cat = self._get_categorical_distance(
          x1.categorical, x2.categorical, example_ndims,
          self.feature_ndims.categorical)
    return self.kernel._apply_with_distance(  # pylint: disable=protected-access
        x1, x2,
        pairwise_square_distance_cont + pairwise_square_distance_cat,
        example_ndims=example_ndims)

  def _matrix(self, x1, x2):
    isd = self.continuous_inverse_scale_diag_parameters()
    isd_cont_padded = util.pad_shape_with_ones(
        isd,
        ndims=1,
        start=-(self.feature_ndims.continuous + 1))
    pairwise_square_distance_cont = util.pairwise_square_distance_matrix(
        x1.continuous * isd_cont_padded,
        x2.continuous * isd_cont_padded,
        feature_ndims=self.feature_ndims.continuous)
    pairwise_square_distance_cat = self._cat_pairwise_square_distance_tensor(
        x1.categorical, x2.categorical, x1_example_ndims=1, x2_example_ndims=1,
        feature_ndims=self.feature_ndims.categorical,
        inverse_scale_diag=self.categorical_embedding_operators)
    return self.kernel._apply_with_distance(  # pylint: disable=protected-access
        x1, x2,
        pairwise_square_distance_cont + pairwise_square_distance_cat,
        example_ndims=2)

  def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    isd = self.continuous_inverse_scale_diag_parameters()
    isd_cont_x1 = util.pad_shape_with_ones(
        isd,
        ndims=x1_example_ndims,
        start=-(self.feature_ndims.continuous + 1))
    isd_cont_x2 = util.pad_shape_with_ones(
        isd,
        ndims=x2_example_ndims,
        start=-(self.feature_ndims.continuous + 1))
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
        inverse_scale_diag=self.categorical_embedding_operators)
    return self.kernel._apply_with_distance(  # pylint: disable=protected-access
        x1, x2,
        pairwise_square_distance_cont + pairwise_square_distance_cat,
        example_ndims=x1_example_ndims+x2_example_ndims)

  def _get_categorical_distance(self, x1, x2, example_ndims, feature_ndims):
    x_batch, _ = ps.split(
        ps.broadcast_shape(ps.shape(x1), ps.shape(x2)),
        num_or_size_splits=[-1, example_ndims + 1])
    bcast_shape = ps.broadcast_shape(x_batch, self.batch_shape_tensor())
    batch_rank = ps.size(bcast_shape)

    def _get_categorical_distance_one_feature(x1_, x2_, isd):
      if isinstance(isd, tf.linalg.LinearOperatorIdentity):
        return tf.cast(tf.not_equal(x1_, x2_), dtype=isd.dtype) * 2.
      if isinstance(isd, tf.linalg.LinearOperatorScaledIdentity):
        return tf.where(
            tf.equal(x1_, x2_),
            tf.zeros([], dtype=isd.dtype),
            2. * isd.multiplier ** 2)

      x1_ = x1_[..., tf.newaxis]
      x2_ = x2_[..., tf.newaxis]
      x1_bcast = ps.broadcast_to(
          x1_,
          ps.concat([bcast_shape, ps.shape(x1_)[-(example_ndims + 1):]], axis=0)
      )
      x2_bcast = ps.broadcast_to(
          x2_,
          ps.concat([bcast_shape, ps.shape(x2_)[-(example_ndims + 1):]], axis=0)
      )
      if isinstance(isd, tf.linalg.LinearOperatorDiag):
        diag_bcast = tf.broadcast_to(isd.diag, ps.concat(
            [bcast_shape, ps.shape(isd.diag)[-1:]], axis=0))
        x1_embedding = tf.gather_nd(diag_bcast, x1_bcast, batch_dims=batch_rank)
        x2_embedding = tf.gather_nd(diag_bcast, x2_bcast, batch_dims=batch_rank)
        return tf.where(
            tf.equal(x1_[..., 0], x2_[..., 0]),
            tf.zeros([], dtype=isd.dtype),
            x1_embedding ** 2 + x2_embedding ** 2)

      isd_mat = isd.to_dense()
      isd_bcast = tf.broadcast_to(
          isd_mat,
          ps.concat([bcast_shape, ps.shape(isd_mat)[-2:]], axis=0))
      x1_embedding = tf.gather_nd(isd_bcast, x1_bcast, batch_dims=batch_rank)
      x2_embedding = tf.gather_nd(isd_bcast, x2_bcast, batch_dims=batch_rank)
      # TODO(emilyaf): Use `util.pairwise_square_distance_tensor` if necessary
      # for high-cardinality categorical features.
      return util.sum_rightmost_ndims_preserving_shape(
          tf.math.squared_difference(x1_embedding, x2_embedding),
          1)

    if feature_ndims == 0:
      return _get_categorical_distance_one_feature(
          x1, x2, self.categorical_embedding_operators[0]
      )

    distances = tf.nest.map_structure(
        _get_categorical_distance_one_feature,
        ps.unstack(x1, axis=-1),
        ps.unstack(x2, axis=-1),
        self.categorical_embedding_operators
        )
    return util.sum_rightmost_ndims_preserving_shape(
        tf.stack(distances, axis=-1), feature_ndims
    )

  def _cat_pairwise_square_distance_tensor(
      self, x1, x2, x1_example_ndims, x2_example_ndims, feature_ndims,
      inverse_scale_diag):
    if not inverse_scale_diag:
      return 0.
    x1 = util.pad_shape_with_ones(
        x1,
        ndims=x2_example_ndims,
        start=-(feature_ndims + 1))
    x2 = util.pad_shape_with_ones(
        x2,
        ndims=x1_example_ndims,
        start=-(feature_ndims + 1 + x2_example_ndims))
    example_ndims = x1_example_ndims + x2_example_ndims
    return self._get_categorical_distance(x1, x2, example_ndims, feature_ndims)
