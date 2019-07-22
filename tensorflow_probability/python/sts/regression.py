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
"""Regression components."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries

tfl = tf.linalg


def _zero_dimensional_mvndiag(dtype):
  """Build a zero-dimensional MVNDiag object."""
  dummy_mvndiag = tfd.MultivariateNormalDiag(
      scale_diag=tf.ones([0], dtype=dtype))
  dummy_mvndiag.covariance = lambda: dummy_mvndiag.variance()[..., tf.newaxis]
  return dummy_mvndiag


def _observe_timeseries_fn(timeseries):
  """Build an observation_noise_fn that observes a Tensor timeseries."""
  def observation_noise_fn(t):
    current_slice = timeseries[..., t, :]
    return tfd.MultivariateNormalDiag(
        loc=current_slice,
        scale_diag=tf.zeros_like(current_slice))
  return observation_noise_fn


class LinearRegression(StructuralTimeSeries):
  """Formal representation of a linear regression from provided covariates.

  This model defines a time series given by a linear combination of
  covariate time series provided in a design matrix:

  ```python
  observed_time_series = matmul(design_matrix, weights)
  ```

  The design matrix has shape `[num_timesteps, num_features]`. The weights
  are treated as an unknown random variable of size `[num_features]` (both
  components also support batch shape), and are integrated over using the same
  approximate inference tools as other model parameters, i.e., generally HMC or
  variational inference.

  This component does not itself include observation noise; it defines a
  deterministic distribution with mass at the point
  `matmul(design_matrix, weights)`. In practice, it should be combined with
  observation noise from another component such as `tfp.sts.Sum`, as
  demonstrated below.

  #### Examples

  Given `series1`, `series2` as `Tensors` each of shape `[num_timesteps]`
  representing covariate time series, we create a regression model that
  conditions on these covariates:

  ```python
  regression = tfp.sts.LinearRegression(
    design_matrix=tf.stack([series1, series2], axis=-1),
    weights_prior=tfd.Normal(loc=0., scale=1.))
  ```

  Here we've also demonstrated specifying a custom prior, using an informative
  `Normal(0., 1.)` prior instead of the default weakly-informative prior.

  As a more advanced application, we might use the design matrix to encode
  holiday effects. For example, suppose we are modeling data from the month of
  December. We can combine day-of-week seasonality with special effects for
  Christmas Eve (Dec 24), Christmas (Dec 25), and New Year's Eve (Dec 31),
  by constructing a design matrix with indicators for those dates.

  ```python
  holiday_indicators = np.zeros([31, 3])
  holiday_indicators[23, 0] = 1  # Christmas Eve
  holiday_indicators[24, 1] = 1  # Christmas Day
  holiday_indicators[30, 2] = 1  # New Year's Eve

  holidays = tfp.sts.LinearRegression(design_matrix=holiday_indicators,
                                      name='holidays')
  day_of_week = tfp.sts.Seasonal(num_seasons=7,
                                 observed_time_series=observed_time_series,
                                 name='day_of_week')
  model = tfp.sts.Sum(components=[holidays, seasonal],
                      observed_time_series=observed_time_series)
  ```

  Note that the `Sum` component in the above model also incorporates observation
  noise, with prior scale heuristically inferred from `observed_time_series`.

  In these examples, we've used a single design matrix, but batching is
  also supported. If the design matrix has batch shape, the default behavior
  constructs weights with matching batch shape, which will fit a separate
  regression for each design matrix. This can be overridden by passing an
  explicit weights prior with appropriate batch shape. For example, if each
  design matrix in a batch contains features with the same semantics
  (e.g., if they represent per-group or per-observation covariates), we might
  choose to share statistical strength by fitting a single weight vector that
  broadcasts across all design matrices:

  ```python
  design_matrix = get_batch_of_inputs()
  design_matrix.shape  # => concat([batch_shape, [num_timesteps, num_features]])

  # Construct a prior with batch shape `[]` and event shape `[num_features]`,
  # so that it describes a single vector of weights.
  weights_prior = tfd.Independent(
      tfd.StudentT(df=5,
                   loc=tf.zeros([num_features]),
                   scale=tf.ones([num_features])),
      reinterpreted_batch_ndims=1)
  linear_regression = LinearRegression(design_matrix=design_matrix,
                                       weights_prior=weights_prior)
  ```

  """

  def __init__(self,
               design_matrix,
               weights_prior=None,
               name=None):
    """Specify a linear regression model.

    Note: the statistical behavior of the regression is determined by
    the broadcasting behavior of the `weights` `Tensor`:

    * `weights_prior.batch_shape == []`: shares a single set of weights across
      all design matrices and observed time series. This may make sense if
      the features in each design matrix have the same semantics (e.g.,
      grouping observations by country, with per-country design matrices
      capturing the same set of national economic indicators per country).
    * `weights_prior.batch_shape == `design_matrix.batch_shape`: fits separate
      weights for each design matrix. If there are multiple observed time series
      for each design matrix, this shares statistical strength over those
      observations.
    * `weights_prior.batch_shape == `observed_time_series.batch_shape`: fits a
      separate regression for each individual time series.

    When modeling batches of time series, you should think carefully about
    which behavior makes sense, and specify `weights_prior` accordingly:
    the defaults may not do what you want!

    Args:
      design_matrix: float `Tensor` of shape `concat([batch_shape,
        [num_timesteps, num_features]])`. This may also optionally be
        an instance of `tf.linalg.LinearOperator`.
      weights_prior: `tfd.Distribution` representing a prior over the regression
        weights. Must have event shape `[num_features]` and batch shape
        broadcastable to the design matrix's `batch_shape`. Alternately,
        `event_shape` may be scalar (`[]`), in which case the prior is
        internally broadcast as `TransformedDistribution(weights_prior,
        tfb.Identity(), event_shape=[num_features],
        batch_shape=design_matrix.batch_shape)`. If `None`,
        defaults to `StudentT(df=5, loc=0., scale=10.)`, a weakly-informative
        prior loosely inspired by the [Stan prior choice recommendations](
        https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations).
        Default value: `None`.
      name: the name of this model component.
        Default value: 'LinearRegression'.
    """
    with tf.compat.v1.name_scope(
        name, 'LinearRegression', values=[design_matrix]) as name:

      if not isinstance(design_matrix, tfl.LinearOperator):
        design_matrix = tfl.LinearOperatorFullMatrix(
            tf.convert_to_tensor(value=design_matrix, name='design_matrix'),
            name='design_matrix_linop')

      if tf.compat.dimension_value(design_matrix.shape[-1]) is not None:
        num_features = design_matrix.shape[-1]
      else:
        num_features = design_matrix.shape_tensor()[-1]

      # Default to a weakly-informative StudentT(df=5, 0., 10.) prior.
      if weights_prior is None:
        weights_prior = tfd.StudentT(
            df=5,
            loc=tf.zeros([], dtype=design_matrix.dtype),
            scale=10 * tf.ones([], dtype=design_matrix.dtype))
      # Sugar: if prior is static scalar, broadcast it to a default shape.
      if weights_prior.event_shape.ndims == 0:
        if design_matrix.batch_shape.is_fully_defined():
          design_matrix_batch_shape_ = design_matrix.batch_shape
        else:
          design_matrix_batch_shape_ = design_matrix.batch_shape_tensor()
        weights_prior = tfd.TransformedDistribution(
            weights_prior,
            bijector=tfb.Identity(),
            batch_shape=design_matrix_batch_shape_,
            event_shape=[num_features])

      tf.debugging.assert_same_float_dtype([design_matrix, weights_prior])

      self._design_matrix = design_matrix

      super(LinearRegression, self).__init__(
          parameters=[
              Parameter('weights', weights_prior, tfb.Identity()),
          ],
          latent_size=0,
          name=name)

  @property
  def design_matrix(self):
    """LinearOperator representing the design matrix."""
    return self._design_matrix

  def _make_state_space_model(self,
                              num_timesteps,
                              param_map,
                              initial_state_prior=None,
                              initial_step=0):

    weights = param_map['weights']  # shape: [B, num_features]
    predicted_timeseries = self.design_matrix.matmul(weights[..., tf.newaxis])

    dtype = self.design_matrix.dtype

    # Since this model has `latent_size=0`, the latent prior and
    # transition model are dummy objects (zero-dimensional MVNs).
    dummy_mvndiag = _zero_dimensional_mvndiag(dtype)
    if initial_state_prior is None:
      initial_state_prior = dummy_mvndiag

    return tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=tf.zeros([0, 0], dtype=dtype),
        transition_noise=dummy_mvndiag,
        observation_matrix=tf.zeros([1, 0], dtype=dtype),
        observation_noise=_observe_timeseries_fn(predicted_timeseries),
        initial_state_prior=initial_state_prior,
        initial_step=initial_step)


class SparseLinearRegression(StructuralTimeSeries):
  """Formal representation of a sparse linear regression.

  This model defines a time series given by a sparse linear combination of
  covariate time series provided in a design matrix:

  ```python
  observed_time_series = matmul(design_matrix, weights)
  ```

  This is identical to `tfp.sts.LinearRegression`, except that
  `SparseLinearRegression` uses a parameterization of a Horseshoe
  prior [1][2] to encode the assumption that many of the `weights` are zero,
  i.e., many of the covariate time series are irrelevant. See the mathematical
  details section below for further discussion. The prior parameterization used
  by `SparseLinearRegression` is more suitable for inference than that
  obtained by simply passing the equivalent `tfd.Horseshoe` prior to
  `LinearRegression`; when sparsity is desired, `SparseLinearRegression` will
  likely yield better results.

  This component does not itself include observation noise; it defines a
  deterministic distribution with mass at the point
  `matmul(design_matrix, weights)`. In practice, it should be combined with
  observation noise from another component such as `tfp.sts.Sum`, as
  demonstrated below.

  #### Examples

  Given `series1`, `series2` as `Tensors` each of shape `[num_timesteps]`
  representing covariate time series, we create a regression model that
  conditions on these covariates:

  ```python
  regression = tfp.sts.SparseLinearRegression(
    design_matrix=tf.stack([series1, series2], axis=-1),
    weights_prior_scale=0.1)
  ```

  The `weights_prior_scale` determines the level of sparsity; small
  scales encourage the weights to be sparse. In some cases, such as when
  the likelihood is iid Gaussian with known scale, the prior scale can be
  analytically related to the expected number of nonzero weights [2]; however,
  this is not the case in general for STS models.

  If the design matrix has batch dimensions, by default the model will create a
  matching batch of weights. For example, if `design_matrix.shape == [
  num_users, num_timesteps, num_features]`, by default the model will fit
  separate weights for each user, i.e., it will internally represent
  `weights.shape == [num_users, num_features]`. To share weights across some or
  all batch dimensions, you can manually specify the batch shape for the
  weights:

  ```python
  # design_matrix.shape == [num_users, num_timesteps, num_features]
  regression = tfp.sts.SparseLinearRegression(
    design_matrix=design_matrix,
    weights_batch_shape=[])  # weights.shape -> [num_features]
  ```

  #### Mathematical Details

  The basic horseshoe prior [1] is defined as a Cauchy-normal scale mixture:

  ```
  scales[i] ~ HalfCauchy(loc=0, scale=1)
  weights[i] ~ Normal(loc=0., scale=scales[i] * global_scale)`
  ```

  The Cauchy scale parameters puts substantial mass near zero, encouraging
  weights to be sparse, but their heavy tails allow weights far from zero to be
  estimated without excessive shrinkage. The horseshoe can be thought of as a
  continuous relaxation of a traditional 'spike-and-slab' discrete sparsity
  prior, in which the latent Cauchy scale mixes between 'spike'
  (`scales[i] ~= 0`) and 'slab' (`scales[i] >> 0`) regimes.

  Following the recommendations in [2], `SparseLinearRegression` implements
  a horseshoe with the following adaptations:

  - The Cauchy prior on `scales[i]` is represented as an InverseGamma-Normal
    compound.
  - The `global_scale` parameter is integrated out following a `Cauchy(0.,
    scale=weights_prior_scale)` hyperprior, which is also represented as an
    InverseGamma-Normal compound.
  - All compound distributions are implemented using a non-centered
    parameterization.

  The compound, non-centered representation defines the same marginal prior as
  the original horseshoe (up to integrating out the global scale),
  but allows samplers to mix more efficiently through the heavy tails; for
  variational inference, the compound representation implicity expands the
  representational power of the variational model.

  Note that we do not yet implement the regularized ('Finnish') horseshoe,
  proposed in [2] for models with weak likelihoods, because the likelihood
  in STS models is typically Gaussian, where it's not clear that additional
  regularization is appropriate. If you need this functionality, please
  email tfprobability@tensorflow.org.

  The full prior parameterization implemented in `SparseLinearRegression` is
  as follows:

  ```
  # Sample global_scale from Cauchy(0, scale=weights_prior_scale).
  global_scale_variance ~ InverseGamma(alpha=0.5, beta=0.5)
  global_scale_noncentered ~ HalfNormal(loc=0, scale=1)
  global_scale = (global_scale_noncentered *
                  sqrt(global_scale_variance) *
                  weights_prior_scale)

  # Sample local_scales from Cauchy(0, 1).
  local_scale_variances[i] ~ InverseGamma(alpha=0.5, beta=0.5)
  local_scales_noncentered[i] ~ HalfNormal(loc=0, scale=1)
  local_scales[i] = local_scales_noncentered[i] * sqrt(local_scale_variances[i])

  weights[i] ~ Normal(loc=0., scale=local_scales[i] * global_scale)
  ```

  #### References

  [1]: Carvalho, C., Polson, N. and Scott, J. Handling Sparsity via the
    Horseshoe. AISTATS (2009).
    http://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf
  [2]: Juho Piironen, Aki Vehtari. Sparsity information and regularization in
    the horseshoe and other shrinkage priors (2017).
    https://arxiv.org/abs/1707.01694

  """

  def __init__(self,
               design_matrix,
               weights_prior_scale=0.1,
               weights_batch_shape=None,
               name=None):
    """Specify a sparse linear regression model.

    Args:
      design_matrix: float `Tensor` of shape `concat([batch_shape,
        [num_timesteps, num_features]])`. This may also optionally be
        an instance of `tf.linalg.LinearOperator`.
      weights_prior_scale: float `Tensor` defining the scale of the Horseshoe
        prior on regression weights. Small values encourage the weights to be
        sparse. The shape must broadcast with `weights_batch_shape`.
        Default value: `0.1`.
      weights_batch_shape: if `None`, defaults to
        `design_matrix.batch_shape_tensor()`. Must broadcast with the batch
        shape of `design_matrix`.
        Default value: `None`.
      name: the name of this model component.
        Default value: 'SparseLinearRegression'.
    """
    with tf.compat.v1.name_scope(
        name, 'SparseLinearRegression',
        values=[design_matrix, weights_prior_scale]) as name:

      if not isinstance(design_matrix, tfl.LinearOperator):
        design_matrix = tfl.LinearOperatorFullMatrix(
            tf.convert_to_tensor(value=design_matrix, name='design_matrix'),
            name='design_matrix_linop')

      if tf.compat.dimension_value(design_matrix.shape[-1]) is not None:
        num_features = design_matrix.shape[-1]
      else:
        num_features = design_matrix.shape_tensor()[-1]

      if weights_batch_shape is None:
        weights_batch_shape = design_matrix.batch_shape_tensor()
      else:
        weights_batch_shape = tf.convert_to_tensor(value=weights_batch_shape,
                                                   dtype=tf.int32)
      weights_shape = tf.concat([weights_batch_shape, [num_features]], axis=0)

      dtype = design_matrix.dtype

      self._design_matrix = design_matrix
      self._weights_prior_scale = weights_prior_scale

      ones_like_weights_batch = tf.ones(weights_batch_shape, dtype=dtype)
      ones_like_weights = tf.ones(weights_shape, dtype=dtype)
      super(SparseLinearRegression, self).__init__(
          parameters=[
              Parameter('global_scale_variance',
                        prior=tfd.InverseGamma(
                            0.5 * ones_like_weights_batch,
                            0.5 * ones_like_weights_batch),
                        bijector=tfb.Softplus()),
              Parameter('global_scale_noncentered',
                        prior=tfd.HalfNormal(
                            scale=ones_like_weights_batch),
                        bijector=tfb.Softplus()),
              Parameter('local_scale_variances',
                        prior=tfd.Independent(tfd.InverseGamma(
                            0.5 * ones_like_weights,
                            0.5 * ones_like_weights),
                                              reinterpreted_batch_ndims=1),
                        bijector=tfb.Softplus()),
              Parameter('local_scales_noncentered',
                        prior=tfd.Independent(tfd.HalfNormal(
                            scale=ones_like_weights),
                                              reinterpreted_batch_ndims=1),
                        bijector=tfb.Softplus()),
              Parameter('weights_noncentered',
                        prior=tfd.Independent(tfd.Normal(
                            loc=tf.zeros_like(ones_like_weights),
                            scale=ones_like_weights),
                                              reinterpreted_batch_ndims=1),
                        bijector=tfb.Identity())
          ],
          latent_size=0,
          name=name)

  @property
  def design_matrix(self):
    """LinearOperator representing the design matrix."""
    return self._design_matrix

  @property
  def weights_prior_scale(self):
    return self._weights_prior_scale

  def params_to_weights(self,
                        global_scale_variance,
                        global_scale_noncentered,
                        local_scale_variances,
                        local_scales_noncentered,
                        weights_noncentered):
    """Build regression weights from model parameters."""
    global_scale = (global_scale_noncentered *
                    tf.sqrt(global_scale_variance) *
                    self.weights_prior_scale)

    local_scales = local_scales_noncentered * tf.sqrt(local_scale_variances)
    return weights_noncentered * local_scales * global_scale[..., tf.newaxis]

  def _make_state_space_model(self,
                              num_timesteps,
                              param_map,
                              initial_state_prior=None,
                              initial_step=0):

    weights = self.params_to_weights(**param_map)
    predicted_timeseries = self.design_matrix.matmul(weights[..., tf.newaxis])

    dtype = self.design_matrix.dtype

    # Since this model has `latent_size=0`, the latent prior and
    # transition model are dummy objects (zero-dimensional MVNs).
    dummy_mvndiag = _zero_dimensional_mvndiag(dtype)
    if initial_state_prior is None:
      initial_state_prior = dummy_mvndiag

    return tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=tf.zeros([0, 0], dtype=dtype),
        transition_noise=dummy_mvndiag,
        observation_matrix=tf.zeros([1, 0], dtype=dtype),
        observation_noise=_observe_timeseries_fn(predicted_timeseries),
        initial_state_prior=initial_state_prior,
        initial_step=initial_step)
