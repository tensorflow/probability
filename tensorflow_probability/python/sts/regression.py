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
    """Build a state space model implementing linear regression.

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
    dummy_mvndiag = tfd.MultivariateNormalDiag(
        scale_diag=tf.ones([0], dtype=dtype))
    dummy_mvndiag.covariance = lambda: dummy_mvndiag.variance()[..., tf.newaxis]
    if initial_state_prior is None:
      initial_state_prior = dummy_mvndiag

    def observation_noise_fn(t):
      predicted_slice = predicted_timeseries[..., t, :]
      return tfd.MultivariateNormalDiag(
          loc=predicted_slice,
          scale_diag=tf.zeros_like(predicted_slice))

    return tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=tf.zeros([0, 0], dtype=dtype),
        transition_noise=dummy_mvndiag,
        observation_matrix=tf.zeros([1, 0], dtype=dtype),
        observation_noise=observation_noise_fn,
        initial_state_prior=initial_state_prior,
        initial_step=initial_step)
