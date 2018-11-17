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
"""Student's t distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops


__all__ = [
    "StudentT",
]


class StudentT(distribution.Distribution):
  """Student's t-distribution.

  This distribution has parameters: degree of freedom `df`, location `loc`,
  and `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, mu, sigma) = (1 + y**2 / df)**(-0.5 (df + 1)) / Z
  where,
  y = (x - mu) / sigma
  Z = abs(sigma) sqrt(df pi) Gamma(0.5 df) / Gamma(0.5 (df + 1))
  ```

  where:
  * `loc = mu`,
  * `scale = sigma`, and,
  * `Z` is the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The StudentT distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ StudentT(df, loc=0, scale=1)
  Y = loc + scale * X
  ```

  Notice that `scale` has semantics more similar to standard deviation than
  variance. However it is not actually the std. deviation; the Student's
  t-distribution std. dev. is `scale sqrt(df / (df - 2))` when `df > 2`.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in the paper

  [Michael Figurnov, Shakir Mohamed, Andriy Mnih.
  Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Student t distribution.
  single_dist = tfd.StudentT(df=3)

  # Evaluate the pdf at 1, returning a scalar Tensor.
  single_dist.prob(1.)

  # Define a batch of two scalar valued Student t's.
  # The first has degrees of freedom 2, mean 1, and scale 11.
  # The second 3, 2 and 22.
  multi_dist = tfd.StudentT(df=[2, 3], loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  multi_dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  multi_dist.sample(3)
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two Student's t distributions.
  # Both have df 2 and mean 1, but different scales.
  dist = tfd.StudentT(df=2, loc=1, scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  df = tf.constant(2.0)
  loc = tf.constant(2.0)
  scale = tf.constant(11.0)
  dist = tfd.StudentT(df=df, loc=loc, scale=scale)
  samples = dist.sample(5)  # Shape [5]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, [df, loc, scale])
  ```

  """

  def __init__(self,
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="StudentT"):
    """Construct Student's t distributions.

    The distributions have degree of freedom `df`, mean `loc`, and scale
    `scale`.

    The parameters `df`, `loc`, and `scale` must be shaped in a way that
    supports broadcasting (e.g. `df + loc + scale` is a valid operation).

    Args:
      df: Floating-point `Tensor`. The degrees of freedom of the
        distribution(s). `df` must contain only positive values.
      loc: Floating-point `Tensor`. The mean(s) of the distribution(s).
      scale: Floating-point `Tensor`. The scaling factor(s) for the
        distribution(s). Note that `scale` is not technically the standard
        deviation of this distribution but has semantics more similar to
        standard deviation than variance.
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
      TypeError: if loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name, values=[df, loc, scale]) as name:
      dtype = dtype_util.common_dtype([df, loc, scale], tf.float32)
      df = tf.convert_to_tensor(df, name="df", dtype=dtype)
      loc = tf.convert_to_tensor(loc, name="loc", dtype=dtype)
      scale = tf.convert_to_tensor(scale, name="scale", dtype=dtype)
      with tf.control_dependencies([tf.assert_positive(df)]
                                   if validate_args else []):
        self._df = tf.identity(df)
        self._loc = tf.identity(loc)
        self._scale = tf.identity(scale)
        tf.assert_same_float_dtype(
            (self._df, self._loc, self._scale))
    super(StudentT, self).__init__(
        dtype=self._scale.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._df, self._loc, self._scale],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("df", "loc", "scale"), (
            [tf.convert_to_tensor(
                sample_shape, dtype=tf.int32)] * 3)))

  @property
  def df(self):
    """Degrees of freedom in these Student's t distribution(s)."""
    return self._df

  @property
  def loc(self):
    """Locations of these Student's t distribution(s)."""
    return self._loc

  @property
  def scale(self):
    """Scaling factors of these Student's t distribution(s)."""
    return self._scale

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(self.df),
        tf.broadcast_dynamic_shape(
            tf.shape(self.loc), tf.shape(self.scale)))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        tf.broadcast_static_shape(self.df.shape,
                                  self.loc.shape),
        self.scale.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # The sampling method comes from the fact that if:
    #   X ~ Normal(0, 1)
    #   Z ~ Chi2(df)
    #   Y = X / sqrt(Z / df)
    # then:
    #   Y ~ StudentT(df).
    seed = seed_stream.SeedStream(seed, "student_t")
    shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    normal_sample = tf.random_normal(shape, dtype=self.dtype, seed=seed())
    df = self.df * tf.ones(self.batch_shape_tensor(), dtype=self.dtype)
    gamma_sample = tf.random_gamma(
        [n],
        0.5 * df,
        beta=0.5,
        dtype=self.dtype,
        seed=seed())
    samples = normal_sample * tf.rsqrt(gamma_sample / df)
    return samples * self.scale + self.loc  # Abs(scale) not wanted.

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _log_unnormalized_prob(self, x):
    y = (x - self.loc) / self.scale  # Abs(scale) superfluous.
    return -0.5 * (self.df + 1.) * tf.log1p(y**2. / self.df)

  def _log_normalization(self):
    return (tf.log(tf.abs(self.scale)) +
            0.5 * tf.log(self.df) +
            0.5 * np.log(np.pi) +
            tf.lgamma(0.5 * self.df) -
            tf.lgamma(0.5 * (self.df + 1.)))

  def _cdf(self, x):
    # Take Abs(scale) to make subsequent where work correctly.
    y = (x - self.loc) / tf.abs(self.scale)
    x_t = self.df / (y**2. + self.df)
    neg_cdf = 0.5 * tf.betainc(0.5 * self.df, 0.5, x_t)
    return tf.where(tf.less(y, 0.), neg_cdf, 1. - neg_cdf)

  def _entropy(self):
    v = tf.ones(self.batch_shape_tensor(),
                dtype=self.dtype)[..., tf.newaxis]
    u = v * self.df[..., tf.newaxis]
    beta_arg = tf.concat([u, v], -1) / 2.
    return (tf.log(tf.abs(self.scale)) +
            0.5 * tf.log(self.df) +
            tf.lbeta(beta_arg) +
            0.5 * (self.df + 1.) *
            (tf.digamma(0.5 * (self.df + 1.)) -
             tf.digamma(0.5 * self.df)))

  @distribution_util.AppendDocstring(
      """The mean of Student's T equals `loc` if `df > 1`, otherwise it is
      `NaN`. If `self.allow_nan_stats=True`, then an exception will be raised
      rather than returning `NaN`.""")
  def _mean(self):
    mean = self.loc * tf.ones(self.batch_shape_tensor(),
                              dtype=self.dtype)
    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return tf.where(
          tf.greater(
              self.df,
              tf.ones(self.batch_shape_tensor(), dtype=self.dtype)),
          mean,
          tf.fill(self.batch_shape_tensor(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies(
          [
              tf.assert_less(
                  tf.ones([], dtype=self.dtype),
                  self.df,
                  message="mean not defined for components of df <= 1"),
          ],
          mean)

  @distribution_util.AppendDocstring("""
      The variance for Student's T equals

      ```
      df / (df - 2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      ```
      """)
  def _variance(self):
    # We need to put the tf.where inside the outer tf.where to ensure we never
    # hit a NaN in the gradient.
    denom = tf.where(tf.greater(self.df, 2.),
                     self.df - 2.,
                     tf.ones_like(self.df))
    # Abs(scale) superfluous.
    var = (tf.ones(self.batch_shape_tensor(), dtype=self.dtype) *
           tf.square(self.scale) * self.df / denom)
    # When 1 < df <= 2, variance is infinite.
    inf = np.array(np.inf, dtype=self.dtype.as_numpy_dtype())
    result_where_defined = tf.where(
        self.df > tf.fill(self.batch_shape_tensor(), 2.),
        var,
        tf.fill(self.batch_shape_tensor(), inf, name="inf"))

    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return tf.where(
          tf.greater(
              self.df,
              tf.ones(self.batch_shape_tensor(), dtype=self.dtype)),
          result_where_defined,
          tf.fill(self.batch_shape_tensor(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies(
          [
              tf.assert_less(
                  tf.ones([], dtype=self.dtype),
                  self.df,
                  message="variance not defined for components of df <= 1"),
          ],
          result_where_defined)

  def _mode(self):
    return tf.identity(self.loc)
