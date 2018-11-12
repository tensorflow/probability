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
"""The Beta distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import distribution_util as util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops


__all__ = [
    "Beta",
]


_beta_sample_note = """Note: `x` must have dtype `self.dtype` and be in
`[0, 1].` It must have a shape compatible with `self.batch_shape()`."""


class Beta(distribution.Distribution):
  """Beta distribution.

  The Beta distribution is defined over the `(0, 1)` interval using parameters
  `concentration1` (aka "alpha") and `concentration0` (aka "beta").

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
  Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
  ```

  where:

  * `concentration1 = alpha`,
  * `concentration0 = beta`,
  * `Z` is the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The concentration parameters represent mean total counts of a `1` or a `0`,
  i.e.,

  ```none
  concentration1 = alpha = mean * total_concentration
  concentration0 = beta  = (1. - mean) * total_concentration
  ```

  where `mean` in `(0, 1)` and `total_concentration` is a positive real number
  representing a mean `total_count = concentration1 + concentration0`.

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Warning: The samples can be zero due to finite precision.
  This happens more often when some of the concentrations are very small.
  Make sure to round the samples to `np.finfo(dtype).tiny` before computing the
  density.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in the paper

  [Michael Figurnov, Shakir Mohamed, Andriy Mnih.
  Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Create a batch of three Beta distributions.
  alpha = [1, 2, 3]
  beta = [1, 2, 3]
  dist = tfd.Beta(alpha, beta)

  dist.sample([4, 5])  # Shape [4, 5, 3]

  # `x` has three batch entries, each with two samples.
  x = [[.1, .4, .5],
       [.2, .3, .5]]
  # Calculate the probability of each pair of samples under the corresponding
  # distribution in `dist`.
  dist.prob(x)         # Shape [2, 3]
  ```

  ```python
  # Create batch_shape=[2, 3] via parameter broadcast:
  alpha = [[1.], [2]]      # Shape [2, 1]
  beta = [3., 4, 5]        # Shape [3]
  dist = tfd.Beta(alpha, beta)

  # alpha broadcast as: [[1., 1, 1,],
  #                      [2, 2, 2]]
  # beta broadcast as:  [[3., 4, 5],
  #                      [3, 4, 5]]
  # batch_Shape [2, 3]
  dist.sample([4, 5])  # Shape [4, 5, 2, 3]

  x = [.2, .3, .5]
  # x will be broadcast as [[.2, .3, .5],
  #                         [.2, .3, .5]],
  # thus matching batch_shape [2, 3].
  dist.prob(x)         # Shape [2, 3]
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  alpha = tf.constant(1.0)
  beta = tf.constant(2.0)
  dist = tfd.Beta(alpha, beta)
  samples = dist.sample(5)  # Shape [5]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, [alpha, beta])
  ```

  """

  def __init__(self,
               concentration1=None,
               concentration0=None,
               validate_args=False,
               allow_nan_stats=True,
               name="Beta"):
    """Initialize a batch of Beta distributions.

    Args:
      concentration1: Positive floating-point `Tensor` indicating mean
        number of successes; aka "alpha". Implies `self.dtype` and
        `self.batch_shape`, i.e.,
        `concentration1.shape = [N1, N2, ..., Nm] = self.batch_shape`.
      concentration0: Positive floating-point `Tensor` indicating mean
        number of failures; aka "beta". Otherwise has same semantics as
        `concentration1`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name, values=[concentration1, concentration0]) as name:
      dtype = dtype_util.common_dtype([concentration1, concentration0],
                                      tf.float32)
      self._concentration1 = self._maybe_assert_valid_concentration(
          tf.convert_to_tensor(
              concentration1, name="concentration1", dtype=dtype),
          validate_args)
      self._concentration0 = self._maybe_assert_valid_concentration(
          tf.convert_to_tensor(
              concentration0, name="concentration0", dtype=dtype),
          validate_args)
      tf.assert_same_float_dtype([self._concentration1, self._concentration0])
      self._total_concentration = self._concentration1 + self._concentration0
    super(Beta, self).__init__(
        dtype=dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[
            self._concentration1, self._concentration0,
            self._total_concentration
        ],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(zip(
        ["concentration1", "concentration0"],
        [tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2))

  @property
  def concentration1(self):
    """Concentration parameter associated with a `1` outcome."""
    return self._concentration1

  @property
  def concentration0(self):
    """Concentration parameter associated with a `0` outcome."""
    return self._concentration0

  @property
  def total_concentration(self):
    """Sum of concentration parameters."""
    return self._total_concentration

  def _batch_shape_tensor(self):
    return tf.shape(self.total_concentration)

  def _batch_shape(self):
    return self.total_concentration.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    seed = seed_stream.SeedStream(seed, "beta")
    expanded_concentration1 = tf.ones_like(
        self.total_concentration, dtype=self.dtype) * self.concentration1
    expanded_concentration0 = tf.ones_like(
        self.total_concentration, dtype=self.dtype) * self.concentration0
    gamma1_sample = tf.random_gamma(
        shape=[n],
        alpha=expanded_concentration1,
        dtype=self.dtype,
        seed=seed())
    gamma2_sample = tf.random_gamma(
        shape=[n],
        alpha=expanded_concentration0,
        dtype=self.dtype,
        seed=seed())
    beta_sample = gamma1_sample / (gamma1_sample + gamma2_sample)
    return beta_sample

  @util.AppendDocstring(_beta_sample_note)
  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  @util.AppendDocstring(_beta_sample_note)
  def _prob(self, x):
    return tf.exp(self._log_prob(x))

  @util.AppendDocstring(_beta_sample_note)
  def _log_cdf(self, x):
    return tf.log(self._cdf(x))

  @util.AppendDocstring(_beta_sample_note)
  def _cdf(self, x):
    return tf.betainc(self.concentration1, self.concentration0, x)

  def _log_unnormalized_prob(self, x):
    x = self._maybe_assert_valid_sample(x)
    return (tf.math.xlogy(self.concentration1 - 1., x) +
            (self.concentration0 - 1.) * tf.math.log1p(-x))

  def _log_normalization(self):
    return (tf.lgamma(self.concentration1)
            + tf.lgamma(self.concentration0)
            - tf.lgamma(self.total_concentration))

  def _entropy(self):
    return (
        self._log_normalization()
        - (self.concentration1 - 1.) * tf.digamma(self.concentration1)
        - (self.concentration0 - 1.) * tf.digamma(self.concentration0)
        + ((self.total_concentration - 2.) *
           tf.digamma(self.total_concentration)))

  def _mean(self):
    return self._concentration1 / self._total_concentration

  def _variance(self):
    return self._mean() * (1. - self._mean()) / (1. + self.total_concentration)

  @util.AppendDocstring(
      """Note: The mode is undefined when `concentration1 <= 1` or
      `concentration0 <= 1`. If `self.allow_nan_stats` is `True`, `NaN`
      is used for undefined modes. If `self.allow_nan_stats` is `False` an
      exception is raised when one or more modes are undefined.""")
  def _mode(self):
    mode = (self.concentration1 - 1.) / (self.total_concentration - 2.)
    if self.allow_nan_stats:
      nan = tf.fill(
          self.batch_shape_tensor(),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
          name="nan")
      is_defined = tf.logical_and(self.concentration1 > 1.,
                                  self.concentration0 > 1.)
      return tf.where(is_defined, mode, nan)
    return control_flow_ops.with_dependencies([
        tf.assert_less(
            tf.ones([], dtype=self.dtype),
            self.concentration1,
            message="Mode undefined for concentration1 <= 1."),
        tf.assert_less(
            tf.ones([], dtype=self.dtype),
            self.concentration0,
            message="Mode undefined for concentration0 <= 1.")
    ], mode)

  def _maybe_assert_valid_concentration(self, concentration, validate_args):
    """Checks the validity of a concentration parameter."""
    if not validate_args:
      return concentration
    return control_flow_ops.with_dependencies([
        tf.assert_positive(
            concentration,
            message="Concentration parameter must be positive."),
    ], concentration)

  def _maybe_assert_valid_sample(self, x):
    """Checks the validity of a sample."""
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        tf.assert_positive(x, message="sample must be positive"),
        tf.assert_less(
            x,
            tf.ones([], self.dtype),
            message="sample must be less than `1`."),
    ], x)


# TODO(b/117098119): Remove tf.distribution references once they're gone.
@kullback_leibler.RegisterKL(Beta, tf.distributions.Beta)
@kullback_leibler.RegisterKL(tf.distributions.Beta, Beta)
@kullback_leibler.RegisterKL(Beta, Beta)
def _kl_beta_beta(d1, d2, name=None):
  """Calculate the batchwise KL divergence KL(d1 || d2) with d1 and d2 Beta.

  Args:
    d1: instance of a Beta distribution object.
    d2: instance of a Beta distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_beta_beta".

  Returns:
    Batchwise KL(d1 || d2)
  """
  def delta(fn, is_property=True):
    fn1 = getattr(d1, fn)
    fn2 = getattr(d2, fn)
    return (fn2 - fn1) if is_property else (fn2() - fn1())
  with tf.name_scope(name, "kl_beta_beta", values=[
      d1.concentration1,
      d1.concentration0,
      d1.total_concentration,
      d2.concentration1,
      d2.concentration0,
      d2.total_concentration,
  ]):
    return (delta("_log_normalization", is_property=False)
            - tf.digamma(d1.concentration1) * delta("concentration1")
            - tf.digamma(d1.concentration0) * delta("concentration0")
            + (tf.digamma(d1.total_concentration)
               * delta("total_concentration")))
