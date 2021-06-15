# Copyright 2020 The TensorFlow Probability Authors.
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
"""The SigmoidBeta distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'SigmoidBeta',
]


class SigmoidBeta(distribution.AutoCompositeTensorDistribution):
  """SigmoidBeta Distribution.

  The SigmoidBeta distribution is defined over the real line using parameters
  `concentration1` (aka 'alpha') and `concentration0` (aka 'beta').

  This distribution is the transformation of the Beta distribution such that
  Sigmoid(X) ~ Beta(...) => X ~ SigmoidBeta(...).

  #### Mathematical Details

  The probability density function (pdf) can be derived from the change of
  variables rule. We begin with `g(X) = Sigmoid(X)`, and note that

  ```none
  p_x(x) = p_y(g(y)) | g'(y) |.
  ```
  With `g'(y) = Sigmoid(x) ( 1 - Sigmoid(x))`, we arrive at

  ```none
  pdf(x; alpha, beta) = Sigmoid(x)^alpha (1 - Sigmoid(x))^beta / B(alpha, beta)
  B(alpha, beta) = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
  ```
  where:

  * `concentration1 = alpha`
  * `concentration0 = beta`
  * `B(alpha, beta` is the [beta function](
    https://en.wikipedia.org/wiki/Beta_function)
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  Critically, these parameters lose the relationship to the mean that they have
  under the untransformed Beta distribution. We choose to keep the names to
  draw analogy to the original Beta distribution.

  ```none
  concentration1 = alpha
  concentration0 = beta
  ```

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.



  The cumlative density function (cdf) can be found by integrating the pdf
  directly from `-infinity` to x:

  ```none
   cdf(x; alpha, beta) = I_Sigmoid(x)(alpha + 1, beta + 1) / B(alpha, beta),
  ```

  where `I_x(alpha, beta)` is the [incomplete beta function](
  https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function).

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in
  [(Figurnov et al., 2018)][1].

  #### Examples

   ```python
   tfd = tfp.distributions

   dist = tfd.SigmoidBeta(concentration0=1.,
                          concentration1=2.)

   dist.sample([4, 5])  # Shape [4, 5, 3]

   # `x` has three batch entries, each with two samples.
   x = [[.1, .4, .5],
        [.2, .3, .5]]
   # Calculate the probability of each pair of samples under the corresponding
   # distribution in `dist`.
   dist.prob(x)         # Shape [2, 3]
   ```

  #### References

  [1]: Michael Figurnov, Shakir Mohamed, Andriy Mnih.
       Implicit Reparameterization Gradients. _arXiv preprint arXiv:1805.08498_,
       2018. https://arxiv.org/abs/1805.08498

  """

  def __init__(self,
               concentration1,
               concentration0,
               validate_args=False,
               allow_nan_stats=True,
               name='SigmoidBeta'):
    """Initialize a batch of SigmoidBeta distributions.

    Args:
      concentration1: Positive floating-point `Tensor` indicating mean
        number of successes; aka 'alpha'.
      concentration0: Positive floating-point `Tensor` indicating mean
        number of failures; aka 'beta'.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([concentration1, concentration0],
                                      dtype_hint=tf.float32)
      self._concentration1 = tensor_util.convert_nonref_to_tensor(
          concentration1, dtype=dtype, name='concentration1')
      self._concentration0 = tensor_util.convert_nonref_to_tensor(
          concentration0, dtype=dtype, name='concentration0')

    super(SigmoidBeta, self).__init__(
        dtype=dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        parameters=parameters,
        name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        concentration1=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration0=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def concentration1(self):
    """Concentration parameter associated with a `1` outcome."""
    return self._concentration1

  @property
  def concentration0(self):
    """Concentration parameter associated with a `0` outcome."""
    return self._concentration0

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    for i, concentration in enumerate([self.concentration0,
                                       self.concentration1]):
      if is_init != tensor_util.is_ref(concentration):
        assertions.append(
            assert_util.assert_positive(
                concentration,
                message=f'`concentration{i}` parameter must be positive.'))
    return assertions

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    seed1, seed2 = samplers.split_seed(seed, salt='sigmoid_beta')
    concentration1 = tf.convert_to_tensor(self.concentration1)
    concentration0 = tf.convert_to_tensor(self.concentration0)
    shape = self._batch_shape_tensor(concentration1=concentration1,
                                     concentration0=concentration0)
    expanded_concentration1 = tf.broadcast_to(concentration1, shape)
    expanded_concentration0 = tf.broadcast_to(concentration0, shape)
    log_gamma1 = gamma_lib.random_gamma(
        shape=[n],
        concentration=expanded_concentration1,
        seed=seed1,
        log_space=True)
    log_gamma2 = gamma_lib.random_gamma(
        shape=[n],
        concentration=expanded_concentration0,
        seed=seed2,
        log_space=True)

    return log_gamma1 - log_gamma2

  def _log_normalization(self, concentration0, concentration1):
    return tfp_math.lbeta(concentration0, concentration1)

  def _log_unnormalized_prob(self, concentration0, concentration1, x):
    return (-concentration0 * tf.math.softplus(-x)
            -concentration1 * tf.math.softplus(x))

  def _log_prob(self, x):
    a = tf.convert_to_tensor(self.concentration1)
    b = tf.convert_to_tensor(self.concentration0)
    return (self._log_unnormalized_prob(a, b, x) -
            self._log_normalization(a, b))

  def _cdf(self, x):
    concentration1 = tf.convert_to_tensor(self.concentration1)
    concentration0 = tf.convert_to_tensor(self.concentration0)
    sig_x = tf.math.sigmoid(x)
    shape = functools.reduce(ps.broadcast_shape, [
        ps.shape(concentration1),
        ps.shape(concentration0),
        ps.shape(sig_x)
    ])
    concentration1 = tf.broadcast_to(concentration1, shape)
    concentration0 = tf.broadcast_to(concentration0, shape)
    sig_x = tf.broadcast_to(sig_x, shape)
    return tf.math.betainc(concentration1, concentration0, sig_x)

  def _mode(self):
    return tf.math.log(self.concentration1 / self.concentration0)
