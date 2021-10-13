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
"""The RelaxedBernoulli distribution class."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import logistic
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util


class RelaxedBernoulli(distribution.AutoCompositeTensorDistribution):
  """RelaxedBernoulli distribution with temperature and logits parameters.

  The RelaxedBernoulli is a distribution over the unit interval (0,1), which
  continuously approximates a Bernoulli. The degree of approximation is
  controlled by a temperature: as the temperature goes to 0 the
  RelaxedBernoulli becomes discrete with a distribution described by the
  `logits` or `probs` parameters, as the temperature goes to infinity the
  RelaxedBernoulli becomes the constant distribution that is identically 0.5.

  The RelaxedBernoulli distribution is a reparameterized continuous
  distribution that is the binary special case of the RelaxedOneHotCategorical
  distribution (Maddison et al., 2016; Jang et al., 2016). For details on the
  binary special case see the appendix of Maddison et al. (2016) where it is
  referred to as BinConcrete. If you use this distribution, please cite both
  papers.

  Some care needs to be taken for loss functions that depend on the
  log-probability of RelaxedBernoullis, because computing log-probabilities of
  the RelaxedBernoulli can suffer from underflow issues. In many case loss
  functions such as these are invariant under invertible transformations of
  the random variables. The KL divergence, found in the variational autoencoder
  loss, is an example. Because RelaxedBernoullis are sampled by a Logistic
  random variable followed by a `tf.sigmoid` op, one solution is to treat
  the Logistic as the random variable and `tf.sigmoid` as downstream. The
  KL divergences of two Logistics, which are always followed by a `tf.sigmoid`
  op, is equivalent to evaluating KL divergences of RelaxedBernoulli samples.
  See Maddison et al., 2016 for more details where this distribution is called
  the BinConcrete.

  An alternative approach is to evaluate Bernoulli log probability or KL
  directly on relaxed samples, as done in Jang et al., 2016. In this case,
  guarantees on the loss are usually violated. For instance, using a Bernoulli
  KL in a relaxed ELBO is no longer a lower bound on the log marginal
  probability of the observation. Thus care and early stopping are important.

  #### Examples

  Creates three continuous distributions, which approximate 3 Bernoullis with
  probabilities (0.1, 0.5, 0.4). Samples from these distributions will be in
  the unit interval (0,1).

  ```python
  temperature = 0.5
  p = [0.1, 0.5, 0.4]
  dist = RelaxedBernoulli(temperature, probs=p)
  ```

  Creates three continuous distributions, which approximate 3 Bernoullis with
  logits (-2, 2, 0). Samples from these distributions will be in
  the unit interval (0,1).

  ```python
  temperature = 0.5
  logits = [-2, 2, 0]
  dist = RelaxedBernoulli(temperature, logits=logits)
  ```

  Creates three continuous distributions, whose sigmoid approximate 3 Bernoullis
  with logits (-2, 2, 0).

  ```python
  temperature = 0.5
  logits = [-2, 2, 0]
  dist = Logistic(logits/temperature, 1./temperature)
  samples = dist.sample()
  sigmoid_samples = tf.sigmoid(samples)
  # sigmoid_samples has the same distribution as samples from
  # RelaxedBernoulli(temperature, logits=logits)
  ```

  Creates three continuous distributions, which approximate 3 Bernoullis with
  logits (-2, 2, 0). Samples from these distributions will be in
  the unit interval (0,1). Because the temperature is very low, samples from
  these distributions are almost discrete, usually taking values very close to 0
  or 1.

  ```python
  temperature = 1e-5
  logits = [-2, 2, 0]
  dist = RelaxedBernoulli(temperature, logits=logits)
  ```

  Creates three continuous distributions, which approximate 3 Bernoullis with
  logits (-2, 2, 0). Samples from these distributions will be in
  the unit interval (0,1). Because the temperature is very high, samples from
  these distributions are usually close to the (0.5, 0.5, 0.5) vector.

  ```python
  temperature = 100
  logits = [-2, 2, 0]
  dist = RelaxedBernoulli(temperature, logits=logits)
  ```

  Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution:
  A Continuous Relaxation of Discrete Random Variables. 2016.

  Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with
  Gumbel-Softmax. 2016.
  """

  def __init__(self,
               temperature,
               logits=None,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name='RelaxedBernoulli'):
    """Construct RelaxedBernoulli distributions.

    Args:
      temperature: A `Tensor`, representing the temperature of a set of
        RelaxedBernoulli distributions. The temperature values should be
        positive.
      logits: An N-D `Tensor` representing the log-odds
        of a positive event. Each entry in the `Tensor` parameterizes
        an independent RelaxedBernoulli distribution where the probability of an
        event is sigmoid(logits). Only one of `logits` or `probs` should be
        passed in.
      probs: An N-D `Tensor` representing the probability of a positive event.
        Each entry in the `Tensor` parameterizes an independent Bernoulli
        distribution. Only one of `logits` or `probs` should be passed in.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: If both `probs` and `logits` are passed, or if neither.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([logits, probs, temperature], tf.float32)

      self._temperature = tensor_util.convert_nonref_to_tensor(
          temperature, name='temperature', dtype=dtype)
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, name='probs', dtype=dtype)
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, name='logits', dtype=dtype)

      super(RelaxedBernoulli, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  def _transformed_logistic(self):
    logistic_scale = tf.math.reciprocal(self._temperature)
    logits_parameter = self._logits_parameter_no_checks()
    logistic_loc = logits_parameter * logistic_scale
    return transformed_distribution.TransformedDistribution(
        distribution=logistic.Logistic(
            logistic_loc,
            logistic_scale,
            allow_nan_stats=self.allow_nan_stats),
        bijector=sigmoid_bijector.Sigmoid())

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        temperature=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        logits=parameter_properties.ParameterProperties(),
        probs=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=sigmoid_bijector.Sigmoid,
            is_preferred=False))
    # pylint: enable=g-long-lambda

  @property
  def temperature(self):
    """Distribution parameter for the location."""
    return self._temperature

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def logits_parameter(self, name=None):
    """Logits computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      return self._logits_parameter_no_checks()

  def _logits_parameter_no_checks(self):
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      return tf.math.log(probs) - tf.math.log1p(-probs)
    return tensor_util.identity_as_tensor(self._logits)

  def probs_parameter(self, name=None):
    """Probs computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self):
    if self._logits is None:
      return tensor_util.identity_as_tensor(self._probs)
    return tf.math.sigmoid(self._logits)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None, **kwargs):
    return self._transformed_logistic().sample(n, seed=seed, **kwargs)

  def _log_prob(self, y, **kwargs):
    # The computation below is the same as
    # `self._transformed_logistic().log_prob(y, **kwargs)`. However, when `y`
    # approaches `0` or `1` it encounters numerical problems. Namely,
    # the Jacobian correction in `TransformedDistribution` becomes large in
    # absolute value, and catastrophically cancels against a similar term in the
    # `log_prob`. Instead, we collapse this computation below, and do the
    # cancellation symbolically.
    # The below also handles the case where `logits` goes to
    # `+-inf`, which also returns `NaN` when using `TransformedDistribution`

    logits_parameter = self._logits_parameter_no_checks()
    logits_y = tf.math.log(y) - tf.math.log1p(-y)
    t = tf.convert_to_tensor(self._temperature)
    z = logits_parameter - t * logits_y
    result = tf.where(
        z > 0,
        -logits_parameter + tf.math.xlogy(t - 1., y) - (
            t + 1.) * tf.math.log1p(-y) - 2 * tf.math.softplus(-z),
        logits_parameter - (t + 1.) * tf.math.log(y) + tf.math.xlog1py(
            t - 1., -y) - 2 * tf.math.softplus(z)) + tf.math.log(t)
    return tf.where(
        # Handle the case where `logits_parameter` is infinite. This corresponds
        # to a `Bernoulli` with all mass centered at `0` or `1`. The above
        # computation returns `NaN` values when `y = 0` or `y = 1`, so we
        # explicitly handle this here.
        tf.math.is_inf(logits_parameter) & tf.math.is_inf(logits_y) &
        ~tf.math.is_nan(logits_parameter) & ~tf.math.is_nan(logits_y),
        tf.where(
            tf.math.equal(
                tf.math.sign(logits_parameter), tf.math.sign(logits_y)),
            dtype_util.as_numpy_dtype(self.dtype)(np.inf),
            dtype_util.as_numpy_dtype(self.dtype)(-np.inf)),
        result)

  def _log_survival_function(self, y, **kwargs):
    return self._transformed_logistic().log_survival_function(y, **kwargs)

  def _cdf(self, y, **kwargs):
    return tf.math.exp(self._log_cdf(y, **kwargs))

  def _log_cdf(self, y, **kwargs):
    logits_y = tf.math.log(y) - tf.math.log1p(-y)
    logistic_scale = tf.math.reciprocal(self._temperature)
    logits_parameter = self._logits_parameter_no_checks()

    inf_logits_same_sign = (
        tf.math.is_inf(logits_y / logistic_scale) &
        tf.math.is_inf(logits_parameter) &
        (tf.math.equal(tf.math.sign(logits_y / logistic_scale),
                       tf.math.sign(logits_parameter))))

    numpy_dtype = dtype_util.as_numpy_dtype(self.dtype)

    safe_logits_y = tf.where(inf_logits_same_sign, numpy_dtype(0.), logits_y)
    safe_logistic_scale = tf.where(
        inf_logits_same_sign, numpy_dtype(1.), logistic_scale)

    # Handle the case when both logit parameters are infinite and opposite
    # signs.
    # When logits_parameter = +inf and logits_y = -inf, this results in
    # probability 0.
    # When logits_parameter = -inf and logits_y = +inf, this results in
    # probability 1.
    numpy_dtype = dtype_util.as_numpy_dtype(self.dtype)

    return tf.where(
        inf_logits_same_sign,
        numpy_dtype(0.),
        -tf.math.softplus(
            -(safe_logits_y  / safe_logistic_scale - logits_parameter)))

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    return sigmoid_bijector.Sigmoid(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []

    assertions = []
    if is_init != tensor_util.is_ref(self._temperature):
      msg1 = 'Argument `temperature` must be positive.'
      temperature = tf.convert_to_tensor(self._temperature)
      assertions.append(assert_util.assert_positive(temperature, message=msg1))

    if self._probs is not None:
      if is_init != tensor_util.is_ref(self._probs):
        probs = tf.convert_to_tensor(self._probs)
        one = tf.constant(1., probs.dtype)
        assertions.extend([
            assert_util.assert_non_negative(
                probs, message='Argument `probs` has components less than 0.'),
            assert_util.assert_less_equal(
                probs, one,
                message='Argument `probs` has components greater than 1.')
        ])

    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    assertions.append(assert_util.assert_less_equal(
        x, tf.ones([], dtype=x.dtype),
        message='Sample must be less than or equal to `1`.'))
    return assertions
