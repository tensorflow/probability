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
"""The stopping ratio logistic distribution class."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import ascending
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import generic


class StoppingRatioLogistic(distribution.AutoCompositeTensorDistribution):
  """Stopping ratio logistic distribution.

  The StoppingRatioLogistic distribution is parameterized by a location and a
  set of non-decreasing cutpoints. It is defined over the integers
  `{0, 1, ..., K}` for `K` non-decreasing cutpoints.

  The difference to the OrderedLogistic is that categories can only be reached
  one after another, i.e., sequentially. Specifically, while the probability
  of an ordinal random variable `X` to be in category `c`
  for the OrderedLogistic reads as

  ```none
  P(X = c; cutpoints, loc) = P(X > c - 1) - P(X > c)
                       = sigmoid(loc - concat([-inf, cutpoints, inf])[c]) -
                         sigmoid(loc - concat([-inf, cutpoints, inf])[c + 1])
  ```

  the StoppingRatioLogistic distribution models the probability of an ordinal
  random variable `X` to be in category `c` given `X >= c` as

  ```none
  P(X = c; X >= c, cutpoints, loc) = sigmoid(cutpoints[c] - loc)
  ```

  The sequential mechanism for `X` starts in category `c = 0` where a binary
  decision between `c = 0` and `c > 0` is made:

  ```none
  P(X = 0; cutpoints, loc) = sigmoid(cutpoints[0] - loc)
  ```

  If `X = 0`, the process stops. Otherwise the process continues with

  ```none
  P(X = 1; X >= 1, cutpoints, loc) = sigmoid(cutpoints[1] - loc)
  ```

  The process continues to move on to higher level categories until it stops at
  some category `X = c`.

  This distribution is useful for ordinal variables where lower categories
  need to be reached first, for instance modelling the degree of a person
  where the categories are `[Bachelor, Master, PhD]`. In order to obtain a PhD
  title, first the degrees `Bachelor` and `Master` need to be reached.

  #### Mathematical Details

  The probability mass function (pmf) is

  ```none
  pmf(x; cutpoints, loc) =
                    sigmoid(cutpoints[x] - loc) *
                    prod_{s=0}^{x - 1} (1 - sigmoid(cutpoints[s] - loc))
  ```

  where `loc` is the location of a latent logistic distribution and
  `cutpoints` define points to split up this latent distribution.

  #### Examples

  To expand on the `[Bachelor, Master, PhD]` from above, create a distribution
  of three ordered categories:

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dist = tfd.StoppingRatioLogistic(cutpoints=[-1.0, 1.0], loc=0.)

  dist.categorical_probs()
  # ==> array([0.2689414  0.53444666 0.19661193], dtype=float32)
  ```

  Here, the probability of finishing one's education with a Bachelor would be
  approx. 26% in this example, while the probability of continuing to pursue
  a Master's would be approx. 53% and the probability of even attaining a PhD
  would be 20%.

  Some further functionality:

  ```python
  dist = tfd.StoppingRatioLogistic(cutpoints=[-2., 0., 2.], loc=0.)

  dist.prob([0, 3])
  # ==> array([0.11920291, 0.05249681], dtype=float32)

  dist.log_prob(1)
  # ==> -0.82007515

  dist.sample(3)
  # ==> array([2, 1, 2], dtype=int32)
  ```

  """

  def __init__(
      self,
      cutpoints,
      loc,
      dtype=tf.int32,
      validate_args=False,
      allow_nan_stats=True,
      name='StoppingRatioLogistic',
  ):
    """Initialize Stopping Ratio Logistic distributions.

    Args:
      cutpoints: A floating-point `Tensor` with shape `(K,)` where
        `K` is the number of cutpoints. The vector of cutpoints should be
        non-decreasing, which is only checked if `validate_args=True`.
      loc: A floating-point `Tensor` with shape `()`. The entry represents the
        mean of the latent logistic distribution.
      dtype: The type of the event samples (default: int32).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g. mode) use the value "`NaN`" to indicate the result is
        undefined. When `False`, an exception is raised if one or more of the
        statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """

    parameters = dict(locals())

    with tf.name_scope(name) as name:

      float_dtype = dtype_util.common_dtype(
          [cutpoints, loc],
          dtype_hint=tf.float32)

      self._cutpoints = tensor_util.convert_nonref_to_tensor(
          cutpoints, dtype_hint=float_dtype, name='cutpoints')
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype_hint=float_dtype, name='loc')

      super(StoppingRatioLogistic, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        cutpoints=parameter_properties.ParameterProperties(
            event_ndims=1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
            default_constraining_bijector_fn=lambda: ascending.Ascending()),  # pylint:disable=unnecessary-lambda
        loc=parameter_properties.ParameterProperties())

  @property
  def cutpoints(self):
    """Cutpoints param separating the latent distribution into categories."""
    return self._cutpoints

  @property
  def loc(self):
    """Mean parameter of the latent logistic distribution."""
    return self._loc

  def categorical_log_probs(self):
    """Log probabilities for the `K + 1` sequential categories."""

    cutpoints = tf.convert_to_tensor(self.cutpoints)
    loc = tf.convert_to_tensor(self.loc)
    num_cat = self._num_categories()

    # For the StoppingRatioLogistic, we have:
    # P(X = c; X >= c, cutpoints, loc) = sigmoid(cutpoints[c] - loc)
    # Given these conditional probabilities, we would like to retrieve
    # P(X = c; cutpoints, loc).
    # Let F(c) = P(X = c; X >= c, cutpoints, loc) and
    # G(c) = P(X = c; cutpoints, loc)

    # Conditional probabilities. These are log(F(k)) and log(1 - F(k))
    conditional_log_probs = tf.math.log_sigmoid(
        cutpoints - loc[..., tf.newaxis])
    conditional_log_probs_complement = generic.log1mexp(conditional_log_probs)

    # Note that F(0) = G(0).
    # G(1) = P(X = 1; cutpoints, loc) =
    #   P(X = 1; X >= 1, cutpoints, loc) * P(X >= 1) = F(1) * (1 - G(0))
    # G(2) = P(X = 2; cutpoints, loc) =
    #   P(X = 2; X >= 2, cutpoints, loc) * P(X >= 2) = F(2) * (1 - G(0) - G(1))
    # In general, G(k) = F(k) * (1 - \sum_{k-1} G(i))

    # We rewrite this recurrence in terms of F(k)
    # G(1) = F(1) * (1 - G(0)) = F(1) * (1 - F(0))
    # G(2) = F(2) * (1 - G(0) - G(1)) = (1 - F(0) - F(1) * (1 - F(0))
    #      = F(2) * (1 - F(0)) * (1 - F(1))
    # G(k) = F(k) * \prod_{k-1} (1 - F(i))

    # log(F(k)) + log(\prod (1 - F(i)))
    categorical_log_probs = conditional_log_probs + tf.math.cumsum(
        conditional_log_probs_complement[..., :(num_cat - 1)],
        axis=-1, exclusive=True)
    # Finally we need to handle the last category.
    return tf.concat([
        categorical_log_probs,
        tf.math.reduce_sum(
            conditional_log_probs_complement[
                ..., :num_cat], axis=-1, keepdims=True)], axis=-1)

  def categorical_probs(self):
    """Probabilities for the `K + 1` sequential categories."""
    return tf.math.exp(self.categorical_log_probs())

  def _num_categories(self):
    return ps.shape(self.cutpoints, out_type=self.dtype)[-1] + 1

  def _sample_n(self, n, seed=None):
    return categorical.Categorical(
        logits=self.categorical_log_probs()).sample(n, seed=seed)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    return categorical.Categorical(
        logits=self.categorical_log_probs()).log_prob(x)

  def _cdf(self, x):
    return categorical.Categorical(
        logits=self.categorical_log_probs()).cdf(x)

  def _mode(self):
    log_probs = self.categorical_log_probs()
    mode = tf.argmax(log_probs, axis=-1, output_type=self.dtype)
    tensorshape_util.set_shape(mode, log_probs.shape[:-1])
    return mode

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    # In init, we can always build shape and dtype checks because
    # we assume shape doesn't change for Variable backed args.
    if is_init:

      if not dtype_util.is_floating(self.cutpoints.dtype):
        raise TypeError('Argument `cutpoints` must having floating type.')

      if not dtype_util.is_floating(self.loc.dtype):
        raise TypeError('Argument `loc` must having floating type.')

      cutpoint_dims = tensorshape_util.rank(self.cutpoints.shape)
      msg = 'Argument `cutpoints` must have rank at least 1.'
      if cutpoint_dims is not None:
        if cutpoint_dims < 1:
          raise ValueError(msg)
      elif self.validate_args:
        cutpoints = tf.convert_to_tensor(self.cutpoints)
        assertions.append(
            assert_util.assert_rank_at_least(cutpoints, 1, message=msg))

    if not self.validate_args:
      return []

    if is_init != tensor_util.is_ref(self.cutpoints):
      cutpoints = tf.convert_to_tensor(self.cutpoints)
      assertions.append(distribution_util.assert_nondecreasing(
          cutpoints, message='Argument `cutpoints` must be non-decreasing.'))

    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    assertions.append(
        assert_util.assert_less_equal(
            x, tf.cast(self._num_categories(), x.dtype),
            message=('StoppingRatioLogistic samples must be `>= 0` and `<= K` '
                     'where `K` is the number of cutpoints.')))
    return assertions


@kullback_leibler.RegisterKL(StoppingRatioLogistic, StoppingRatioLogistic)
def _kl_stopping_ratio_logistic_stopping_ratio_logistic(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b), both StoppingRatioLogistic.

  This function utilises the `StoppingRatioLogistic` `categorical_log_probs`
  member function to implement KL divergence for discrete probability
  distributions as described in
  e.g. [Wikipedia](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence).

  Args:
    a: instance of a StoppingRatioLogistic distribution object.
    b: instance of a StoppingRatioLogistic distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None`

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or
                     'kl_stopping_ratio_logistic_stopping_ratio_logistic'):
    a_log_probs = a.categorical_log_probs()
    b_log_probs = b.categorical_log_probs()
    return tf.reduce_sum(
        tf.math.multiply_no_nan(
            tf.math.exp(a_log_probs),
            a_log_probs - b_log_probs),
        axis=-1)
