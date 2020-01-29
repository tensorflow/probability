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
"""The ordered logistic distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


class OrderedLogistic(distribution.Distribution):
  """Ordered logistic distribution

  The OrderedLogistic distribution is parameterized by a location and a set of
  cutpoints. It is defined over the integers `{0, 1, ..., K}` for `K-1`
  non-decreasing cutpoints.

  One often useful way to interpret this distribution is by imagining a draw
  from a latent/unobserved logistic distribution with location `location` and
  scale `1` and then only considering the index of the bin defined by the `K-1`
  cutpoints this draw falls between.

  This distribution can be useful for modelling outcomes which have inherent
  ordering but no real numerical values, for example modelling the outcome of a
  survey question where the responses are `[bad, mediocre, good]`, which would
  be coded as `[0, 1, 2]`.

  #### Mathematical Details

  The survival function (s) is:

  ```none
  s(x; c, eta) = P(X > x)
               = sigmoid(eta - concat([-inf, c, inf])[x+1])
  ```

  where `location = eta` is the location of a latent logistic distribution and
  `cutpoints = c` define points to split up this latent distribution. The
  concatenation of the cutpoints, `concat([-inf, c, inf])`, ensures that `s(K) =
  P(X > K) = 0` and `s(-1) = P(X > -1) = 1` which aids in the definition of the
  probability mass function (pmf):

  ```none
  pmf(x; c, eta) = P(X > x-1) - P(x > x)
                 = s(x-1; c, eta) - s(x; c, eta)
  ```

  #### Examples

  Create a symmetric 4-class distribution:

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dist = tfd.OrderedLogistic(cutpoints=[-2., 0., 2.], location=0.)

  dist.categorical_probs()
  # ==> array([0.11920293, 0.38079706, 0.3807971 , 0.11920291], dtype=float32)
  ```

  Create a batch of 3 4-class distributions via batching the location of the
  underlying latent logistic distribution. Additionally, compared to the above
  example, the cutpoints have moved closer together/to zero, thus the
  probability of a latent draw falling in the inner two categories has shrunk
  for the `location = 0` case:

  ```python
  dist = tfd.OrderedLogistic(cutpoints=[-1., 0., 1.], location=[-1., 0., 1.])

  dist.categorical_probs()
  # ==> array([[0.5       , 0.23105855, 0.1497385 , 0.11920291],
               [0.2689414 , 0.23105861, 0.23105855, 0.26894143],
               [0.11920293, 0.14973842, 0.23105861, 0.5       ]], dtype=float32)

  ```

  Some further functionallity:

  ```python
  dist = tfd.OrderedLogistic(cutpoints=[-1., 0., 2.], location=0.)

  dist.prob([0, 3])
  # ==> array([0.2689414 , 0.11920291], dtype=float32)

  dist.log_prob(1)
  # ==> -1.4650838

  dist.sample(3)
  # ==> array([0, 1, 1], dtype=int32)

  dist.entropy()
  # ==> 1.312902
  ```

  """

  def __init__(
      self,
      cutpoints,
      location,
      dtype=tf.int32,
      validate_args=False,
      allow_nan_stats=True,
      name="OrderedLogistic",
  ):
    """Initialize Ordered Logistic distributions

    Args:
      cutpoints: An N-D floating point `Tensor`, `N >= 1`, representing the
        points which split up a latent logistic distribution. The first `N - 1`
        dimensions index into a batch of independent distributions and the last
        dimension represents a vector of cutpoints. The vector of cutpoints
        should be non-decreasing, which is only checked if `validate_args=True`.
      location: A floating point `Tensor`, representing the mean(s) of the
        latent logistic distribution(s).
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
          [cutpoints, location],
          dtype_hint=tf.float32)

      self._cutpoints = tensor_util.convert_nonref_to_tensor(
          cutpoints, dtype_hint=float_dtype, name="cutpoints")
      self._location = tensor_util.convert_nonref_to_tensor(
          location, dtype_hint=float_dtype, name="location")

      inf = tf.broadcast_to(
          tf.constant(np.inf, dtype=float_dtype),
          prefer_static.shape(self._cutpoints[..., :1]))

      self._augmented_cutpoints = tf.concat([-inf, cutpoints, inf], axis=-1)
      self._num_categories = tf.shape(cutpoints, out_type=dtype)[-1] + 1

      super(OrderedLogistic, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(cutpoints=1, location=0)

  @property
  def cutpoints(self):
    """Input argument `cutpoints`"""
    return self._cutpoints

  @property
  def location(self):
    """Input argument `location`"""
    return self._location

  def categorical_log_probs(self):
    """Matrix of predicted log probabilities for each category"""
    log_survival = tf.math.log_sigmoid(
        self.location[..., tf.newaxis] -
        self._augmented_cutpoints[tf.newaxis, ...])
    log_probs = tfp_math.log_sub_exp(
        log_survival[..., :-1], log_survival[..., 1:])
    shape = tf.concat(
        [self._batch_shape_tensor(), [self._num_categories]], axis=0)
    return tf.reshape(log_probs, shape)

  def categorical_probs(self):
    """Matrix of predicted log probabilities for each category"""
    return tf.math.exp(self.categorical_log_probs())

  def _sample_n(self, n, seed=None):
    logits = tf.reshape(
        self.categorical_log_probs(), [-1, self._num_categories])
    draws = tf.random.categorical(logits, n, dtype=self.dtype, seed=seed)
    return tf.reshape(
        tf.transpose(draws),
        shape=tf.concat([[n], self._batch_shape_tensor()], axis=0))

  def _batch_shape_tensor(self, cutpoints=None, location=None):
    cutpoints = self.cutpoints if cutpoints is None else cutpoints
    location = self.location if location is None else location
    return prefer_static.broadcast_shape(
        prefer_static.shape(cutpoints)[:-1],
        prefer_static.shape(location))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.location.shape, self.cutpoints.shape[:-1])

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    log_survival_xm1 = self._log_survival_function(x - 1)
    log_survival_x = self._log_survival_function(x)
    return tfp_math.log_sub_exp(log_survival_xm1, log_survival_x)

  def _log_cdf(self, x):
    return tfp_math.log1mexp(self._log_survival_function(x))

  def _log_survival_function(self, x):
    return tf.math.log_sigmoid(
        self.location -
        tf.gather(self._augmented_cutpoints, x + 1, axis=-1))

  def _entropy(self):
    log_probs = self.categorical_log_probs()
    return -tf.reduce_sum(
        tf.math.multiply_no_nan(log_probs, tf.math.exp(log_probs)),
        axis=-1)

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

      if not dtype_util.is_floating(self.location.dtype):
        raise TypeError('Argument `location` must having floating type.')

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
      assert not assertions  # Should never happen.
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
            message=('OrderedLogistic samples must be between `0` and `K-1` '
                     'where `K` is the number of categories.')))
    return assertions


@kullback_leibler.RegisterKL(OrderedLogistic, OrderedLogistic)
def _kl_ordered_logistic_ordered_logistic(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b), a and b OrderedLogistic.

  Args:
    a: instance of a OrderedLogistic distribution object.
    b: instance of a OrderedLogistic distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None` (i.e., `'kl_ordered_logistic_ordered_logistic'`).

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_ordered_logistic_ordered_logistic'):
    a_log_probs = a.categorical_log_probs()
    b_log_probs = b.categorical_log_probs()
    return tf.reduce_sum(
        tf.math.exp(a_log_probs) * (a_log_probs - b_log_probs),
        axis=-1)
