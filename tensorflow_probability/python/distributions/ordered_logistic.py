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
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

# TODO(b/149334734): Consider rewriting this underlying class via using
# QuantizedDistribution.


def _broadcast_cat_event_and_params(event, params, base_dtype):
  """Broadcasts the event or distribution parameters."""
  if dtype_util.is_floating(event.dtype):
    # When `validate_args=True` we've already ensured int/float casting
    # is closed.
    event = tf.cast(event, dtype=tf.int32)
  elif not dtype_util.is_integer(event.dtype):
    raise TypeError('`value` should have integer `dtype` or '
                    '`self.dtype` ({})'.format(base_dtype))
  shape_known_statically = (
      tensorshape_util.rank(params.shape) is not None and
      tensorshape_util.is_fully_defined(params.shape[:-1]) and
      tensorshape_util.is_fully_defined(event.shape))
  if not shape_known_statically or params.shape[:-1] != event.shape:
    params = params * tf.ones_like(event[..., tf.newaxis],
                                   dtype=params.dtype)
    params_shape = tf.shape(params)[:-1]
    event = event * tf.ones(params_shape, dtype=event.dtype)
    if tensorshape_util.rank(params.shape) is not None:
      tensorshape_util.set_shape(event, params.shape[:-1])

  return event, params


class OrderedLogistic(distribution.Distribution):
  """Ordered logistic distribution.

  The OrderedLogistic distribution is parameterized by a location and a set of
  cutpoints. It is defined over the integers `{0, 1, ..., K}` for `K`
  non-decreasing cutpoints.

  One often useful way to interpret this distribution is by imagining a draw
  from a latent/unobserved logistic distribution with location `loc` and
  scale `1` and then only considering the index of the bin defined by the `K`
  cutpoints this draw falls between. An example implementation of this idea is
  as follows:

  ```python
  cutpoints = [0.0, 1.0]
  loc = 0.5

  def probs_from_latent(latent):
    augmented_cutpoints = tf.concat([[-np.inf], cutpoints, [np.inf]], axis=0)
    below = latent[..., tf.newaxis] < augmented_cutpoints[1:]
    above = latent[..., tf.newaxis] > augmented_cutpoints[:-1]
    return tf.cast(below & above, tf.float32)

  latent_and_ordered_logistic = tfd.JointDistributionSequential([
    tfd.Logistic(loc=loc, scale=1.),
    lambda l: tfd.Categorical(probs=probs_from_latent(l), dtype=tf.float32)
  ])

  tf.stack(latent_and_ordered_logistic.sample(5), axis=1)
  # ==> array([[ 0.6434291,  1.       ],
               [ 3.0963311,  2.       ],
               [-1.2692463,  0.       ],
               [-3.3595495,  0.       ],
               [ 0.8468886,  1.       ]], dtype=float32)
  ```

  which displays that latent draws < `cutpoints[0] = 0.` are category 0, latent
  draws between `cutpoints[0] = 0.` and `cutpoints[1] = 1.` are category 1, and
  finally latent draws > `cutpoints[1] = 1.` (the final cutpoint in this
  example) are the top category of 2.

  This distribution can be useful for modelling outcomes which have inherent
  ordering but no real numerical values, for example modelling the outcome of a
  survey question where the responses are `[bad, mediocre, good]`, which would
  be coded as `[0, 1, 2]` and the model would contain two cutpoints (`K = 2`).

  #### Mathematical Details

  The survival function (s) is:

  ```none
  s(x; c, eta) = P(X > x)
               = sigmoid(eta - concat([-inf, c, inf])[x+1])
  ```

  where `loc = eta` is the location of a latent logistic distribution and
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

  dist = tfd.OrderedLogistic(cutpoints=[-2., 0., 2.], loc=0.)

  dist.categorical_probs()
  # ==> array([0.11920293, 0.38079706, 0.3807971 , 0.11920291], dtype=float32)
  ```

  Create a batch of 3 4-class distributions via batching the location of the
  underlying latent logistic distribution. Additionally, compared to the above
  example, the cutpoints have moved closer together/to zero, thus the
  probability of a latent draw falling in the inner two categories has shrunk
  for the `loc = 0` case:

  ```python
  dist = tfd.OrderedLogistic(cutpoints=[-1., 0., 1.], loc=[-1., 0., 1.])

  dist.categorical_probs()
  # ==> array([[0.5       , 0.23105855, 0.1497385 , 0.11920291],
               [0.2689414 , 0.23105861, 0.23105855, 0.26894143],
               [0.11920293, 0.14973842, 0.23105861, 0.5       ]], dtype=float32)

  ```

  Some further functionallity:

  ```python
  dist = tfd.OrderedLogistic(cutpoints=[-1., 0., 2.], loc=0.)

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
      loc,
      dtype=tf.int32,
      validate_args=False,
      allow_nan_stats=True,
      name='OrderedLogistic',
  ):
    """Initialize Ordered Logistic distributions.

    Args:
      cutpoints: A floating-point `Tensor` with shape `[B1, ..., Bb, K]` where
        `b >= 0` indicates the number of batch dimensions. Each entry is then a
        `K`-length vector of cutpoints. The vector of cutpoints should be
        non-decreasing, which is only checked if `validate_args=True`.
      loc: A floating-point `Tensor` with shape `[B1, ..., Bb]` where `b >=
        0` indicates the number of batch dimensions. The entries represent the
        mean(s) of the latent logistic distribution(s). Different batch shapes
        for `cutpoints` and `loc` are permitted, with the distribution
        `batch_shape` being `tf.shape(loc[..., tf.newaxis] -
        cutpoints)[:-1]` assuming the subtraction is a valid broadcasting
        operation.
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

      super(OrderedLogistic, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(cutpoints=1, loc=0)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('loc', 'scale'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

  @property
  def cutpoints(self):
    """Input argument `cutpoints`."""
    return self._cutpoints

  @property
  def loc(self):
    """Input argument `loc`."""
    return self._loc

  def categorical_log_probs(self):
    """Log probabilities for the `K+1` ordered categories."""
    log_survival = tf.math.log_sigmoid(
        self.loc[..., tf.newaxis] - self._augmented_cutpoints())
    return tfp_math.log_sub_exp(
        log_survival[..., :-1], log_survival[..., 1:])

  def categorical_probs(self):
    """Probabilities for the `K+1` ordered categories."""
    return tf.math.exp(self.categorical_log_probs())

  def _augmented_cutpoints(self):
    cutpoints = tf.convert_to_tensor(self.cutpoints)
    inf = tf.fill(
        ps.shape(cutpoints[..., :1]),
        tf.constant(np.inf, dtype=cutpoints.dtype))
    return tf.concat([-inf, cutpoints, inf], axis=-1)

  def _num_categories(self):
    return ps.shape(self.cutpoints, out_type=self.dtype)[-1] + 1

  def _sample_n(self, n, seed=None):
    # TODO(b/149334734): Consider sampling from the Logistic distribution,
    # and then truncating based on cutpoints. This could be faster than the
    # given sampling scheme.
    logits = tf.reshape(
        self.categorical_log_probs(), [-1, self._num_categories()])
    draws = samplers.categorical(logits, n, dtype=self.dtype, seed=seed)
    return tf.reshape(
        tf.transpose(draws),
        shape=ps.concat([[n], self._batch_shape_tensor()], axis=0))

  def _batch_shape_tensor(self, cutpoints=None, loc=None):
    cutpoints = self.cutpoints if cutpoints is None else cutpoints
    loc = self.loc if loc is None else loc
    return ps.broadcast_shape(
        ps.shape(cutpoints)[:-1],
        ps.shape(loc))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.loc.shape, self.cutpoints.shape[:-1])

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    # TODO(b/149334734): Consider using QuantizedDistribution for the log_prob
    # computation for better precision.
    num_categories = self._num_categories()
    x, augmented_log_survival = _broadcast_cat_event_and_params(
        event=x,
        params=tf.math.log_sigmoid(
            self.loc[..., tf.newaxis] - self._augmented_cutpoints()),
        base_dtype=dtype_util.base_dtype(self.dtype))
    x_flat = tf.reshape(x, [-1, 1])
    augmented_log_survival_flat = tf.reshape(
        augmented_log_survival, [-1, num_categories + 1])
    log_survival_flat_xm1 = tf.gather(
        params=augmented_log_survival_flat,
        indices=tf.clip_by_value(x_flat, 0, num_categories),
        batch_dims=1)
    log_survival_flat_x = tf.gather(
        params=augmented_log_survival_flat,
        indices=tf.clip_by_value(x_flat + 1, 0, num_categories),
        batch_dims=1)
    log_prob_flat = tfp_math.log_sub_exp(
        log_survival_flat_xm1, log_survival_flat_x)
    # Deal with case where both survival probabilities are -inf, which gives
    # `log_prob_flat = nan` when it should be -inf.
    minus_inf = tf.constant(-np.inf, dtype=log_prob_flat.dtype)
    log_prob_flat = tf.where(
        x_flat > num_categories - 1, minus_inf, log_prob_flat)
    return tf.reshape(log_prob_flat, shape=ps.shape(x))

  def _log_cdf(self, x):
    return tfp_math.log1mexp(self._log_survival_function(x))

  def _log_survival_function(self, x):
    num_categories = self._num_categories()
    x, augmented_log_survival = _broadcast_cat_event_and_params(
        event=x,
        params=tf.math.log_sigmoid(
            self.loc[..., tf.newaxis] - self._augmented_cutpoints()),
        base_dtype=dtype_util.base_dtype(self.dtype))
    x_flat = tf.reshape(x, [-1, 1])
    augmented_log_survival_flat = tf.reshape(
        augmented_log_survival, [-1, num_categories + 1])
    log_survival_flat = tf.gather(
        params=augmented_log_survival_flat,
        indices=tf.clip_by_value(x_flat + 1, 0, num_categories),
        batch_dims=1)
    return tf.reshape(log_survival_flat, shape=ps.shape(x))

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
    assertions.append(distribution_util.assert_casting_closed(
        x, target_dtype=tf.int32))
    assertions.append(assert_util.assert_non_negative(x))
    assertions.append(
        assert_util.assert_less_equal(
            x, tf.cast(self._num_categories(), x.dtype),
            message=('OrderedLogistic samples must be `>= 0` and `<= K` '
                     'where `K` is the number of cutpoints.')))
    return assertions


@kullback_leibler.RegisterKL(OrderedLogistic, OrderedLogistic)
def _kl_ordered_logistic_ordered_logistic(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b), a and b OrderedLogistic.

  This function utilises the `OrderedLogistic` `categorical_log_probs` member
  function to implement KL divergence for discrete probability distributions as
  described in
  e.g. [Wikipedia](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence).

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
