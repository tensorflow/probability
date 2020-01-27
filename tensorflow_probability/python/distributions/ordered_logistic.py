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
  cutpoints. It is defined over the integers `{0, 1, ..., K}` when there are
  `K-1` cutpoints.

  One often useful way to interpret this distribution is by imagining a draw
  from a latent/unobserved logistic distribution with location `location` and
  scale `1` and then only considering which of the `cutpoints` this draw falls
  in.

  #### Mathematical Details

  TODO

  #### Examples

  TODO
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

    TODO
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
          name=name,
      )

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
    if cutpoints is None:
      first_cutpoint = self.cutpoints[..., 0]
    else:
      first_cutpoint = cutpoints[..., 0]

    if location is None:
      location = self.location

    return prefer_static.broadcast_shape(
        prefer_static.shape(first_cutpoint),
        prefer_static.shape(location))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.location.shape, self.cutpoints[..., 0].shape)

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
            message=('OrderedLogistic samples must be between `0` and `n-1` '
                     'where `n` is the number of categories.')))
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
