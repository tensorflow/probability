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


class StoppingRatioLogistic(distribution.Distribution):
  """Stopping ratio logistic distribution.

  The StoppingRatioLogistic distribution is parameterized by a location and a
  set of non-decreasing cutpoints.  t is defined over the integers
 `{0, 1, ..., K}` for `K` non-decreasing cutpoints.

  The difference to the OrderedLogistic is that categories can only be reached
  one after another, i.e., sequentially. Specifically, the StoppingRatioLogistic
  distribution models the probability of an ordinal random variable `X` to be in
  category `c` given `X >= c` as

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
                    \prod_{s=0}^{x = 1} (1 - sigmoid(cutpoints[s] - loc))
  ```

  where `loc` is the location of a latent logistic distribution and
  `cutpoints` define points to split up this latent distribution.

  #### Examples

  Create a symmetric 4-class distribution:

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dist = tfd.StoppingRatioLogistic(cutpoints=[-2., 0., 2.], loc=0.)

  dist.categorical_probs()
  # ==> array([0.11920293, 0.38079706, 0.3807971 , 0.11920291], dtype=float32)
  ```

  Some further functionallity:

  ```python
  dist = tfd.StoppingRatioLogistic(cutpoints=[-2., 0., 2.], loc=0.)

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
      name='StoppingRatioLogistic',
  ):
    """Initialize Stopping Ratio Logistic distributions.

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

      super(StoppingRatioLogistic, self).__init__(
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
    """Log probabilities for the `K+1` sequential categories."""
    return tf.math.log(self.categorical_probs())

  def categorical_probs(self):
    """Probabilities for the `K+1` sequential categories."""
    K = self._num_categories()
    cutpoints = self.cutpoints
    loc = self.loc
    one = tf.ones(1, dtype=cutpoints.dtype)
    q = one - tf.math.sigmoid(cutpoints - loc)
    p = tf.zeros(K, dtype=cutpoints.dtype)

    idx = [[i] for i in range(K - 1)]
    p = tf.tensor_scatter_nd_update(p, tf.constant(idx), one - q)
    p = tf.tensor_scatter_nd_update(
      p,
      tf.constant([[K - 1]]),
      [tf.reduce_prod(q[..., :K])]
    )

    for i in range(1, K - 1):
      p = tf.tensor_scatter_nd_update(
          p,
          tf.constant([[i]]),
          [p[i] * tf.math.reduce_prod(q[..., :i])]
      )

    return p

  def _num_categories(self):
    return prefer_static.shape(self.cutpoints, out_type=self.dtype)[-1] + 1

  def _sample_n(self, n, seed=None):
    logits = tf.reshape(
        self.categorical_log_probs(), [-1, self._num_categories()])
    draws = tf.random.categorical(logits, n, dtype=self.dtype, seed=seed)
    return tf.reshape(
        tf.transpose(draws),
        shape=tf.concat([[n], self._batch_shape_tensor()], axis=0))

  def _batch_shape_tensor(self, cutpoints=None, loc=None):
    cutpoints = self.cutpoints if cutpoints is None else cutpoints
    loc = self.loc if loc is None else loc
    return prefer_static.broadcast_shape(
        prefer_static.shape(cutpoints)[:-1],
        prefer_static.shape(loc))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.loc.shape, self.cutpoints.shape[:-1])

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    log_prob = self.categorical_log_probs()
    return tf.gather(log_prob, x)

  def _log_cdf(self, x):
    cdf = tf.cumsum(self.categorical_probs())
    return tf.math.log(tf.gather(cdf, x))

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
  """Calculate the batched KL divergence KL(a || b), a and b StoppingRatioLogistic.

  This function utilises the `StoppingRatioLogistic` `categorical_log_probs` member
  function to implement KL divergence for discrete probability distributions as
  described in
  e.g. [Wikipedia](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence).

  Args:
    a: instance of a StoppingRatioLogistic distribution object.
    b: instance of a StoppingRatioLogistic distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None`

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_stopping_ratio_logistic_stopping_ratio_logistic'):
    a_log_probs = a.categorical_log_probs()
    b_log_probs = b.categorical_log_probs()
    return tf.reduce_sum(
        tf.math.exp(a_log_probs) * (a_log_probs - b_log_probs),
        axis=-1)

