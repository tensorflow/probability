# Copyright 2019 The TensorFlow Probability Authors.
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
"""The PlackettLuce distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gumbel
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


class PlackettLuce(distribution.Distribution):
  """Plackett-Luce distribution over permutations.

  The Plackett-Luce distribution is defined over permutations of
  fixed length. It is parameterized by a positive score vector of same length.

  This class provides methods to create indexed batches of PlackettLuce
  distributions. If the provided `scores` is rank 2 or higher, for
  every fixed set of leading dimensions, the last dimension represents one
  single PlackettLuce distribution. When calling distribution
  functions (e.g. `dist.log_prob(x)`), `scores` and `x` are broadcast to the
  same shape (if possible). In all cases, the last dimension of `scores, x`
  represents single PlackettLuce distributions.

  #### Mathematical Details

  The Plackett-Luce is a distribution over permutation vectors `p` of length `k`
  where the permutation `p` is an arbitrary ordering of `k` indices
  `{0, 1, ..., k-1}`.

  The probability mass function (pmf) is,

  ```none
  pmf(p; s) = prod_i s_{p_i} / (Z - Z_i)
  Z = sum_{j=0}^{k-1} s_j
  Z_i = sum_{j=0}^{i-1} s_{p_j} for i>0 and 0 for i=0
  ```

  where:

  * `scores = s = [s_0, ..., s_{k-1}]`, `s_i >= 0`.

  Samples from Plackett-Luce distribution are generated sequentially as follows.

  Initialize normalization `N_0 = Z`
  For `i` in `{0, 1, ..., k-1}`

    1. Sample i-th element of permutation
       `p_i ~ Categorical(probs=[s_0/N_i, ..., s_{k-1}/N_i])`
    2. Update normalization
      `N_{i+1} = N_i-s_{p_i}`
    3. Mask out sampled index for subsequent rounds
       `s_{p_i} = 0`

  Return p

  Alternately, an equivalent way to sample from this distribution is to sort
  Gumbel perturbed log-scores [1].

  ```none
  p = argsort(log s + g) ~ PlackettLuce(s)
  g = [g_0, ..., g_{k-1}], g_i~ Gumbel(0, 1)
  ```

  #### Examples

  ```python
  scores = [0.1, 2., 5.]
  dist = PlackettLuce(scores)
  ```

  Creates a distribution over permutations of length 3, with the 3rd index
  likely to appear first in the permutation.
  The distribution function can be evaluated on permutations as follows.

  ```python
  # permutations same shape as scores.
  permutations = [2, 1, 0]
  dist.prob(permutations) # Shape []

  # scores broadcast to [[0.1, 2.3, 5.], [0.1, 2.3, 5.]] to match permutations.
  permutations = [[2, 1, 0], [1, 0, 2]]
  dist.prob(permutations) # Shape [2]

  # scores broadcast to shape [5, 7, 3] to match permutations.
  permutations = [[...]]  # Shape [5, 7, 3]
  dist.prob(permutaions)  # Shape [5, 7]
  ```

  Creates a 2-batch of 3-class distributions.

  ```python
  scores = [[0.1, 2.3, 5.], [4.2, 0.5, 3.1]]  # Shape [2, 3]
  dist = PlackettLuce(scores)

  # permutations broadcast to [[2, 1, 0], [2, 1, 0]] to match shape of scores.
  permutations = [2, 1, 0]
  dist.prob(permutations) # Shape [2]
  ```

  #### References

  [1]: Aditya Grover, Eric Wang, Aaron Zweig, Stefano Ermon. Stochastic
  Optimization of Sorting Networks via Continuous Relaxations. ICLR 2019.
  """

  def __init__(self,
               scores,
               dtype=tf.int32,
               validate_args=False,
               allow_nan_stats=True,
               name='PlackettLuce'):
    """Initialize a batch of PlackettLuce distributions.

    Args:
      scores: An N-D `Tensor`, `N >= 1`, representing the scores of a set of
        elements to be permuted. The first `N - 1` dimensions index into a
        batch of independent distributions and the last dimension represents a
        vector of scores for the elements.
      dtype: The type of the event samples (default: int32).
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
    with tf.name_scope(name) as name:
      self._scores = tensor_util.convert_nonref_to_tensor(
          scores, dtype_hint=tf.float32, name='scores')

      super(PlackettLuce, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        scores=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  def _event_size(self, scores=None):
    if scores is None:
      scores = self._scores
    if scores.shape is not None:
      event_size = tf.compat.dimension_value(scores.shape[-1])
      if event_size is not None:
        return event_size
    return tf.shape(scores)[-1]

  @property
  def scores(self):
    """Input argument `scores`.

    Each element is a non-negative value for which the sorted permutation is
    an ordering supported by this distribution.

    Returns:
      scores: A batch of scores used for initializing the distribution.
    """
    return self._scores

  def _event_shape_tensor(self, scores=None):
    scores = self._scores if scores is None else scores
    return ps.shape(scores)[-1:]

  def _event_shape(self, scores=None):
    scores = self._scores if scores is None else scores
    return tensorshape_util.with_rank_at_least(scores.shape, 1)[-1:]

  def _mode(self):
    return tf.cast(
        tf.argsort(self.scores, axis=-1, direction='DESCENDING'),
        self.dtype)

  def _log_prob(self, x):
    scores = tf.convert_to_tensor(self.scores)
    event_size = self._event_size(scores)

    x = tf.cast(x, self.dtype)
    # Broadcast scores or x if need be.
    if (not tensorshape_util.is_fully_defined(x.shape) or
        not tensorshape_util.is_fully_defined(scores.shape) or
        x.shape != scores.shape):
      broadcast_shape = ps.broadcast_shape(
          ps.shape(scores), ps.shape(x))
      scores = tf.broadcast_to(scores, broadcast_shape)
      x = tf.broadcast_to(x, broadcast_shape)
    scores_shape = ps.shape(scores)[:-1]
    scores_2d = tf.reshape(scores, [-1, event_size])
    x_2d = tf.reshape(x, [-1, event_size])

    rearranged_scores = tf.gather(scores_2d, x_2d, batch_dims=1)
    normalization_terms = tf.cumsum(rearranged_scores, axis=-1, reverse=True)
    ret = tf.math.reduce_sum(
        tf.math.log(rearranged_scores / normalization_terms), axis=-1)
    # Reshape back to user-supplied batch and sample dims prior to 2D reshape.
    ret = tf.reshape(ret, scores_shape)
    return ret

  def _sample_n(self, n, seed=None):
    scores = tf.convert_to_tensor(self.scores)
    sample_shape = ps.concat([[n], ps.shape(scores)], axis=0)
    gumbel_noise = gumbel.Gumbel(loc=0, scale=1).sample(sample_shape,
                                                        seed=seed)
    noisy_log_scores = gumbel_noise + tf.math.log(scores)
    return tf.cast(
        tf.argsort(noisy_log_scores, axis=-1, direction='DESCENDING'),
        self.dtype)

  def scores_parameter(self, name=None):
    """Scores vec computed from non-`None` input arg (`scores`)."""
    with self._name_and_control_scope(name or 'scores_parameter'):
      return tf.identity(self._scores)

  def _default_event_space_bijector(self):
    return

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_equal(
        tf.range(self._event_size(), dtype=x.dtype),
        tf.sort(x, axis=-1),
        message='Sample must be a permutation of `{0, ..., k-1}`, where `k` is '
                'the size of the last dimension of `scores`.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    scores = self._scores
    param, name = (scores, 'scores')

    # In init, we can always build shape and dtype checks because
    # we assume shape doesn't change for Variable backed args.
    if is_init:
      if not dtype_util.is_floating(param.dtype):
        raise TypeError('Argument `{}` must having floating type.'.format(name))

      msg = 'Argument `{}` must have rank at least 1.'.format(name)
      shape_static = tensorshape_util.dims(param.shape)
      if shape_static is not None:
        if len(shape_static) < 1:
          raise ValueError(msg)
      elif self.validate_args:
        param = tf.convert_to_tensor(param)
        assertions.append(
            assert_util.assert_rank_at_least(param, 1, message=msg))
        with tf.control_dependencies(assertions):
          param = tf.identity(param)

      msg1 = 'Argument `{}` must have final dimension >= 1.'.format(name)
      msg2 = 'Argument `{}` must have final dimension <= {}.'.format(
          name, dtype_util.max(tf.int32))
      event_size = shape_static[-1] if shape_static is not None else None
      if event_size is not None:
        if event_size < 1:
          raise ValueError(msg1)
        if event_size > dtype_util.max(tf.int32):
          raise ValueError(msg2)
      elif self.validate_args:
        param = tf.convert_to_tensor(param)
        assertions.append(assert_util.assert_greater_equal(
            tf.shape(param)[-1], 1, message=msg1))
        # NOTE: For now, we leave out a runtime assertion that
        # `tf.shape(param)[-1] <= tf.int32.max`.  An earlier `tf.shape` call
        # will fail before we get to this point.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(scores):
      scores = tf.convert_to_tensor(scores)
      assertions.extend([
          assert_util.assert_positive(scores),
      ])

    return assertions
