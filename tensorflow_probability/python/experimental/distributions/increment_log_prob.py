# Copyright 2021 The TensorFlow Probability Authors.
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
"""The `IncrementLogProb` class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'IncrementLogProb',
]

JAX_MODE = False  # Overwritten by rewrite script.


class IncrementLogProb(object):
  """A distribution-like object representing an unnormalized density at a point.

  `IncrementLogProb` is a distribution-like class that represents an
  unnormalized density, sometimes called a "factor".  Its raison d'être is to
  provide an overall offset to the log probability of a JointDensity.

  It has no `log_prob` method, instead surfacing the unnormalized log
  probability/density through its `unnormalized_log_prob` method.  It retains a
  `sample` method, which returns an empty tensor with the same `dtype` as the
  `increment_log_prob` argument provided to it originally.
  """

  def __init__(
      self,
      log_prob_increment,
      validate_args=False,  # pylint: disable=unused-argument
      allow_nan_stats=False,  # pylint: disable=unused-argument
      reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,  # pylint: disable=unused-argument
      name='IncrementLogProb'):
    """Construct a `IncrementLogProb` distribution-like object.

    Args:
      log_prob_increment: Log probability/density to increment by.
      validate_args: This argument is ignored, but is present because it is used
        in certain situations where `Distribution`s are expected.
      allow_nan_stats: This argument is ignored, but is present because it is
        used in certain situations where `Distribution`s are expected.
      reparameterization_type: This argument is ignored, but is present because
        it is used in certain situations where `Distribution`s are expected.
      name: Python `str` name prefixed to Ops created by this class.
    """
    self._parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._log_prob_increment = tensor_util.convert_nonref_to_tensor(
          log_prob_increment)
      self._dtype = self._log_prob_increment.dtype
      self._name = name

  @property
  def log_prob_increment(self):
    """The amount to increment log probability by."""
    return self._log_prob_increment

  @property
  def name(self):
    return self._name

  @property
  def parameters(self):
    return dict(self._parameters)

  @property
  def dtype(self):
    return self._dtype

  def unnormalized_log_prob(self, _):
    return self.log_prob_increment

  @property
  def batch_shape(self):
    return self.log_prob_increment.shape

  def batch_shape_tensor(self, name='batch_shape_tensor'):
    """Shape of a single sample from a single event index as a 1-D `Tensor`.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Args:
      name: name to give to the op

    Returns:
      batch_shape: `Tensor`.
    """
    with tf.name_scope(name):

      def conversion_fn(s):
        return tf.identity(
            tf.convert_to_tensor(s, dtype=tf.int32), name='batch_shape')

      if JAX_MODE:
        conversion_fn = ps.convert_to_shape_tensor
      return nest.map_structure_up_to(
          None, conversion_fn, self.batch_shape, check_types=False)

  @property
  def event_shape(self):
    return tf.TensorShape([0])

  def event_shape_tensor(self, name='event_shape_tensor'):
    """Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      event_shape: `Tensor`.
    """
    with tf.name_scope(name):

      def conversion_fn(s):
        return tf.identity(
            tf.convert_to_tensor(s, dtype=tf.int32), name='event_shape')

      if JAX_MODE:
        conversion_fn = ps.convert_to_shape_tensor
      return nest.map_structure_up_to(
          None, conversion_fn, self.event_shape, check_types=False)

  @property
  def allow_nan_stats(self):
    return False

  @property
  def reparameterization_type(self):
    return False

  @classmethod
  def _parameter_control_dependencies(cls, *_, **__):
    return []

  @property
  def experimental_shard_axis_names(self):
    """The list or structure of lists of active shard axis names."""
    return []

  def sample(self, sample_shape=(), seed=None, name='sample'):  # pylint: disable=unused-argument
    return tf.zeros(
        ps.concat(
            [
                # sample_shape might be a scalar
                ps.reshape(
                    ps.convert_to_shape_tensor(sample_shape, tf.int32),
                    shape=[-1]),
                self.batch_shape_tensor(),
                self.event_shape_tensor()
            ],
            axis=0))
