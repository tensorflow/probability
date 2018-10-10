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
"""Registration and usage mechanisms for KL-divergences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect


_DIVERGENCES = {}


__all__ = [
    "RegisterKL",
    "kl_divergence",
]


def _registered_kl(type_a, type_b):
  """Get the KL function registered for classes a and b."""
  hierarchy_a = tf_inspect.getmro(type_a)
  hierarchy_b = tf_inspect.getmro(type_b)
  dist_to_children = None
  kl_fn = None
  for mro_to_a, parent_a in enumerate(hierarchy_a):
    for mro_to_b, parent_b in enumerate(hierarchy_b):
      candidate_dist = mro_to_a + mro_to_b
      candidate_kl_fn = _DIVERGENCES.get((parent_a, parent_b), None)
      if not kl_fn or (candidate_kl_fn and candidate_dist < dist_to_children):
        dist_to_children = candidate_dist
        kl_fn = candidate_kl_fn
  return kl_fn


def kl_divergence(distribution_a, distribution_b,
                  allow_nan_stats=True, name=None):
  """Get the KL-divergence KL(distribution_a || distribution_b).

  If there is no KL method registered specifically for `type(distribution_a)`
  and `type(distribution_b)`, then the class hierarchies of these types are
  searched.

  If one KL method is registered between any pairs of classes in these two
  parent hierarchies, it is used.

  If more than one such registered method exists, the method whose registered
  classes have the shortest sum MRO paths to the input types is used.

  If more than one such shortest path exists, the first method
  identified in the search is used (favoring a shorter MRO distance to
  `type(distribution_a)`).

  Args:
    distribution_a: The first distribution.
    distribution_b: The second distribution.
    allow_nan_stats: Python `bool`, default `True`. When `True`,
      statistics (e.g., mean, mode, variance) use the value "`NaN`" to
      indicate the result is undefined. When `False`, an exception is raised
      if one or more of the statistic's batch members are undefined.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
    A Tensor with the batchwise KL-divergence between `distribution_a`
    and `distribution_b`.

  Raises:
    NotImplementedError: If no KL method is defined for distribution types
      of `distribution_a` and `distribution_b`.
  """
  kl_fn = _registered_kl(type(distribution_a), type(distribution_b))
  if kl_fn is None:
    # TODO(b/117098119): For backwards compatibility, we check TF's registry as
    # well. This typically happens when this function is called on a pair of
    # TF's distributions.
    with deprecation.silence():
      return tf.distributions.kl_divergence(distribution_a, distribution_b)

  with tf.name_scope("KullbackLeibler"):
    kl_t = kl_fn(distribution_a, distribution_b, name=name)
    if allow_nan_stats:
      return kl_t

    # Check KL for NaNs
    kl_t = tf.identity(kl_t, name="kl")

    with tf.control_dependencies([
        control_flow_ops.Assert(
            tf.logical_not(
                tf.reduce_any(tf.is_nan(kl_t))),
            ["KL calculation between %s and %s returned NaN values "
             "(and was called with allow_nan_stats=False). Values:"
             % (distribution_a.name, distribution_b.name), kl_t])]):
      return tf.identity(kl_t, name="checked_kl")


def cross_entropy(ref, other,
                  allow_nan_stats=True, name=None):
  """Computes the (Shannon) cross entropy.

  Denote two distributions by `P` (`ref`) and `Q` (`other`). Assuming `P, Q`
  are absolutely continuous with respect to one another and permit densities
  `p(x) dr(x)` and `q(x) dr(x)`, (Shanon) cross entropy is defined as:

  ```none
  H[P, Q] = E_p[-log q(X)] = -int_F p(x) log q(x) dr(x)
  ```

  where `F` denotes the support of the random variable `X ~ P`.

  Args:
    ref: `tfd.Distribution` instance.
    other: `tfd.Distribution` instance.
    allow_nan_stats: Python `bool`, default `True`. When `True`,
      statistics (e.g., mean, mode, variance) use the value "`NaN`" to
      indicate the result is undefined. When `False`, an exception is raised
      if one or more of the statistic's batch members are undefined.
    name: Python `str` prepended to names of ops created by this function.

  Returns:
    cross_entropy: `ref.dtype` `Tensor` with shape `[B1, ..., Bn]`
      representing `n` different calculations of (Shanon) cross entropy.
  """
  with tf.name_scope(name, "cross_entropy"):
    return ref.entropy() + kl_divergence(
        ref, other, allow_nan_stats=allow_nan_stats)


class RegisterKL(object):
  """Decorator to register a KL divergence implementation function.

  Usage:

  @distributions.RegisterKL(distributions.Normal, distributions.Normal)
  def _kl_normal_mvn(norm_a, norm_b):
    # Return KL(norm_a || norm_b)
  """

  def __init__(self, dist_cls_a, dist_cls_b):
    """Initialize the KL registrar.

    Args:
      dist_cls_a: the class of the first argument of the KL divergence.
      dist_cls_b: the class of the second argument of the KL divergence.
    """
    self._key = (dist_cls_a, dist_cls_b)

  def __call__(self, kl_fn):
    """Perform the KL registration.

    Args:
      kl_fn: The function to use for the KL divergence.

    Returns:
      kl_fn

    Raises:
      TypeError: if kl_fn is not a callable.
      ValueError: if a KL divergence function has already been registered for
        the given argument classes.
    """
    if not callable(kl_fn):
      raise TypeError("kl_fn must be callable, received: %s" % kl_fn)
    if self._key in _DIVERGENCES:
      raise ValueError("KL(%s || %s) has already been registered to: %s"
                       % (self._key[0].__name__, self._key[1].__name__,
                          _DIVERGENCES[self._key]))
    _DIVERGENCES[self._key] = kl_fn
    # TODO(b/117098119): For backwards compatibility, we register the
    # distributions in both this registry and the deprecated TF's registry.
    #
    # Additionally, for distributions which have deprecated copies, we register
    # all 3 combinations in their respective files (see test for the list).
    with deprecation.silence():
      tf.distributions.RegisterKL(*self._key)(kl_fn)
    return kl_fn
