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

import inspect
import six

import tensorflow.compat.v2 as tf
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


_DIVERGENCES = {}


__all__ = [
    "RegisterKL",
    "kl_divergence",
    "augment_kl_xent_docs",
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
  # NOTE: We use `d.__class__` instead of `type(d)` for objects that override
  # their `__class__` attribute.
  kl_fn = _registered_kl(distribution_a.__class__, distribution_b.__class__)
  if kl_fn is None:
    raise NotImplementedError(
        "No KL(distribution_a || distribution_b) registered for distribution_a "
        "type {} and distribution_b type {}".format(
            distribution_a.__class__.__name__,
            distribution_b.__class__.__name__))

  name = name or "KullbackLeibler"
  with tf.name_scope(name):
    # pylint: disable=protected-access
    with distribution_a._name_and_control_scope(name + "_a"):
      with distribution_b._name_and_control_scope(name + "_b"):
        kl_t = kl_fn(distribution_a, distribution_b, name=name)
        if allow_nan_stats:
          return kl_t

    # Check KL for NaNs
    kl_t = tf.identity(kl_t, name="kl")

    with tf.control_dependencies([
        tf.debugging.Assert(
            tf.logical_not(tf.reduce_any(tf.math.is_nan(kl_t))),
            [("KL calculation between {} and {} returned NaN values "
              "(and was called with allow_nan_stats=False). Values:".format(
                  distribution_a.name, distribution_b.name)), kl_t])
    ]):
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
  with tf.name_scope(name or "cross_entropy"):
    return ref.entropy() + kl_divergence(
        ref, other, allow_nan_stats=allow_nan_stats)


class RegisterKL(object):
  """Decorator to register a KL divergence implementation function.

  Usage:

  ```python
  @distributions.RegisterKL(distributions.Normal, distributions.Normal)
  def _kl_normal_mvn(norm_a, norm_b):
    # Return KL(norm_a || norm_b)
  ```

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
    return kl_fn


def _dist_classes(distributions_module):
  dist_classes = []
  for attr in dir(distributions_module):
    value = getattr(distributions_module, attr)
    if (inspect.isclass(value) and
        issubclass(value, distributions_module.Distribution)):
      dist_classes.append(value)
  return dist_classes


def _summarize_registered_kls(dist_classes):
  """Returns a str of registered KLs, to append to the doc for kl_divergence."""
  maxdists = 6  # Only show up to N subclasses per p/q.
  ps, qs = [], []
  for p, q in sorted(_DIVERGENCES,
                     key=lambda p_q: (p_q[0].__name__, p_q[1].__name__)):
    subps = sorted([d for d in dist_classes if issubclass(d, p) and d is not p],
                   key=lambda d: d.__name__)
    subqs = sorted([d for d in dist_classes if issubclass(d, q) and d is not q],
                   key=lambda d: d.__name__)
    ps.append(p.__name__)
    for subp in subps[:maxdists]:
      ps.append("{} +".format(subp.__name__))
    if len(subps) > maxdists:
      ps.append("{} more +".format(len(subps) - maxdists))
    ps.extend([""] * (len(subqs[:maxdists + 1]) - len(subps[:maxdists + 1])))

    qs.append(q.__name__)
    for subq in subqs[:maxdists]:
      qs.append("+ {}".format(subq.__name__))
    if len(subqs) > maxdists:
      qs.append("+ {} more".format(len(subqs) - maxdists))
    qs.extend([""] * (len(subps[:maxdists + 1]) - len(subqs[:maxdists + 1])))
  maxp = max(map(len, ps))
  rows = []
  for p, q in [("distribution_a", "distribution_b")] + list(zip(ps, qs)):
    rows.append("{}{} {} {}".format(
        " " * (maxp - len(p)), p, "  " if "+" in (p + q) else "||", q))
  return """
  Built-in KL(distribution_a || distribution_b) registrations:

  ```text
  {}
  {}
  {}
  ```
  """.format(rows[0], "=" * max(map(len, rows)), "\n  ".join(rows[1:]))


def augment_kl_xent_docs(distributions_module):
  """Augments doc on tfd.kl_divergence, EachDist.kl_divergence/cross_entropy."""
  dist_classes = _dist_classes(distributions_module)
  kl_divergence.__doc__ += _summarize_registered_kls(dist_classes)

  if not six.PY3:
    return  # Cannot update __doc__ on instancemethod objects in PY2.

  for dist_class in dist_classes:
    others = []
    for p, q in _DIVERGENCES:
      if issubclass(dist_class, p):
        others.extend(
            subq.__name__ for subq in dist_classes if issubclass(subq, q))
    others = sorted(set(others))

    def merge_doc(original, additional):
      for line in original.split("\n"):
        if "args:" == line.strip().lower():
          indent = line.lower().split("args")[0]
          yield "{}{}\n{}".format(indent, additional, indent)
        yield line

    if others:
      others_str = ", ".join("`{}`".format(other) for other in others)
      dist_class.kl_divergence.__doc__ = "\n".join(
          merge_doc(
              dist_class.kl_divergence.__doc__,
              "`other` types with built-in registrations: {}".format(
                  others_str)))
      dist_class.cross_entropy.__doc__ = "\n".join(
          merge_doc(
              dist_class.cross_entropy.__doc__,
              "`other` types with built-in registrations: {}".format(
                  others_str)))
