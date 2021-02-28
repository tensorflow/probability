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
"""Inline bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector


__all__ = [
    'Inline',
]


class Inline(bijector.Bijector):
  """Bijector constructed from custom callables.

  Example Use:

  ```python
  exp = Inline(
      forward_fn=tf.exp,
      inverse_fn=tf.math.log,
      inverse_log_det_jacobian_fn=lambda y: -tf.math.log(y),
      forward_min_event_ndims=0,
      is_increasing=True,
      name='exp')
  ```

  The above example is equivalent to the `Bijector` `Exp()`.
  """

  def __init__(self,
               forward_fn=None,
               inverse_fn=None,
               inverse_log_det_jacobian_fn=None,
               forward_log_det_jacobian_fn=None,
               forward_event_shape_fn=None,
               forward_event_shape_tensor_fn=None,
               inverse_event_shape_fn=None,
               inverse_event_shape_tensor_fn=None,
               is_constant_jacobian=False,
               is_increasing=None,
               validate_args=False,
               forward_min_event_ndims=bijector.UNSPECIFIED,
               inverse_min_event_ndims=bijector.UNSPECIFIED,
               name='inline'):
    """Creates a `Bijector` from callables.

    At the minimum, you must supply one of `forward_min_event_ndims` or
    `inverse_min_event_ndims`. To be fully functional, a typical bijector will
    also require `forward_fn`, `inverse_fn` and at least one of
    `inverse_log_det_jacobian_fn` or `forward_log_det_jacobian_fn`.

    Args:
      forward_fn: Python callable implementing the forward transformation.
      inverse_fn: Python callable implementing the inverse transformation.
      inverse_log_det_jacobian_fn: Python callable implementing the
        `log o det o jacobian` of the inverse transformation.
      forward_log_det_jacobian_fn: Python callable implementing the
        `log o det o jacobian` of the forward transformation.
      forward_event_shape_fn: Python callable implementing non-identical
        static event shape changes. Default: shape is assumed unchanged.
      forward_event_shape_tensor_fn: Python callable implementing non-identical
        event shape changes. Default: shape is assumed unchanged.
      inverse_event_shape_fn: Python callable implementing non-identical
        static event shape changes. Default: shape is assumed unchanged.
      inverse_event_shape_tensor_fn: Python callable implementing non-identical
        event shape changes. Default: shape is assumed unchanged.
      is_constant_jacobian: Python `bool` indicating that the Jacobian is
        constant for all input arguments.
      is_increasing: `bool` `Tensor` indicating a scalar bijector function is
        increasing for all input arguments, or a callable returning a `bool`
        `Tensor` specifying such truth values.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      forward_min_event_ndims: Python `int` indicating the minimal
        dimensionality this bijector acts on.
      inverse_min_event_ndims: Python `int` indicating the minimal
        dimensionality this bijector acts on.
      name: Python `str`, name given to ops managed by this object.

    Raises:
      TypeError: If any of the non-`None` `*_fn` arguments are not callable.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._maybe_implement(forward_fn, '_forward', 'forward_fn')
      self._maybe_implement(inverse_fn, '_inverse', 'inverse_fn')
      self._maybe_implement(inverse_log_det_jacobian_fn,
                            '_inverse_log_det_jacobian',
                            'inverse_log_det_jacobian_fn')
      self._maybe_implement(forward_log_det_jacobian_fn,
                            '_forward_log_det_jacobian',
                            'forward_log_det_jacobian_fn')
      if is_increasing is not None and not callable(is_increasing):
        is_increasing_val = is_increasing
        is_increasing = lambda: is_increasing_val
      self._maybe_implement(is_increasing, '_is_increasing', 'is_increasing')

      # By default assume shape doesn't change.
      self._forward_event_shape = _maybe_impute_as_identity(
          forward_event_shape_fn, 'forward_event_shape_fn')
      self._forward_event_shape_tensor = _maybe_impute_as_identity(
          forward_event_shape_tensor_fn, 'forward_event_shape_tensor_fn')
      self._inverse_event_shape = _maybe_impute_as_identity(
          inverse_event_shape_fn, 'inverse_event_shape_fn')
      self._inverse_event_shape_tensor = _maybe_impute_as_identity(
          inverse_event_shape_tensor_fn, 'inverse_event_shape_tensor_fn')

      super(Inline, self).__init__(
          forward_min_event_ndims=forward_min_event_ndims,
          inverse_min_event_ndims=inverse_min_event_ndims,
          is_constant_jacobian=is_constant_jacobian,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  def _maybe_implement(self, fn, lhs_name, rhs_name):
    if not fn:
      return
    if not callable(fn):
      raise TypeError('`{}` is not a callable function.'.format(rhs_name))
    setattr(self, lhs_name, fn)


def _maybe_impute_as_identity(fn, name):
  if fn is None:
    return lambda x: x
  if not callable(fn):
    raise TypeError('`{}` is not a callable function.'.format(name))
  return fn
