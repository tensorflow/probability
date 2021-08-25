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
"""Chain bijector."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import composition
from tensorflow_probability.python.internal import prefer_static as ps


__all__ = [
    'Chain',
]


class _Chain(composition.Composition):
  """Bijector which applies a sequence of bijectors.

  Example Use:

  ```python
  chain = Chain([Exp(), Softplus()], name="one_plus_exp")
  ```

  Results in:

  * Forward:

    ```python
    exp = Exp()
    softplus = Softplus()
    Chain([exp, softplus]).forward(x)
    = exp.forward(softplus.forward(x))
    = tf.exp(tf.log(1. + tf.exp(x)))
    = 1. + tf.exp(x)
    ```

  * Inverse:

    ```python
    exp = Exp()
    softplus = Softplus()
    Chain([exp, softplus]).inverse(y)
    = softplus.inverse(exp.inverse(y))
    = tf.log(tf.exp(tf.log(y)) - 1.)
    = tf.log(y - 1.)
    ```

  Keyword arguments can be passed to the inner bijectors by utilizing the inner
  bijector names, e.g.:

  ```python
  chain = Chain([Bijector1(name='b1'), Bijector2(name='b2')])
  y = chain.forward(x, b1={'arg': 1}, b2={'arg': 2})

  # Equivalent to:
  z = Bijector2().forward(x, arg=1)
  y = Bijector1().forward(z, arg=2)
  ```

  """

  def __init__(self,
               bijectors=None,
               validate_args=False,
               validate_event_size=False,
               parameters=None,
               name=None):
    """Instantiates `Chain` bijector.

    Args:
      bijectors: Python `list` of bijector instances. An empty list makes this
        bijector equivalent to the `Identity` bijector.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      validate_event_size: Checks that bijectors are not applied to inputs with
        incomplete support (that is, inputs where one or more elements are a
        deterministic transformation of the others). For example, the following
        LDJ would be incorrect:
        `Chain([Scale(), SoftmaxCentered()]).forward_log_det_jacobian([1], [1])`
        The jacobian contribution from `Scale` applies to a 2-dimensional input,
        but the output from `SoftMaxCentered` is a 1-dimensional input embedded
        in a 2-dimensional space. Setting `validate_event_size=True` (default)
        prints warnings in these cases. When `validate_args` is also `True`, the
        warning is promoted to an exception.
      parameters: Locals dict captured by subclass constructor, to be used for
        copy/slice re-instantiation operators.
      name: Python `str`, name given to ops managed by this object. Default:
        E.g., `Chain([Exp(), Softplus()]).name == "chain_of_exp_of_softplus"`.

    Raises:
      ValueError: if bijectors have different dtypes.
    """
    parameters = dict(locals()) if parameters is None else parameters

    if name is None:
      name = ('identity' if not bijectors else
              '_of_'.join(['chain'] + [b.name for b in bijectors]))
      name = name.replace('/', '')

    # If there are no bijectors, treat this like a single-part Identity.
    forward_min_event_ndims = 0
    inverse_min_event_ndims = 0
    if bijectors:
      forward_min_event_ndims = None  # Inferred by base class.
      inverse_min_event_ndims = None  # Inferred by base class.

    with tf.name_scope(name) as name:
      super(_Chain, self).__init__(
          bijectors=bijectors or (),
          validate_args=validate_args,
          validate_event_size=validate_event_size,
          forward_min_event_ndims=forward_min_event_ndims,
          inverse_min_event_ndims=inverse_min_event_ndims,
          parameters=parameters,
          name=name)

  def _is_increasing(self, **kwargs):
    # desc(desc)=>asc, asc(asc)=>asc, other cases=>desc.
    is_increasing = True
    for b in self._bijectors:
      is_increasing = ps.equal(
          is_increasing, b._internal_is_increasing(**kwargs.get(b.name, {})))  # pylint: disable=protected-access
    return is_increasing

  def _walk_forward(self, step_fn, x, **kwargs):
    """Applies `transform_fn` to `x` sequentially over nested bijectors."""
    for bij in reversed(self._bijectors):
      x = step_fn(bij, x, **kwargs.get(bij.name, {}))
    return x  # Now `y`

  def _walk_inverse(self, step_fn, y, **kwargs):
    """Applies `transform_fn` to `y` sequentially over nested bijectors."""
    for bij in self._bijectors:
      y = step_fn(bij, y, **kwargs.get(bij.name, {}))
    return y  # Now `x`


class Chain(_Chain, bijector_lib.AutoCompositeTensorBijector):

  def __new__(cls, *args, **kwargs):
    """Returns a `_Chain` if any of `bijectors` is not a `CompositeTensor."""
    if cls is Chain:
      if args:
        bijectors = args[0]
      else:
        bijectors = kwargs.get('bijectors')

      if bijectors is not None:
        if not all(isinstance(b, tf.__internal__.CompositeTensor)
                   for b in bijectors):
          return _Chain(*args, **kwargs)
    return super(Chain, cls).__new__(cls)


Chain.__doc__ = _Chain.__doc__ + '\n' + (
    'If every element of the `bijectors` list is a `CompositeTensor`, the '
    'resulting `Chain` bijector is a `CompositeTensor` as well. If any element '
    'of `bijectors` is not a `CompositeTensor`, then a non-`CompositeTensor` '
    '`_Chain` instance is created instead. Bijector subclasses that inherit '
    'from `Chain` will also inherit from `CompositeTensor`.')
