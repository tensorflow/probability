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
"""JointMap bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import composition
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'JointMap',
]


class _JointMap(composition.Composition):
  """Bijector which applies a structure of bijectors in parallel.

  This is the "structured" counterpart to `Chain`. Whereas `Chain` applies an
  ordered sequence, JointMap applies a structure of transformations to a
  matching structure of inputs.

  Example Use:

  ```python
  exp = Exp()
  scale = Scale(2.)
  parallel = JointMap({'a': exp, 'b': scale})
  x = {'a': 1., 'b': 2.}

  parallel.forward(x)
  # = {'a': exp.forward(x['a']), 'b': scale.forward(x['b'])}
  # = {'a': tf.exp(1.), 'b': 2. * 2.}

  parallel.inverse(x)
  # = {'a': exp.inverse(x['a']), 'b': scale.inverse(x['b'])}
  # = {'a': tf.log(1.), 'b': 2. / 2.}
  ```

  Bijectors need not be a dictionary; it could be a list, tuple, list of
  dictionaries, or anything else supported by `tf.nest.map_structure`.
  """

  def __init__(self,
               bijectors=None,
               validate_args=False,
               parameters=None,
               name=None):
    """Instantiates `JointMap` bijector.

    Args:
      bijectors: Structure of bijector instances to apply in parallel.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      parameters: Locals dict captured by subclass constructor, to be used for
        copy/slice re-instantiation operators.
      name: Python `str`, name given to ops managed by this object.
        Default value: automatically constructed, e.g.,
          `jointmap_of_exp_and_softplus`.

    Raises:
      ValueError: if bijectors have different dtypes.
    """
    parameters = dict(locals()) if parameters is None else parameters

    if not bijectors:
      raise ValueError('`bijectors` must not be empty.')

    if name is None:
      name = ('jointmap_of_' +
              '_and_'.join([b.name for b in nest.flatten(bijectors)]))
      name = name.replace('/', '')
    with tf.name_scope(name) as name:
      # Structured dtypes are based on the non-wrapped input.
      # Keep track of the non-wrapped structure of bijectors to correctly
      # wrap inputs/outputs in _walk methods.
      self._nested_structure = self._no_dependency(
          nest.map_structure(lambda b: None, bijectors))

      super(_JointMap, self).__init__(
          bijectors=bijectors,
          validate_args=validate_args,
          forward_min_event_ndims=self._nested_structure,
          inverse_min_event_ndims=self._nested_structure,
          parameters=parameters,
          name=name)

  def _walk_forward(self, step_fn, xs, **kwargs):
    """Applies `transform_fn` to `x` in parallel over nested bijectors."""
    # Set check_types to False to support bij-structures wrapped by Trackable.
    return nest.map_structure_up_to(
        self._nested_structure,
        lambda bij, x: step_fn(bij, x, **kwargs.get(bij.name, {})),  # pylint: disable=unnecessary-lambda
        self._bijectors, xs, check_types=False)

  def _walk_inverse(self, step_fn, ys, **kwargs):
    """Applies `transform_fn` to `y` in parallel over nested bijectors."""
    # Set check_types to False to support bij-structures wrapped by Trackable.
    return nest.map_structure_up_to(
        self._nested_structure,
        lambda bij, y: step_fn(bij, y, **kwargs.get(bij.name, {})),  # pylint: disable=unnecessary-lambda
        self._bijectors, ys, check_types=False)


class JointMap(_JointMap, bijector_lib.AutoCompositeTensorBijector):

  def __new__(cls, *args, **kwargs):
    """Returns a `_JointMap` any of `bijectors` is not a `CompositeTensor."""
    if cls is JointMap:
      if args:
        bijectors = args[0]
      else:
        bijectors = kwargs.get('bijectors')
      if bijectors is not None:
        if not all(isinstance(b, tf.__internal__.CompositeTensor)
                   for b in tf.nest.flatten(bijectors)):
          return _JointMap(*args, **kwargs)
    return super(JointMap, cls).__new__(cls)


JointMap.__doc__ = _JointMap.__doc__ + '\n' + (
    'If every element of `bijectors` is a `CompositeTensor`, the resulting '
    '`JointMap` bijector is a `CompositeTensor` as well. If any element of '
    '`bijectors` is not a `CompositeTensor`, then a non-`CompositeTensor` '
    '`_JointMap` instance is created instead. Bijector subclasses that inherit '
    'from `JointMap` will also inherit from `CompositeTensor`.')
