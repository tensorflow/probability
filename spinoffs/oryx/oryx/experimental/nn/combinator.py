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
"""Contains combinator layers."""

from jax import random

from oryx.core import state
from oryx.experimental.nn import base

__all__ = [
    'Serial',
]


class Serial(base.Layer):
  """Layer that executes a sequence of child layers."""

  @classmethod
  def initialize(cls, init_key, *args):
    """Initializes Serial Layer.

    Args:
      init_key: Random key.
      *args: Contains input specs and layer_inits.

    Returns:
      Tuple with the output spec and the LayerParams.
    """
    in_specs, layer_inits = args[:-1], args[-1]
    layers = state.init(list(layer_inits), name='layers')(init_key, *in_specs)
    return base.LayerParams(tuple(layers))  # pytype: disable=wrong-arg-types

  @classmethod
  def spec(cls, *args):
    in_specs, layer_inits = args[:-1], args[-1]
    return state.spec(list(layer_inits))(*in_specs)

  @property
  def state(self):
    return tuple(l.state for l in self.params)

  def _call(self, *args, rng=None, **kwargs):
    """Applies the serial sequence of layers to the input x.

    Args:
      *args: inputs to the Serial call.
      rng: an optional PRNGKey that will be threaded through the layers.
      **kwargs: keyword arguments to be passed to the layers.
    Returns:
      The result of applying a sequence of layers to args.
    """
    return self._call_and_update(*args, rng=rng, **kwargs)[0]

  def _update(self, *args, rng=None, **kwargs):
    return self._call_and_update(*args, rng=rng, **kwargs)[1]

  def _call_and_update(self, *args, rng=None, **kwargs):
    """Returns a Serial object with updated layer states."""
    layers_out = []
    for layer in self.params:
      if not isinstance(args, tuple):
        args = (args,)
      if rng is not None:
        rng, subrng = random.split(rng)
      else:
        subrng = None
      args, new_layer = state.call_and_update(layer, *args, rng=subrng,
                                              **kwargs)  # pylint: disable=assignment-from-no-return
      layers_out.append(new_layer)
    return args, self.replace(params=tuple(layers_out))

  def flatten(self):
    """Converts the Layer to a tuple suitable for PyTree."""
    children_cls = tuple(l.__class__ for l in self.params)
    xs, children_data = zip(*tuple(l.flatten() for l in self.params))
    data = (children_cls, children_data, self.name)
    return xs, data

  @classmethod
  def unflatten(cls, data, xs):
    """Reconstruct the Layer from the PyTree tuple."""
    children_cls, children_data, name = data[0], data[1], data[2]
    layers = tuple(c.unflatten(d, x) for c, x, d in
                   zip(children_cls, xs, children_data))
    layer = object.__new__(cls)
    layer_params = base.LayerParams(layers)
    layer.__init__(layer_params, name=name)
    return layer

  def __str__(self):
    """String representation of the Layer."""
    return ' >> '.join(map(str, self.params))
