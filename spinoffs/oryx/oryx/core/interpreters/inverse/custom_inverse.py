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
# Lint as: python3
"""Contains logic for defining custom inverses for functions.

Automatic inversion works for only a certain class of functions (see
`core.inverse` documentation for more details). For example, an
autoregressive neural network will likely use masked weight matrices in order
to be invertible but the automatic inversion is not aware of autoregressive
masked matrices (yet!). Furthermore, we may want more numerically stable
inverses for functions like softmax or sigmoid.

This module provides a `custom_inverse` decorator for Python functions that
enables overriding the default programmatic inversion. See `custom_inverse`
for further documentation.
"""
from jax import util as jax_util
from jax._src import tree_util
from oryx.core import kwargs_util
from oryx.core import primitive

from oryx.core.interpreters.inverse import core
from oryx.core.interpreters.inverse import slice as slc

__all__ = [
    'custom_inverse',
    'NonInvertibleError'
]

ildj_registry = core.ildj_registry
NDSlice = slc.NDSlice
InverseAndILDJ = core.InverseAndILDJ
safe_zip = jax_util.safe_zip


class CustomInverse:
  """Wrapper class for functions with a custom inverse."""

  def __init__(self, func, prim):
    self.func = func
    self.prim = prim

  def __call__(self, *args, **kwargs):
    return primitive.initial_style_bind(self.prim)(self.func)(
        *args, **kwargs)

  def def_inverse_unary(self, f_inv=None, f_ildj=None):
    """Defines a unary inverse rule.

    Args:
      f_inv: An optional unary function that returns the inverse of this
        function. If not provided, we automatically invert the forward function.
      f_ildj: An optional unary function that computes the ILDJ of this
        function. If it is not provided, we automatically compute it from
        `f_inv`.
    """
    f_inv_ = f_inv or core.inverse(self.func)
    f_ildj_ = f_ildj or core.ildj(core.inverse(f_inv_), reduce_ildj=False)

    def f_inverse_and_ildj(invals, outval, outildj):
      inval = invals[0]
      if outval is None:
        raise NonInvertibleError()
      if inval is None:
        new_inval = f_inv_(outval)
        new_inildj = f_ildj_(outval) + outildj
        return (new_inval,), (new_inildj,)
      raise NonInvertibleError()

    self.def_inverse_and_ildj(f_inverse_and_ildj)

  def def_inverse_and_ildj(self, f_ildj):
    """Defines a general inverse and ILDJ rule.

    Args:
      f_ildj: A function from `invals, outvals, out_ildjs` to `new_invals,
        new_ildjs`. Unknown values are provided as `None`.
    """

    def ildj_rule(incells, outcells, *, in_tree, out_tree, num_consts, **_):
      # First incell is a wrapped function because prim is a call primitive.
      const_incells, incells = jax_util.split_list(incells, [num_consts])
      if (all(outcell.top() for outcell in outcells) and
          any(not incell.top() for incell in incells)):
        flat_outvals = [outcell.val for outcell in outcells]
        flat_outildjs = [outcell.ildj for outcell in outcells]
        outvals = tree_util.tree_unflatten(out_tree, flat_outvals)
        outildjs = tree_util.tree_unflatten(out_tree, flat_outildjs)
        flat_invals = [
            None if not incell.top() else incell.val for incell in incells
        ]
        invals = tree_util.tree_unflatten(in_tree, flat_invals)
        try:
          new_invals, new_ildjs = f_ildj(invals, outvals, outildjs)
        except NonInvertibleError:
          return const_incells + incells, outcells, None
        # We need to flatten the output from `f_ildj` using
        # `tree_util.tree_flatten` but if the user returns `None` (when
        # inversion is not possible), JAX will remove `None`s from the flattened
        # version and the number of `new_incells` will not match the old
        # `incells`. We use the private `_replace_nones` feature in JAX to
        # replace it with a sentinel that won't be removed when flattening.
        none_ = object()
        new_invals = tree_util._replace_nones(none_, new_invals)  # pylint: disable=protected-access
        new_ildjs = tree_util._replace_nones(none_, new_ildjs)  # pylint: disable=protected-access
        new_flat_invals = tree_util.tree_leaves(new_invals)
        new_flat_ildjs = tree_util.tree_leaves(new_ildjs)
        inslices = [
            NDSlice.new(inval, ildj)
            for inval, ildj in zip(new_flat_invals, new_flat_ildjs)
        ]
        new_incells = []
        for new_flat_inval, old_incell, inslice in zip(
            new_flat_invals, incells, inslices):
          if new_flat_inval is not none_:
            new_incells.append(InverseAndILDJ(old_incell.aval, [inslice]))
          else:
            new_incells.append(old_incell)
        return const_incells + new_incells, outcells, None
      elif (all(incell.top() for incell in incells) and
            any(not outcell.top() for outcell in outcells)):
        flat_invals = [incell.val for incell in incells]
        invals = tree_util.tree_unflatten(in_tree, flat_invals)
        outvals = self(*invals)
        flat_outvals = tree_util.tree_leaves(outvals)
        outcells = [InverseAndILDJ.new(outval) for outval in flat_outvals]
        return const_incells + incells, outcells, None
      return const_incells + incells, outcells, None

    core.ildj_registry[self.prim] = ildj_rule


def custom_inverse(f):
  """Decorates a function to enable defining a custom inverse.

  A `custom_inverse`-decorated function is semantically identical to the
  original except when it is inverted with `core.inverse`. By default,
  `core.inverse(custom_inverse(f))` will programmatically invert the body of
  `f`, but `f` has two additional methods that can override that behavior:
  `def_inverse_unary` and `def_inverse_ildj`.

  ## `def_inverse_unary`

  `def_inverse_unary` is applicable if `f` is a unary function.
  `def_inverse_unary` takes in an optional `f_inv` function, which is a unary
  function from the output of `f` to the input of `f`.

  Example:
  ```python
  @custom_inverse
  def add_one(x):
    return x + 1.
  add_one.def_inverse_unary(lambda x: x * 2)  # Define silly custom inverse.
  inverse(add_one)(2.)  # ==> 4.
  ```

  With a unary `f_inv` function, Oryx will automatically compute an inverse
  log-det Jacobian using `core.ildj(core.inverse(f_inv))`, but a user can
  also override the Jacobian term by providing the optional `f_ildj` keyword
  argument to `def_inverse_unary`.

  Example:
  ```python
  @custom_inverse
  def add_one(x):
    return x + 1.
  add_one.def_inverse_unary(lambda x: x * 2, f_ildj=lambda x: jnp.ones_like(x))
  ildj(add_one)(2.)  # ==> 1.
  ```

  ## `def_inverse_and_ildj`

  A more general way of defining a custom inverse or ILDJ is to use
  `def_inverse_and_ildj`, which will enable the user to invert functions with
  partially known inputs and outputs. Take an example like
  `add = lambda x, y: x + y`, which cannot be inverted with just the output,
  but can be inverted when just one input is known. `def_inverse_and_ildj`
  takes a single function `f_ildj` as an argument. `f_ildj` is a function from
  `invals` (a set of values corresponding to `f`'s inputs), `outvals` (a set
  of values corresponding to `f`'s outputs) and `out_ildjs` (a set of inverse
  diagonal log-Jacobian values for each of the `outvals`). If any are unknown,
  they will be `None`. `f_ildj` should return a tuple
  `(new_invals, new_inildjs)` which corresponds to known values of the inputs
  and any corresponding diagonal Jacobian values (which should be the same shape
  as `invals`). If these values cannot be computed (e.g. too many values are
  `None`) the user can raise a `NonInvertibleError` which will signal to Oryx to
  give up trying to invert the function for this set of values.

  Example:
  ```python
  @custom_inverse
  def add(x, y):
    return x + y

  def add_ildj(invals, outvals, out_ildjs):
    x, y = invals
    z = outvals
    z_ildj = outildjs
    if x is None and y is None:
      raise NonInvertibleError()
    if x is None:
      return (z - y, y), (z_ildj + jnp.zeros_like(z), jnp.zeros_like(z))
    if y is None:
      return (x, z - x), (jnp.zeros_like(z), z_ildj + jnp.zeros_like(z))

  add.def_inverse_and_ildj(add_ildj)
  inverse(partial(add, 1.))(2.)  # ==> 1.
  inverse(partial(add, 1.))(2.)  # ==> 0.
  ```

  Args:
    f: a function for which we'd like to define a custom inverse.

  Returns:
    A `CustomInverse` object whose inverse can be overridden with
    `def_inverse_unary` or `def_inverse`.
  """
  return CustomInverse(f, primitive.InitialStylePrimitive(f.__name__))


class NonInvertibleError(Exception):
  """Raised by a custom inverse definition when values are unknown."""


@kwargs_util.argspec_and_keywords.register(CustomInverse)
def custom_inverse_argspec_and_keywords(ci):
  return kwargs_util.argspec_and_keywords(ci.func)
