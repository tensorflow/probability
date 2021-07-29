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
"""Annotations of special functions."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import builtins
import functools
import math
import operator
import sys

import tensorflow.compat.v2 as tf


# According to:
#
# https://docs.python.org/3/reference/datamodel.html#special-lookup
#
# "implicit invocations of special methods are only guaranteed to work
# correctly if defined on an object's type, not in the object's instance
# dictionary".
#
# Additionally:
#
# "In addition to bypassing any instance attributes in the interest of
# correctness, implicit special method lookup generally also bypasses the
# __getattribute__() method even of the object's metaclass"
#
# We therefore use:
#   https://docs.python.org/3/reference/datamodel.html#special-method-names
# to compile sets of names which we will explicitly handle if defined in the
# static class.

PY2_OR_OLDER = sys.version_info[0] < 3


def _reverse(fn):
  @functools.wraps(fn)
  def _wrapped(a, b, *args, **kwargs):
    return fn(b, a, *args, **kwargs)
  return _wrapped


def _defer(fn, name=None, reverse=False):
  """Wraps `fn` by instead calling `self.__action__`."""
  if name is None:
    name = fn.__name__
    if not name.startswith('__'):
      name = '__' + name
    if not name.endswith('__'):
      name = name + '__'
  if reverse:
    fn = _reverse(fn)
    if name.startswith('__'):
      name = '__r' + name[2:]
    else:
      name = 'r' + name
  @functools.wraps(fn)
  def _wrapped_fn(self, *args, **kwargs):
    return self.__action__(fn, *args, _action_name=name, **kwargs)
  return _wrapped_fn


def _enter(self):
  return self.__enter__()


def _exit(self, exc_type, exc_value, traceback):
  return self.__exit__(exc_type, exc_value, traceback)


def _call(self, *args, **kwargs):
  # Note: it is essential to use `self(...)` rather than `self.__call__(...)`
  # since the latter fails to correctly forward to `self.__init__(...)`.
  return self(*args, **kwargs)


class SpecialMethods(object):
  """Special methods to intercept."""

  __slots__ = ('_name',)

  def __action__(self, fn, *args, **kwargs):
    action_name = kwargs.pop('_action_name', None)
    name = try_get_name(fn) if action_name is None else action_name
    raise NotImplementedError(
        'Subclass must implement `__action__` ({}).'.format(name))

  __repr__ = _defer(builtins.repr)
  __str__ = _defer(builtins.str)
  __bytes__ = _defer(builtins.bytes)
  __format__ = _defer(builtins.format)

  __lt__ = _defer(operator.lt)
  __le__ = _defer(operator.le)
  __eq__ = _defer(operator.eq)
  __ne__ = _defer(operator.ne)
  __gt__ = _defer(operator.gt)
  __ge__ = _defer(operator.ge)

  __hash__ = _defer(builtins.hash)
  __bool__ = _defer(builtins.bool)
  __len__ = _defer(builtins.len)
  __getitem__ = _defer(operator.getitem)
  __setitem__ = _defer(operator.setitem)
  __delitem__ = _defer(operator.delitem)
  __iter__ = _defer(builtins.iter)
  __next__ = _defer(builtins.next)
  __reversed__ = _defer(builtins.reversed)
  __contains__ = _defer(operator.contains)

  __neg__ = _defer(operator.neg)
  __pos__ = _defer(operator.pos)
  __abs__ = _defer(builtins.abs)
  __invert__ = _defer(operator.invert)

  __complex__ = _defer(builtins.complex)
  __int__ = _defer(builtins.int)
  __float__ = _defer(builtins.float)
  __index__ = _defer(operator.index)

  __round__ = _defer(builtins.round)
  __trunc__ = _defer(math.trunc)
  __floor__ = _defer(math.floor)
  __ceil__ = _defer(math.ceil)

  __enter__ = _defer(_enter, '__enter__')
  __exit__ = _defer(_exit, '__exit__')
  __call__ = _defer(_call, '__call__')

  if PY2_OR_OLDER:

    def next(self, *default):
      # We don't call __action__ since __next__ will do it for us.
      return self.__next__(*default)

    # '__coerce__',
    __inv__ = _defer(operator.inv)
    __nonzero__ = _defer(builtins.bool, '__nonzero__')

    __long__ = _defer(builtins.long)
    __hex__ = _defer(builtins.hex)
    __oct__ = _defer(builtins.oct)

    # Old PY2:
    # __getslice__ = _defer(builtins, '__getslice__')
    # __setslice__ = _defer(builtins, '__setslice__')
    # __delslice__ = _defer(builtins, '__delslice__')

  else:

    __length_hint__ = _defer(operator.length_hint)

  __add__ = _defer(operator.add)
  __sub__ = _defer(operator.sub)
  __mul__ = _defer(operator.mul)
  __truediv__ = _defer(operator.truediv)
  __floordiv__ = _defer(operator.floordiv)
  __mod__ = _defer(operator.mod)
  __divmod__ = _defer(builtins.divmod)
  __pow__ = _defer(builtins.pow)
  __lshift__ = _defer(operator.lshift)
  __rshift__ = _defer(operator.rshift)
  __and__ = _defer(operator.and_, '__and__')
  __xor__ = _defer(operator.xor)
  __or__ = _defer(operator.or_, '__or__')

  __radd__ = _defer(operator.add, reverse=True)
  __rsub__ = _defer(operator.sub, reverse=True)
  __rmul__ = _defer(operator.mul, reverse=True)
  __rtruediv__ = _defer(operator.truediv, reverse=True)
  __rfloordiv__ = _defer(operator.floordiv, reverse=True)
  __rmod__ = _defer(operator.mod, reverse=True)
  __rdivmod__ = _defer(builtins.divmod, reverse=True)
  __rpow__ = _defer(builtins.pow, reverse=True)
  __rlshift__ = _defer(operator.lshift, reverse=True)
  __rrshift__ = _defer(operator.rshift, reverse=True)
  __rand__ = _defer(operator.and_, '__and__', reverse=True)
  __rxor__ = _defer(operator.xor, reverse=True)
  __ror__ = _defer(operator.or_, '__or__', reverse=True)

  __iadd__ = _defer(operator.iadd)
  __isub__ = _defer(operator.isub)
  __imul__ = _defer(operator.imul)
  __itruediv__ = _defer(operator.itruediv)
  __ifloordiv__ = _defer(operator.ifloordiv)
  __imod__ = _defer(operator.imod)
  __ipow__ = _defer(operator.ipow)
  __ilshift__ = _defer(operator.ilshift)
  __irshift__ = _defer(operator.irshift)
  __iand__ = _defer(operator.iand)
  __ixor__ = _defer(operator.ixor)
  __ior__ = _defer(operator.ior)

  if PY2_OR_OLDER:

    __cmp__ = _defer(builtins.cmp)
    __rcmp__ = _defer(builtins.cmp, reverse=True)
    __div__ = _defer(operator.div)
    __rdiv__ = _defer(operator.div, reverse=True)
    __idiv__ = _defer(operator.idiv)

  else:

    __matmul__ = _defer(operator.matmul)
    __rmatmul__ = _defer(operator.matmul, reverse=True)
    __imatmul__ = _defer(operator.imatmul)

  def __getattr__(self, attr):
    """Implements `__getattr__`."""
    # By implementing __getattr__, attributes will first be accessed from self,
    # otherwise will be accessed from the deferred object.
    if (attr in _GETATTRIBUTE_PASSTHROUGH_OVERRIDE or
        # For some reason we can't use generators here because they behave
        # differently in Ipython REPL execution regime.
        any(tuple(fn(attr)
                  for fn in _GETATTRIBUTE_PASSTHROUGH_OVERRIDE_CALLABLES))):
      raise AttributeError()
    return self.__action__(getattr, attr, _action_name=attr)


# If the following attributes are not found in the DeferredBase subclass then
# they will raise AttributeError on access.
# Note: Most of these functions will always be defined in DeferredBase. For
# those which are in DeferredBase, inclusion here has no extra overhead.
# pylint: disable=line-too-long
_GETATTRIBUTE_PASSTHROUGH_OVERRIDE = {
    # https://docs.python.org/3/reference/datamodel.html

    '__annotations__',    # inspect: method: mapping of parameters names to annotations; "return" key is reserved for return annotations.
    '__code__',           # inspect: method: code object containing compiled function bytecode
    '__defaults__',       # inspect: method: tuple of any default values for positional or keyword parameters
    '__doc__',            # inspect: class/method/module: documentation string
    '__file__',           # inspect: module: filename (missing for built-in modules)
    '__func__',           # inspect: method: function object containing implementation of method
    '__globals__',        # inspect: method: global namespace in which this function was defined
    '__kwdefaults__',     # inspect: method: mapping of any default values for keyword-only parameters
    '__module__',         # inspect: class/method: name of module in which this class was defined
    '__name__',           # inspect: class/method: name with which this class was defined
    '__qualname__',       # inspect: class/method: qualified name
    '__self__',           # inspect: method: instance to which this method is bound, or None

    '__closure__',
    '__signature__',
    '__text_signature__',

    '__dict__',
    '__slots__',
    '__weakref__',
    '__class__',

    '__hash__',
    '__eq__',
    '__ne__',
    '__ge__',
    '__gt__',
    '__le__',
    '__lt__',

    '__copy__',           # serialization
    '__deepcopy__',       # serialization
    '__getnewargs__',     # serialization: pickle
    '__reduce__',         # serialization: pickle
    '__reduce_ex__',      # serialization: pickle
    '__setstate__',       # serialization: pickle

    '__delattr__',
    '__getattr__',
    '__getattribute__',
    '__setattr__',

    '_ipython_canary_method_should_not_exist_',
    '_ipython_display_',  # print: Queried by Jupyter Notebook.
    '__format__',         # print
    '__dir__',            # print
    '__repr__',           # print
    '__str__',            # print

    '__new__',
    '__init__',
    '__prepare__',
    '__classcell__',
    '__class_getitem__',
    '__delete__',
    '__init_subclass__',
    '__instancecheck__',
    '__mro__',
    '__mro_entries__',
    '__set_name__',
    '__sizeof__',
    '__subclasscheck__',
    '__subclasshook__',
    '__traceback__',

    '__del__',            # descriptors
    '__get__',            # descriptors
    '__set__',            # descriptors

    # '_partialmethod',    # Might not be needed if we exclude __signature__.
}


# pylint: disable=g-long-lambda
_GETATTRIBUTE_PASSTHROUGH_OVERRIDE_CALLABLES = [
    lambda x: (len(x) > 2
               and x.startswith('_repr_')
               and x[-2] != '_'
               and x[-1] == '_'),  # Queried by Jupyter Notebook, eg,
                                   # "_repr_latex_".
]
# pylint: enable=g-long-lambda

# pylint: enable=line-too-long


# --- The following is for reference purposes. -------------


SPECIAL_PROPERTIES = {
    '__module__',
    '__doc__',
    '__dict__',
    '__weakref__',
    '__name__',
    '__class__',

    '__closure__',
    '__code__',
    '__defaults__',
    '__globals__',
    '__qualname__',
}


IGNORED_SPECIAL_METHODS = {
    '__new__',
    '__init__',
    '__slots__',
    '__call__',

    '__get__',
    '__set__',
    '__del__',

    '__getattr__',
    '__getattribute__',
    '__setattr__',
    '__delattr__',

    '__dir__',
    '__delete__',
    '__set_name__',
    '__init_subclass__',
    '__class_getitem__',

    '__instancecheck__',
    '__subclasscheck__',
    '__subclasshook__',

    '__missing__',
    '__sizeof__',

    # Class serialization:
    # (Pretty sure only only `__copy__` is magic.)
    '__copy__',
    '__deepcopy__',
    '__reduce__',
    '__reduce_ex__',
    '__getnewargs__',
    '__setstate__',
}


class ObjectProxy(SpecialMethods):
  """Like `wrapt.ObjectProxy` except using our way."""
  slots = ('__wrapped__', '__unpack__')

  def __init__(self, wrapped, unpack=True):
    self.__wrapped__ = wrapped
    self.__unpack__ = unpack

  def __action__(self, fn, *args, **kwargs):
    kwargs.pop('_action_name', None)
    self, args, kwargs = tf.nest.map_structure(
        lambda x: (  # pylint: disable=g-long-lambda
            x.__wrapped__ if isinstance(x, ObjectProxy) and x.__unpack__
            else x),
        [self, args, kwargs])
    return fn(self, *args, **kwargs)


def try_get_name(fn, name_fallback='unknown'):
  return str(getattr(fn, 'name', None) or
             getattr(fn, '__name__', None) or
             getattr(type(fn), '__name__', name_fallback))
