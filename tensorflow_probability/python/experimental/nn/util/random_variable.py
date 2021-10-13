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
"""Functions for creating objects with RandomVariable semantics."""
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as tfd
from tensorflow_probability.python.util.deferred_tensor import DeferredTensor


__all__ = [
    'CallOnce',
    'RandomVariable',
]


class RandomVariable(DeferredTensor):
  """`RandomVariable` supports random variable semantics for TFP distributions.

  The `RandomVariable` class memoizes concretizations of TFP distribution-like
  objects so that random draws can be re-triggered on-demand, i.e., by calling
  `reset`. For more details type `help(tfp.util.DeferredTensor)`.

  #### Examples

  ```python
  # In this example we see the memoization semantics in action.
  tfd = tfp.distributions
  tfn = tfp.experimental.nn
  x = tfn.util.RandomVariable(tfd.Normal(0, 1))
  x_ = tf.convert_to_tensor(x)
  x _ + 1. == x + 1.
  # ==> True; `x` always has the same value until reset.
  x.reset()
  tf.convert_to_tensor(x) == x_
  # ==> False; `x` was reset which triggers a new sample.
  ```

  ```python
  # In this example we see how to concretize with different semantics.
  tfd = tfp.distributions
  tfn = tfp.experimental.nn
  x = tfn.util.RandomVariable(
      tfd.Bernoulli(probs=[[0.25], [0.5]]),
      convert_to_tensor_fn=tfd.Distribution.mean,
      dtype=tf.float32,
      shape=[2, 1],
      name='x')
  tf.convert_to_tensor(x)
  # ==> [[0.25], [0.5]]
  x.shape
  # ==> [2, 1]
  x.dtype
  # ==> tf.float32
  x.name
  # ==> 'x'
  ```

  ```python
  # In this example we see a common pitfall: accessing the memoized value from a
  # different graph context.
  tfd = tfp.distributions
  tfn = tfp.experimental.nn
  x = tfn.util.RandomVariable(tfd.Normal(0, 1))
  @tf.function(autograph=False, jit_compile=True)
  def run():
    return tf.convert_to_tensor(x)
  first = run()
  second = tf.convert_to_tensor(x)
  # raises ValueError:
  #   "You are attempting to access a memoized value from a different
  #   graph context. Please call `this.reset()` before accessing a
  #   memoized value from a different graph context."
  x.reset()
  third = tf.convert_to_tensor(x)
  # ==> No exception.
  first == third
  # ==> False
  ```

  """

  def __init__(self, distribution, convert_to_tensor_fn=tfd.Distribution.sample,
               dtype=None, shape=None, name=None):
    """Creates the `RandomVariable` object.

    Args:
      distribution: TFP distribution-like object which is passed into the
        `convert_to_tensor_fn` whenever this object is evaluated in
        `Tensor`-like contexts.
      convert_to_tensor_fn: Python `callable` which takes one argument, the
        `distribution` and returns a `Tensor` of type `dtype` and shape `shape`.
        Default value: `tfp.distributions.Distribution.sample`.
      dtype: TF `dtype` equivalent to what would otherwise be
        `convert_to_tensor_fn(distribution).dtype`.
        Default value: `None` (i.e., `distribution.dtype`).
      shape: `tf.TensorShape`-like object compatible with what would otherwise
        be `convert_to_tensor_fn(distribution).shape`.
        Default value: `'None'` (i.e., unspecified static shape).
      name: Python `str` representing this object's `name`; used only in graph
        mode.
        Default value: `None` (i.e., `distribution.name`)
    """

    self._distribution = distribution
    self._convert_to_tensor_fn = convert_to_tensor_fn
    super(RandomVariable, self).__init__(
        tf.constant([], tf.bool),  # Dummy.
        CallOnce(lambda _: convert_to_tensor_fn(distribution)),
        shape=shape,
        dtype=dtype or distribution.dtype,
        name=name or distribution.name)

  @property
  def distribution(self):
    return self._distribution

  @property
  def convert_to_tensor_fn(self):
    return self._convert_to_tensor_fn

  def reset(self):
    """Removes memoized value which triggers re-eval on subsequent reads."""
    self.transform_fn.reset()

  def is_unset(self):
    """Returns `True` if there is no memoized value and `False` otherwise."""
    return self.transform_fn.is_unset()


class CallOnce(tf.Module):
  """Function object which memoizes the result of `create_value_fn()`.

  This object is used to memoize the computation of some function. Upon first
  call, the user provided `create_value_fn` is called and with the args/kwargs
  provided to this object's `__call__`. On subsequent calls the previous result
  is returned and **regardless of the args/kwargs provided to this object's
  `__call__`**. To trigger a new evaluation, invoke `this.reset()` and to
  identify if a new evaluation will execute (on-demand) invoke
  `this.is_unset()`. For an example application of this object, see
  `help(tfp.experimental.nn.util.RandomVariable)` and/or
  `help(tfp.util.DeferredTensor)`.
  """

  def __init__(self, create_value_fn):
    """Creates the `CallOnce` object.

    Args:
      create_value_fn: Python `callable` which takes any input args/kwargs and
        returns a value to memoize. (The value is not presumed to be of any
        particular type.)
    """
    self._create_value_fn = create_value_fn
    self._value = _UNSET
    # TODO(b/156185251): We have to set `__name__` because of a
    # not-really-necessary requirement of DeferredTensor.
    self.__name__ = str(getattr(create_value_fn, 'name', None) or
                        getattr(type(create_value_fn), '__name__', 'unknown'))
    super(CallOnce, self).__init__(name=self.__name__)

  @property
  def create_value_fn(self):
    return self._create_value_fn

  @property
  def value(self):
    return self._value

  def is_unset(self):
    """Returns `True` if there is no memoized value and `False` otherwise."""
    return self._value is _UNSET

  def __call__(self, *args, **kwargs):
    """Return the memoized value."""
    if self.is_unset():
      self._value = self._create_value_fn(*args, **kwargs)
      return self._value  # No need to to go through checks on first call.
    my_graph = getattr(self._value, 'graph', None)
    my_graph_was_deleted = (
        my_graph is None and
        # Don't check subclass; check parent only.
        type(self._value) is tf.Tensor)  # pylint: disable=unidiomatic-typecheck
    # We write `your_graph` as a lambda to ensure it's only evaluated when
    # necessary.
    your_graph = lambda: getattr(tf.constant([], dtype=tf.bool), 'graph', None)
    if my_graph_was_deleted or (
        my_graph is not None and my_graph is not your_graph()):
      raise ValueError(
          'You are attempting to access a memoized value from a different '
          'graph context. Please call `this.reset()` before accessing a '
          'memoized value from a different graph context.')
    return self._value

  def reset(self):
    """Removes memoized value which triggers re-eval on subsequent reads."""
    self._value = _UNSET


class _Unset(object):
  """Dummy object which exists to be unique from any possible user value."""

  def __repr__(self):
    return 'unset'


_UNSET = _Unset()
