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
"""Deferred initialization of tf.Modules (distributions, bijectors, etc.)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.lazybones.utils import special_methods
from tensorflow_probability.python.internal import nest_util


class DeferredModule(tf.Module, special_methods.SpecialMethods):
  """Wrapper to defer initialization of a `tf.Module` instance.

  `DeferredModule` is a general-purpose mechanism for creating objects that are
  'tape safe', meaning that computation occurs only when an instance
  method is called, not at construction. This ensures that method calls inside
  of a `tf.GradientTape` context will produce gradients to any underlying
  `tf.Variable`s.

  ### Examples

  TFP's built-in Distributions and Bijectors are tape-safe by contract, but
  this does not extend to cases where computation is required
  to construct an object's parameters prior to initialization.
  For example, suppose we want to construct a Gamma
  distribution with a given mean and variance. In a naive implementation,
  we would convert these to the Gamma's native `concentration` and
  `rate` parameters when the distribution is constructed. Any future method
  calls would produce gradients to `concentration` and `rate`, but not to the
  underlying mean and variance:

  ```python
  mean, variance = tf.Variable(3.2), tf.Variable(9.1)
  dist = tfd.Gamma(concentration=mean**2 / variance,
                   rate=mean / variance)

  with tf.GradientTape() as tape:
    lp = dist.log_prob(5.0)
  grads = tape.gradient(lp, [mean, variance])
  # ==> `grads` are `[None, None]` !! :-(
  ```

  To preserve the gradients, we can defer the parameter transformation using
  `DeferredModule`. The resulting object behaves just like a
  `tfd.Gamma` instance, however, instead of running the `Gamma` constructor just
  once, it internally applies the parameter transformation and constructs a
  new, temporary instance of `tfd.Gamma` on *every method invocation*.
  This ensures that all operations needed to compute a method's return value
  from any underlying variables are performed every time the method is invoked.
  A surrounding `GradientTape` context will therefore be able to trace the full
  computation.

  ```python
  def gamma_params_from_mean_and_variance(mean, variance, **kwargs):
    rate = mean / variance
    return dict(concentration=mean * rate, rate=rate, **kwargs)

  mean, variance = tf.Variable(3.2), tf.Variable(9.1)
  deferred_dist = tfp.experimental.util.DeferredModule(
    tfd.Gamma,
    args_fn=gamma_params_from_mean_and_variance,
    mean=mean,  # May be passed by position or by name.
    variance=variance)

  with tf.GradientTape() as tape:
    lp = deferred_dist.log_prob(5.0)
  grads = tape.gradient(lp, [mean, variance])
  # ==> `grads` are defined!
  ```

  Note that we could have achieved a similar effect by using
  `tfp.util.DeferredTensor` to individually defer the `concentration` and `rate`
  parameters. However, this would have been significantly more verbose, and
  would not share any computation between the two parameter transformations.
  In general, `DeferredTensor` is often idiomatic for simple transformations of
  a single value, while `DeferredModule` may be preferred for transformations
  that operate on multiple values and/or contain multiple steps.

  ### Caveats

  Objects derived from a `DeferredModule` are no longer deferred, so
  they will not preserve gradients. For example, slicing into a deferred
  Distribution yields a new, concrete Distribution instance:

  ```python
  dist = tfp.experimental.util.DeferredModule(
    tfd.Normal,
    args_fn=lambda scaled_loc, log_scale: (5 * scaled_loc, tf.exp(log_scale)),
    scaled_loc=tf.Variable([1., 2., 3.]),
    log_scale=tf.Variable([1., 1., 1.]))
  dist.batch_shape  # ==> [3]
  len(dist.trainable_variables)  # ==> 2

  slice = dist[:2]  # Instantiates a new, non-deferred Distribution.
  slice.batch_shape  # ==> [2]
  len(slice.trainable_variables)  # ==> 0 (!)

  # If needed, we could defer the slice with another layer of wrapping.
  deferred_slice = tfp.experimental.util.DeferredModule(
    base_class=lambda d: d[:2],
    args_fn=lambda d: d,
    d=dist)
  len(deferred_slice.trainable_variables)  # ==> 2
  ```

  """

  def __init__(self, base_class, args_fn, *args, **kwargs):
    """Defers initialization of an object with transformed arguments.

    Args:
      base_class: Python type or callable such that `base_class(**args_fn(...))`
        is an instance of `tf.Module`---for example, a TFP Distribution or
        Bijector.
      args_fn: Python callable specifying a deferred transformation of the
        provided arguments. This must have signature
        `base_class_init_args = args_fn(*args, **kwargs)`. The return value
        `base_class_init_args` may be either a dictionary or an iterable
        (list/tuple), in which case the class will be initialized as
        `base_class(**base_class_init_args)` or
        `base_class(*base_class_init_args)`, respectively.
      *args: Optional positional arguments to `args_fn`.
      **kwargs: Optional keyword arguments to `args_fn`.
    """
    self._base_class = base_class
    self._args_fn = args_fn
    self._param_args = args
    self._param_kwargs = kwargs

    # In order for DeferredModule to work as a tf.Module, we need to ensure that
    # attrs used by tf.Module are handled directly, rather than being forwarded
    # to the inner class.
    self._module_attrs = set(dir(tf.Module()))

    super(DeferredModule, self).__init__()

  def __action__(self, fn, *args, **kwargs):
    kwargs.pop('_action_name', None)
    return fn(self._build_module(), *args, **kwargs)

  def _build_module(self):
    return nest_util.call_fn(self._base_class,
                             self._args_fn(*self._param_args,
                                           **self._param_kwargs))

  def __getattr__(self, attr, **kwargs):
    if attr in ('_base_class',
                '_args_fn',
                '_param_args',
                '_param_kwargs',
                '_module_attrs'):
      raise AttributeError()
    if attr in self._module_attrs:
      raise AttributeError()
    return super(DeferredModule, self).__getattr__(attr, **kwargs)
