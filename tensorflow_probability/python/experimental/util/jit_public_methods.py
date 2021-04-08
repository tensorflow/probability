# Copyright 2021 The TensorFlow Probability Authors.
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
"""A wrapper to XLA-compile an object's public methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler

__all__ = [
    'DEFAULT_METHODS_EXCLUDED_FROM_JIT',
    'JitPublicMethods'
]

JAX_MODE = False

DEFAULT_METHODS_EXCLUDED_FROM_JIT = (
    # tfd.Distribution
    'event_shape',
    'event_shape_tensor',
    'batch_shape',
    'batch_shape_tensor',
    'dtype',
    'kl_divergence',  # Wrapping applied explicitly in `_traced_kl_divergence`.
    'experimental_default_event_space_bijector',
    # tfb.Bijector
    # TODO(davmre): Test wrapping bijectors.
    'forward_event_shape',
    'forward_event_shape_tensor',
    'inverse_event_shape',
    'inverse_event_shape_tensor',
    'forward_dtype',
    'inverse_dtype',
    'forward_event_ndims',
    'inverse_event_ndims'
)


class JitPublicMethods(object):
  """Wrapper to compile an object's public methods using XLA."""

  def __init__(self,
               object_to_wrap,
               trace_only=False,
               methods_to_exclude=DEFAULT_METHODS_EXCLUDED_FROM_JIT):
    """Wraps an object's public methods using `tf.function`/`jax.jit`.

    Args:
      object_to_wrap: Any Python object; for example, a
        `tfd.Distribution` instance.
      trace_only: Python `bool`; if `True`, the object's methods are
        not compiled, but only traced with `tf.function(jit_compile=False)`.
        This is only valid in the TensorFlow backend; in JAX, passing
        `trace_only=True` will raise an exception.
        Default value: `False`.
      methods_to_exclude: List of Python `str` method names not to wrap.
        For example, these may include methods that do not take or return
        Tensor values. By default, a number of `tfd.Distribution` and
        `tfb.Bijector` methods and properties are excluded (e.g., `event_shape`,
        `batch_shape`, `dtype`, etc.).
        Default value: `tfp.experimental.util.DEFAULT_METHODS_EXCLUDED_FROM_JIT`

    """
    if JAX_MODE and trace_only:
      raise ValueError('JitPublicMethods with `trace_only=True` is not valid '
                       'in the JAX backend.')
    self._object_to_wrap = object_to_wrap
    self._methods_to_exclude = methods_to_exclude
    self._trace_only = trace_only

  @property
  def methods_to_exclude(self):
    return self._methods_to_exclude

  @property
  def trace_only(self):
    return self._trace_only

  @property
  def object_to_wrap(self):
    return self._object_to_wrap

  def copy(self, **kwargs):
    return type(self)(self.object_to_wrap.copy(**kwargs),
                      trace_only=self.trace_only,
                      methods_to_exclude=self.methods_to_exclude)

  def __getitem__(self, slices):
    return type(self)(self.object_to_wrap[slices],
                      trace_only=self.trace_only,
                      methods_to_exclude=self.methods_to_exclude)

  def __getattr__(self, name):
    # Note: this method is called only as a fallback if an attribute isn't
    # otherwise set.

    if name == 'object_to_wrap':
      # Avoid triggering an infinite loop if __init__ hasn't run yet.
      raise AttributeError()
    attr = getattr(self.object_to_wrap, name)

    if callable(attr):
      if not (name.startswith('_') or name in self.methods_to_exclude):
        # On the first call to a method, wrap it, and store the wrapped
        # function to be reused by future calls.
        attr = tf.function(autograph=False,
                           jit_compile=not self.trace_only)(attr)
        setattr(self, name, attr)

    return attr


@kullback_leibler.RegisterKL(JitPublicMethods, distribution_lib.Distribution)
@kullback_leibler.RegisterKL(distribution_lib.Distribution, JitPublicMethods)
@kullback_leibler.RegisterKL(JitPublicMethods, JitPublicMethods)
def _compiled_kl_divergence(d1, d2, name=None):
  """Compiled KL divergence between two distributions."""
  trace_only = True
  if isinstance(d1, JitPublicMethods):
    trace_only &= d1.trace_only
    d1 = d1.object_to_wrap
  if isinstance(d2, JitPublicMethods):
    trace_only &= d2.trace_only
    d2 = d2.object_to_wrap
  return tf.function(autograph=False, jit_compile=not trace_only)(
      d1.kl_divergence)(d2, name=name)
