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
"""TF and JAX compatible custom gradients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow.python.ops import array_ops  # pylint: disable=g-direct-tensorflow-import


JAX_MODE = False

if JAX_MODE:
  from jax import custom_jvp  # pylint: disable=g-import-not-at-top


def custom_gradient(vjp_fwd=None, vjp_bwd=None, jvp_fn=None):
  """Decorates a function and adds custom derivatives.

  TF only supports VJPs, so we decorate with tf.custom_gradient.

  JAX supports either JVP or VJP. If a custom JVP is provided, then JAX can
  transpose to derive a VJP rule. Therefore we prefer jvp_fn if given, but fall
  back to the vjp functions otherwise.

  Args:
    vjp_fwd: A function (*args) => (output, auxiliaries).
    vjp_bwd: A function (auxiliaries, output_gradient) => args_gradients.
    jvp_fn: A function (primals, tangents) => (primal_out, tangent_out).

  Returns:
    A decorator to be applied to a function f(*args) => output.
  """

  def finalize(f):
    """Decorate f with a custom gradient."""

    if JAX_MODE:

      # https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html

      # For JAX, we prefer to specify a custom JVP, as JAX can use a function
      # transform to transpose a JVP (must be linear in the tangents) to a VJP.
      if jvp_fn is not None:
        f_jvp = custom_jvp(f)
        f_jvp.defjvp(jvp_fn)
        return f_jvp

      else:
        from jax import custom_vjp  # pylint: disable=g-import-not-at-top
        f_vjp = custom_vjp(f)
        f_vjp.defvjp(vjp_fwd, vjp_bwd)
        return f_vjp

    else:
      # TF custom gradients support only custom VJPs.
      @tf.custom_gradient
      def f_wrapped(*args, **kwargs):
        val, aux = vjp_fwd(*args, **kwargs)
        return val, lambda *g: vjp_bwd(aux, tf.nest.pack_sequence_as(val, g))

      return f_wrapped

  return finalize


def prevent_gradient(x, message='', name=None):
  return array_ops.prevent_gradient(x, message=message, name=name)


if JAX_MODE:

  def _prevent_gradient_helper_jvp(primals, tangents):
    # The custom error message is passed in as the key of the single item in
    # the dict `primals`.
    message, _ = primals[0].popitem()
    raise LookupError(
        'Gradient explicitly disabled. Reason: \'{}\''.format(message))

  @custom_jvp
  def _prevent_gradient_helper(d):
    return d

  _prevent_gradient_helper.defjvp(_prevent_gradient_helper_jvp)

  def prevent_gradient(x, message='', name=None):  # pylint: disable=unused-argument,function-redefined
    return _prevent_gradient_helper({message: x})[message]
