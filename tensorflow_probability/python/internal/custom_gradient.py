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

import tensorflow.compat.v2 as tf

from tensorflow.python.ops import array_ops  # pylint: disable=g-direct-tensorflow-import


JAX_MODE = False

if JAX_MODE:
  import jax  # pylint: disable=g-import-not-at-top
  from jax import custom_jvp  # pylint: disable=g-import-not-at-top


def custom_gradient(vjp_fwd=None, vjp_bwd=None, jvp_fn=None,
                    nondiff_argnums=()):
  """Decorates a function and adds custom derivatives.

  TF only supports VJPs, so we decorate with tf.custom_gradient.

  JAX supports either JVP or VJP. If a custom JVP is provided, then JAX can
  transpose to derive a VJP rule. Therefore we prefer jvp_fn if given, but fall
  back to the vjp functions otherwise.

  Args:
    vjp_fwd: A function (*args) => (output, auxiliaries).
    vjp_bwd: A function (auxiliaries, output_gradient) =>
      nondiff_args_gradients. `None` gradients will be inserted into the correct
      positions for `nondiff_argnums`.
    jvp_fn: A function (*nondiff_args, primals, tangents) =>
      (primal_out, tangent_out).
    nondiff_argnums: Tuple of argument indices which are not differentiable.
      These must integers or other non-Tensors. Tensors with no gradient should
      be indicated with a None in the result of vjp_bwd.

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
        f_jvp = custom_jvp(f, nondiff_argnums=nondiff_argnums)
        f_jvp.defjvp(jvp_fn)
        return f_jvp

      else:
        from jax import custom_vjp  # pylint: disable=g-import-not-at-top
        f_vjp = custom_vjp(f, nondiff_argnums=nondiff_argnums)
        f_vjp.defvjp(vjp_fwd, vjp_bwd)
        return f_vjp

    else:
      # TF custom gradients support only custom VJPs.
      def none_wrapper(*args, **kwargs):  # custom_gradient can't handle None.
        closure = {i: a for i, a in enumerate(args)
                   if i in nondiff_argnums or a is None}
        trimmed_args = [a for i, a in enumerate(args) if i not in closure]
        # Convert DeferredTensors to Tensors.
        trimmed_args = tf.nest.map_structure(tf.convert_to_tensor, trimmed_args)

        @tf.custom_gradient
        def f_wrapped(*args, **kwargs):
          reconstruct_args = []
          args_structure = tf.nest.map_structure(lambda _: 0, args)
          for i in range(len(args) + len(closure)):
            if i in closure:
              reconstruct_args.append(closure[i])
            else:
              reconstruct_args.append(args[0])
              args = args[1:]
          val, aux = vjp_fwd(*reconstruct_args, **kwargs)

          def vjp_bwd_wrapped(*g, **kwargs):
            # We don't want to use an explicit `variables` arg, because TF will
            # complain if the wrapped function doesn't actually have variables
            # in it. TF will only specify this arg if there are variables.
            variables = kwargs.get('variables', ())
            nondiff_args = [closure[i] for i in nondiff_argnums]
            result = vjp_bwd(*nondiff_args, aux,
                             tf.nest.pack_sequence_as(val, g), **kwargs)
            if variables:
              result, variables = result
            result = tf.nest.flatten(result)
            for i in nondiff_argnums:
              result = tuple(result[:i]) + (None,) + tuple(result[i:])
            result = [a for i, a in enumerate(result) if i not in closure]
            result = tf.nest.pack_sequence_as(args_structure, result)
            if variables:
              return result, variables
            else:
              return result

          return val, vjp_bwd_wrapped

        return f_wrapped(*trimmed_args, **kwargs)

      return none_wrapper

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


def is_valid_gradient(grad):
  if JAX_MODE:
    return grad.dtype != jax.float0
  return grad is not None
