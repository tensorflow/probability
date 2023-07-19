# Copyright 2019 The TensorFlow Probability Authors.
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
"""Experimental tools for linear algebra."""

from tensorflow_probability.python.experimental.linalg.linear_operator_interpolated_psd_kernel import LinearOperatorInterpolatedPSDKernel
from tensorflow_probability.python.experimental.linalg.linear_operator_psd_kernel import LinearOperatorPSDKernel
from tensorflow_probability.python.experimental.linalg.linear_operator_row_block import LinearOperatorRowBlock
from tensorflow_probability.python.experimental.linalg.linear_operator_unitary import LinearOperatorUnitary
from tensorflow_probability.python.experimental.linalg.no_pivot_ldl import no_pivot_ldl
from tensorflow_probability.python.experimental.linalg.no_pivot_ldl import simple_robustified_cholesky
from tensorflow_probability.python.internal import all_util


_allowed_symbols = [
    'LinearOperatorInterpolatedPSDKernel',
    'LinearOperatorPSDKernel',
    'LinearOperatorRowBlock',
    'LinearOperatorUnitary',
    'no_pivot_ldl',
    'simple_robustified_cholesky',
]


JAX_MODE = False

if JAX_MODE:

  def register_pytrees(env):
    """Registers all LinearOperators in a scope as Pytrees."""
    non_shape_params = {
        'LinearOperatorRowBlock': ('operators',),
        'LinearOperatorUnitary': ('matrix',),
    }
    from jax import tree_util  # pylint:disable=g-import-not-at-top
    import inspect  # pylint:disable=g-import-not-at-top
    for value in env.values():
      if not inspect.isclass(value):
        continue
      if value.__name__ not in non_shape_params:
        continue

      def register(cls):
        """Registers a class as a JAX pytree node."""
        def flatten(linop):
          param_names = set(non_shape_params[cls.__name__])
          components = {param_name: value for param_name, value
                        in linop.parameters.items()
                        if param_name in param_names}
          metadata = {param_name: value for param_name, value
                      in linop.parameters.items()
                      if param_name not in param_names}
          if components:
            keys, values = zip(*sorted(components.items()))
          else:
            keys, values = (), ()
          return values, (keys, metadata)
        def unflatten(info, xs):
          keys, metadata = info
          parameters = dict(list(zip(keys, xs)), **metadata)
          return cls(**parameters)
        tree_util.register_pytree_node(cls, flatten, unflatten)
      register(value)
  register_pytrees(dict(locals()))
  del register_pytrees


all_util.remove_undocumented(__name__, _allowed_symbols)
