# Copyright 2018 The TensorFlow Probability Authors.
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
"""Numpy implementations of `tf.linalg` functions."""

# pylint: disable=g-import-not-at-top
# pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.linalg_impl import *

JAX_MODE = False  # pylint: disable=g-statement-before-imports

# Don't crash if we can't import bazel-generated numpy versions of TF
# LinearOperators. This allows external developers to run TF tests without
# installing bazel.
try:
  # pylint: disable=unused-import
  from tensorflow_probability.python.internal.backend.numpy.gen import adjoint_registrations as _adjoint_registrations
  from tensorflow_probability.python.internal.backend.numpy.gen import cholesky_registrations as _cholesky_registrations
  from tensorflow_probability.python.internal.backend.numpy.gen import inverse_registrations as _inverse_registrations
  from tensorflow_probability.python.internal.backend.numpy.gen import linear_operator_algebra as _linear_operator_algebra
  from tensorflow_probability.python.internal.backend.numpy.gen import matmul_registrations as _matmul_registrations
  from tensorflow_probability.python.internal.backend.numpy.gen import solve_registrations as _solve_registrations

  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_block_diag import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_block_lower_triangular import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_circulant import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_composition import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_diag import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_full_matrix import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_householder import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_identity import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_kronecker import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_low_rank_update import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_lower_triangular import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_toeplitz import *
  from tensorflow_probability.python.internal.backend.numpy.gen.linear_operator_zeros import *

  if JAX_MODE:

    def register_pytrees(env):
      """Registers all LinearOperators in a scope as Pytrees."""
      non_shape_params = {
          'LinearOperator': (),
          'LinearOperatorAdjoint': ('operator',),
          'LinearOperatorBlockDiag': ('operators',),
          'LinearOperatorBlockLowerTriangular': ('operators',),
          'LinearOperatorCirculant': ('spectrum',),
          'LinearOperatorCirculant2D': ('spectrum',),
          'LinearOperatorCirculant3D': ('spectrum',),
          'LinearOperatorComposition': ('operators',),
          'LinearOperatorDiag': ('diag',),
          'LinearOperatorFullMatrix': ('matrix',),
          'LinearOperatorHouseholder': ('reflection_axis',),
          'LinearOperatorIdentity': (),
          'BaseLinearOperatorIdentity': (),
          'LinearOperatorScaledIdentity': ('multiplier',),
          'LinearOperatorInversion': ('operator',),
          'LinearOperatorKronecker': ('operators',),
          'LinearOperatorLowRankUpdate': ('base_operator', 'diag_update'),
          'LinearOperatorLowerTriangular': ('tril',),
          'LinearOperatorToeplitz': ('col', 'row'),
          'LinearOperatorZeros': (),
      }
      from jax import tree_util
      import inspect

      for value in env.values():
        if not inspect.isclass(value):
          continue
        if not issubclass(value, LinearOperator):  # pylint: disable=undefined-variable
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

except ImportError:
  import sys
  # Backend should only fail importing if we're running pytest.
  if 'pytest' not in sys.modules:
    raise
