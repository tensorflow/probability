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
"""JAX backend."""

from discussion.fun_mcmc import tf_on_jax
from discussion.fun_mcmc import util_jax as util
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.internal import prefer_static

tf = tf_on_jax.tf

__all__ = [
    'BACKEND_NAME',
    'multi_backend_test',
    'prefer_static',
    'tf',
    'tfp',
    'util',
]

BACKEND_NAME = 'jax'


def multi_backend_test(globals_dict,
                       relative_module_name,
                       backends=('jax', 'tf'),
                       test_case=None):
  """See backend.multi_backend_test."""
  if test_case is None:
    return lambda test_case: multi_backend_test(  # pylint: disable=g-long-lambda
        globals_dict=globals_dict,
        relative_module_name=relative_module_name,
        backends=backends,
        test_case=test_case)
  return test_case
