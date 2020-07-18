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

tf = tf_on_jax.tf

__all__ = [
    'tf',
    'tfp',
    'util',
]
