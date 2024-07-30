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
"""JAX backend."""

import jax
import jax.numpy as jnp

from fun_mc.dynamic.backend_jax import util
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.internal import distribute_lib
from tensorflow_probability.substrates.jax.internal import prefer_static

__all__ = [
    'distribute_lib',
    'prefer_static',
    'jax',
    'jnp',
    'tfp',
    'util',
]
