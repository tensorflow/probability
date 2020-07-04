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
"""Functional MCMC API."""

from discussion.fun_mcmc import api
from discussion.fun_mcmc import using_jax
from discussion.fun_mcmc import using_tensorflow
from discussion.fun_mcmc.api import *
from discussion.fun_mcmc.backend import get_backend
from discussion.fun_mcmc.backend import JAX
from discussion.fun_mcmc.backend import MANUAL_TRANSFORMS
from discussion.fun_mcmc.backend import set_backend
from discussion.fun_mcmc.backend import TENSORFLOW

__all__ = api.__all__ + [
    'get_backend',
    'JAX',
    'MANUAL_TRANSFORMS',
    'set_backend',
    'TENSORFLOW',
    'using_jax',
    'using_tensorflow',
]
