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
"""Module for probability bijectors and related functions."""
import inspect

from oryx.bijectors import bijector_extensions
from tensorflow_probability.substrates import jax as tfp

__all__ = [
    'bijector_extensions'
]

tfb = tfp.bijectors

_bijectors = {}

for name in dir(tfb):
  bij = getattr(tfb, name)
  if inspect.isclass(bij) and issubclass(bij, tfb.Bijector):
    if bij is not tfb.Bijector:
      bij = bijector_extensions.make_type(bij)
  _bijectors[name] = bij


for key, val in _bijectors.items():
  locals()[key] = val


del _bijectors
