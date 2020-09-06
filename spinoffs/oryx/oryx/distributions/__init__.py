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
"""Module for probability distributions and related functions."""
import inspect

from oryx.distributions import distribution_extensions
from tensorflow_probability.substrates import jax as tfp

__all__ = [
    'distribution_extensions'
]

tfd = tfp.distributions

_distributions = {}

for name in dir(tfd):
  dist = getattr(tfd, name)
  if inspect.isclass(dist) and issubclass(dist, tfd.Distribution):
    if dist is not tfd.Distribution:
      dist = distribution_extensions.make_type(dist)
  _distributions[name] = dist


for key, val in _distributions.items():
  locals()[key] = val


del _distributions
