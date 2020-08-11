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
# Lint as: python3
"""Utility functions for tfp.experimental.lazybones."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

from tensorflow_probability.python.experimental.lazybones.utils.probability import distribution_measure
from tensorflow_probability.python.experimental.lazybones.utils.probability import log_prob
from tensorflow_probability.python.experimental.lazybones.utils.probability import prob
from tensorflow_probability.python.experimental.lazybones.utils.special_methods import ObjectProxy
from tensorflow_probability.python.experimental.lazybones.utils.special_methods import SpecialMethods
from tensorflow_probability.python.experimental.lazybones.utils.utils import get_leaves
from tensorflow_probability.python.experimental.lazybones.utils.utils import get_roots
from tensorflow_probability.python.experimental.lazybones.utils.utils import is_any_ancestor
from tensorflow_probability.python.experimental.lazybones.utils.utils import iter_edges
from tensorflow_probability.python.experimental.lazybones.utils.utils import plot_graph
from tensorflow_probability.python.experimental.lazybones.utils.weak_container import HashableWeakRef
from tensorflow_probability.python.experimental.lazybones.utils.weak_container import WeakKeyDictionary
from tensorflow_probability.python.experimental.lazybones.utils.weak_container import WeakSet


__all__ = [
    'HashableWeakRef',
    'ObjectProxy',
    'SpecialMethods',
    'WeakKeyDictionary',
    'WeakSet',
    'distribution_measure',
    'get_leaves',
    'get_roots',
    'is_any_ancestor',
    'iter_edges',
    'log_prob',
    'plot_graph',
    'prob',
]
