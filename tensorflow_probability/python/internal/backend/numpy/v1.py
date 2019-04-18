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
"""Experimental Numpy backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal.backend.numpy.internal import utils
from tensorflow_probability.python.internal.backend.numpy.ops import name_scope


__all__ = [
    'assert_equal',
    'assert_greater',
    'assert_greater_equal',
    'assert_integer',
    'assert_less',
    'assert_less_equal',
    'assert_near',
    'assert_non_negative',
    'assert_non_positive',
    'assert_none_equal',
    'assert_positive',
    'assert_positive',
    'assert_rank',
    'assert_rank_at_least',
    'name_scope',
    'placeholder_with_default',
    'set_random_seed',
]


# --- Begin Public Functions --------------------------------------------------


assert_equal = tf.compat.v1.assert_equal
assert_greater = tf.compat.v1.assert_greater
assert_less = tf.compat.v1.assert_less
assert_rank = tf.compat.v1.assert_rank

assert_greater_equal = tf.compat.v1.assert_greater_equal
assert_integer = tf.compat.v1.assert_integer
assert_less_equal = tf.compat.v1.assert_less_equal
assert_near = tf.compat.v1.assert_near
assert_non_negative = tf.compat.v1.assert_non_negative
assert_non_positive = tf.compat.v1.assert_non_positive
assert_none_equal = tf.compat.v1.assert_none_equal
assert_positive = tf.compat.v1.assert_positive
assert_positive = tf.compat.v1.assert_positive
assert_rank_at_least = tf.compat.v1.assert_rank_at_least

placeholder_with_default = utils.copy_docstring(
    tf.compat.v1.placeholder_with_default,
    lambda input, shape, name=None: np.array(input).reshape(shape))  # pylint: disable=redefined-builtin,unused-argument

set_random_seed = utils.copy_docstring(
    tf.compat.v1.set_random_seed,
    np.random.seed)

del tf
