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
"""Edward2 probabilistic programming language."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from tensorflow_probability.python.edward2.generated_random_variables import *
from tensorflow_probability.python.edward2.generated_random_variables import as_random_variable
from tensorflow_probability.python.edward2.generated_random_variables import rv_all
from tensorflow_probability.python.edward2.interceptor import get_interceptor
from tensorflow_probability.python.edward2.interceptor import interception
from tensorflow_probability.python.edward2.program_transformations import make_log_joint_fn
from tensorflow_probability.python.edward2.random_variable import RandomVariable
# pylint: enable=wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = rv_all + [
    "RandomVariable",
    "as_random_variable",
    "get_interceptor",
    "interception",
    "make_log_joint_fn",
]

remove_undocumented(__name__, _allowed_symbols)
