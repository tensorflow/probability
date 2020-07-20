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

"""Edward2 probabilistic programming language.

For user guides, see:

+ [Overview](
   https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/README.md)
+ [Upgrading from Edward to Edward2](
   https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/Upgrading_From_Edward_To_Edward2.md)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from tensorflow_probability.python.experimental.edward2.generated_random_variables import *
from tensorflow_probability.python.experimental.edward2.generated_random_variables import as_random_variable
from tensorflow_probability.python.experimental.edward2.generated_random_variables import rv_dict
from tensorflow_probability.python.experimental.edward2.interceptor import get_next_interceptor
from tensorflow_probability.python.experimental.edward2.interceptor import interceptable
from tensorflow_probability.python.experimental.edward2.interceptor import interception
from tensorflow_probability.python.experimental.edward2.interceptor import tape
from tensorflow_probability.python.experimental.edward2.program_transformations import make_log_joint_fn
from tensorflow_probability.python.experimental.edward2.program_transformations import make_value_setter
from tensorflow_probability.python.experimental.edward2.random_variable import RandomVariable
from tensorflow_probability.python.internal import all_util
# pylint: enable=wildcard-import


_allowed_symbols = list(rv_dict.keys()) + [
    "RandomVariable",
    "as_random_variable",
    "interception",
    "get_next_interceptor",
    "interceptable",
    "make_log_joint_fn",
    "make_value_setter",
    "tape",
]

all_util.remove_undocumented(__name__, _allowed_symbols)
