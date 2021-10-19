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
"""Experimental math."""

from tensorflow_probability.python.experimental.math.manual_special_functions import exp_pade_4_4
from tensorflow_probability.python.experimental.math.manual_special_functions import expm1_pade_4_4
from tensorflow_probability.python.experimental.math.manual_special_functions import log1p_pade_4_4
from tensorflow_probability.python.experimental.math.manual_special_functions import log_pade_4_4
from tensorflow_probability.python.experimental.math.manual_special_functions import patch_manual_special_functions
from tensorflow_probability.python.experimental.math.manual_special_functions import reduce_logsumexp
from tensorflow_probability.python.experimental.math.manual_special_functions import softplus

__all__ = [
    'exp_pade_4_4',
    'expm1_pade_4_4',
    'log1p_pade_4_4',
    'log_pade_4_4',
    'patch_manual_special_functions',
    'reduce_logsumexp',
    'softplus',
]
