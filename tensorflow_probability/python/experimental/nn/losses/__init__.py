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
"""Loss functions for neural networks."""
from tensorflow_probability.python.experimental.nn.losses.losses import compute_extra_loss
from tensorflow_probability.python.experimental.nn.losses.losses import kl_divergence_exact
from tensorflow_probability.python.experimental.nn.losses.losses import kl_divergence_monte_carlo
from tensorflow_probability.python.experimental.nn.losses.losses import negloglik
from tensorflow_probability.python.internal import all_util


_allowed_symbols = [
    'compute_extra_loss',
    'kl_divergence_exact',
    'kl_divergence_monte_carlo',
    'negloglik',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
