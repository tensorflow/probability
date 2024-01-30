# Copyright 2024 The TensorFlow Probability Authors.
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
"""Package for training GP-like Bayesian Neural Nets w/ composite structure."""

from tensorflow_probability.python.experimental.autobnn import bnn
from tensorflow_probability.python.experimental.autobnn import bnn_tree
# estimators causes vectorized_stochastic_volatility_test to fail
# because it imports training_util
from tensorflow_probability.python.experimental.autobnn import estimators
from tensorflow_probability.python.experimental.autobnn import kernels
from tensorflow_probability.python.experimental.autobnn import likelihoods
from tensorflow_probability.python.experimental.autobnn import models
from tensorflow_probability.python.experimental.autobnn import operators
# training_util causes vectorized_stochastic_volatility_test to fail
# Suspects: JaxTyping, bayeux, matplotlib, pandas.
# And the culprit is ... bayeux.
from tensorflow_probability.python.experimental.autobnn import training_util
from tensorflow_probability.python.experimental.autobnn import util
from tensorflow_probability.python.internal import all_util

_allowed_symbols = [
    'bnn',
    'bnn_tree',
    'estimators',
    'kernels',
    'likelihoods',
    'models',
    'operators',
    'training_util',
    'util',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
