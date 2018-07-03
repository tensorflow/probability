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
"""TensorFlow Probability Optimizer python package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.optimizer import linesearch
from tensorflow_probability.python.optimizer.bfgs import minimize as bfgs_minimize
from tensorflow_probability.python.optimizer.nelder_mead import minimize as nelder_mead_minimize
from tensorflow_probability.python.optimizer.nelder_mead import nelder_mead_one_step
from tensorflow_probability.python.optimizer.sgld import StochasticGradientLangevinDynamics
from tensorflow_probability.python.optimizer.variational_sgd import VariationalSGD


from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'bfgs_minimize',
    'nelder_mead_minimize',
    'nelder_mead_one_step',
    'linesearch',
    'StochasticGradientLangevinDynamics',
    'VariationalSGD',
]

remove_undocumented(__name__, _allowed_symbols)
