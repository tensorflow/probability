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
"""TensorFlow Probability GLM python package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.glm.family import Bernoulli
from tensorflow_probability.python.glm.family import BernoulliNormalCDF
from tensorflow_probability.python.glm.family import CustomExponentialFamily
from tensorflow_probability.python.glm.family import ExponentialFamily
from tensorflow_probability.python.glm.family import GammaExp
from tensorflow_probability.python.glm.family import GammaSoftplus
from tensorflow_probability.python.glm.family import LogNormal
from tensorflow_probability.python.glm.family import LogNormalSoftplus
from tensorflow_probability.python.glm.family import Normal
from tensorflow_probability.python.glm.family import NormalReciprocal
from tensorflow_probability.python.glm.family import Poisson
from tensorflow_probability.python.glm.family import PoissonSoftplus
from tensorflow_probability.python.glm.fisher_scoring import convergence_criteria_small_relative_norm_weights_change
from tensorflow_probability.python.glm.fisher_scoring import fit
from tensorflow_probability.python.glm.fisher_scoring import fit_one_step
from tensorflow_probability.python.glm.proximal_hessian import fit_sparse
from tensorflow_probability.python.glm.proximal_hessian import fit_sparse_one_step

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'fit',
    'fit_one_step',
    'fit_sparse',
    'fit_sparse_one_step',
    'convergence_criteria_small_relative_norm_weights_change',
    'Bernoulli',
    'BernoulliNormalCDF',
    'CustomExponentialFamily',
    'ExponentialFamily',
    'GammaExp',
    'GammaSoftplus',
    'LogNormal',
    'LogNormalSoftplus',
    'Normal',
    'NormalReciprocal',
    'Poisson',
    'PoissonSoftplus',
]

remove_undocumented(__name__, _allowed_symbols)
