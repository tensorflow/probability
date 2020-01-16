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
"""Probabilistic Layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.layers.conv_variational import Convolution1DFlipout
from tensorflow_probability.python.layers.conv_variational import Convolution1DReparameterization
from tensorflow_probability.python.layers.conv_variational import Convolution2DFlipout
from tensorflow_probability.python.layers.conv_variational import Convolution2DReparameterization
from tensorflow_probability.python.layers.conv_variational import Convolution3DFlipout
from tensorflow_probability.python.layers.conv_variational import Convolution3DReparameterization
from tensorflow_probability.python.layers.dense_variational import DenseFlipout
from tensorflow_probability.python.layers.dense_variational import DenseLocalReparameterization
from tensorflow_probability.python.layers.dense_variational import DenseReparameterization
from tensorflow_probability.python.layers.dense_variational_v2 import DenseVariational
from tensorflow_probability.python.layers.distribution_layer import CategoricalMixtureOfOneHotCategorical
from tensorflow_probability.python.layers.distribution_layer import DistributionLambda
from tensorflow_probability.python.layers.distribution_layer import IndependentBernoulli
from tensorflow_probability.python.layers.distribution_layer import IndependentLogistic
from tensorflow_probability.python.layers.distribution_layer import IndependentNormal
from tensorflow_probability.python.layers.distribution_layer import IndependentPoisson
from tensorflow_probability.python.layers.distribution_layer import KLDivergenceAddLoss
from tensorflow_probability.python.layers.distribution_layer import KLDivergenceRegularizer
from tensorflow_probability.python.layers.distribution_layer import MixtureLogistic
from tensorflow_probability.python.layers.distribution_layer import MixtureNormal
from tensorflow_probability.python.layers.distribution_layer import MixtureSameFamily
from tensorflow_probability.python.layers.distribution_layer import MultivariateNormalTriL
from tensorflow_probability.python.layers.distribution_layer import OneHotCategorical
from tensorflow_probability.python.layers.distribution_layer import VariationalGaussianProcess
from tensorflow_probability.python.layers.initializers import BlockwiseInitializer
from tensorflow_probability.python.layers.masked_autoregressive import AutoregressiveTransform
from tensorflow_probability.python.layers.util import default_loc_scale_fn
from tensorflow_probability.python.layers.util import default_mean_field_normal_fn
from tensorflow_probability.python.layers.util import default_multivariate_normal_fn
from tensorflow_probability.python.layers.variable_input import VariableLayer

_allowed_symbols = [
    'AutoregressiveTransform',
    'BlockwiseInitializer',
    'CategoricalMixtureOfOneHotCategorical',
    'Convolution1DFlipout',
    'Convolution1DReparameterization',
    'Convolution2DFlipout',
    'Convolution2DReparameterization',
    'Convolution3DFlipout',
    'Convolution3DReparameterization',
    'DenseFlipout',
    'DenseLocalReparameterization',
    'DenseReparameterization',
    'DenseVariational',
    'DistributionLambda',
    'IndependentBernoulli',
    'IndependentLogistic',
    'IndependentNormal',
    'IndependentPoisson',
    'KLDivergenceAddLoss',
    'KLDivergenceRegularizer',
    'MixtureLogistic',
    'MixtureNormal',
    'MixtureSameFamily',
    'MultivariateNormalTriL',
    'OneHotCategorical',
    'VariableLayer',
    'default_loc_scale_fn',
    'default_mean_field_normal_fn',
    'default_multivariate_normal_fn',
]
