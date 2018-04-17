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
from tensorflow_probability.python.layers.util import default_loc_scale_fn
from tensorflow_probability.python.layers.util import default_mean_field_normal_fn
from tensorflow_probability.python.layers.util import default_multivariate_normal_fn

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'Convolution1DFlipout',
    'Convolution1DReparameterization',
    'Convolution2DFlipout',
    'Convolution2DReparameterization',
    'Convolution3DFlipout',
    'Convolution3DReparameterization',
    'DenseFlipout',
    'DenseLocalReparameterization',
    'DenseReparameterization',
    'default_loc_scale_fn',
    'default_mean_field_normal_fn',
    'default_multivariate_normal_fn',
]

remove_undocumented(__name__, _allowed_symbols)
