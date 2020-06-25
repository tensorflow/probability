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
"""Module for neural network layers."""
from oryx.experimental.nn import function
from oryx.experimental.nn.base import Layer
from oryx.experimental.nn.base import LayerParams
from oryx.experimental.nn.base import Template
from oryx.experimental.nn.combinator import Serial
from oryx.experimental.nn.convolution import Conv
from oryx.experimental.nn.convolution import Deconv
from oryx.experimental.nn.core import Dense
from oryx.experimental.nn.core import Dropout
from oryx.experimental.nn.core import LogSoftmax
from oryx.experimental.nn.core import Relu
from oryx.experimental.nn.core import Softmax
from oryx.experimental.nn.core import Softplus
from oryx.experimental.nn.core import Tanh
from oryx.experimental.nn.normalization import BatchNorm
from oryx.experimental.nn.pooling import AvgPooling
from oryx.experimental.nn.pooling import MaxPooling
from oryx.experimental.nn.pooling import SumPooling
from oryx.experimental.nn.reshape import Flatten
from oryx.experimental.nn.reshape import Reshape

del function  # Only needed for registration
