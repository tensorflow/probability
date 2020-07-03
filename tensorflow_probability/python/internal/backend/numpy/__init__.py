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
"""Experimental Numpy backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.internal.backend.numpy import bitwise
from tensorflow_probability.python.internal.backend.numpy import compat
from tensorflow_probability.python.internal.backend.numpy import debugging
from tensorflow_probability.python.internal.backend.numpy import dtype as dtypes
from tensorflow_probability.python.internal.backend.numpy import errors
from tensorflow_probability.python.internal.backend.numpy import keras
from tensorflow_probability.python.internal.backend.numpy import linalg
from tensorflow_probability.python.internal.backend.numpy import nest
from tensorflow_probability.python.internal.backend.numpy import nn
from tensorflow_probability.python.internal.backend.numpy import numpy_array as array
from tensorflow_probability.python.internal.backend.numpy import numpy_logging as logging
from tensorflow_probability.python.internal.backend.numpy import numpy_math as math
from tensorflow_probability.python.internal.backend.numpy import numpy_signal as signal
from tensorflow_probability.python.internal.backend.numpy import random_generators as random
from tensorflow_probability.python.internal.backend.numpy import raw_ops
from tensorflow_probability.python.internal.backend.numpy import sets_lib as sets
from tensorflow_probability.python.internal.backend.numpy import sparse_lib as sparse
from tensorflow_probability.python.internal.backend.numpy import test_lib as test
from tensorflow_probability.python.internal.backend.numpy.control_flow import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.dtype import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.functional_ops import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.gen.tensor_shape import TensorShape
from tensorflow_probability.python.internal.backend.numpy.misc import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.numpy_array import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.numpy_math import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.ops import *  # pylint: disable=wildcard-import


Assert = debugging.Assert
Module = compat.v2.Module
TensorArray = compat.v2.TensorArray
Variable = compat.v2.Variable
check_numerics = debugging.check_numerics
eye = linalg.eye
function = compat.function
matmul = linalg.matmul
