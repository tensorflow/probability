# Copyright 2019 The TensorFlow Probability Authors.
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
"""TFP for Numpy."""


# pylint: disable=g-bad-import-order

# from tensorflow_probability.substrates.numpy.google import staging  # DisableOnExport  # pylint:disable=line-too-long

from tensorflow_probability.python.version import __version__
from tensorflow_probability.substrates.numpy import bijectors
from tensorflow_probability.substrates.numpy import distributions
from tensorflow_probability.substrates.numpy import experimental
from tensorflow_probability.substrates.numpy import internal
from tensorflow_probability.substrates.numpy import math
from tensorflow_probability.substrates.numpy import mcmc
from tensorflow_probability.substrates.numpy import optimizer
from tensorflow_probability.substrates.numpy import random
from tensorflow_probability.substrates.numpy import sts
from tensorflow_probability.substrates.numpy import stats
from tensorflow_probability.substrates.numpy import util

from tensorflow_probability.python.internal.backend import numpy as tf2numpy
