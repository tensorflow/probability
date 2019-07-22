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
"""TensorFlow Probability benchmarking library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.debugging.benchmarking.benchmark_tf_function import benchmark_tf_function
from tensorflow_probability.python.debugging.benchmarking.benchmark_tf_function import BenchmarkTfFunctionConfig
from tensorflow_probability.python.debugging.benchmarking.benchmark_tf_function import default_benchmark_config
from tensorflow_probability.python.debugging.benchmarking.benchmark_tf_function import HARDWARE_CPU
from tensorflow_probability.python.debugging.benchmarking.benchmark_tf_function import HARDWARE_GPU
from tensorflow_probability.python.debugging.benchmarking.benchmark_tf_function import RUNTIME_EAGER
from tensorflow_probability.python.debugging.benchmarking.benchmark_tf_function import RUNTIME_FUNCTION
from tensorflow_probability.python.debugging.benchmarking.benchmark_tf_function import RUNTIME_XLA


__all__ = [
    "benchmark_tf_function",
    "RUNTIME_EAGER",
    "RUNTIME_FUNCTION",
    "RUNTIME_XLA",
    "HARDWARE_CPU",
    "HARDWARE_GPU",
    "BenchmarkTfFunctionConfig",
    "default_benchmark_config",
    ]
