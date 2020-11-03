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
# Description:
#   TransitionKernel for NeuTra.

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "neutra",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":neutra_kernel",
    ],
)

py_library(
    name = "neutra_kernel",
    srcs = ["neutra_kernel.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:unnest",
    ],
)

py_test(
    name = "neutra_kernel_test",
    size = "medium",
    srcs = ["neutra_kernel_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        ":neutra",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
