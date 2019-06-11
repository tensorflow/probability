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
#   Numpy backend.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "numpy",
    srcs = ["__init__.py"],
    deps = [
        ":compat",
        ":control_flow",
        ":debugging",
        ":dtype",
        ":errors",
        ":functional_ops",
        ":keras",
        ":linalg",
        ":misc",
        ":nn",
        ":numpy_array",
        ":numpy_logging",
        ":numpy_math",
        ":ops",
        ":random_generators",
        ":test_lib",
        # tensorflow dep,
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
    deps = [
        ":v1",
        ":v2",
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "control_flow",
    srcs = ["control_flow.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "debugging",
    srcs = ["debugging.py"],
    deps = [
        ":v1",
        ":v2",
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "dtype",
    srcs = ["dtype.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "functional_ops",
    srcs = ["functional_ops.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    deps = [
        ":keras_layers",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "keras_layers",
    srcs = ["keras_layers.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "linalg",
    srcs = ["linalg.py"],
    deps = [
        ":linear_operator",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "linear_operator",
    srcs = ["linear_operator.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "misc",
    srcs = ["misc.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "nn",
    srcs = ["nn.py"],
    deps = [
        ":numpy_math",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "numpy_array",
    srcs = ["numpy_array.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "numpy_logging",
    srcs = ["numpy_logging.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "numpy_math",
    srcs = ["numpy_math.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "random_generators",
    srcs = ["random_generators.py"],
    deps = [
        ":numpy_math",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "test_lib",
    srcs = ["test_lib.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_test(
    name = "numpy_test",
    srcs = ["numpy_test.py"],
    deps = [
        ":numpy",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "v1",
    srcs = ["v1.py"],
    deps = [
        ":initializers",
        ":ops",
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "v2",
    srcs = ["v2.py"],
    deps = [
        ":ops",
    ],
)
