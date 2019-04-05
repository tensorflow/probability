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
        ":array",
        ":control_flow",
        ":dtype",
        ":linalg",
        ":math",
        ":misc",
        ":ops",
        ":test",
    ],
)

py_library(
    name = "array",
    srcs = ["array.py"],
    deps = [
        # numpy dep,
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
    name = "dtype",
    srcs = ["dtype.py"],
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
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "math",
    srcs = ["math.py"],
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
    name = "ops",
    srcs = ["ops.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)

py_library(
    name = "test",
    srcs = ["test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy/internal:utils",
    ],
)
