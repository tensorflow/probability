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
#   TensorFlow Probability ODE solvers.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:custom_gradient",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "bdf",
    srcs = ["bdf.py"],
    deps = [
        ":base",
        ":bdf_util",
        ":util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "dormand_prince",
    srcs = ["dormand_prince.py"],
    deps = [
        ":base",
        ":runge_kutta_util",
        ":util",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
    ],
)

multi_substrate_py_library(
    name = "bdf_util",
    srcs = ["bdf_util.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "bdf_util_test",
    size = "small",
    srcs = ["bdf_util_test.py"],
    deps = [
        ":bdf",
        ":bdf_util",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "runge_kutta_util",
    srcs = ["runge_kutta_util.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "runge_kutta_util_test",
    size = "small",
    srcs = ["runge_kutta_util_test.py"],
    deps = [
        ":runge_kutta_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "ode",
    srcs = ["__init__.py"],
    deps = [
        ":base",
        ":bdf",
        ":dormand_prince",
    ],
)

multi_substrate_py_test(
    name = "ode_test",
    size = "large",
    srcs = ["ode_test.py"],
    shard_count = 10,
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "xla_test_cpu",
    srcs = ["xla_test.py"],
    args = ["--test_device=cpu"],
    main = "xla_test.py",
    shard_count = 1,
    tags = [
        "nozapfhahn",
    ],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_test(
    name = "xla_test_gpu",
    srcs = ["xla_test.py"],
    args = ["--test_device=gpu"],
    main = "xla_test.py",
    shard_count = 1,
    tags = [
        "nozapfhahn",
        "requires-gpu-nvidia",
    ],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    deps = [
        ":util",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)
