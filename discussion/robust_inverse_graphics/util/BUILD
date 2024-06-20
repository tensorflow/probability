# Copyright 2024 The TensorFlow Probability Authors.
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
# Utilities.

# [internal] load pytype.bzl (pytype_strict_library, pytype_strict_test)
# [internal] load strict.bzl
# Placeholder: py_library

package(
    # default_applicable_licenses
    default_visibility = ["//discussion/robust_inverse_graphics:__subpackages__"],
)

licenses(["notice"])

# pytype_strict
py_library(
    name = "array_util",
    srcs = ["array_util.py"],
    deps = [
        # jax dep,
    ],
)

# pytype_strict
py_library(
    name = "camera_util",
    srcs = ["camera_util.py"],
    deps = [
        # numpy dep,
        # pyquaternion dep,
    ],
)

# pytype_strict
py_library(
    name = "gin_utils",
    srcs = ["gin_utils.py"],
    deps = [
        # absl/flags dep,
        # gin dep,
        # yaml dep,
    ],
)

# pytype_strict
py_library(
    name = "math_util",
    srcs = ["math_util.py"],
    deps = [
        # jax dep,
    ],
)

# pytype_strict
py_test(
    name = "math_util_test",
    srcs = ["math_util_test.py"],
    deps = [
        ":math_util",
        ":test_util",
        # google/protobuf:use_fast_cpp_protos dep,
        # jax dep,
    ],
)

# pytype_strict
py_library(
    name = "plot_util",
    srcs = ["plot_util.py"],
    deps = [
        # jax dep,
        # matplotlib dep,
        # numpy dep,
        "//fun_mc:using_jax",
    ],
)

# Not strict or pytype due to the test_util.jax dep.
py_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.py"],
    deps = [
        # absl/testing:absltest dep,
        # jax dep,
        "//tensorflow_probability/python/internal:test_util.jax",
    ],
)

# py_strict
py_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    deps = [
        ":test_util",
        # flax dep,
        # google/protobuf:use_fast_cpp_protos dep,
        # numpy dep,
    ],
)

# pytype_strict
py_library(
    name = "tree2",
    srcs = ["tree2.py"],
    deps = [
        # etils/epath dep,
        # immutabledict dep,
        # numpy dep,
    ],
)

# py_strict
py_test(
    name = "tree2_test",
    srcs = ["tree2_test.py"],
    deps = [
        ":test_util",
        ":tree2",
        # absl/testing:parameterized dep,
        # flax:core dep,
        # google/protobuf:use_fast_cpp_protos dep,
        # immutabledict dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability:jax",
    ],
)

# pytype_strict
py_library(
    name = "tree_util",
    srcs = ["tree_util.py"],
    deps = [
        # jax dep,
    ],
)

# py_strict
py_test(
    name = "tree_util_test",
    srcs = ["tree_util_test.py"],
    deps = [
        ":test_util",
        ":tree_util",
        # flax:core dep,
        # google/protobuf:use_fast_cpp_protos dep,
        # jax dep,
    ],
)
