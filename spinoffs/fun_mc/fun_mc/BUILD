# Copyright 2021 The TensorFlow Probability Authors.
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
#   Functional MC API.

# [internal] load pytype.bzl (pytype_library)
# Placeholder: py_test

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = [
        "//visibility:public",
    ],
)

# pytype
py_library(
    name = "fun_mc",
    srcs = ["__init__.py"],
    deps = [
        ":api",
        ":backend",
        ":using_jax",
        ":using_tensorflow",
    ],
)

# pytype
py_library(
    name = "api",
    srcs = ["api.py"],
    deps = [
        ":fun_mc_lib",
        ":prefab",
        ":smc",
        ":types",
        ":util_tfp",
    ],
)

# pytype
py_library(
    name = "backend",
    srcs = ["backend.py"],
    deps = [
        "//fun_mc/dynamic/backend_jax:backend",
    ],
)

# pytype
py_library(
    name = "fun_mc_lib",
    srcs = ["fun_mc_lib.py"],
    deps = [
        ":backend",
        # numpy dep,
    ],
)

# pytype
py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        ":using_jax",
        ":using_tensorflow",
    ],
)

# pytype
py_library(
    name = "using_jax",
    srcs = ["using_jax.py"],
    deps = [
        ":api",
        # jax dep,
        "//fun_mc/backends:rewrite",
        "//fun_mc/dynamic/backend_jax:backend",
        # tensorflow_probability/substrates:jax dep,
    ],
)

# pytype
py_library(
    name = "using_tensorflow",
    srcs = ["using_tensorflow.py"],
    deps = [
        ":api",
        # tensorflow dep,
        # tensorflow_probability dep,
        "//fun_mc/backends:rewrite",
        "//fun_mc/dynamic/backend_tensorflow:backend",
    ],
)

py_test(
    name = "fun_mc_test",
    size = "large",
    srcs = ["fun_mc_test.py"],
    shard_count = 8,
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":fun_mc",
        ":prefab",
        ":test_util",
        # absl/testing:parameterized dep,
        # scipy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
        # tensorflow_probability/python/internal:test_util dep,
        # tensorflow/compiler/jit dep,
    ],
)

# pytype
py_library(
    name = "malt",
    srcs = ["malt.py"],
    deps = [
        ":backend",
        ":fun_mc_lib",
    ],
)

py_test(
    name = "malt_test",
    srcs = ["malt_test.py"],
    shard_count = 2,
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":fun_mc",
        ":malt",
        ":test_util",
        # jax dep,
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        # tensorflow/compiler/jit dep,
    ],
)

# pytype
py_library(
    name = "prefab",
    srcs = ["prefab.py"],
    deps = [
        ":backend",
        ":fun_mc_lib",
        ":malt",
        ":sga_hmc",
    ],
)

py_test(
    name = "prefab_test",
    srcs = ["prefab_test.py"],
    shard_count = 2,
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":fun_mc",
        ":prefab",
        ":test_util",
        # jax dep,
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        # tensorflow/compiler/jit dep,
    ],
)

# pytype
py_library(
    name = "sga_hmc",
    srcs = ["sga_hmc.py"],
    deps = [
        ":backend",
        ":fun_mc_lib",
        # immutabledict dep,
    ],
)

py_test(
    name = "sga_hmc_test",
    srcs = ["sga_hmc_test.py"],
    shard_count = 2,
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":fun_mc",
        ":sga_hmc",
        ":test_util",
        # jax dep,
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        # tensorflow/compiler/jit dep,
    ],
)

# pytype
py_library(
    name = "smc",
    srcs = ["smc.py"],
    deps = [
        ":backend",
        ":fun_mc_lib",
        ":types",
    ],
)

pytype_strict_contrib_test(
    name = "smc_test",
    srcs = ["smc_test.py"],
    shard_count = 4,
    deps = [
        ":backend",
        ":fun_mc_lib",
        ":smc",
        ":test_util",
        ":types",
        # absl/testing:parameterized dep,
        # jax dep,
        # mock dep,
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        # tensorflow/compiler/jit dep,
    ],
)

# pytype
py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        ":backend",
        # jaxtyping dep,
        # typeguard dep,
    ],
)

# pytype
py_library(
    name = "util_tfp",
    srcs = ["util_tfp.py"],
    deps = [
        ":backend",
        ":fun_mc_lib",
    ],
)

py_test(
    name = "util_tfp_test",
    srcs = ["util_tfp_test.py"],
    shard_count = 2,
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":fun_mc",
        ":test_util",
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
        # tensorflow_probability/python/internal:test_util dep,
        # tensorflow/compiler/jit dep,
    ],
)
