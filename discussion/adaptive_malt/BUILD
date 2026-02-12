# Copyright 2022 The TensorFlow Probability Authors.
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
# Adaptive MALT files.

# [internal] load pytype.bzl (pytype_strict_binary, pytype_strict_library)
# Placeholder: py_test

package(
    # default_applicable_licenses
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# pytype_strict
py_library(
    name = "adaptive_malt",
    srcs = ["adaptive_malt.py"],
    deps = [
        ":utils",
        # etils/epath dep,
        # gin dep,
        # immutabledict dep,
        # jax dep,
        # numpy dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        "//fun_mc:using_jax",
        "//inference_gym:using_jax",
        "//tensorflow_probability/substrates:jax",
    ],
)

py_test(
    name = "adaptive_malt_test",
    srcs = ["adaptive_malt_test.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":adaptive_malt",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//inference_gym:using_jax",
        "//tensorflow_probability/substrates:jax",
    ],
)

# pytype_strict
py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        # absl/flags dep,
        # etils/epath dep,
        # gin dep,
        # h5py dep,
        # jax dep,
        # numpy dep,
        # tree dep,
        # yaml dep,
    ],
)

# pytype_strict
py_binary(
    name = "grid_search_runner",
    srcs = ["grid_search_runner.py"],
    deps = [
        ":adaptive_malt",
        ":utils",
        # JAX gpu dep,
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        # etils/epath dep,
        # gin dep,
        # numpy dep,
    ],
)

# pytype_strict
py_binary(
    name = "trial_runner",
    srcs = ["trial_runner.py"],
    deps = [
        ":adaptive_malt",
        ":utils",
        # JAX gpu dep,
        # absl:app dep,
        # absl/flags dep,
        # etils/epath dep,
        # gin dep,
    ],
)

# pytype_strict
py_library(
    name = "grid_search_runner_experiments",
    srcs = ["grid_search_runner_experiments.py"],
    deps = [
        # numpy dep,
    ],
)

# pytype_strict
py_library(
    name = "trial_runner_experiments",
    srcs = ["trial_runner_experiments.py"],
)
