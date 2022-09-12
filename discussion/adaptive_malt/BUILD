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

# [internal] load pytype.bzl (pytype_strict_library)
load("//third_party/py/pytest:pytest_defs.bzl", "pytest_test")

licenses(["notice"])

# pytype_strict
py_library(
    name = "adaptive_malt",
    srcs = ["adaptive_malt.py"],
    deps = [
        # gin dep,
        # immutabledict dep,
        # jax dep,
        "//fun_mc:using_jax",
        "//inference_gym:using_jax",
        "//tensorflow_probability/substrates:jax",
    ],
)

pytest_test(
    name = "adaptive_malt_test",
    srcs = ["adaptive_malt_test.py"],
    deps = [
        ":adaptive_malt",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # inference_gym:using_jax dep,
        # jax dep,
        # numpy dep,
        "//inference_gym:using_jax",
        "//tensorflow_probability/substrates:jax",
    ],
)
