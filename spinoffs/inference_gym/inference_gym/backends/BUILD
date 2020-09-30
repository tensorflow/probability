# Copyright 2020 The TensorFlow Probability Authors.
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
#   A package for target densities and benchmarking of inference algorithms
#   against the same.

# [internal] load pytype.bzl (pytype_strict_library)
# [internal] load strict.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
        "//inference_gym:__subpackages__",
    ],
)

licenses(["notice"])

# pytype_strict
py_library(
    name = "rewrite",
    srcs = ["rewrite.py"],
    srcs_version = "PY3",
)

# pytype_strict
py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
)

# py_strict
py_test(
    name = "jax_integration_test",
    srcs = ["jax_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//inference_gym:using_jax",
    ],
)

# py_strict
py_test(
    name = "numpy_integration_test",
    srcs = ["numpy_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//inference_gym:using_numpy",
    ],
)

# py_strict
py_test(
    name = "tensorflow_integration_test",
    srcs = ["tensorflow_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        # tensorflow dep,
        "//inference_gym:using_tensorflow",
    ],
)
