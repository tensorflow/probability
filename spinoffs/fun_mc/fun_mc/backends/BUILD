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
# [internal] load pytype.bzl (pytype_strict_library)
# [internal] load strict.bzl

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

# pytype_strict
py_library(
    name = "rewrite",
    srcs = ["rewrite.py"],
)

# py_strict
py_test(
    name = "tensorflow_integration_test",
    srcs = ["tensorflow_integration_test.py"],
    deps = [
        # absl/testing:absltest dep,
        # tensorflow dep,
        "//fun_mc:using_tensorflow",
    ],
)

# py_strict
py_test(
    name = "jax_integration_test",
    srcs = ["jax_integration_test.py"],
    deps = [
        # absl/testing:absltest dep,
        # jax dep,
        "//fun_mc:using_jax",
    ],
)
