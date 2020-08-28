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
#   Experimental code for symbolic execution/analysis.

# [internal] load strict.bzl

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "lazybones",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":deferred",
        ":deferred_scope",
        "//tensorflow_probability/python/experimental/lazybones/utils",
    ],
)

py_library(
    name = "deferred",
    srcs = ["deferred.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":deferred_scope",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/lazybones/utils:special_methods",
        "//tensorflow_probability/python/experimental/lazybones/utils:weak_container",
    ],
)

py_library(
    name = "deferred_scope",
    srcs = ["deferred_scope.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
        "//tensorflow_probability/python/experimental/lazybones/utils:weak_container",
    ],
)

py_test(
    name = "deferred_np_test",
    size = "small",
    srcs = ["deferred_np_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        # numpy dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "deferred_tf_test",
    size = "small",
    srcs = ["deferred_tf_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "deferred_jax_test",
    size = "medium",
    srcs = ["deferred_jax_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["tfp_jax"],
    deps = [
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability",
    ],
)
