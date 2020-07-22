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
#   Functional MCMC API.

# [internal] load pytype.bzl (pytype_library)

# py_test rules are rewritten to py3_test on export to OSS, for reasons of
# reversibility of import<->export transforms. This is only necessary until we
# eliminate all py2and3_test targets, after which everything can safely use
# py_test, interanlly and externally, and no rewrites will be needed.
# @unused  <-- satisfy buildifier errors
load("//tensorflow_probability/python:build_defs.bzl", "py3_test")

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//visibility:public",
    ],
)

exports_files(["LICENSE"])

# pytype
py_library(
    name = "fun_mcmc",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
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
    srcs_version = "PY2AND3",
    deps = [
        ":fun_mcmc_lib",
        ":prefab",
        ":util_tfp",
    ],
)

# pytype
py_library(
    name = "util_jax",
    srcs = ["util_jax.py"],
    srcs_version = "PY3",
    deps = [
        # jax dep,
        # jax:stax dep,
    ],
)

# pytype
py_library(
    name = "tf_on_jax",
    srcs = ["tf_on_jax.py"],
    srcs_version = "PY3",
    deps = [
        # jax dep,
        # jax:stax dep,
    ],
)

# pytype
py_library(
    name = "util_tf",
    srcs = ["util_tf.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

# pytype
py_library(
    name = "backend",
    srcs = ["backend.py"],
    srcs_version = "PY3",
    deps = [
        ":tf_on_jax",
        ":util_jax",
        ":util_tf",
        "//tensorflow_probability",
        "//tensorflow_probability/substrates:jax",
    ],
)

# pytype
py_library(
    name = "fun_mcmc_lib",
    srcs = ["fun_mcmc_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":backend",
        # numpy dep,
    ],
)

# pytype
py_library(
    name = "using_jax",
    srcs = ["using_jax.py"],
    srcs_version = "PY3",
    deps = [
        ":api",
        ":backend_jax",
        ":dynamic_backend_jax",
        ":rewrite",
    ],
)

# pytype
py_library(
    name = "using_tensorflow",
    srcs = ["using_tensorflow.py"],
    srcs_version = "PY3",
    deps = [
        ":api",
        ":backend_tf",
        ":dynamic_backend_tf",
        ":rewrite",
    ],
)

# pytype
py_library(
    name = "backend_jax",
    srcs = ["backend_jax.py"],
    srcs_version = "PY3",
    deps = [
        ":tf_on_jax",
        ":util_jax",
        "//tensorflow_probability/substrates:jax",
    ],
)

# pytype
py_library(
    name = "backend_tf",
    srcs = ["backend_tf.py"],
    srcs_version = "PY3",
    deps = [
        ":util_tf",
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

# pytype
py_library(
    name = "dynamic_backend_jax",
    srcs = ["dynamic/backend_jax/__init__.py"],
)

# pytype
py_library(
    name = "dynamic_backend_tf",
    srcs = ["dynamic/backend_tf/__init__.py"],
)

# pytype
py_library(
    name = "rewrite",
    srcs = ["rewrite.py"],
    srcs_version = "PY3",
)

# py_test
py3_test(
    name = "fun_mcmc_test",
    srcs = ["fun_mcmc_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        ":fun_mcmc",
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

# pytype
py_library(
    name = "prefab",
    srcs = ["prefab.py"],
    deps = [
        ":backend",
        ":fun_mcmc_lib",
    ],
)

# py_test
py3_test(
    name = "prefab_test",
    srcs = ["prefab_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        ":fun_mcmc",
        # jax dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

# pytype
py_library(
    name = "util_tfp",
    srcs = ["util_tfp.py"],
    deps = [
        ":backend",
        ":fun_mcmc_lib",
        "//tensorflow_probability",
    ],
)

# py_test
py3_test(
    name = "util_tfp_test",
    srcs = ["util_tfp_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        ":fun_mcmc",
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)
