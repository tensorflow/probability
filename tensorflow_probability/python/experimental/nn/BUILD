# Copyright 2019 The TensorFlow Probability Authors.
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
#   tf.Module tools for building neural architectures.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "nn",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":affine_layers",
        ":convolutional_layers",
        ":convolutional_transpose_layers",
        ":layers",
        ":variational_base",
        "//tensorflow_probability/python/experimental/nn/initializers",
        "//tensorflow_probability/python/experimental/nn/util",
    ],
)

py_library(
    name = "affine_layers",
    srcs = ["affine_layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        ":variational_base",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/experimental/nn/util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "affine_layers_test",
    size = "small",
    srcs = ["affine_layers_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "convolutional_layers",
    srcs = ["convolutional_layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        ":variational_base",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/experimental/nn/util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "convolutional_layers_test",
    size = "small",
    srcs = ["convolutional_layers_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "convolutional_transpose_layers",
    srcs = ["convolutional_transpose_layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convolutional_layers",
        ":layers",
        ":variational_base",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/experimental/nn/util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "convolutional_transpose_layers_test",
    size = "small",
    srcs = ["convolutional_transpose_layers_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/nn/util",
        "//tensorflow_probability/python/internal:name_util",
    ],
)

py_test(
    name = "layers_test",
    size = "small",
    srcs = ["layers_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "variational_base",
    srcs = ["variational_base.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/math:random_ops",
        "//tensorflow_probability/python/monte_carlo:expectation",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)
