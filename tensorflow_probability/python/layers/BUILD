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
#   TensorFlow Probability layers.

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "layers",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":conv_variational",
        ":dense_variational",
        ":dense_variational_v2",
        ":distribution_layer",
        ":initializers",
        ":util",
        ":variable_input",
    ],
)

py_library(
    name = "conv_variational",
    srcs = [
        "conv_variational.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:seed_stream",
        "//tensorflow_probability/python/math",
        "//tensorflow_probability/python/util:docstring",
    ],
)

py_test(
    name = "conv_variational_test",
    size = "medium",
    srcs = ["conv_variational_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "dense_variational",
    srcs = [
        "dense_variational.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:seed_stream",
        "//tensorflow_probability/python/math",
        "//tensorflow_probability/python/util:docstring",
    ],
)

py_test(
    name = "dense_variational_test",
    size = "small",
    srcs = ["dense_variational_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "dense_variational_v2",
    srcs = [
        "dense_variational_v2.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/distributions:seed_stream",
    ],
)

py_test(
    name = "dense_variational_v2_test",
    size = "small",
    srcs = ["dense_variational_v2_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "distribution_layer",
    srcs = ["distribution_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/layers/internal",
    ],
)

py_test(
    name = "distribution_layer_test",
    size = "medium",
    srcs = ["distribution_layer_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "initializers",
    srcs = [
        "initializers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "initializers_test",
    srcs = ["initializers_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "variable_input",
    srcs = [
        "variable_input.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/layers/internal",
    ],
)

py_test(
    name = "variable_input_test",
    srcs = ["variable_input_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)
