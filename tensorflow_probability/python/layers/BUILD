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
        ":masked_autoregressive",
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
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/util:seed_stream",
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
        "//tensorflow_probability/python/internal:test_util",
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
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

py_test(
    name = "dense_variational_test",
    size = "medium",
    srcs = ["dense_variational_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
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
        "//tensorflow_probability/python/util:seed_stream",
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
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "distribution_layer",
    srcs = ["distribution_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        # cloudpickle dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:fill_scale_tril",
        "//tensorflow_probability/python/bijectors:transpose",
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/distributions:logistic",
        "//tensorflow_probability/python/distributions:mixture_same_family",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:onehot_categorical",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/distributions:variational_gaussian_process",
        "//tensorflow_probability/python/layers/internal",
    ],
)

py_test(
    name = "distribution_layer_test",
    size = "medium",
    srcs = ["distribution_layer_test.py"],
    shard_count = 5,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
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
    size = "small",
    srcs = ["initializers_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "masked_autoregressive",
    srcs = ["masked_autoregressive.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:masked_autoregressive",
        "//tensorflow_probability/python/distributions:transformed_distribution",
    ],
)

py_test(
    name = "masked_autoregressive_test",
    size = "small",
    srcs = ["masked_autoregressive_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
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
        "//tensorflow_probability/python/distributions:deterministic",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:normal",
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
    ],
)

py_test(
    name = "variable_input_test",
    size = "small",
    srcs = ["variable_input_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "weight_norm",
    srcs = [
        "weight_norm.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "weight_norm_test",
    size = "small",
    srcs = ["weight_norm_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/layers:weight_norm",
    ],
)
