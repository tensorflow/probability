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

# Placeholder: py_library
# Placeholder: py_test

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "layers",
    srcs = ["__init__.py"],
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
    deps = [
        ":util",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/random",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

py_test(
    name = "conv_variational_test",
    size = "medium",
    srcs = ["conv_variational_test.py"],
    shard_count = 2,
    deps = [
        ":conv_variational",
        ":util",
        # keras/testing_infra:test_utils dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/random:random_ops",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

py_library(
    name = "dense_variational",
    srcs = [
        "dense_variational.py",
    ],
    deps = [
        ":util",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/random",
        "//tensorflow_probability/python/util",
    ],
)

py_test(
    name = "dense_variational_test",
    size = "medium",
    srcs = ["dense_variational_test.py"],
    deps = [
        ":dense_variational",
        ":util",
        # keras/testing_infra:test_utils dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

py_library(
    name = "dense_variational_v2",
    srcs = [
        "dense_variational_v2.py",
    ],
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
        ":dense_variational_v2",
        ":distribution_layer",
        ":variable_input",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "distribution_layer",
    srcs = ["distribution_layer.py"],
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
        ":distribution_layer",
        ":variable_input",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:fill_scale_tril",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/distributions:logistic",
        "//tensorflow_probability/python/distributions:mixture_same_family",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
        "//tensorflow_probability/python/util:deferred_tensor",
    ],
)

py_library(
    name = "initializers",
    srcs = [
        "initializers.py",
    ],
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "initializers_test",
    size = "small",
    srcs = ["initializers_test.py"],
    deps = [
        ":initializers",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "masked_autoregressive",
    srcs = ["masked_autoregressive.py"],
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
        ":distribution_layer",
        ":masked_autoregressive",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:masked_autoregressive",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    deps = [
        # keras dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:deterministic",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/util",
    ],
)

py_library(
    name = "variable_input",
    srcs = [
        "variable_input.py",
    ],
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "variable_input_test",
    size = "small",
    srcs = ["variable_input_test.py"],
    deps = [
        ":distribution_layer",
        ":variable_input",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "weight_norm",
    srcs = [
        "weight_norm.py",
    ],
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "weight_norm_test",
    size = "medium",
    srcs = ["weight_norm_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/layers:weight_norm",
    ],
)
