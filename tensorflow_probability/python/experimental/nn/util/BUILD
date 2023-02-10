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

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow_probability:license"],
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

py_library(
    name = "convolution_util",
    srcs = ["convolution_util.py"],
    deps = [
        ":utils",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "convolution_util_test",
    size = "medium",
    srcs = ["convolution_util_test.py"],
    shard_count = 4,
    deps = [
        ":convolution_util",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "kernel_bias",
    srcs = ["kernel_bias.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:mixture_same_family",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/experimental/nn/initializers:initializers_lib",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "kernel_bias_test",
    size = "small",
    srcs = ["kernel_bias_test.py"],
    shard_count = 4,
    deps = [
        ":kernel_bias",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/nn/initializers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "util",
    srcs = ["__init__.py"],
    deps = [
        ":convolution_util",
        ":kernel_bias",
        ":random_variable",
        ":utils",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

py_library(
    name = "random_variable",
    srcs = ["random_variable.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/util:deferred_tensor",
    ],
)

py_test(
    name = "random_variable_test",
    size = "medium",
    srcs = ["random_variable_test.py"],
    deps = [
        ":random_variable",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)
