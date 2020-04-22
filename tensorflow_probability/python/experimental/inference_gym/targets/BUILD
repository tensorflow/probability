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
#   A package for target densities.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_test",
    "multi_substrate_py_library",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "targets",
    srcs = ["__init__.py"],
    deps = [
        ":banana",
        ":bayesian_model",
        ":ill_conditioned_gaussian",
        ":logistic_regression",
        ":sparse_logistic_regression",
    ],
)

multi_substrate_py_library(
    name = "banana",
    srcs = ["banana.py"],
    deps = [
        ":bayesian_model",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
    ],
)

multi_substrate_py_test(
    name = "banana_test",
    srcs = ["banana_test.py"],
    deps = [
        ":banana",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "bayesian_model",
    srcs = ["bayesian_model.py"],
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "bayesian_model_test",
    srcs = ["bayesian_model_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "ill_conditioned_gaussian",
    srcs = ["ill_conditioned_gaussian.py"],
    deps = [
        ":bayesian_model",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
    ],
)

multi_substrate_py_test(
    name = "ill_conditioned_gaussian_test",
    srcs = ["ill_conditioned_gaussian_test.py"],
    deps = [
        ":ill_conditioned_gaussian",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "logistic_regression",
    srcs = ["logistic_regression.py"],
    deps = [
        ":bayesian_model",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/experimental/inference_gym/internal:data",
    ],
)

multi_substrate_py_test(
    name = "logistic_regression_test",
    srcs = ["logistic_regression_test.py"],
    deps = [
        ":logistic_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "sparse_logistic_regression",
    srcs = ["sparse_logistic_regression.py"],
    deps = [
        ":bayesian_model",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/experimental/inference_gym/internal:data",
    ],
)

multi_substrate_py_test(
    name = "sparse_logistic_regression_test",
    srcs = ["sparse_logistic_regression_test.py"],
    jax_size = "medium",
    deps = [
        ":sparse_logistic_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
    ],
)
