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
#   Automatic marginalization of latent variables.

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
    name = "logeinsumexp",
    srcs = ["logeinsumexp.py"],
    deps = [
        # numpy dep,
        # opt_einsum dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "logeinsumexp_test",
    size = "medium",
    srcs = [
        "logeinsumexp_test.py",
    ],
    deps = [
        ":logeinsumexp",
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "marginalize",
    srcs = ["__init__.py"],
    deps = [
        ":logeinsumexp",
        ":marginalizable",
    ],
)

py_library(
    name = "marginalizable",
    srcs = ["marginalizable.py"],
    deps = [
        ":logeinsumexp",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:sample",
    ],
)

py_test(
    name = "marginalizable_test",
    size = "medium",
    srcs = ["marginalizable_test.py"],
    deps = [
        ":marginalize",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:deterministic",
        "//tensorflow_probability/python/distributions:hidden_markov_model",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
