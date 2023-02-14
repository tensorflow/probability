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
#   MCMC methods, diagnostics, and related utilities.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "internal",
    srcs = ["__init__.py"],
    deps = [
        ":leapfrog_integrator",
        ":slice_sampler_utils",
        ":util",
    ],
)

multi_substrate_py_library(
    name = "slice_sampler_utils",
    srcs = ["slice_sampler_utils.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:gamma",
    ],
)

multi_substrate_py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:broadcast_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    deps = [
        ":util",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:lognormal",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "leapfrog_integrator",
    srcs = ["leapfrog_integrator.py"],
    deps = [
        ":util",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "leapfrog_integrator_test",
    size = "medium",
    srcs = ["leapfrog_integrator_test.py"],
    disabled_substrates = ["numpy"],
    deps = [
        ":leapfrog_integrator",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
