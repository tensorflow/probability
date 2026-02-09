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
#   Internal helper libraries for layers.

# Placeholder: py_library
# Placeholder: py_test

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

py_library(
    name = "internal",
    srcs = ["__init__.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":distribution_tensor_coercible",
        ":tensor_tuple",
    ],
)

py_library(
    name = "distribution_tensor_coercible",
    srcs = ["distribution_tensor_coercible.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/util",
    ],
)

py_test(
    name = "distribution_tensor_coercible_test",
    size = "small",
    srcs = ["distribution_tensor_coercible_test.py"],
    strict_deps = False,
    deps = [
        ":distribution_tensor_coercible",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/layers:distribution_layer",
    ],
)

py_library(
    name = "tensor_tuple",
    srcs = ["tensor_tuple.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "tensor_tuple_test",
    size = "small",
    srcs = ["tensor_tuple_test.py"],
    strict_deps = False,
    deps = [
        ":tensor_tuple",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)
