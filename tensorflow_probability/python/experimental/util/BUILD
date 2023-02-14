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
#   TensorFlow Probability experimental utility functions.

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
    name = "util",
    srcs = ["__init__.py"],
    substrates_omit_deps = [
        ":composite_tensor",
        ":deferred_module",
    ],
    deps = [
        ":composite_tensor",
        ":deferred_module",
        ":jit_public_methods",
        ":trainable",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

py_library(
    name = "deferred_module",
    srcs = ["deferred_module.py"],
    deps = [
        ":special_methods",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:nest_util",
    ],
)

py_test(
    name = "deferred_module_test",
    size = "small",
    srcs = ["deferred_module_test.py"],
    deps = [
        ":deferred_module",
        # tensorflow:tensorflow_no_contrib dep,
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/distributions:gamma",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_library(
    name = "jit_public_methods",
    srcs = ["jit_public_methods.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:kullback_leibler",
    ],
)

multi_substrate_py_test(
    name = "jit_public_methods_test",
    size = "small",
    srcs = ["jit_public_methods_test.py"],
    disabled_substrates = ["numpy"],
    deps = [
        ":jit_public_methods",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

py_library(
    name = "special_methods",
    srcs = ["special_methods.py"],
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "trainable",
    srcs = ["trainable.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:trainable_state_util",
    ],
)

multi_substrate_py_test(
    name = "trainable_test",
    srcs = ["trainable_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    deps = [
        ":util",
        # optax dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:fill_scale_tril",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:scale_matvec_tril",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:dirichlet",
        "//tensorflow_probability/python/distributions:gamma",
        "//tensorflow_probability/python/distributions:geometric",
        "//tensorflow_probability/python/distributions:kumaraswamy",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/distributions:wishart",
        "//tensorflow_probability/python/internal:structural_tuple",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/math:minimize",
    ],
)

py_library(
    name = "composite_tensor",
    srcs = ["composite_tensor.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "composite_tensor_test",
    srcs = ["composite_tensor_test.py"],
    deps = [
        ":composite_tensor",
        # numpy dep,
        # six dep,
        # tensorflow:tensorflow_no_contrib dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:finite_discrete",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:mixture_same_family",
        "//tensorflow_probability/python/distributions:mvn_linear_operator",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:onehot_categorical",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/layers:distribution_layer",
    ],
)
