# Copyright 2023 The TensorFlow Probability Authors.
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
# Code for AutoBNN.  See README.md for more information.

# Placeholder: py_library
# Placeholder: py_test

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "bnn",
    srcs = ["bnn.py"],
    deps = [
        ":likelihoods",
        # flax:core dep,
        # jax dep,
        # jaxtyping dep,
        "//tensorflow_probability/python/distributions:distribution.jax",
    ],
)

py_test(
    name = "bnn_test",
    srcs = ["bnn_test.py"],
    # TODO(b/320723712): enable this test in OSS.
    tags = ["no-oss-ci"],
    deps = [
        ":bnn",
        # absl/testing:absltest dep,
        # google/protobuf:use_fast_cpp_protos dep,
        # jax dep,
        "//tensorflow_probability:jax",
        "//tensorflow_probability/python/distributions:lognormal.jax",
        "//tensorflow_probability/python/distributions:normal.jax",
    ],
)

py_library(
    name = "bnn_tree",
    srcs = ["bnn_tree.py"],
    deps = [
        ":bnn",
        ":kernels",
        ":operators",
        ":util",
        # flax:core dep,
        # jax dep,
    ],
)

py_test(
    name = "bnn_tree_test",
    timeout = "long",
    srcs = ["bnn_tree_test.py"],
    shard_count = 3,
    deps = [
        ":bnn_tree",
        ":kernels",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # flax dep,
        # google/protobuf:use_fast_cpp_protos dep,
        # jax dep,
    ],
)

py_library(
    name = "estimators",
    srcs = ["estimators.py"],
    deps = [
        ":bnn",
        ":likelihoods",
        ":models",
        ":training_util",
        # jax dep,
        # jaxtyping dep,
    ],
)

py_test(
    name = "estimators_test",
    srcs = ["estimators_test.py"],
    deps = [
        ":estimators",
        ":kernels",
        ":operators",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "kernels",
    srcs = ["kernels.py"],
    deps = [
        ":bnn",
        # flax dep,
        # flax:core dep,
        # jax dep,
        "//tensorflow_probability/python/distributions:lognormal.jax",
        "//tensorflow_probability/python/distributions:normal.jax",
        "//tensorflow_probability/python/distributions:student_t.jax",
        "//tensorflow_probability/python/distributions:uniform.jax",
    ],
)

py_test(
    name = "kernels_test",
    srcs = ["kernels_test.py"],
    deps = [
        ":kernels",
        ":util",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # google/protobuf:use_fast_cpp_protos dep,
        # jax dep,
        "//tensorflow_probability/python/distributions:lognormal.jax",
    ],
)

py_library(
    name = "likelihoods",
    srcs = ["likelihoods.py"],
    deps = [
        # flax:core dep,
        # jax dep,
        # jaxtyping dep,
        "//tensorflow_probability:jax",
        "//tensorflow_probability/python/bijectors:softplus.jax",
        "//tensorflow_probability/python/distributions:distribution.jax",
        "//tensorflow_probability/python/distributions:inflated.jax",
        "//tensorflow_probability/python/distributions:logistic.jax",
        "//tensorflow_probability/python/distributions:lognormal.jax",
        "//tensorflow_probability/python/distributions:negative_binomial.jax",
        "//tensorflow_probability/python/distributions:normal.jax",
        "//tensorflow_probability/python/distributions:transformed_distribution.jax",
    ],
)

py_test(
    name = "likelihoods_test",
    srcs = ["likelihoods_test.py"],
    deps = [
        ":likelihoods",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
        ":bnn",
        ":bnn_tree",
        ":kernels",
        ":likelihoods",
        ":operators",
        # jax dep,
    ],
)

py_test(
    name = "models_test",
    timeout = "long",
    srcs = ["models_test.py"],
    shard_count = 3,
    deps = [
        ":likelihoods",
        ":models",
        ":operators",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
    ],
)

py_library(
    name = "operators",
    srcs = ["operators.py"],
    deps = [
        ":bnn",
        ":likelihoods",
        # flax:core dep,
        # jax dep,
        "//tensorflow_probability:jax",
        "//tensorflow_probability/python/bijectors:chain.jax",
        "//tensorflow_probability/python/bijectors:scale.jax",
        "//tensorflow_probability/python/bijectors:shift.jax",
        "//tensorflow_probability/python/distributions:beta.jax",
        "//tensorflow_probability/python/distributions:dirichlet.jax",
        "//tensorflow_probability/python/distributions:half_normal.jax",
        "//tensorflow_probability/python/distributions:normal.jax",
        "//tensorflow_probability/python/distributions:transformed_distribution.jax",
    ],
)

py_test(
    name = "operators_test",
    srcs = ["operators_test.py"],
    deps = [
        ":kernels",
        ":operators",
        ":util",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # google/protobuf:use_fast_cpp_protos dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability/python/distributions:distribution.jax",
    ],
)

py_library(
    name = "training_util",
    srcs = ["training_util.py"],
    deps = [
        # bayeux dep,
        # jax dep,
        # jaxtyping dep,
        # matplotlib dep,
        # numpy dep,
        # pandas dep,
        "//tensorflow_probability:jax",
        "//tensorflow_probability/python/experimental/autobnn:bnn",
        "//tensorflow_probability/python/experimental/autobnn:util",
        "//tensorflow_probability/python/experimental/timeseries:metrics",
    ],
)

py_test(
    name = "training_util_test",
    srcs = ["training_util_test.py"],
    deps = [
        ":training_util",
        # chex dep,
        # google/protobuf:use_fast_cpp_protos dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability/python/experimental/autobnn:kernels",
        "//tensorflow_probability/python/experimental/autobnn:operators",
        "//tensorflow_probability/python/experimental/autobnn:util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        ":bnn",
        # jax dep,
        # numpy dep,
        # scipy dep,
        "//tensorflow_probability/python/distributions:distribution.jax",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        ":kernels",
        ":util",
        # google/protobuf:use_fast_cpp_protos dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)
