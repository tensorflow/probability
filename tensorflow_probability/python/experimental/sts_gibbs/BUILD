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
#   Gibbs sampling for Bayesian structural time series models

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

multi_substrate_py_library(
    name = "sts_gibbs",
    srcs = ["__init__.py"],
    deps = [
        ":dynamic_spike_and_slab",
        ":gibbs_sampler",
        ":spike_and_slab",
    ],
)

multi_substrate_py_test(
    name = "benchmarks_test",
    size = "medium",
    srcs = ["benchmarks_test.py"],
    disabled_substrates = ["numpy"],
    tags = ["notap"],
    deps = [
        ":gibbs_sampler",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:inverse_gamma",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/sts/internal:missing_values_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "gibbs_sampler",
    srcs = ["gibbs_sampler.py"],
    deps = [
        ":dynamic_spike_and_slab",
        ":spike_and_slab",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:square",
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:gamma",
        "//tensorflow_probability/python/distributions:inverse_gamma",
        "//tensorflow_probability/python/distributions:mixture_same_family",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_linear_operator",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:normal_conjugate_posteriors",
        "//tensorflow_probability/python/experimental/distributions:mvn_precision_factor_linop",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/sts/components:local_level",
        "//tensorflow_probability/python/sts/components:local_linear_trend",
        "//tensorflow_probability/python/sts/components:regression",
        "//tensorflow_probability/python/sts/components:seasonal",
        "//tensorflow_probability/python/sts/components:sum",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "gibbs_sampler_test",
    size = "large",
    srcs = ["gibbs_sampler_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 4,
    deps = [
        ":gibbs_sampler",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:inverse_gamma",
        "//tensorflow_probability/python/distributions:linear_gaussian_ssm",
        "//tensorflow_probability/python/distributions:lognormal",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:student_t",
        "//tensorflow_probability/python/experimental/distributions:mvn_precision_factor_linop",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/stats:sample_stats",
        "//tensorflow_probability/python/sts:forecast",
        "//tensorflow_probability/python/sts/components:local_level",
        "//tensorflow_probability/python/sts/components:local_linear_trend",
        "//tensorflow_probability/python/sts/components:regression",
        "//tensorflow_probability/python/sts/components:seasonal",
        "//tensorflow_probability/python/sts/components:sum",
        "//tensorflow_probability/python/sts/internal:missing_values_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "dynamic_spike_and_slab",
    srcs = ["dynamic_spike_and_slab.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:gamma",
        "//tensorflow_probability/python/distributions:inverse_gamma",
        "//tensorflow_probability/python/distributions:joint_distribution_auto_batched",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/experimental/distributions:mvn_precision_factor_linop",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:vectorization_util",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "dynamic_spike_and_slab_test",
    size = "medium",
    srcs = ["dynamic_spike_and_slab_test.py"],
    disabled_substrates = [
        "numpy",
        "jax",
    ],
    deps = [
        ":dynamic_spike_and_slab",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:inverse_gamma",
        "//tensorflow_probability/python/distributions:normal_conjugate_posteriors",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc/internal:util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "spike_and_slab",
    srcs = ["spike_and_slab.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:inverse_gamma",
        "//tensorflow_probability/python/distributions:joint_distribution_auto_batched",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/experimental/distributions:mvn_precision_factor_linop",
        "//tensorflow_probability/python/internal:broadcast_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:vectorization_util",
        "//tensorflow_probability/python/math:linalg",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "spike_and_slab_test",
    size = "medium",
    srcs = ["spike_and_slab_test.py"],
    disabled_substrates = ["numpy"],
    deps = [
        ":sts_gibbs",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:inverse_gamma",
        "//tensorflow_probability/python/distributions:normal_conjugate_posteriors",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)
