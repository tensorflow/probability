# Copyright 2024 The TensorFlow Probability Authors.
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
# Algorithms for training Gaussian Processes in time less than O(n^3)

# Placeholder: py_library
# Placeholder: py_test

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = [
        "//health/population/haigiography/temporal:__subpackages__",
        "//tensorflow_probability:__subpackages__",
        # vizier:__subpackages__ dep,
    ],
)

py_library(
    name = "fastgp.jax",
    srcs = ["__init__.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":fast_gp",
        ":fast_gprm",
        ":fast_log_det",
        ":fast_mtgp",
        ":linalg",
        ":linear_operator_sum",
        ":mbcg",
        ":partial_lanczos",
        ":preconditioners",
        ":schur_complement",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

# Dummy libraries to satisfy the multi_substrate_py_library deps of
# tfp/python/experimental:experimental.
py_library(
    name = "fastgp",
#     strict_deps = False,  # DisableOnExport
    deps = [],
)

py_library(
    name = "fastgp.numpy",
#     strict_deps = False,  # DisableOnExport
    deps = [],
)

py_library(
    name = "mbcg",
    srcs = ["mbcg.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
    ],
)

py_test(
    name = "mbcg_test",
    srcs = ["mbcg_test.py"],
    deps = [
        ":mbcg",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "fast_gp",
    srcs = ["fast_gp.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":fast_log_det",
        ":mbcg",
        ":preconditioners",
        "//tensorflow_probability/python/bijectors:softplus.jax",
        "//tensorflow_probability/python/distributions:distribution.jax",
        "//tensorflow_probability/python/distributions:gaussian_process_regression_model.jax",
        "//tensorflow_probability/python/distributions/internal:stochastic_process_util.jax",
        "//tensorflow_probability/python/internal:dtype_util.jax",
        "//tensorflow_probability/python/internal:parameter_properties.jax",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util.jax",
        "//tensorflow_probability/python/internal/backend/jax",
        "//tensorflow_probability/python/mcmc:sample_halton_sequence.jax",
    ],
)

py_test(
    name = "fast_gp_test",
    srcs = ["fast_gp_test.py"],
    shard_count = 3,
    deps = [
        ":fast_gp",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability/substrates:jax",
    ],
)

py_library(
    name = "fast_mtgp",
    srcs = ["fast_mtgp.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":fast_gp",
        ":fast_log_det",
        ":linear_operator_sum",
        ":mbcg",
        ":preconditioners",
        "//tensorflow_probability/python/distributions:distribution.jax",
        "//tensorflow_probability/python/distributions/internal:stochastic_process_util.jax",
        "//tensorflow_probability/python/experimental/psd_kernels:multitask_kernel.jax",
        "//tensorflow_probability/python/internal:dtype_util.jax",
        "//tensorflow_probability/python/internal:prefer_static.jax",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util.jax",
        "//tensorflow_probability/python/internal/backend/jax",
    ],
)

py_test(
    name = "fast_mtgp_test",
    srcs = ["fast_mtgp_test.py"],
    shard_count = 3,
    deps = [
        ":fast_mtgp",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability/substrates:jax",
    ],
)

py_library(
    name = "fast_gprm",
    srcs = ["fast_gprm.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":fast_gp",
        ":preconditioners",
        ":schur_complement",
        "//tensorflow_probability/python/bijectors:softplus.jax",
        "//tensorflow_probability/python/distributions/internal:stochastic_process_util.jax",
        "//tensorflow_probability/python/internal:dtype_util.jax",
        "//tensorflow_probability/python/internal:nest_util.jax",
        "//tensorflow_probability/python/internal:parameter_properties.jax",
    ],
)

py_test(
    name = "fast_gprm_test",
    srcs = ["fast_gprm_test.py"],
    deps = [
        ":fast_gprm",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability/python/internal:test_util.jax",
        "//tensorflow_probability/substrates:jax",
    ],
)

py_library(
    name = "linalg",
    srcs = ["linalg.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":partial_lanczos",
        # jax/experimental:sparse dep,
        "//tensorflow_probability/python/internal/backend/jax",
    ],
)

py_test(
    name = "linalg_test",
    srcs = ["linalg_test.py"],
    shard_count = 3,
    deps = [
        ":linalg",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        "//tensorflow_probability/substrates:jax",
    ],
)

py_library(
    name = "partial_lanczos",
    srcs = ["partial_lanczos.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":mbcg",
        # scipy dep,
        "//tensorflow_probability/python/internal/backend/jax",
    ],
)

py_test(
    name = "partial_lanczos_test",
    srcs = ["partial_lanczos_test.py"],
    deps = [
        ":mbcg",
        ":partial_lanczos",
        # jax dep,
        # numpy dep,
        "//tensorflow_probability/python/internal:test_util.jax",
        "//tensorflow_probability/substrates:jax",
    ],
)

py_library(
    name = "fast_log_det",
    srcs = ["fast_log_det.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":mbcg",
        ":partial_lanczos",
        ":preconditioners",
        # numpy dep,
        # scipy dep,
    ],
)

py_test(
    name = "fast_log_det_test",
    srcs = ["fast_log_det_test.py"],
    shard_count = 3,
    deps = [
        ":fast_log_det",
        ":preconditioners",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability/python/internal:test_util.jax",
        "//tensorflow_probability/substrates:jax",
    ],
)

py_library(
    name = "linear_operator_sum",
    srcs = ["linear_operator_sum.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
    ],
)

py_library(
    name = "preconditioners",
    srcs = ["preconditioners.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":linalg",
        ":linear_operator_sum",
        "//tensorflow_probability/python/internal/backend/jax",
        "//tensorflow_probability/python/math:linalg.jax",
    ],
)

py_test(
    name = "preconditioners_test",
    srcs = ["preconditioners_test.py"],
    shard_count = 3,
    deps = [
        ":preconditioners",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        "//tensorflow_probability/python/internal/backend/jax",
    ],
)

py_library(
    name = "schur_complement",
    srcs = ["schur_complement.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":preconditioners",
        "//tensorflow_probability/python/bijectors:softplus.jax",
        "//tensorflow_probability/python/internal:distribution_util.jax",
        "//tensorflow_probability/python/internal:dtype_util.jax",
        "//tensorflow_probability/python/internal:nest_util.jax",
    ],
)

py_test(
    name = "schur_complement_test",
    srcs = ["schur_complement_test.py"],
    deps = [
        ":schur_complement",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability/substrates:jax",
    ],
)
