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
#   TensorFlow Probability examples.

# py_test rules are rewritten to py3_test on export to OSS, for reasons of
# reversibility of import<->export transforms. This is only necessary until we
# eliminate all py2and3_test targets, after which everything can safely use
# py_test, interanlly and externally, and no rewrites will be needed.
# @unused  <-- satisfy buildifier errors
load("//tensorflow_probability/python:build_defs.bzl", "py3_test")

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "bayesian_neural_network",
    srcs = ["bayesian_neural_network.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bayesian_neural_network_lib",
    ],
)

py_library(
    name = "bayesian_neural_network_lib",
    srcs = ["bayesian_neural_network.py"],
    srcs_version = "PY3",
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # seaborn dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py3_test(
    name = "bayesian_neural_network_test",
    size = "medium",
    srcs = ["bayesian_neural_network.py"],
    args = [
        "--fake_data",
        "--num_epochs=1",
        "--viz_steps=1000",
    ],
    main = "bayesian_neural_network.py",
    python_version = "PY3",
    tags = [
        "tf2-kokoro-broken",
        # TODO(b/147689726) Re-enable this test after contrib references are
        # rm'ed.
        "notap",
    ],
    deps = [
        ":bayesian_neural_network_lib",
    ],
)

py_binary(
    name = "grammar_vae",
    srcs = ["grammar_vae.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":grammar_vae_lib",
    ],
)

py_library(
    name = "grammar_vae_lib",
    srcs = ["grammar_vae.py"],
    srcs_version = "PY3",
    deps = [
        # absl/flags dep,
        # numpy dep,
        # scipy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py3_test(
    name = "grammar_vae_test",
    size = "small",
    srcs = ["grammar_vae.py"],
    args = [
        "--max_steps=5",
        "--latent_size=2",
        "--num_units=3",
    ],
    main = "grammar_vae.py",
    python_version = "PY3",
    deps = [
        ":grammar_vae_lib",
    ],
)

py_binary(
    name = "disentangled_vae",
    srcs = ["disentangled_vae.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":disentangled_vae_lib",
    ],
)

py_library(
    name = "disentangled_vae_lib",
    srcs = ["disentangled_vae.py"],
    srcs_version = "PY3",
    deps = [
        ":sprites_dataset",
        # absl:app dep,
        # absl/flags dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py3_test(
    name = "disentangled_vae_dry_run_test",
    size = "medium",
    srcs = ["disentangled_vae.py"],
    args = [
        "--fake_data",
        "--batch_size=2",
        "--hidden_size=3",
        "--latent_size_static=4",
        "--latent_size_dynamic=5",
        "--log_steps=1",
        "--max_steps=2",
        "--enable_debug_logging",
    ],
    main = "disentangled_vae.py",
    python_version = "PY3",
    tags = [
        "no-oss-ci",
        "tf2-kokoro-broken",
        # TODO(b/147689726) Re-enable this test after contrib references are
        # rm'ed.
        "notap",
    ],
    deps = [
        ":disentangled_vae_lib",
    ],
)

py3_test(
    name = "disentangled_vae_test",
    size = "medium",
    srcs = ["disentangled_vae_test.py"],
    main = "disentangled_vae_test.py",
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "no-oss-ci",
        "tf2-broken",
        "tf2-kokoro-broken",
    ],
    deps = [
        ":disentangled_vae_lib",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_binary(
    name = "deep_exponential_family",
    srcs = ["deep_exponential_family.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deep_exponential_family_lib",
    ],
)

py_library(
    name = "deep_exponential_family_lib",
    srcs = ["deep_exponential_family.py"],
    srcs_version = "PY3",
    deps = [
        # absl/flags dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py3_test(
    name = "deep_exponential_family_test",
    size = "small",
    srcs = ["deep_exponential_family.py"],
    args = [
        "--fake_data",
        "--max_steps=5",
        "--layer_sizes=5,3,2",
    ],
    main = "deep_exponential_family.py",
    python_version = "PY3",
    tags = [
        "tf2-broken",
        "tf2-kokoro-broken",
    ],
    deps = [
        ":deep_exponential_family_lib",
    ],
)

py_binary(
    name = "latent_dirichlet_allocation_distributions",
    srcs = ["latent_dirichlet_allocation_distributions.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":latent_dirichlet_allocation_distributions_lib",
    ],
)

py_library(
    name = "latent_dirichlet_allocation_distributions_lib",
    srcs = ["latent_dirichlet_allocation_distributions.py"],
    srcs_version = "PY3",
    deps = [
        # absl/flags dep,
        # numpy dep,
        # scipy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions",
    ],
)

py3_test(
    name = "latent_dirichlet_allocation_distributions_test",
    size = "small",
    srcs = ["latent_dirichlet_allocation_distributions.py"],
    args = [
        "--fake_data",
        "--max_steps=5",
        "--delete_existing",
        "--viz_steps=5",
        "--learning_rate=1e-7",
    ],
    main = "latent_dirichlet_allocation_distributions.py",
    python_version = "PY3",
    deps = [
        ":latent_dirichlet_allocation_distributions_lib",
    ],
)

py_binary(
    name = "logistic_regression",
    srcs = ["logistic_regression.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":logistic_regression_lib",
    ],
)

py_library(
    name = "logistic_regression_lib",
    srcs = ["logistic_regression.py"],
    srcs_version = "PY3",
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py3_test(
    name = "logistic_regression_test",
    size = "small",
    srcs = [
        "logistic_regression.py",
    ],
    args = [
        "--num_examples=32",
        "--batch_size=8",
    ],
    main = "logistic_regression.py",
    python_version = "PY3",
    tags = ["tf2-kokoro-broken"],
    deps = [
        ":logistic_regression_lib",
    ],
)

py_library(
    name = "sprites_dataset",
    srcs = ["sprites_dataset.py"],
    srcs_version = "PY3",
    deps = [
        # absl/flags dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_binary(
    name = "vae",
    srcs = ["vae.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":vae_lib",
    ],
)

py_library(
    name = "vae_lib",
    srcs = ["vae.py"],
    srcs_version = "PY3",
    deps = [
        # absl/flags dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions",
    ],
)

py3_test(
    name = "vae_test",
    size = "medium",
    srcs = ["vae.py"],
    args = [
        "--fake_data",
        "--max_steps=5",
        "--delete_existing",
        "--viz_steps=5",
        "--learning_rate=1e-7",
    ],
    main = "vae.py",
    python_version = "PY3",
    deps = [
        ":vae_lib",
    ],
)

py_binary(
    name = "vq_vae",
    srcs = ["vq_vae.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":vq_vae_lib",
    ],
)

py_library(
    name = "vq_vae_lib",
    srcs = ["vq_vae.py"],
    srcs_version = "PY3",
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py3_test(
    name = "vq_vae_test",
    size = "medium",
    srcs = ["vq_vae.py"],
    args = [
        "--mnist_type='fake_data'",
        "--max_steps=2",
        "--base_depth=2",
    ],
    main = "vq_vae.py",
    python_version = "PY3",
    tags = [
        "tf2-kokoro-broken",
        # TODO(b/147689726) Re-enable this test after contrib references are
        # rm'ed.
        "notap",
    ],
    deps = [
        ":vq_vae_lib",
    ],
)

py_binary(
    name = "cifar10_bnn",
    srcs = ["cifar10_bnn.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cifar10_bnn_lib",
    ],
)

py_library(
    name = "cifar10_bnn_lib",
    srcs = ["cifar10_bnn.py"],
    srcs_version = "PY3",
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/examples/models",
    ],
)

py3_test(
    name = "cifar10_bnn_test",
    size = "medium",
    srcs = ["cifar10_bnn.py"],
    args = [
        "--fake_data",
        "--epochs=1",
        "--batch_size=5",
    ],
    main = "cifar10_bnn.py",
    python_version = "PY3",
    tags = [
        "tf2-broken",
        "tf2-kokoro-broken",
    ],
    deps = [
        ":cifar10_bnn_lib",
    ],
)
