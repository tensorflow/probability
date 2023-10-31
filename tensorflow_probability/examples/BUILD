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

# Placeholder: py_library
# Placeholder: py_test
# Placeholder: py_binary

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

py_binary(
    name = "bayesian_neural_network",
    srcs = ["bayesian_neural_network.py"],
    deps = [
        ":bayesian_neural_network_lib",
    ],
)

py_library(
    name = "bayesian_neural_network_lib",
    srcs = ["bayesian_neural_network.py"],
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # seaborn dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "bayesian_neural_network_test",
    size = "medium",
    srcs = ["bayesian_neural_network.py"],
    args = [
        "--fake_data",
        "--num_epochs=1",
        "--num_monte_carlo=4",
    ],
    main = "bayesian_neural_network.py",
    tags = ["not_run:arm"],
    deps = [
        ":bayesian_neural_network_lib",
    ],
)

py_binary(
    name = "disentangled_vae",
    srcs = ["disentangled_vae.py"],
    deps = [
        ":disentangled_vae_lib",
    ],
)

py_library(
    name = "disentangled_vae_lib",
    srcs = ["disentangled_vae.py"],
    deps = [
        ":sprites_dataset",
        # absl:app dep,
        # absl/flags dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tf_keras",
    ],
)

py_test(
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
    deps = [
        ":disentangled_vae_lib",
    ],
)

py_test(
    name = "disentangled_vae_test",
    size = "medium",
    srcs = ["disentangled_vae_test.py"],
    main = "disentangled_vae_test.py",
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
    name = "logistic_regression",
    srcs = ["logistic_regression.py"],
    deps = [
        ":logistic_regression_lib",
    ],
)

py_library(
    name = "logistic_regression_lib",
    srcs = ["logistic_regression.py"],
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
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
    tags = ["tf2-kokoro-broken"],
    deps = [
        ":logistic_regression_lib",
    ],
)

py_library(
    name = "sprites_dataset",
    srcs = ["sprites_dataset.py"],
    deps = [
        # absl/flags dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_binary(
    name = "vq_vae",
    srcs = ["vq_vae.py"],
    deps = [
        ":vq_vae_lib",
    ],
)

py_library(
    name = "vq_vae_lib",
    srcs = ["vq_vae.py"],
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "vq_vae_test",
    size = "medium",
    srcs = ["vq_vae.py"],
    args = [
        "--mnist_type='fake_data'",
        "--max_steps=2",
        "--base_depth=2",
    ],
    main = "vq_vae.py",
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
    deps = [
        ":cifar10_bnn_lib",
    ],
)

py_library(
    name = "cifar10_bnn_lib",
    srcs = ["cifar10_bnn.py"],
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/examples/models",
    ],
)

py_test(
    name = "cifar10_bnn_test",
    size = "medium",
    srcs = ["cifar10_bnn.py"],
    args = [
        "--fake_data",
        "--epochs=1",
        "--batch_size=5",
    ],
    main = "cifar10_bnn.py",
    tags = [
        "tf2-broken",
        "tf2-kokoro-broken",
    ],
    deps = [
        ":cifar10_bnn_lib",
    ],
)
