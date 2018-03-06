# TensorFlow Probability

This package collects tools for probabilistic reasoning in TensorFlow. It is
intended to serve as a hub for development of modeling tools, inference
algorithms, useful models, and general statistical computation. Taking
advantage of the TensorFlow ecosystem allows straightforward combination of
probabilistic methods with deep networks, gradient-based inference via
automatic differentiation, and scalability to large datasets and models via
hardware acceleration (e.g., GPUs) and distributed computation.

Contents of this package currently include:

* *Sampling algorithms* E.g., `tfp.mcmc.sample_chain`,
  `tfp.mcmc.HamiltonianMonteCarlo`, `tfp.monte_carlo.expectation`.
* *Example models* (`tfp.examples`): implementations of common probability
  models in TensorFlow using tools from this package and from
  `tf.contrib.distributions`.

Contents of this repository should be considered under active development;
interfaces may change at any time. We welcome external contributions! See
[Contributing](#contributing) for details.

## Installation

As simple as:

```shell
pip install tfp-nightly --user --upgrade     # depends on tensorflow (CPU-only)
```

We also provide a GPU-enabled flavor:

```shell
pip install tfp-nightly-gpu --user --upgrade # depends on tensorflow-gpu (GPU enabled)
```

TensorFlow Probability does not currently contain any GPU-specific code; the
primary difference between these packages is that `tensorflow-probability-gpu`
depends on a GPU-enabled version of TensorFlow.

To force a Python 3-specific install, replace `pip` with `pip3` in the above
commands. For additional installation help, guidance installing prerequisites,
and (optionally) setting up virtual environments, see the [TensorFlow
installation guide](https://www.tensorflow.org/install).

You can also install from source. This requires the [Bazel](
https://bazel.build/) build system.

```shell
git clone https://github.com/tensorflow/probability.git
cd probability
bazel build --config=opt --copt=-O3 --copt=-march=native :pip_pkg
./bazel-bin/pip_pkg /tmp/tensorflow_probability_pkg
pip install /tmp/tensorflow_probability_pkg/*.whl --user --upgrade
```

## Examples

It is often easiest to learn by example. The `examples/` directory contains
reference implementations of common probabilistic models and demonstrates
idiomatic styles for building probability models in TensorFlow. Example code may
be run directly from the command line, e.g.,

`python -m tensorflow_probability.examples.weight_uncertainty.mnist_deep_nn`

to train a Bayesian deep network to classify MNIST digits. See the
[examples](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/)
directory for the list of example implementations.

## Usage

After you've installed `tensorflow_probability`, functions can be accessed as:

```python
import tensorflow_probability as tfp
```

## Contributing

We're eager to collaborate with you! Feel free to [open an issue on
GitHub](https://github.com/tensorflow/probability/issues) and/or send us your
pull requests. See [our contribution doc](CONTRIBUTING.md) for more details.
