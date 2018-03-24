# TensorFlow Probability

TensorFlow Probability is a library for probabilistic reasoning and statistical
analysis in TensorFlow. As part of the TensorFlow ecosystem, TensorFlow
Probability provides integration of probabilistic methods with deep networks,
gradient-based inference via automatic differentiation, and scalability to
large datasets and models via hardware acceleration (e.g., GPUs) and distributed
computation.

The library consists of the following modules:

* *Distributions* (`tfp.distributions`, `tfp.trainable_distributions`):
  Probability distributions with efficient, composable manipulations.
* *Edward2* (`tfp.edward2`): A probabilistic programming language, which enables
  flexible probabilistic models and flexible computation for their training and
  testing.
* *Layers* (`tfp.layers`): Neural network layers with uncertainty
  over the functions they represent, extending TensorFlow Layers.
* *Monte Carlo* (`tfp.mcmc`, `tfp.optimizers`, `tfp.monte_carlo`): Algorithms
  for approximate Bayesian inference via sampling.
* *Variational Inference* (`tfp.vi`): Algorithms for approximate Bayesian
  inference via optimization.
* *Examples* (`tfp.examples`): End-to-end implementations of probabilistic
  reasoning using TensorFlow Probability.

TensorFlow Probability is under active development. Interfaces may change at any
time.

## Installation

To install the latest version, run the following:

```shell
pip install --user --upgrade tfp-nightly    # depends on tensorflow (CPU-only)
```

We also provide a GPU-enabled version.

```shell
pip install --user --upgrade tfp-nightly-gpu  # depends on tensorflow-gpu (GPU enabled)
```

Currently, TensorFlow Probability does not contain any GPU-specific code. The
primary difference between these packages is that `tensorflow-probability-gpu`
depends on a GPU-enabled version of TensorFlow.

To force a Python 3-specific install, replace `pip` with `pip3` in the above
commands. For additional installation help, guidance installing prerequisites,
and (optionally) setting up virtual environments, see the [TensorFlow
installation guide](https://www.tensorflow.org/install).

You can also install from source. This requires the [Bazel](
https://bazel.build/) build system.

```shell
# sudo apt-get install bazel git python-pip  # Ubuntu; others, see above links.
git clone https://github.com/tensorflow/probability.git
cd probability
bazel build --config=opt --copt=-O3 --copt=-march=native :pip_pkg
PKGDIR=$(mktemp -d)
./bazel-bin/pip_pkg $PKGDIR
pip install --user --upgrade $PKGDIR/*.whl
```

## Usage

Access the library using

```python
import tensorflow_probability as tfp
```

## Examples

See the `tfp.examples` module for examples of end-to-end implementations. They
can also be run under command line: for example, run

`python -m tensorflow_probability.examples.vae`

to train a variational auto-encoder to generate MNIST digits. See the
[`examples/`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/)
directory for more details.

## Contributing

We're eager to collaborate with you! Feel free to [open an issue on
GitHub](https://github.com/tensorflow/probability/issues) and/or send us your
pull requests. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more details.
This project adheres to TensorFlow's [code of conduct](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold this code.
