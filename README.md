# TensorFlow Probability

TensorFlow Probability is a library for probabilistic reasoning and statistical
analysis in TensorFlow. As part of the TensorFlow ecosystem, TensorFlow
Probability provides integration of probabilistic methods with deep networks,
gradient-based inference via automatic differentiation, and scalability to
large datasets and models via hardware acceleration (e.g., GPUs) and distributed
computation.

Our probabilistic machine learning tools are structured as follows.

__Layer 0: TensorFlow.__ Numerical operations. In particular, the LinearOperator
class enables matrix-free implementations that can exploit special structure
(diagonal, low-rank, etc.) for efficient computation. It is built and maintained
by the TensorFlow Probability team and is now part of
[`tf.linalg`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops/linalg)
in core TF.

__Layer 1: Statistical Building Blocks__

* Distributions ([`tf.contrib.distributions`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distributions/python/ops),
  [`tf.distributions`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops/distributions)):
  A large collection of probability
  distributions and related statistics with batch and
  [broadcasting](https://docs.scipy.org/doc/numpy-1.14.0/user/basics.broadcasting.html)
  semantics.
* Bijectors ([`tf.contrib.distributions.bijectors`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distributions/python/ops/bijectors)):
  Reversible and composable transformations of random variables. Bijectors
  provide a rich class of transformed distributions, from classical examples
  like the
  [log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution)
  to sophisticated deep learning models such as
  [masked autoregressive flows](https://arxiv.org/abs/1705.07057).

__Layer 2: Model Building__

* Edward2 ([`tfp.edward2`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2)):
  A probabilistic programming language for specifying flexible probabilistic
  models as programs.
* Probabilistic Layers ([`tfp.layers`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers)):
  Neural network layers with uncertainty over the functions they represent,
  extending TensorFlow Layers.
* Trainable Distributions ([`tfp.trainable_distributions`](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/trainable_distributions.py)):
  Probability distributions parameterized by a single Tensor, making it easy to
  build neural nets that output probability distributions.

__Layer 3: Probabilistic Inference__

* Markov chain Monte Carlo ([`tfp.mcmc`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc)):
  Algorithms for approximating integrals via sampling. Includes
  [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo),
  random-walk Metropolis-Hastings, and the ability to build custom transition
  kernels.
* Variational Inference ([`tfp.vi`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/vi)):
  Algorithms for approximating integrals via optimization.
* Optimizers ([`tfp.optimizer`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/optimizer)):
  Stochastic optimization methods, extending TensorFlow Optimizers. Includes
  [Stochastic Gradient Langevin Dynamics](http://www.icml-2011.org/papers/398_icmlpaper.pdf).
* Monte Carlo ([`tfp.monte_carlo`](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/monte_carlo.py)):
  Tools for computing Monte Carlo expectations.

TensorFlow Probability is under active development. Interfaces may change at any
time.

## Examples

See [`tensorflow_probability/examples/`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/)
for end-to-end examples. It includes tutorial notebooks such as:

* [Linear Mixed Effects Models](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Linear_Mixed_Effects_Models.ipynb).
  A hierarchical linear model for sharing statistical strength across examples.
* [Eight Schools](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Eight_Schools.ipynb).
  A hierarchical normal model for exchangeable treatment effects.
* [Gaussian Copulas](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Gaussian_Copula.ipynb).
  Probability distributions for capturing dependence across random variables.
* [Understanding TensorFlow Distributions Shapes](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb).
  How to distinguish between samples, batches, and events for arbitrarily shaped
  probabilistic computations.

It also includes example scripts such as:

* [Variational Autoencoders](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/vae.py).
  Representation learning with a latent code and variational inference.
* [Bayesian Neural Networks](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/bayesian_neural_network.py).
  Neural networks with uncertainty over their weights.
* [Bayesian Logistic Regression](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/logistic_regression.py).
  Bayesian inference for binary classification.

## Installation

To install the latest version, run the following:

```shell
pip install --user --upgrade tfp-nightly    # depends on tensorflow (CPU-only)
```

TensorFlow Probability depends on a current nightly release of TensorFlow
(`tf-nightly`); the `--upgrade` flag ensures you'll automatically get the latest
version.

We also provide a GPU-enabled package:

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

## Community

As part of TensorFlow, we're committed to fostering an open and welcoming
environment.

* [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow): Ask
  or answer technical questions.
* [GitHub](https://github.com/tensorflow/probability/issues): Report bugs or
  make feature requests.
* [TensorFlow Blog](https://medium.com/tensorflow): Stay up to date on content
  from the TensorFlow team and best articles from the community.
* [Youtube Channel](http://youtube.com/tensorflow/): Follow TensorFlow shows.
* Mailing list: Stay tuned!

See the [TensorFlow Community](https://www.tensorflow.org/community/) page for
more details. Check out our latest publicity here:

+ [Coffee with a Googler: Probabilistic Machine Learning in TensorFlow](
  https://www.youtube.com/watch?v=BjUkL8DFH5Q)
+ [Introducing TensorFlow Probability](
  https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245)

## Contributing

We're eager to collaborate with you! See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for more details. This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

## References

+ _TensorFlow Distributions._ Joshua V. Dillon, Ian Langmore, Dustin Tran,
Eugene Brevdo, Srinivas Vasudevan, Dave Moore, Brian Patton, Alex Alemi, Matt
Hoffman, Rif A. Saurous.
[arXiv preprint arXiv:1711.10604, 2017](https://arxiv.org/abs/1711.10604).
