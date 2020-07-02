# TensorFlow Probability

TensorFlow Probability is a library for probabilistic reasoning and statistical
analysis in TensorFlow. As part of the TensorFlow ecosystem, TensorFlow
Probability provides integration of probabilistic methods with deep networks,
gradient-based inference using automatic differentiation, and scalability to
large datasets and models with hardware acceleration (GPUs) and distributed
computation.

To get started with TensorFlow Probability, see the [install guide](./install)
and view the
[Python notebook tutorials](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/){:.external}.

## Components

Our probabilistic machine learning tools are structured as follows:

### Layer 0: TensorFlow

*Numerical operations*—in particular, the `LinearOperator`
class—enables matrix-free implementations that can exploit a particular structure
(diagonal, low-rank, etc.) for efficient computation. It is built and maintained
by the TensorFlow Probability team and is part of
[`tf.linalg`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops/linalg)
in core TensorFlow.

### Layer 1: Statistical Building Blocks

*   *Distributions*
    ([`tfp.distributions`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions)):
    A large collection of probability distributions and related statistics with
    batch and
    [broadcasting](https://docs.scipy.org/doc/numpy-1.14.0/user/basics.broadcasting.html){:.external}
    semantics.
*   *Bijectors*
    ([`tfp.bijectors`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/bijectors)):
    Reversible and composable transformations of random variables. Bijectors
    provide a rich class of transformed distributions, from classical examples
    like the
    [log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution){:.external}
    to sophisticated deep learning models such as
    [masked autoregressive flows](https://arxiv.org/abs/1705.07057){:.external}.

### Layer 2: Model Building

*   Joint Distributions (e.g.,
    [`tfp.distributions.JointDistributionSequential`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/joint_distribution_sequential.py)):
    Joint distributions over one or more possibly-interdependent distributions.
    For an introduction to modeling with TFP's `JointDistribution`s, check out
    [this colab](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Modeling_with_JointDistribution.ipynb)
*   *Probabilistic layers*
    ([`tfp.layers`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers)):
    Neural network layers with uncertainty over the functions they represent,
    extending TensorFlow layers.

### Layer 3: Probabilistic Inference

*   *Markov chain Monte Carlo*
    ([`tfp.mcmc`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc)):
    Algorithms for approximating integrals via sampling. Includes
    [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo){:.external},
    random-walk Metropolis-Hastings, and the ability to build custom transition
    kernels.
*   *Variational Inference*
    ([`tfp.vi`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/vi)):
    Algorithms for approximating integrals through optimization.
*   *Optimizers*
    ([`tfp.optimizer`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/optimizer)):
    Stochastic optimization methods, extending TensorFlow Optimizers. Includes
    [Stochastic Gradient Langevin Dynamics](http://www.icml-2011.org/papers/398_icmlpaper.pdf){:.external}.
*   *Monte Carlo*
    ([`tfp.monte_carlo`](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/monte_carlo)):
    Tools for computing Monte Carlo expectations.

TensorFlow Probability is under active development and interfaces may change.

## Examples

In addition to the
[Python notebook tutorials](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/){:.external}
listed in the navigation, there are some example scripts available:

* [Variational Autoencoders](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/vae.py)
  —Representation learning with a latent code and variational inference.
* [Vector-Quantized Autoencoder](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/vq_vae.py)
  —Discrete representation learning with vector quantization.
* [Bayesian Neural Networks](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/bayesian_neural_network.py)
  —Neural networks with uncertainty over their weights.
* [Bayesian Logistic Regression](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/logistic_regression.py)
  —Bayesian inference for binary classification.

## Report issues

Report bugs or feature requests using the
[TensorFlow Probability issue tracker](https://github.com/tensorflow/probability/issues).
