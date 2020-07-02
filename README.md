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

* Distributions ([`tfp.distributions`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions)):
  A large collection of probability
  distributions and related statistics with batch and
  [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  semantics. See the
  [Distributions Tutorial](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/TensorFlow_Distributions_Tutorial.ipynb).
* Bijectors ([`tfp.bijectors`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/bijectors)):
  Reversible and composable transformations of random variables. Bijectors
  provide a rich class of transformed distributions, from classical examples
  like the
  [log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution)
  to sophisticated deep learning models such as
  [masked autoregressive flows](https://arxiv.org/abs/1705.07057).

__Layer 2: Model Building__

* Joint Distributions (e.g., [`tfp.distributions.JointDistributionSequential`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/joint_distribution_sequential.py)):
    Joint distributions over one or more possibly-interdependent distributions.
    For an introduction to modeling with TFP's `JointDistribution`s, check out
    [this colab](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Modeling_with_JointDistribution.ipynb)
* Probabilistic Layers ([`tfp.layers`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers)):
  Neural network layers with uncertainty over the functions they represent,
  extending TensorFlow Layers.

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
* Monte Carlo ([`tfp.monte_carlo`](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/monte_carlo)):
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
* [Hierarchical Linear Models](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/HLM_TFP_R_Stan.ipynb).
  Hierarchical linear models compared among TensorFlow Probability, R, and Stan.
* [Bayesian Gaussian Mixture Models](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Bayesian_Gaussian_Mixture_Model.ipynb).
  Clustering with a probabilistic generative model.
* [Probabilistic Principal Components Analysis](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb).
  Dimensionality reduction with latent variables.
* [Gaussian Copulas](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Gaussian_Copula.ipynb).
  Probability distributions for capturing dependence across random variables.
* [TensorFlow Distributions: A Gentle Introduction](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/TensorFlow_Distributions_Tutorial.ipynb).
  Introduction to TensorFlow Distributions.
* [Understanding TensorFlow Distributions Shapes](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb).
  How to distinguish between samples, batches, and events for arbitrarily shaped
  probabilistic computations.
* [TensorFlow Probability Case Study: Covariance Estimation](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/TensorFlow_Probability_Case_Study_Covariance_Estimation.ipynb).
  A user's case study in applying TensorFlow Probability to estimate covariances.

It also includes example scripts such as:

* [Variational Autoencoders](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/vae.py).
  Representation learning with a latent code and variational inference.
* [Vector-Quantized Autoencoder](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/vq_vae.py).
  Discrete representation learning with vector quantization.
* [Disentangled Sequential Variational Autoencoder](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/disentangled_vae.py)
  Disentangled representation learning over sequences with variational inference.
* [Grammar Variational Autoencoder](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/grammar_vae.py).
  Representation learning over productions in a context-free grammar.
* Latent Dirichlet Allocation
  ([Distributions version](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/latent_dirichlet_allocation_distributions.py),
  Mixed membership modeling for capturing topics in a document.
+ [Deep Exponential Family](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/deep_exponential_family.py).
  A deep, sparse generative model for discovering a hierarchy of topics.
* [Bayesian Neural Networks](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/bayesian_neural_network.py).
  Neural networks with uncertainty over their weights.
* [Bayesian Logistic Regression](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/logistic_regression.py).
  Bayesian inference for binary classification.

## Installation

For additional details on installing TensorFlow, guidance installing
prerequisites, and (optionally) setting up virtual environments, see the
[TensorFlow installation guide](https://www.tensorflow.org/install).

### Stable Builds

To install the latest stable version, run the following:

```shell
# Notes:

# - The `--upgrade` flag ensures you'll get the latest version.
# - The `--user` flag ensures the packages are installed to your user directory
#   rather than the system directory.
# - TensorFlow 2 packages require a pip >= 19.0
python -m pip install --upgrade --user pip
python -m pip install --upgrade --user tensorflow tensorflow_probability
```

For CPU-only usage (and a smaller install), install with `tensorflow-cpu`.

To use a pre-2.0 version of TensorFlow, run:

```shell
python -m pip install --upgrade --user "tensorflow<2" "tensorflow_probability<0.9"
```

Note: Since [TensorFlow](https://www.tensorflow.org/install) is *not* included
as a dependency of the TensorFlow Probability package (in `setup.py`), you must
explicitly install the TensorFlow package (`tensorflow` or `tensorflow-cpu`).
This allows us to maintain one package instead of separate packages for CPU and
GPU-enabled TensorFlow. See the
[TFP release notes](https://github.com/tensorflow/probability/releases) for more
details about dependencies between TensorFlow and TensorFlow Probability.


### Nightly Builds

There are also nightly builds of TensorFlow Probability under the pip package
`tfp-nightly`, which depends on one of `tf-nightly` or `tf-nightly-cpu`.
Nightly builds include newer features, but may be less stable than the
versioned releases. Both stable and nightly docs are available
[here](https://www.tensorflow.org/probability/api_docs/python/tfp?version=nightly).

```shell
python -m pip install --upgrade --user tf-nightly tfp-nightly
```

### Installing from Source

You can also install from source. This requires the [Bazel](
https://bazel.build/) build system. It is highly recommended that you install
the nightly build of TensorFlow (`tf-nightly`) before trying to build
TensorFlow Probability from source.

```shell
# sudo apt-get install bazel git python-pip  # Ubuntu; others, see above links.
python -m pip install --upgrade --user tf-nightly
git clone https://github.com/tensorflow/probability.git
cd probability
bazel build --copt=-O3 --copt=-march=native :pip_pkg
PKGDIR=$(mktemp -d)
./bazel-bin/pip_pkg $PKGDIR
python -m pip install --upgrade --user $PKGDIR/*.whl
```

## Community

As part of TensorFlow, we're committed to fostering an open and welcoming
environment.

* [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow): Ask
  or answer technical questions.
* [GitHub](https://github.com/tensorflow/probability/issues): Report bugs or
  make feature requests.
* [TensorFlow Blog](https://blog.tensorflow.org/): Stay up to date on content
  from the TensorFlow team and best articles from the community.
* [Youtube Channel](http://youtube.com/tensorflow/): Follow TensorFlow shows.
* [tfprobability@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/tfprobability):
  Open mailing list for discussion and questions.

See the [TensorFlow Community](https://www.tensorflow.org/community/) page for
more details. Check out our latest publicity here:

+ [Coffee with a Googler: Probabilistic Machine Learning in TensorFlow](
  https://www.youtube.com/watch?v=BjUkL8DFH5Q)
+ [Introducing TensorFlow Probability](
  https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245)

## Contributing

We're eager to collaborate with you! See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for a guide on how to contribute. This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

## References

If you use TensorFlow Probability in a paper, please cite:

+ _TensorFlow Distributions._ Joshua V. Dillon, Ian Langmore, Dustin Tran,
Eugene Brevdo, Srinivas Vasudevan, Dave Moore, Brian Patton, Alex Alemi, Matt
Hoffman, Rif A. Saurous.
[arXiv preprint arXiv:1711.10604, 2017](https://arxiv.org/abs/1711.10604).

(We're aware there's a lot more to TensorFlow Probability than Distributions, but the Distributions paper lays out our vision and is a fine thing to cite for now.)
