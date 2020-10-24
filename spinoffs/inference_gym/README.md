# Inference Gym

## Overview

The Inference Gym is the place to exercise inference methods to help make them
faster, leaner and more robust. The goal of the Inference Gym is to provide
a set of probabilistic inference problems with a standardized interface, making
it easy to test new inference techniques across a variety of challenging tasks.

Currently it provides a repository of probabilistic models that can be used to
benchmark (the computational and statistical performance of) inference
algorithms. Probabilistic models are implemented as subclasses of the
[`Model`][model] class, which minimally provides the following faculties:

- A description of the shapes and dtypes of the parameters of the model.
- Event space bijectors which map from the unconstrained real space, to the
  support of the model's associated density.
- Ability to compute the log un-normalized density at a certain parameter
  setting.
- Name of the model.
- Sample transformations, which when applied to samples from the model's density
  represent quantities with a useful interpretation.

Each model can additionally provide:

- Ground truth quantities associated with each sample transformation. This can
  include mean, variance and other statistics. If these are estimated via
  Monte-Carlo methods, a standard error is also provided. This can be used to
  verify the algorithm's level of bias.

## Getting started

Check out the [`tutorial`][tutorial].

## Usage

```bash
pip install tfp-nightly inference_gym
# Install at least one the folowing
pip install tf-nightly  # For the TensorFlow backend.
pip install jax jaxlib  # For the JAX backend.
# Install to support external datasets
pip install tfds-nightly
```

```python
import matplotlib.pyplot as plt
import numpy as np
from inference_gym import using_tensorflow as
inference_gym

model = inference_gym.targets.GermanCreditNumericLogisticRegression()

samples = inference_method(
  model.unnormalized_log_prob,
  model.default_event_space_bijector,
  model.event_shape,
  model.dtype)

plt.figure()
plt.suptitle(str(model))  # 'German Credit Numeric Logistic Regression'
for i, (name, sample_transformation) in enumerate(
    model.sample_transformations.items()):
  transformed_samples = sample_transformation(samples)
  bias_sq = tf.square(
      tf.reduce_mean(transformed_samples, 0) -
      sample_transformation.ground_truth_mean)
  ess = compute_ess(  # E.g. tfp.mcmc.effective_sample_size if using MCMC.
      transformed_samples,
      tf.square(sample_transformation.ground_truth_standard_deviation))
  plt.subplot(len(model.sample_transformations), 2, 2 * i + 1)
  plt.title('{} bias^2'.format(sample_transformation))  # e.g. 'Identity bias^2'
  plt.bar(np.arange(bias_sq.shape[-1]), bias_sq)
  plt.subplot(len(model.sample_transformations), 2, 2 * i + 2)
  plt.title('{} ess'.format(sample_transformation))
  plt.bar(np.arange(ess.shape[-1]), ess)
```

Also, see [`VectorModel`][vector_model] which can be used to simplify the
interface requirements for the inference method.


## What makes for a good Inference Gym Model?

A good model should ideally do one or more of these:

- Help build intuition (usually 1D or 2D for ease of visualization)
- Represent a generally important application of Bayesian inference
- Pose a challenge for inference, e.g.
  - high dimensionality
  - poor or pathological conditioning
  - mixing continuous and discrete latents
  - multimodality
  - non-identifiability
  - expensive gradients

Naturally, a model shouldn’t have all of those properties so users can more
easily do experiments to tease out which complication has what effect on the
inference procedure. This isn’t an exhaustive list.

## Making changes

### Adding a new model

It's easiest to mimic an existing example. Here's a small table to help you
find an example. If your model isn't described well by these possibilities,
feel free to ask for help.

| Bayesian Model? | Real dataset? | Analytic Ground Truth? | Stan Implementation? | Multiple RVs? | Example Model                                                            |
|-----------------|---------------|------------------------|----------------------|---------------|--------------------------------------------------------------------------|
| Yes             | Real          | No                     | Yes                  | Yes           | [`GermanCreditNumericSparseLogicRegression`][sparse_logistic_regression] |
| Yes             | Real          | No                     | Yes                  | No            | [`GermanCreditLogicRegression`][logistic_regression]                     |
| Yes             | Synthetic     | No                     | Yes                  | Yes           | [`SyntheticItemResponseTheory`][irt]                                     |
| No              | None          | Yes                    | No                   | No            | [`IllConditionedGaussian`][gaussian]                                     |

A Bayesian model in the table above refers to models whose density over the
parameters is computed using the product of a prior and a likelihood function
(i.e. using Bayes' theorem). These models should inherit from the
[`BayesianModel`][bayesian_model] class, as it provides some utilities for such
models.

Currently we have a little tooling to help use `cmdstanpy` to generate ground
truth values (in the correct format) for models without analytic ground truth.
Using this requires adding a model implementation inside the
[`inference_gym/tools/stan`][ground_truth_dir]
directory.

New (and existing) models should follow the [Model Contract][contract].

### Adding a new real dataset

We strongly encourage you to add your dataset to TensorFlow Datasets first.
Then, you can follow the example of the `German Credit (numeric)` dataset used
in the `GermanCreditLogicRegression`.

### Adding a new synthetic dataset

Follow the example of the [`SyntheticItemResponseTheory`][irt] model.

### Generating ground truth files.

See [`inference_gym/tools/get_ground_truth.py`][get_ground_truth].

[model]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/inference_gym/targets/model.py
[get_ground_truth]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/inference_gym/tools/get_ground_truth.py
[ground_truth_dir]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/inference_gym/tools/stan
[bayesian_model]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/inference_gym/targets/bayesian_model.py
[sparse_logistic_regression]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/inference_gym/targets/sparse_logistic_regression.py
[logistic_regression]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/inference_gym/targets/logistic_regression.py
[irt]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/inference_gym/targets/item_response_theory.py
[gaussian]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/inference_gym/targets/ill_conditioned_gaussian.py
[vector_model]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/inference_gym/targets/vector_model.py
[tutorial]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/notebooks/inference_gym_tutorial.ipynb
[contract]: https://github.com/tensorflow/probability/tree/master/spinoffs/inference_gym/model_contract.md

### Citing Inference Gym

To cite the Inference Gym:

```none
@software{inferencegym2020,
  author = {Pavel Sountsov and Alexey Radul and contributors},
  title = {Inference Gym},
  url = {https://pypi.org/project/inference_gym},
  version = {0.0.3},
  year = {2020},
}
```

Make sure to update the `version` attribute to match the actual version you're
using.

