# Inference Gym Model Contract

## Overview

This document describes which features of  models hosted in the Inference Gym a
contributor should prioritize. This document can also be taken as a set of
expectations that a user of the library can have.

Models can have many features, some which are easier to implement than others.
While an ideal model would implement every feature possible for it, in practice
a contributor may prioritize some features for the initial commit, and leave
others for future refinement.

The primary use case of a model is to be able to run an inference algorithm on
it. The secondary goal is to be able to verify the accuracy of the algorithm.
There are other finer points of usability which also matter, but the overarching
princple of the contract for models is that it's better to have a model usable
for its primary use case without all the nice-to-haves, rather than not have
the model available at all.

That said, one of the major goals of the Inference Gym is to support using
the same algorithm across many models. Therefore, contributed models must
implement a uniform API that lets them be instantiated and manipulated
programmatically.

## Mandatory Features

Without these features implemented, a model will not be added to the Inference
Gym.

1. The required methods of the Model class must be implemented, and the base
   class constructor arguments must be provided. This provides the ability
   to programmatically describe the support of the model's density
   (via Tensor shape, dtype and constraints), as well as to evaluate that
   density at a point inside the support. Rationale: These properties are the
   minimal description of a model needed to initialize a black-box inference
   algorithm.

2. There must be at least one sample transformation that describes the standard
   parameterization of the model. The vast majority of the time this is the
   `identity` transformation. Rationale: A standard parameterization aids the
   interpretability of statistics and plots derived from applying an inference
   algorithm to a model. In many situations, the standard parameterization is
   used to inform decisions, so it's important to evaluate inference by
   examining what happens to the standard parameters even if a different
   parameterization is used to aid the algorithm.

3. The model must work on all the supported TFP backends (TensorFlow/JAX/Numpy).
   It is the goal of the maintainers of the Inference Gym to make this an easy
   requirement to fulfill. Rationale: We want to support researchers and other
   users in all the backends.

4. The model output should be reproducible by supplying a PRNG seed.  This can
   typically be achieved by using explicit PRNG objects like
   `numpy.random.RandomState` for all randomness. Rationale: If multiple reports
   are written using the same model, we want the underlying model to describe
   the same density and produce the same transformations across time so that the
   experiments remain commensurable.

5. There is a test verifying the above points. It is the goal of the maintainers
   of the Inference Gym to make this easy. Rationale: Writing tests is a
   standard software development practice that leads to robust code.

## Strongly Recommended Features

Without these features implemented, a contributor will need to make a strong
argument for inclusion of the model.

1. If the model uses a non-synthetic dataset, that dataset is used via
   [TensorFlow Datasets][tfds]. Synthetic datasets can be checked
   in to the Inference Gym if they are of reasonable size. The Inference Gym
   will also accept a reproducible procedure for synthesizing a dataset at
   runtime, as appropriate. Rationale: This absolves the Inference Gym from
   being the warden for dataset loading functionality.

2. The model should support lazy dataset loading. Rationale: While not strictly
   necessary, it makes a huge difference in interactive use of the Inference
   Gym. One example is obtaining model metadata for post-run plots without
   loading all the datasets into the plotting process.

3. The model should have relevant additional sample transformations. For
   example, regression models should support computing held-out negative
   log-likelihood. Rationale: This is similar to having a standard
   parameterization. In this case, there are certain transformations which are
   natural to look at when analyizing a model.

4. If the model has analytic ground truth values, they should be filled in.
   Rationale: Ground truth values enable one way of measuring the bias of an
   inference algorithm.

5. If it is possible to exactly sample from the model and it has analytic ground
   truth values, then there should be a test doing so and verifying that the
   ground truth values correspond to it. Rationale: This verifies that the model
   definition is self-consistent.

## Nice To Haves

These are important features, but often are very difficult to implement. We do
not require them for the initial contribution.

1. If the model doesn't have analytic ground truth values, then there should be
   a reproducible script, a Colab notebook, or integration with our existing
   `cmdstanpy`-based harness to compute the ground truth. It is not acceptable
   to hardcode values that cannot be updated by the maintainers. The ground
   truth values should be accurate, providing ~2-3 significant digits at least.
   Rationale: Ground truth values enable one way of measuring the bias of an
   inference algorithm. The outlined non-analytic methods must be runnable so
   new sample transformations can be added easily, and so the ground truth can
   be refreshed to account for bug-fixes or changes in underlying platforms.

2. There should be a test which uses a reference inference algorithm to verify
   the correspondence of the model definition to its ground truth values. It's
   fine to have models where the reference fails though, that's an exciting
   research opportunity! If the reference can only sample from the posterior
   slowly, a correspondingly slow test would be acceptable. Rationale: This
   verifies that the model definition is self-consistent.

[tfds]: https://www.tensorflow.org/datasets
