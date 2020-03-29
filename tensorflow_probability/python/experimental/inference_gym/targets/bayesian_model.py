# Lint as: python2, python3
# Copyright 2020 The TensorFlow Probability Authors.
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
"""Implementation of the Bayesian model class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'BayesianModel',
]


class BayesianModel(object):
  """Base class for Bayesian models in the Inference Gym.

  Given a Bayesian model described by a joint distribution `P(x, y)` which we
  can sample from, we construct the posterior by conditioning the joint model on
  evidence `y`. The posterior distribution `P(x | y)` is represented as a
  product of the inverse normalization constant and the un-normalized density:
  `1/Z tilde{P}(x | y)`. Note that as a special case the evidence is allowed to
  be empty, in which case both the joint and the posterior are just `P(x)`.

  Given a Bayesian model conditioned on evidence, you can access the associated
  un-normalized density via the `unnormalized_log_prob` method.

  The dtype, shape, and constraints over `x` are returned by the `dtype`,
  `event_shape`, and `default_event_space_bijector` properties. Note that `x`
  could be structured, in which case `dtype` and `shape` will be structured as
  well (parallel to the structure of `x`). `default_event_space_bijector` need
  not be structured, but could operate on the structured `x`. A generic way of
  constructing a random number that is within the event space of this model is
  to do:

  ```python
  model = LogisticRegression(...)
  unconstrained_values = tf.nest.map_structure(
      lambda d, s: tf.random.normal(s, dtype=d),
      model.dtype,
      model.event_shape,
    )
  constrained_values = tf.nest.map_structure_up_to(
      model.default_event_space_bijector,
      lambda b, v: b(v),
      model.default_event_space_bijector,
      unconstrained_values,
    )
  ```

  A model has two names. First, the `name` property is used for various name
  scopes inside the implementation of the model. Second, a pretty name which is
  meant to be suitable for a table inside a publication, accessed via the
  `__str__` method.

  Models come with associated sample transformations, which describe useful ways
  of looking at the samples from the posterior distribution. Each transformation
  optionally comes equipped with various ground truth values (computed
  analytically or via Monte Carlo averages). You can apply the transformations
  to samples from the model like so:

  ```python
  model = LogisticRegression(...)
  for name, sample_transformation in model.sample_transformations.items():
    transformed_samples = sample_transformation(samples)
    if sample_transformation.ground_truth_mean is not None:
      square_diff = tf.nest.map_structure(
          lambda gtm, sm: (gtm - tf.reduce_mean(sm, axis=0))**2,
          sample_transformation.ground_truth_mean,
          transformed_samples,
      )
  ```

  #### Examples

  A simple 2-variable Bayesian model:

  ```python
  class SimpleModel(gym.targets.BayesianModel):

    def __init__(self):
      self._joint_distribution_val = tfd.JointDistributionSequential([
          tfd.Exponential(0.),
          lambda s: tfd.Normal(0., s),
      ])
      self._evidence_val = 1.

      super(TestModel, self).__init__(
          default_event_space_bijector=tfb.Exp(),
          event_shape=self._joint_distribution_val.event_shape[0],
          dtype=self._joint_distribution_val.dtype[0],
          name='simple_model',
          pretty_name='SimpleModel',
          sample_transformations=dict(
              identity=gym.targets.BayesianModel.SampleTransformation(
                  fn=lambda x: x,
                  pretty_name='Identity',
              ),),
      )

    def _joint_distribution(self):
      return self._joint_distribution_val

    def _evidence(self):
      return self._evidence_val

    def _unnormalized_log_prob(self, x):
      return self.joint_distribution().log_prob([x, self.evidence()])
  ```

  Note how we first constructed a joint distribution, and then used its
  properties to specify the Bayesian model. We don't specify the ground truth
  values for the `identity` sample transformation as they're not known
  analytically. See `GermanCreditNumericLogisticRegression` Bayesian model for
  an example of how to incorporate Monte-Carlo derived values for ground truth
  into a sample transformation.
  """

  def __init__(
      self,
      default_event_space_bijector,
      event_shape,
      dtype,
      name,
      pretty_name,
      sample_transformations,
  ):
    """Constructs a BayesianModel.

    Args:
      default_event_space_bijector: A (nest of) bijectors that take
        unconstrained `R**n` tensors to the event space of the posterior.
      event_shape: A (nest of) shapes describing the samples from the posterior.
      dtype: A (nest of) dtypes describing the dtype of the posterior.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.
      sample_transformations: A dictionary of Python strings to
        `SampleTransformation`s.
    """
    self._default_event_space_bijector = default_event_space_bijector
    self._event_shape = event_shape
    self._dtype = dtype
    self._name = name
    self._pretty_name = pretty_name
    if not isinstance(sample_transformations, collections.OrderedDict):
      sample_transformations = collections.OrderedDict(
          sorted(sample_transformations.items()))
    self._sample_transformations = sample_transformations

  # PyLint is confused, this is a new-style class.
  class SampleTransformation(  # pylint: disable=slots-on-old-class
      collections.namedtuple('SampleTransformation', [
          'fn',
          'pretty_name',
          'ground_truth_mean',
          'ground_truth_mean_standard_error',
          'ground_truth_standard_deviation',
          'ground_truth_standard_deviation_standard_error',
      ])):
    """A transformation of samples of the outer `BayesianModel`.

    Specifically, `E_{x~p}[f(x)]` for a model `p` and transformation `f`.  The
    model `p` is implicit, in that the `SampleTransformation` appears in the
    `sample_transformations` field of that `BayesianModel`.  The `f` is given as
    `fn` so that candidate samples may be passed through it.  The `fn` may close
    over the parameters of `p`, and the `ground_truth_mean` will presumably
    depend on `p` implicitly via some sampling process.

    If the `ground_truth_mean` is estimated by sampling, then
    `ground_truth_standard_deviation` and `ground_truth_mean_standard_error` are
    related using the standard formula:
    ```none
    SEM = SD / sqrt(N)
    ```
    where `N` is the number of samples. `ground_truth_standard_deviation`
    describes the distribution of `f(x)`, while
    `ground_truth_mean_standard_error`
    desribes how accurately we know `ground_truth_mean`.

    Attributes:
      fn: Function that takes samples from the target and returns a (nest of)
        `Tensor`. The returned `Tensor` must retain the leading non-event
        dimensions.
      pretty_name: Human readable name, suitable for a table in a paper.
      ground_truth_mean: Ground truth value of this expectation. Can be `None`
        if not available. Default: `None`.
      ground_truth_mean_standard_error: Standard error of the ground truth mean.
        Can be `None` if not available. Default: `None`.
      ground_truth_standard_deviation: Standard deviation of samples transformed
        by `fn`. Can be `None` if not available. Default: `None`.
      ground_truth_standard_deviation_standard_error: Standard error of the
        ground truth standard deviation. Can be `None` if not available.
        Default: `None`.

    #### Examples

    An identity `fn` for a vector-valued target would look like:

    ```python
    fn = lambda x: x
    ```
    """

    __slots__ = ()

    def __call__(self, value):
      """Returns `fn(value)`."""
      return self.fn(value)

    def __str__(self):
      """The prety name of this transformation."""
      return self.pretty_name

  def _unnormalized_log_prob(self, value):
    raise NotImplementedError('_unnormalized_log_prob is not implemented.')

  def unnormalized_log_prob(self, value, name='unnormalized_log_prob'):
    """The un-normalized log density of evaluated at a point.

    This corresponds to the target distribution associated with the model, often
    its posterior.

    Args:
      value: A (nest of) `Tensor` to evaluate the log density at.
      name: Python `str` name prefixed to Ops created by this method.

    Returns:
      unnormalized_log_prob: A floating point `Tensor`.
    """
    with tf.name_scope(self.name):
      with tf.name_scope(name):
        return self._unnormalized_log_prob(value)

  def _joint_distribution(self):
    raise NotImplementedError('_joint_distribution is not implemented.')

  def joint_distribution(self, name='joint_distribution'):
    """The joint distribution before any conditioning."""
    with tf.name_scope(self.name):
      with tf.name_scope(name):
        return self._joint_distribution()

  def _evidence(self):
    raise NotImplementedError('_evidence is not implemented.')

  def evidence(self, name='evidence'):
    """The evidence that the joint model is conditioned on."""
    with tf.name_scope(self.name):
      with tf.name_scope(name):
        return self._evidence()

  @property
  def default_event_space_bijector(self):
    """Bijector mapping the reals (R**n) to the event space of this model."""
    return self._default_event_space_bijector

  @property
  def event_shape(self):
    """Shape of a single sample from as a `TensorShape`.

    May be partially defined or unknown.

    Returns:
      event_shape: `TensorShape`, possibly unknown.
    """
    return nest.map_structure_up_to(self.dtype, tf.TensorShape,
                                    self._event_shape)

  @property
  def dtype(self):
    """The `DType` of `Tensor`s handled by this model."""
    return self._dtype

  @property
  def name(self):
    """Python `str` name prefixed to Ops created by this class."""
    return self._name

  def __str__(self):
    """The prety name of the model, suitable for a figure caption."""
    return self._pretty_name

  @property
  def sample_transformations(self):
    """A dictionary of names to `SampleTransformation`s."""
    return self._sample_transformations


BayesianModel.SampleTransformation.__new__.__defaults__ = (None, None, None,
                                                           None)
