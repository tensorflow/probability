# Copyright 2019 The TensorFlow Probability Authors.
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
"""Calibration metrics for probabilistic predictions.

Calibration is a property of probabilistic prediction models: a model is said to
be well-calibrated if its predicted probabilities over a class of events match
long-term frequencies over the sampling distribution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'brier_decomposition',
    'brier_score',
    'expected_calibration_error',
]


def brier_decomposition(labels, logits, name=None):
  r"""Decompose the Brier score into uncertainty, resolution, and reliability.

  [Proper scoring rules][1] measure the quality of probabilistic predictions;
  any proper scoring rule admits a [unique decomposition][2] as
  `Score = Uncertainty - Resolution + Reliability`, where:

  * `Uncertainty`, is a generalized entropy of the average predictive
    distribution; it can both be positive or negative.
  * `Resolution`, is a generalized variance of individual predictive
    distributions; it is always non-negative.  Difference in predictions reveal
    information, that is why a larger resolution improves the predictive score.
  * `Reliability`, a measure of calibration of predictions against the true
    frequency of events.  It is always non-negative and a lower value here
    indicates better calibration.

  This method estimates the above decomposition for the case of the Brier
  scoring rule for discrete outcomes.  For this, we need to discretize the space
  of probability distributions; we choose a simple partition of the space into
  `nlabels` events: given a distribution `p` over `nlabels` outcomes, the index
  `k` for which `p_k > p_i` for all `i != k` determines the discretization
  outcome; that is, `p in M_k`, where `M_k` is the set of all distributions for
  which `p_k` is the largest value among all probabilities.

  The estimation error of each component is O(k/n), where n is the number
  of instances and k is the number of labels.  There may be an error of this
  order when compared to `brier_score`.

  #### References
  [1]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
  [2]: Jochen Broecker.  Reliability, sufficiency, and the decomposition of
       proper scores.
       Quarterly Journal of the Royal Meteorological Society, Vol. 135, 2009.
       https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.456

  Args:
    labels: Tensor, (n,), with tf.int32 or tf.int64 elements containing ground
      truth class labels in the range [0,nlabels].
    logits: Tensor, (n, nlabels), with logits for n instances and nlabels.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    uncertainty: Tensor, scalar, the uncertainty component of the
      decomposition.
    resolution: Tensor, scalar, the resolution component of the decomposition.
    reliability: Tensor, scalar, the reliability component of the
      decomposition.
  """
  with tf.name_scope(name or 'brier_decomposition'):
    labels = tf.convert_to_tensor(labels)
    logits = tf.convert_to_tensor(logits)
    num_classes = logits.shape[-1]

    # Compute pbar, the average distribution
    pred_class = tf.argmax(logits, axis=-1, output_type=labels.dtype)

    if tensorshape_util.rank(logits.shape) > 2:
      flatten, unflatten = _make_flatten_unflatten_fns(logits.shape[:-2])
      def fn_to_map(args):
        yhat, y = args
        return tf.math.confusion_matrix(yhat, y,
                                        num_classes=num_classes,
                                        dtype=logits.dtype)
      confusion_matrix = tf.map_fn(
          fn_to_map, [flatten(pred_class), flatten(labels)],
          dtype=logits.dtype)
      confusion_matrix = unflatten(confusion_matrix)
    else:
      confusion_matrix = tf.math.confusion_matrix(pred_class, labels,
                                                  num_classes=num_classes,
                                                  dtype=logits.dtype)

    dist_weights = tf.reduce_sum(confusion_matrix, axis=-1)
    dist_weights /= tf.reduce_sum(dist_weights, axis=-1, keepdims=True)
    pbar = tf.reduce_sum(confusion_matrix, axis=-2)
    pbar /= tf.reduce_sum(pbar, axis=-1, keepdims=True)

    eps = np.finfo(dtype_util.as_numpy_dtype(confusion_matrix.dtype)).eps
    # dist_mean[k,:] contains the empirical distribution for the set M_k
    # Some outcomes may not realize, corresponding to dist_weights[k] = 0
    dist_mean = confusion_matrix / (
        eps + tf.reduce_sum(confusion_matrix, axis=-1, keepdims=True))

    # Uncertainty: quadratic entropy of the average label distribution
    uncertainty = -tf.reduce_sum(tf.square(pbar), axis=-1)

    # Resolution: expected quadratic divergence of predictive to mean
    resolution = tf.square(tf.expand_dims(pbar, -1) - dist_mean)
    resolution = tf.reduce_sum(dist_weights *
                               tf.reduce_sum(resolution, axis=-1),
                               axis=-1)

    # Reliability: expected quadratic divergence of predictive to true
    if tensorshape_util.rank(logits.shape) > 2:
      # TODO(b/139094519): Avoid using tf.map_fn here.
      prob_true = tf.map_fn(lambda args: tf.gather(args[0], args[1]),
                            [flatten(dist_mean), flatten(pred_class)],
                            dtype=dist_mean.dtype)
      prob_true = unflatten(prob_true)
    else:
      prob_true = tf.gather(dist_mean, pred_class, axis=0)
    log_prob_true = tf.math.log(prob_true)

    log_prob_pred = logits - tf.math.reduce_logsumexp(logits, axis=-1,
                                                      keepdims=True)

    log_reliability = _reduce_log_l2_exp(log_prob_pred, log_prob_true, axis=-1)
    log_reliability = tf.math.reduce_logsumexp(log_reliability, axis=-1,)

    num_samples = tf.cast(tf.shape(logits)[-2], logits.dtype)
    reliability = tf.exp(log_reliability - tf.math.log(num_samples))

    return uncertainty, resolution, reliability


def brier_score(labels, logits, name=None):
  r"""Compute Brier score for a probabilistic prediction.

  The [Brier score][1] is a loss function for probabilistic predictions over a
  number of discrete outcomes.  For a probability vector `p` and a realized
  outcome `k` the Brier score is `sum_i p[i]*p[i] - 2*p[k]`.  Smaller values are
  better in terms of prediction quality.  The Brier score can be negative.

  Compared to the cross entropy (aka logarithmic scoring rule) the Brier score
  does not strongly penalize events which are deemed unlikely but do occur,
  see [2].  The Brier score is a strictly proper scoring rule and therefore
  yields consistent probabilistic predictions.

  #### References
  [1]: G.W. Brier.
       Verification of forecasts expressed in terms of probability.
       Monthley Weather Review, 1950.
  [2]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

  Args:
    labels: Tensor, (N1, ..., Nk), with tf.int32 or tf.int64 elements containing
      ground truth class labels in the range [0, num_classes].
    logits: Tensor, (N1, ..., Nk, num_classes), with logits for each example.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    brier_score: Tensor, (N1, ..., Nk), containint elementwise Brier scores;
      caller should `reduce_mean()` over examples in a dataset.
  """
  with tf.name_scope(name or 'brier_score'):
    labels = tf.convert_to_tensor(labels)
    logits = tf.convert_to_tensor(logits)
    probabilities = tf.math.softmax(logits, axis=1)

    num_classes = probabilities.shape[-1]
    plabel = probabilities * tf.one_hot(labels, depth=num_classes,
                                        dtype=probabilities.dtype)
    plabel = tf.reduce_sum(plabel, axis=-1)
    return tf.reduce_sum(tf.square(probabilities), axis=-1) - 2. * plabel


def _compute_calibration_bin_statistics(
    num_bins, logits=None, labels_true=None, labels_predicted=None):
  """Compute binning statistics required for calibration measures.

  Args:
    num_bins: int, number of probability bins, e.g. 10.
    logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
    labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
      ground truth class labels in the range [0,nlabels].
    labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
      containing decisions of the predictive system.  If `None`, we will use
      the argmax decision using the `logits`.

  Returns:
    bz: Tensor, shape (2,num_bins), tf.int32, counts of incorrect (row 0) and
      correct (row 1) predictions in each of the `num_bins` probability bins.
    pmean_observed: Tensor, shape (num_bins,), tf.float32, the mean predictive
      probabilities in each probability bin.
  """

  if labels_predicted is None:
    # If no labels are provided, we take the label with the maximum probability
    # decision.  This corresponds to the optimal expected minimum loss decision
    # under 0/1 loss.
    pred_y = tf.argmax(logits, axis=1, output_type=labels_true.dtype)
  else:
    pred_y = labels_predicted

  correct = tf.cast(tf.equal(pred_y, labels_true), tf.int32)

  # Collect predicted probabilities of decisions
  pred = tf.nn.softmax(logits, axis=1)
  prob_y = tf.gather(
      pred, pred_y[:, tf.newaxis], batch_dims=1)  # p(pred_y | x)
  prob_y = tf.reshape(prob_y, (tf.size(prob_y),))

  # Compute b/z histogram statistics:
  # bz[0,bin] contains counts of incorrect predictions in the probability bin.
  # bz[1,bin] contains counts of correct predictions in the probability bin.
  bins = tf.histogram_fixed_width_bins(prob_y, [0.0, 1.0], nbins=num_bins)
  event_bin_counts = tf.math.bincount(
      correct * num_bins + bins,
      minlength=2 * num_bins,
      maxlength=2 * num_bins)
  event_bin_counts = tf.reshape(event_bin_counts, (2, num_bins))

  # Compute mean predicted probability value in each of the `num_bins` bins
  pmean_observed = tf.math.unsorted_segment_sum(prob_y, bins, num_bins)
  tiny = np.finfo(dtype_util.as_numpy_dtype(logits.dtype)).tiny
  pmean_observed = pmean_observed / (
      tf.cast(tf.reduce_sum(event_bin_counts, axis=0), logits.dtype) + tiny)

  return event_bin_counts, pmean_observed


def expected_calibration_error(num_bins, logits=None, labels_true=None,
                               labels_predicted=None, name=None):
  """Compute the Expected Calibration Error (ECE).

  This method implements equation (3) in [1].  In this equation the probability
  of the decided label being correct is used to estimate the calibration
  property of the predictor.

  Note: a trade-off exist between using a small number of `num_bins` and the
  estimation reliability of the ECE.  In particular, this method may produce
  unreliable ECE estimates in case there are few samples available in some bins.
  As an alternative to this method, consider also using
  `bayesian_expected_calibration_error`.

  #### References
  [1]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger,
       On Calibration of Modern Neural Networks.
       Proceedings of the 34th International Conference on Machine Learning
       (ICML 2017).
       arXiv:1706.04599
       https://arxiv.org/pdf/1706.04599.pdf

  Args:
    num_bins: int, number of probability bins, e.g. 10.
    logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
    labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
      ground truth class labels in the range [0,nlabels].
    labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
      containing decisions of the predictive system.  If `None`, we will use
      the argmax decision using the `logits`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    ece: Tensor, scalar, tf.float32.
  """
  with tf.name_scope(name or 'expected_calibration_error'):
    logits = tf.convert_to_tensor(logits)
    labels_true = tf.convert_to_tensor(labels_true)
    if labels_predicted is not None:
      labels_predicted = tf.convert_to_tensor(labels_predicted)

    # Compute empirical counts over the events defined by the sets
    # {incorrect,correct}x{0,1,..,num_bins-1}, as well as the empirical averages
    # of predicted probabilities in each probability bin.
    event_bin_counts, pmean_observed = _compute_calibration_bin_statistics(
        num_bins, logits=logits, labels_true=labels_true,
        labels_predicted=labels_predicted)

    # Compute the marginal probability of observing a probability bin.
    event_bin_counts = tf.cast(event_bin_counts, tf.float32)
    bin_n = tf.reduce_sum(event_bin_counts, axis=0)
    pbins = bin_n / tf.reduce_sum(bin_n)  # Compute the marginal bin probability

    # Compute the marginal probability of making a correct decision given an
    # observed probability bin.
    tiny = np.finfo(np.float32).tiny
    pcorrect = event_bin_counts[1, :] / (bin_n + tiny)

    # Compute the ECE statistic as defined in reference [1].
    ece = tf.reduce_sum(pbins * tf.abs(pcorrect - pmean_observed))

  return ece


def _make_flatten_unflatten_fns(batch_shape):
  """Builds functions for flattening and unflattening batch dimensions."""
  batch_shape = tuple(batch_shape)
  batch_rank = len(batch_shape)
  ndims = np.prod(batch_shape, dtype=np.int32)

  def flatten_fn(x):
    x_shape = tuple(x.shape)
    if x_shape[:batch_rank] != batch_shape:
      raise ValueError('Expected batch-shape=%s; received array of shape=%s' %
                       (batch_shape, x_shape))
    flat_shape = (ndims,) + x_shape[batch_rank:]
    return tf.reshape(x, flat_shape)

  def unflatten_fn(x):
    x_shape = tuple(x.shape)
    if x_shape[0] != ndims:
      raise ValueError('Expected batch-size=%d; received shape=%s' %
                       (ndims, x_shape))
    return tf.reshape(x, batch_shape + x_shape[1:])
  return flatten_fn, unflatten_fn


def _reduce_log_l2_exp(loga, logb, axis=-1):
  return tf.math.reduce_logsumexp(2. * tfp_math.log_sub_exp(loga, logb),
                                  axis=axis)
