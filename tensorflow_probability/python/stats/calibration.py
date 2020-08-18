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
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.stats import quantiles as quantiles_lib


__all__ = [
    'brier_decomposition',
    'brier_score',
    'expected_calibration_error',
    'expected_calibration_error_quantiles',
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
      shape_as_list = tensorshape_util.as_list(logits.shape)
      flatten, unflatten = _make_flatten_unflatten_fns(shape_as_list[:-2])
      def fn_to_map(args):
        yhat, y = args
        return tf.math.confusion_matrix(yhat, y,
                                        num_classes=num_classes,
                                        dtype=logits.dtype)
      confusion_matrix = tf.map_fn(
          fn_to_map, [flatten(pred_class), flatten(labels)],
          fn_output_signature=logits.dtype)
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
                            fn_output_signature=dist_mean.dtype)
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
  prob_y = tf.reshape(prob_y, (ps.size(prob_y),))

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
  ndims = ps.cast(ps.reduce_prod(batch_shape), tf.int32)

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


def expected_calibration_error_quantiles(
    hit, pred_log_prob, num_buckets=20, axis=0, log_space_buckets=False,
    name=None):
  """Expected calibration error via `quantiles(exp(pred_log_prob),num_buckets)`.

  Calibration is a measure of how well a model reports its own uncertainty. A
  model is said to be "calibrated" if buckets of predicted probabilities have
  the same within bucket average accurcy. The exected calibration error is the
  average absolute difference between predicted probability and (bucket) average
  accuracy. That is:

  ```python
  bucket weight = bucket_count / tf.reduce_sum(bucket_count, axis=0)
  bucket_error = abs(bucket_accuracy - bucket_confidence)
  ece = tf.reduce_sum(bucket_weight * bucket_error, axis=0)
  ```

  where `bucket_accuracy, bucket_confidence, bucket_count` are statistics
  aggregated by `num_buckets`-quantiles of `tf.math.exp(pred_log_prob)`. Note:
  `bucket_*` always have `num_buckets` size for the zero-th dimension.

  Args:
    hit: `bool` `Tensor` where `True` means the model prediction was correct
      and `False` means the model prediction was incorrect. Shape must
      broadcast with pred_log_prob.
    pred_log_prob: `Tensor` representing the model's predicted log probability
      for the given `hit`. Shape must broadcast with `hit`.
    num_buckets: `int` representing the number of buckets over which to
      aggregate hits. Buckets are quantiles of `exp(pred_log_prob)`.
      Default value: `20`.
    axis: Dimension over which to compute buckets and aggregate stats.
      Default value: `0`.
    log_space_buckets: When `False` bucket edges are computed from
      `tf.math.exp(pred_log_prob)`; when `True` bucket edges are computed from
      `pred_log_prob`.
      Default value: `False`.
    name: Prefer `str` name used for ops created by this function.
      Default value: `None` (i.e.,
      `"expected_calibration_error_quantiles"`).

  Returns:
    ece: Expected calibration error; `tf.reduce_sum(abs(bucket_accuracy -
      bucket_confidence) * bucket_count, axis=0) / tf.reduce_sum(bucket_count,
      axis=0)`.
    bucket_accuracy: `Tensor` representing the within bucket average hits, i.e.,
      total bucket hits divided by bucket count. Has shape
      `tf.concat([[num_buckets], tf.shape(tf.reduce_sum(pred_log_prob,
      axis=axis))], axis=0)`.
    bucket_confidence: `Tensor` representing the within bucket average
      probability, i.e., total bucket predicted probability divided by bucket
      count. Has shape `tf.concat([[num_buckets],
      tf.shape(tf.reduce_sum(pred_log_prob, axis=axis))], axis=0)`.
    bucket_count: `Tensor` representing the total number of obervations in each
      bucket. Has shape `tf.concat([[num_buckets],
      tf.shape(tf.reduce_sum(pred_log_prob, axis=axis))], axis=0)`.
    bucket_pred_log_prob: `Tensor` representing `pred_log_prob` bucket edges.
      Always in log space, regardless of the value of `log_space_buckets`.
    bucket: `int` `Tensor` representing the bucket within which `pred_log_prob`
      lies.

  #### Examples

  ```python
  # Example 1: Generic use.
  label = tf.cast([0, 0, 1, 0, 1, 1], dtype=tf.bool)
  log_pred = tf.math.log([0.1, 0.05, 0.5, 0.2, 0.99, 0.99])
  (
    ece,
    acc,
    conf,
    cnt,
    edges,
    bucket,
  ) = tfp.stats.expected_calibration_error_quantiles(
      label, log_pred, num_buckets=3)
  # ece  ==> tf.Tensor(0.145, shape=(), dtype=float32)
  # acc  ==> tf.Tensor([0. 0. 1.], shape=(3,), dtype=float32)
  # conf ==> tf.Tensor([0.075, 0.2, 0.826665], shape=(3,), dtype=float32)
  # cnt  ==> tf.Tensor([2. 1. 3.], shape=(3,), dtype=float32)
  ```

  ```python
  # Example 2: Categorgical classification.
  # Assume we have evidence `x`, targets `y`, and model function `dnn`.
  d = tfd.Categorical(logits=dnn(x))
  def all_categories(d):
    num_classes = tf.shape(d.logits_parameter())[-1]
    batch_ndims = tf.size(d.batch_shape_tensor())
    expand_shape = tf.pad(
        [num_classes], paddings=[[0, batch_ndims]], constant_values=1)
    return tf.reshape(tf.range(num_classes, dtype=d.dtype), expand_shape)
  all_pred_log_prob = d.log_prob(all_categories(d))
  yhat = tf.argmax(all_pred_log_prob, axis=0)
  def rollaxis(x, shift):
    return tf.transpose(x, tf.roll(tf.range(tf.rank(x)), shift=shift, axis=0))
  pred_log_prob = tf.gather(rollaxis(all_pred_log_prob, shift=-1),
                            yhat,
                            batch_dims=len(d.batch_shape))
  hit = tf.equal(y, yhat)
  (
    ece,
    acc,
    conf,
    cnt,
    edges,
    bucket,
  ) = tfp.stats.expected_calibration_error_quantiles(
      hit, pred_log_prob, num_buckets=10)
  ```

  """
  with tf.name_scope(name or 'expected_calibration_error_quantiles'):
    pred_log_prob = tf.convert_to_tensor(
        pred_log_prob, dtype_hint=tf.float32, name='pred_log_prob')
    dtype = pred_log_prob.dtype
    hit = tf.cast(hit, dtype, name='hit')
    # Make sure to compute quantiles in "prob" space not "log(prob)".
    if log_space_buckets:
      bucket_pred_log_prob = quantiles_lib.quantiles(
          pred_log_prob,
          num_quantiles=num_buckets,
          axis=axis)
    else:
      bucket_pred_log_prob = tf.math.log(quantiles_lib.quantiles(
          tf.math.exp(pred_log_prob),
          num_quantiles=num_buckets,
          axis=axis))
    bucket = _find_bins(pred_log_prob, bucket_pred_log_prob, axis)
    def _fn(i):
      """`map_fn` body."""
      keep = tf.equal(i, bucket)
      total_hit = tf.math.reduce_sum(
          tf.where(keep, hit, tf.constant(0., dtype)),
          axis=axis)
      total_count = tf.math.reduce_sum(
          tf.cast(keep, dtype),
          axis=axis)
      log_total_pred_prob = tf.math.reduce_logsumexp(
          tf.where(keep, pred_log_prob, tf.constant(-np.inf, dtype)),
          axis=axis)
      return total_hit, log_total_pred_prob, total_count

    # On the following line, we use vectorized_map instead of map_fn not for
    # efficiency reasons but because at the time of writing, map_fn doesn't
    # work correctly on the JAX substrate.  Specifically, it does not like that
    # _fn returns a tuple.
    bucket_total_hit, bucket_log_total_pred_prob, bucket_count = (
        tf.vectorized_map(
            fn=_fn,
            elems=tf.range(num_buckets, dtype=bucket.dtype)))
    n = tf.maximum(bucket_count, 1.)
    bucket_accuracy = bucket_total_hit / n
    bucket_confidence = tf.math.exp(bucket_log_total_pred_prob - tf.math.log(n))
    bucket_error = abs(bucket_accuracy - bucket_confidence)
    n = ps.cast(ps.shape(pred_log_prob)[axis], dtype)
    ece = tf.math.reduce_sum(bucket_count * bucket_error, axis=0) / n
    return (
        ece,
        bucket_accuracy,
        bucket_confidence,
        bucket_count,
        bucket_pred_log_prob,
        bucket,
    )


def _find_bins(x, edges, axis, dtype=tf.int64, name=None):
  """Like `tfp.stats.find_bins` but correctly handles quantiles axis arg."""
  with tf.name_scope(name or 'find_bins'):
    # We can't do this:
    #   return tf.cast(quantiles_lib.find_bins(x, edges=edges), dtype=tf.int64)
    # because it doesn't seem to correctly handle axis!=-1. This is a bug in TFP
    # and should be fixed. Furthermore, the following is probably more efficient
    # than tfp.stats..find_bins anyway.
    num_buckets = ps.size0(edges) - 1
    # First, we need to have `keepdims=True` semantics for edges.
    axis = axis % ps.rank(x)
    edges = tf.expand_dims(edges, axis + 1)
    # We now find the bucket which is is the "first larger", then subtract one
    # to get the bucket which is the "last smaller". Care must be taken for the
    # max element.
    pred = x < edges
    # The following is equivalent to:
    #    tf.argmin(tf.cast(~pred, dtype), axis=0))
    # yet gives the same implementation across TPU/GPU/CPU.
    # As a bonus, we can also leverage the `sorted=True` behavior.
    _, bucket_larger = tf.math.top_k(
        tf.cast(
            tf.transpose(pred, ps.pad(
                ps.range(1, ps.rank(pred)),
                paddings=[[0, 1]])),
            dtype),
        k=1,
        sorted=True)
    bucket_larger = bucket_larger[..., 0]

    bucket_larger = tf.where(
        pred[-1],  # == ~tf.math.reduce_all(pred, axis=0)
        tf.cast(bucket_larger, dtype),
        tf.cast(num_buckets, dtype))
    return bucket_larger - 1
