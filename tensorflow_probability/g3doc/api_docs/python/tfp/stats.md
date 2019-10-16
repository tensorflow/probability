<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.stats


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/stats/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Statistical functions.

<!-- Placeholder for "Used in" -->


## Functions

[`auto_correlation(...)`](../tfp/stats/auto_correlation.md): Auto correlation along one axis.

[`brier_decomposition(...)`](../tfp/stats/brier_decomposition.md): Decompose the Brier score into uncertainty, resolution, and reliability.

[`brier_score(...)`](../tfp/stats/brier_score.md): Compute Brier score for a probabilistic prediction.

[`cholesky_covariance(...)`](../tfp/stats/cholesky_covariance.md): Cholesky factor of the covariance matrix of vector-variate random samples.

[`correlation(...)`](../tfp/stats/correlation.md): Sample correlation (Pearson) between observations indexed by `event_axis`.

[`count_integers(...)`](../tfp/stats/count_integers.md): Counts the number of occurrences of each value in an integer array `arr`.

[`covariance(...)`](../tfp/stats/covariance.md): Sample covariance between observations indexed by `event_axis`.

[`expected_calibration_error(...)`](../tfp/stats/expected_calibration_error.md): Compute the Expected Calibration Error (ECE).

[`find_bins(...)`](../tfp/stats/find_bins.md): Bin values into discrete intervals.

[`histogram(...)`](../tfp/stats/histogram.md): Count how often `x` falls in intervals defined by `edges`.

[`log_average_probs(...)`](../tfp/stats/log_average_probs.md): Computes `log(average(to_probs(logits)))` in a numerically stable manner.

[`log_loomean_exp(...)`](../tfp/stats/log_loomean_exp.md): Computes the log-leave-one-out-mean of `exp(logx)`.

[`log_loosum_exp(...)`](../tfp/stats/log_loosum_exp.md): Computes the log-leave-one-out-sum of `exp(logx)`.

[`log_soomean_exp(...)`](../tfp/stats/log_soomean_exp.md): Computes the log-swap-one-out-mean of `exp(logx)`.

[`log_soosum_exp(...)`](../tfp/stats/log_soosum_exp.md): Computes the log-swap-one-out-sum of `exp(logx)`.

[`percentile(...)`](../tfp/stats/percentile.md): Compute the `q`-th percentile(s) of `x`.

[`quantile_auc(...)`](../tfp/stats/quantile_auc.md): Calculate ranking stats AUROC and AUPRC.

[`quantiles(...)`](../tfp/stats/quantiles.md): Compute quantiles of `x` along `axis`.

[`stddev(...)`](../tfp/stats/stddev.md): Estimate standard deviation using samples.

[`variance(...)`](../tfp/stats/variance.md): Estimate variance using samples.

