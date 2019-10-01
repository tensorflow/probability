<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.brier_score" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.brier_score


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/stats/calibration.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Compute Brier score for a probabilistic prediction.

``` python
tfp.stats.brier_score(
    labels,
    logits,
    name=None
)
```



<!-- Placeholder for "Used in" -->

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

#### Args:


* <b>`labels`</b>: Tensor, (N1, ..., Nk), with tf.int32 or tf.int64 elements containing
  ground truth class labels in the range [0, num_classes].
* <b>`logits`</b>: Tensor, (N1, ..., Nk, num_classes), with logits for each example.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:


* <b>`brier_score`</b>: Tensor, (N1, ..., Nk), containint elementwise Brier scores;
  caller should `reduce_mean()` over examples in a dataset.