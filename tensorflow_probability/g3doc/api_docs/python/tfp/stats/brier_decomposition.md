<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.brier_decomposition" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.brier_decomposition


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/stats/calibration.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Decompose the Brier score into uncertainty, resolution, and reliability.

``` python
tfp.stats.brier_decomposition(
    labels,
    logits,
    name=None
)
```



<!-- Placeholder for "Used in" -->

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

#### Args:


* <b>`labels`</b>: Tensor, (n,), with tf.int32 or tf.int64 elements containing ground
  truth class labels in the range [0,nlabels].
* <b>`logits`</b>: Tensor, (n, nlabels), with logits for n instances and nlabels.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:


* <b>`uncertainty`</b>: Tensor, scalar, the uncertainty component of the
  decomposition.
* <b>`resolution`</b>: Tensor, scalar, the resolution component of the decomposition.
* <b>`reliability`</b>: Tensor, scalar, the reliability component of the
  decomposition.