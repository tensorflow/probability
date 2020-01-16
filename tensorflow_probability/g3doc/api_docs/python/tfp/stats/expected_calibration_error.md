<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.expected_calibration_error" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.expected_calibration_error


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/stats/calibration.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Compute the Expected Calibration Error (ECE).

``` python
tfp.stats.expected_calibration_error(
    num_bins,
    logits=None,
    labels_true=None,
    labels_predicted=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->

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

#### Args:


* <b>`num_bins`</b>: int, number of probability bins, e.g. 10.
* <b>`logits`</b>: Tensor, (n,nlabels), with logits for n instances and nlabels.
* <b>`labels_true`</b>: Tensor, (n,), with tf.int32 or tf.int64 elements containing
  ground truth class labels in the range [0,nlabels].
* <b>`labels_predicted`</b>: Tensor, (n,), with tf.int32 or tf.int64 elements
  containing decisions of the predictive system.  If `None`, we will use
  the argmax decision using the `logits`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:


* <b>`ece`</b>: Tensor, scalar, tf.float32.