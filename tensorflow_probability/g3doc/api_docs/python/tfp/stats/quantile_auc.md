<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.quantile_auc" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.quantile_auc


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/stats/ranking.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Calculate ranking stats AUROC and AUPRC.

``` python
tfp.stats.quantile_auc(
    q0,
    n0,
    q1,
    n1,
    curve='ROC',
    name=None
)
```



<!-- Placeholder for "Used in" -->

Computes AUROC and AUPRC from quantiles (one for positive trials and one for
negative trials).

We use `pi(x)` to denote a score. We assume that if `pi(x) > k` for some
threshold `k` then the event is predicted to be "1", and otherwise it is
predicted to be a "0".  Its actual label is `y`, which may or may not be the
same.

Area Under Curve: Receiver Operator Characteristic (AUROC) is defined as:

/ 1
|  TruePositiveRate(k) d FalsePositiveRate(k)
/ 0

where,

```none
TruePositiveRate(k) = P(pi(x) > k | y = 1)
FalsePositiveRate(k) = P(pi(x) > k | y = 0)
```

Area Under Curve: Precision-Recall (AUPRC) is defined as:

/ 1
|  Precision(k) d Recall(k)
/ 0

where,

```none
  Precision(k) = P(y = 1 | pi(x) > k)
  Recall(k) = TruePositiveRate(k) = P(pi(x) > k | y = 1)
```

Notice that AUROC and AUPRC exchange the role of Recall in the
integration, i.e.,

            Integrand    Measure
          +------------+-----------+
  AUROC   |  Recall    |  FPR      |
          +------------+-----------+
  AUPRC   |  Precision |  Recall   |
          +------------+-----------+

To learn more about the relationship between AUROC and AUPRC see [1].

#### Args:


* <b>`q0`</b>: `N-D` `Tensor` of `float`, Quantiles of predicted probabilities given a
  negative trial. The first `N-1` dimensions are batch dimensions, and the
  AUC is calculated over the final dimension.
* <b>`n0`</b>: `float` or `(N-1)-D Tensor`, Number of negative trials. If `Tensor`,
  dimensions must match the first `N-1` dimensions of `q0`.
* <b>`q1`</b>: `N-D` `Tensor` of `float`, Quantiles of predicted probabilities given a
  positive trial. The first `N-1` dimensions are batch dimensions, which
  must match those of `q0`.
* <b>`n1`</b>: `float` or `(N-1)-D Tensor`, Number of positive trials. If `Tensor`,
  dimensions must match the first `N-1` dimensions of `q0`.
* <b>`curve`</b>: `str`, Specifies the name of the curve to be computed. Must be 'ROC'
  [default] or 'PR' for the Precision-Recall-curve.
* <b>`name`</b>: `str`, An optional name_scope name.


#### Returns:


* <b>`auc`</b>: `Tensor` of `float`, area under the ROC or PR curve.

#### Examples

```python
  n = 1000
  m = 500
  predictions_given_positive = np.random.rand(n)
  predictions_given_negative = np.random.rand(m)
  q1 = tfp.stats.quantiles(predictions_given_positive, num_quantiles=50)
  q0 = tfp.stats.quantiles(predictions_given_negative, num_quantiles=50)
  auroc = tfp.stats.quantile_auc(q0, m, q1, n, curve='ROC')

```

### Mathematical Details

The algorithm proceeds by partitioning the combined quantile data into a
series of intervals `[a, b)`, approximating the probability mass of
predictions conditioned on a positive trial (`d1`) and probability mass of
predictions conditioned on a negative trial (`d0`) in each interval `[a, b)`,
and accumulating the incremental AUROC/AUPRC as functions of `d0` and `d1`.

We assume that pi(x) is uniform within a given bucket of each quantile. Thus
it will also be uniform within an interval [a, b) as long as the interval does
not cross the quantile's bucket boundaries.

A consequence of this assumption is that the cdf is piecewise linear.
That is,

  P( pi(x) > k | y = 0 ), and,
  P( pi(x) > k | y = 1 ),

are linear in `k`.

Standard AUROC is fairly easier to calculate. Under the conditional uniformity
assumptions we have a piece's contribution, [a, b), as:

 / b
 |
 |  P(y = 1 | pi > k) d P(pi > k | y = 0)
 |
 / a

    / b
    |
= - |  P(pi > k | y = 1) P(pi = k | y = 0) d k
    |
    / a

                       / b
   -1 / (len(q0) - 1)  |
= -------------------- |  P(pi > k | y = 1) d k
   q0[j + 1] - q0[j]   |
                       / a

                              / 1
   1 / (len(q0) - 1)          |
= ------------------- (b - a) |  (p1 + u d1) d u
   q0[j + 1] - q0[j]          |
                              / 0

   1 / (len(q0) - 1)
= ------------------- (b - a) (p1 + d1 / 2)
   q0[j + 1] - q0[j]


AUPRC is a bit harder to calculate since the integrand,
`P(y > 0 | pi(x) > k)`, is conditional on `k` rather than a probability over
`k`.

We proceed by formulating Precision in terms of the quantiles we have
available to us.

Precision(k) = P(y = 1 | pi(x) > k)

                   P(pi(x) > delta | y = 1 ) P(y = 1)
= -----------------------------------------------------------------------
  P(pi(x) > delta | y = 1 ) P(y = 1) + P(pi(x) > delta | y = 0 ) P(y = 0)


Since the cdf's are piecewise linear, we calculate this piece's contribution
to AUPRC by the integral:

 / b
 |
 |  P(y = 1 | pi(x) > delta) d P(pi > delta | y = 1)
 |
 / a

   1 / (len(q1) - 1)
= ------------------- (b - a) *
   q1[i + 1] - q1[i]

      / 1
      |             n1 * (u d1 + p1)
    * |  -------------------------------------- du
      |   n1 * (u d1 + p1) +  n0 * (u d0 + p0)
      / 0

                              / 1
   1 / (len(q1) - 1)          |            n1 * (u d1 + p1)
= ------------------- (b - a) |  ------------------------------------ du
   q1[i + 1] - q1[i]          |  n1 * (u d1 + p1) +  n0 * (u d0 + p0)
                              / 0

where the equality is a consequence of the piecewise uniformity assumption.

The solution to the integral is given by Mathematica:

```
Integrate[n1 (u d1 + p1) / (n1 (u d1 + p1) + n0 (u d0 + p0)), {u, 0, 1},
          Assumptions -> {p1 > 0, d1 > 0, p0 > 0, d0 > 0, n1 > 0, n0 > 0}]
```

This integral can be solved by hand by noticing that:

  f(x) / (f(x) + g(x)) = 1 / (1 + g(x)/f(x))

Thus define: u = 1 + g(x)/f(x)
for which: du = [g'(x)h(x) - g(x)f'(x)] / h(x)^2 dx
and solving integral 1/u du.


#### References

  [1]: Jesse Davis and Mark Goadrich. The relationship between
       Precision-Recall and ROC curves. In _International Conference on
       Machine Learning_, 2006. http://dl.acm.org/citation.cfm?id=1143874