<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.pivoted_cholesky" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.pivoted_cholesky

Computes the (partial) pivoted cholesky decomposition of `matrix`.

``` python
tfp.math.pivoted_cholesky(
    matrix,
    max_rank,
    diag_rtol=0.001,
    name=None
)
```



Defined in [`python/math/linalg.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/linalg.py).

<!-- Placeholder for "Used in" -->

The pivoted Cholesky is a low rank approximation of the Cholesky decomposition
of `matrix`, i.e. as described in [(Harbrecht et al., 2012)][1]. The
currently-worst-approximated diagonal element is selected as the pivot at each
iteration. This yields from a `[B1...Bn, N, N]` shaped `matrix` a `[B1...Bn,
N, K]` shaped rank-`K` approximation `lr` such that `lr @ lr.T ~= matrix`.
Note that, unlike the Cholesky decomposition, `lr` is not triangular even in
a rectangular-matrix sense. However, under a permutation it could be made
triangular (it has one more zero in each column as you move to the right).

Such a matrix can be useful as a preconditioner for conjugate gradient
optimization, i.e. as in [(Wang et al. 2019)][2], as matmuls and solves can be
cheaply done via the Woodbury matrix identity, as implemented by
`tf.linalg.LinearOperatorLowRankUpdate`.

#### Args:


* <b>`matrix`</b>: Floating point `Tensor` batch of symmetric, positive definite
  matrices.
* <b>`max_rank`</b>: Scalar `int` `Tensor`, the rank at which to truncate the
  approximation.
* <b>`diag_rtol`</b>: Scalar floating point `Tensor` (same dtype as `matrix`). If the
  errors of all diagonal elements of `lr @ lr.T` are each lower than
  `element * diag_rtol`, iteration is permitted to terminate early.
* <b>`name`</b>: Optional name for the op.


#### Returns:


* <b>`lr`</b>: Low rank pivoted Cholesky approximation of `matrix`.

#### References

[1]: H Harbrecht, M Peters, R Schneider. On the low-rank approximation by the
     pivoted Cholesky decomposition. _Applied numerical mathematics_,
     62(4):428-440, 2012.

[2]: K. A. Wang et al. Exact Gaussian Processes on a Million Data Points.
     _arXiv preprint arXiv:1903.08114_, 2019. https://arxiv.org/abs/1903.08114