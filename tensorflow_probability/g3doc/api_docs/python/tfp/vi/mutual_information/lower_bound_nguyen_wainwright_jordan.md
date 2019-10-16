<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.mutual_information.lower_bound_nguyen_wainwright_jordan" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.mutual_information.lower_bound_nguyen_wainwright_jordan


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/vi/mutual_information.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Lower bound on Kullback-Leibler (KL) divergence from Nguyen at al.

``` python
tfp.vi.mutual_information.lower_bound_nguyen_wainwright_jordan(
    logu,
    joint_sample_mask=None,
    validate_args=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

The lower bound was introduced by Nguyen, Wainwright, Jordan (NWJ) in
[Nguyen et al. (2010)][1], and also known as `f-GAN KL` [(Nowozin et al.,
2016)][2] and `MINE-f` [(Belghazi et al., 2018)][3].
```none
I_NWJ = E_p(x,y)[f(x, y)] - 1/e * E_p(y)[Z(y)],
```
where `f(x, y)` is a critic function that scores pairs of samples `(x, y)`,
and `Z(y)` is the corresponding partition function:
```none
Z(y) = E_p(x)[ exp(f(x, y)) ].
```

#### Example:



`X`, `Y` are samples from a joint Gaussian distribution, with correlation
`0.8` and both of dimension `1`.

```python
batch_size, rho, dim = 10000, 0.8, 1
y, eps = tf.split(
    value=tf.random.normal(shape=(2 * batch_size, dim), seed=7),
    num_or_size_splits=2, axis=0)
mean, conditional_stddev = rho * y, tf.sqrt(1. - tf.square(rho))
x = mean + conditional_stddev * eps

# Scores/unnormalized likelihood of pairs of samples `x[i], y[j]`
# (For NWJ lower bound, the optimal critic is of the form `f(x, y) = 1 +
# log(p(x | y) / p(x))` [(Poole et al., 2018)][4]. )
conditional_dist = tfd.MultivariateNormalDiag(
    mean, scale_identity_multiplier=conditional_stddev)
conditional_scores = conditional_dist.log_prob(y[:, tf.newaxis, :])
marginal_dist = tfd.MultivariateNormalDiag(tf.zeros(dim), tf.ones(dim))
marginal_scores = marginal_dist.log_prob(y)[:, tf.newaxis]
scores = 1 + conditional_scores - marginal_scores

# Mask for joint samples in score tensor
# (The `scores` has its shape [x_batch_size, y_batch_size], i.e.
# `scores[i, j] = f(x[i], y[j]) = log p(x[i] | y[j])`.)
joint_sample_mask = tf.eye(batch_size, dtype=bool)

# Lower bound on KL divergence between p(x,y) and p(x)p(y),
# i.e. the mutual information between `X` and `Y`.
lower_bound_nguyen_wainwright_jordan(
    logu=scores, joint_sample_mask=joint_sample_mask)
```

#### Args:


* <b>`logu`</b>: `float`-like `Tensor` of size `[batch_size_1, batch_size_2]`
  representing critic scores (scores) for pairs of points (x, y) with
  `logu[i, j] = f(x[i], y[j])`.
* <b>`joint_sample_mask`</b>: `bool`-like `Tensor` of the same size as `logu`
  masking the positive samples by `True`, i.e. samples from joint
  distribution `p(x, y)`.
  Default value: `None`. By default, an identity matrix is constructed as
  the mask.
* <b>`validate_args`</b>: Python `bool`, default `False`. Whether to validate input
  with asserts. If `validate_args` is `False`, and the inputs are invalid,
  correct behavior is not guaranteed.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., 'lower_bound_nguyen_wainwright_jordan').


#### Returns:


* <b>`lower_bound`</b>: `float`-like `scalar` for lower bound on KL divergence
  between joint and marginal distrbutions.

#### References:

[1]: XuanLong Nguyen, Martin J. Wainwright, Michael I. Jordan.
     Estimating Divergence Functionals and the Likelihood Ratio
     by Convex Risk Minimization. _IEEE Transactions on Information Theory_,
     56(11):5847-5861, 2010. https://arxiv.org/abs/0809.0853.
[2]: Sebastian Nowozin, Botond Cseke, Ryota Tomioka. f-GAN: Training
     Generative Neural Samplers using Variational Divergence Minimization.
     In _Conference on Neural Information Processing Systems_, 2016.
     https://arxiv.org/abs/1606.00709.
[3]: Mohamed Ishmael Belghazi, et al. MINE: Mutual Information Neural
     Estimation. In _International Conference on Machine Learning_, 2018.
     https://arxiv.org/abs/1801.04062.
[4]: Ben Poole, Sherjil Ozair, Aaron van den Oord, Alexander A. Alemi,
     George Tucker. On Variational Bounds of Mutual Information. In
     _International Conference on Machine Learning_, 2019.
     https://arxiv.org/abs/1905.06922.