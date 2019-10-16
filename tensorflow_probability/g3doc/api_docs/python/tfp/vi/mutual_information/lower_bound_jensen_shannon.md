<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.mutual_information.lower_bound_jensen_shannon" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.mutual_information.lower_bound_jensen_shannon


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/vi/mutual_information.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Lower bound on Jensen-Shannon (JS) divergence.

``` python
tfp.vi.mutual_information.lower_bound_jensen_shannon(
    logu,
    joint_sample_mask=None,
    validate_args=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

This lower bound on JS divergence is proposed in
[Goodfellow et al. (2014)][1] and [Nowozin et al. (2016)][2].
When estimating lower bounds on mutual information, one can also use
different approaches for training the critic w.r.t. estimating
mutual information [(Poole et al., 2018)][3]. The JS lower bound is
used to train the critic with the standard lower bound on the
Jensen-Shannon divergence as used in GANs, and then evaluates the
critic using the NWJ lower bound on KL divergence, i.e. mutual information.
As Eq.7 and Eq.8 of [Nowozin et al. (2016)][2], the bound is given by
```none
I_JS = E_p(x,y)[log( D(x,y) )] + E_p(x)p(y)[log( 1 - D(x,y) )]
```
where the first term is the expectation over the samples from joint
distribution (positive samples), and the second is for the samples
from marginal distributions (negative samples), with
```none
D(x, y) = sigmoid(f(x, y)),
log(D(x, y)) = softplus(-f(x, y)).
```
`f(x, y)` is a critic function that scores all pairs of samples.

#### Example:



`X`, `Y` are samples from a joint Gaussian distribution, with
correlation `0.8` and both of dimension `1`.

```python
batch_size, rho, dim = 10000, 0.8, 1
y, eps = tf.split(
    value=tf.random.normal(shape=(2 * batch_size, dim), seed=7),
    num_or_size_splits=2, axis=0)
mean, conditional_stddev = rho * y, tf.sqrt(1. - tf.square(rho))
x = mean + conditional_stddev * eps

# Scores/unnormalized likelihood of pairs of samples `x[i], y[j]`
# (For JS lower bound, the optimal critic is of the form `f(x, y) = 1 +
# log(p(x | y) / p(x))` [(Poole et al., 2018)][3].)
conditional_dist = tfd.MultivariateNormalDiag(
    mean, scale_identity_multiplier=conditional_stddev)
conditional_scores = conditional_dist.log_prob(y[:, tf.newaxis, :])
marginal_dist = tfd.MultivariateNormalDiag(tf.zeros(dim), tf.ones(dim))
marginal_scores = marginal_dist.log_prob(y)[:, tf.newaxis]
scores = 1 + conditional_scores - marginal_scores

# Mask for joint samples in the score tensor
# (The `scores` has its shape [x_batch_size, y_batch_size], i.e.
# `scores[i, j] = f(x[i], y[j]) = log p(x[i] | y[j])`.)
joint_sample_mask = tf.eye(batch_size, dtype=bool)

# Lower bound on Jensen Shannon divergence
lower_bound_jensen_shannon(logu=scores, joint_sample_mask=joint_sample_mask)
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
  Default value: `None` (i.e., 'lower_bound_jensen_shannon').


#### Returns:


* <b>`lower_bound`</b>: `float`-like `scalar` for lower bound on JS divergence.

#### References:

[1]: Ian J. Goodfellow, et al. Generative Adversarial Nets. In
     _Conference on Neural Information Processing Systems_, 2014.
     https://arxiv.org/abs/1406.2661.
[2]: Sebastian Nowozin, Botond Cseke, Ryota Tomioka. f-GAN: Training
     Generative Neural Samplers using Variational Divergence Minimization.
     In _Conference on Neural Information Processing Systems_, 2016.
     https://arxiv.org/abs/1606.00709.
[3]: Ben Poole, Sherjil Ozair, Aaron van den Oord, Alexander A. Alemi,
     George Tucker. On Variational Bounds of Mutual Information. In
     _International Conference on Machine Learning_, 2019.
     https://arxiv.org/abs/1905.06922.