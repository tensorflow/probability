<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.mutual_information.lower_bound_info_nce" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.mutual_information.lower_bound_info_nce


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/vi/mutual_information.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



InfoNCE lower bound on mutual information.

``` python
tfp.vi.mutual_information.lower_bound_info_nce(
    logu,
    joint_sample_mask=None,
    validate_args=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

InfoNCE lower bound is proposed in [van den Oord et al. (2018)][1]
based on noise contrastive estimation (NCE).
```none
I(X; Y) >= 1/K sum(i=1:K, log( p_joint[i] / p_marginal[i])),
```
where the numerator and the denominator are, respectively,
```none
p_joint[i] = p(x[i] | y[i]) = exp( f(x[i], y[i]) ),
p_marginal[i] = 1/K sum(j=1:K, p(x[i] | y[j]) )
              = 1/K sum(j=1:K, exp( f(x[i], y[j]) ) ),
```
and `(x[i], y[i]), i=1:K` are samples from joint distribution `p(x, y)`.
Pairs of points (x, y) are scored using a critic function `f`.

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

# Conditional distribution of p(x|y)
conditional_dist = tfd.MultivariateNormalDiag(
    mean, scale_identity_multiplier=conditional_stddev)

# Scores/unnormalized likelihood of pairs of samples `x[i], y[j]`
# (The scores has its shape [x_batch_size, distibution_batch_size]
# as the `lower_bound_info_nce` requires `scores[i, j] = f(x[i], y[j])
# = log p(x[i] | y[j])`.)
scores = conditional_dist.log_prob(x[:, tf.newaxis, :])

# Mask for joint samples
joint_sample_mask = tf.eye(batch_size, dtype=bool)

# InfoNCE lower bound on mutual information
lower_bound_info_nce(logu=scores, joint_sample_mask=joint_sample_mask)
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
  Default value: `None` (i.e., 'lower_bound_info_nce').


#### Returns:


* <b>`lower_bound`</b>: `float`-like `scalar` for lower bound on mutual information.

#### References

[1]: Aaron van den Oord, Yazhe Li, Oriol Vinyals. Representation
     Learning with Contrastive Predictive Coding. _arXiv preprint
     arXiv:1807.03748_, 2018. https://arxiv.org/abs/1807.03748.