<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.truthy" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.truthy

Normalizes Tensor ranks for use in `if` conditions.

### Aliases:

* `tfp.experimental.auto_batching.frontend.truthy`
* `tfp.experimental.auto_batching.truthy`

``` python
tfp.experimental.auto_batching.truthy(x)
```



Defined in [`python/internal/auto_batching/frontend.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/frontend.py).

<!-- Placeholder for "Used in" -->

This enables dry-runs of programs with control flow.  Usage: Program the
conditions of `if` statements and `while` loops to have a batch dimension, and
then wrap them with this function.  Example:
```python
ctx = frontend.Context
truthy = frontend.truthy

@ctx.batch(type_inference=...)
def my_abs(x):
  if truthy(x > 0):
    return x
  else:
    return -x

my_abs([-5], dry_run=True)
# returns [5] in Eager mode
```

This is necessary because auto-batched programs still have a leading batch
dimension (of size 1) even in dry-run mode, and a Tensor of shape [1] is not
acceptable as the condition to an `if` or `while`.  However, the leading
dimension is critical during batched execution; so conditions of ifs need to
have rank 1 if running batched and rank 0 if running unbatched (i.e.,
dry-run).  The `truthy` function arranges for this be happen (by detecting
whether it is in dry-run mode or not).

If you missed a spot where you should have used `truthy`, the error message
will say `Non-scalar tensor <Tensor ...> cannot be converted to boolean.`

#### Args:


* <b>`x`</b>: A Tensor.


#### Returns:


* <b>`x`</b>: The Tensor `x` if we are in batch mode, or if the shape of `x` is
  anything other than `[1]`.  Otherwise returns the single scalar in `x` as
  a Tensor of scalar shape (which is acceptable in the conditions of `if`
  and `while` statements.