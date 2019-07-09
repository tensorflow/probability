<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.nelder_mead_one_step" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.optimizer.nelder_mead_one_step

A single iteration of the Nelder Mead algorithm.

``` python
tfp.optimizer.nelder_mead_one_step(
    current_simplex,
    current_objective_values,
    objective_function=None,
    dim=None,
    func_tolerance=None,
    position_tolerance=None,
    batch_evaluate_objective=False,
    reflection=None,
    expansion=None,
    contraction=None,
    shrinkage=None,
    name=None
)
```



Defined in [`python/optimizer/nelder_mead.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/optimizer/nelder_mead.py).

<!-- Placeholder for "Used in" -->
