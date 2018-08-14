<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.nelder_mead_one_step" />
</div>

# tfp.optimizer.nelder_mead_one_step

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

A single iteration of the Nelder Mead algorithm.