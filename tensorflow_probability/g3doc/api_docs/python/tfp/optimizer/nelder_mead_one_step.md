<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.nelder_mead_one_step" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.optimizer.nelder_mead_one_step


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/optimizer/nelder_mead.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



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



<!-- Placeholder for "Used in" -->
