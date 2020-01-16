<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.optimizer


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/optimizer/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



TensorFlow Probability Optimizer python package.

<!-- Placeholder for "Used in" -->


## Modules

[`linesearch`](../tfp/optimizer/linesearch.md) module: Line-search optimizers package.

## Classes

[`class StochasticGradientLangevinDynamics`](../tfp/optimizer/StochasticGradientLangevinDynamics.md): An optimizer module for stochastic gradient Langevin dynamics.

[`class VariationalSGD`](../tfp/optimizer/VariationalSGD.md): An optimizer module for constant stochastic gradient descent.

## Functions

[`bfgs_minimize(...)`](../tfp/optimizer/bfgs_minimize.md): Applies the BFGS algorithm to minimize a differentiable function.

[`converged_all(...)`](../tfp/optimizer/converged_all.md): Condition to stop when all batch members have converged or failed.

[`converged_any(...)`](../tfp/optimizer/converged_any.md): Condition to stop when any batch member converges, or all have failed.

[`differential_evolution_minimize(...)`](../tfp/optimizer/differential_evolution_minimize.md): Applies the Differential evolution algorithm to minimize a function.

[`differential_evolution_one_step(...)`](../tfp/optimizer/differential_evolution_one_step.md): Performs one step of the differential evolution algorithm.

[`lbfgs_minimize(...)`](../tfp/optimizer/lbfgs_minimize.md): Applies the L-BFGS algorithm to minimize a differentiable function.

[`nelder_mead_minimize(...)`](../tfp/optimizer/nelder_mead_minimize.md): Minimum of the objective function using the Nelder Mead simplex algorithm.

[`nelder_mead_one_step(...)`](../tfp/optimizer/nelder_mead_one_step.md): A single iteration of the Nelder Mead algorithm.

[`proximal_hessian_sparse_minimize(...)`](../tfp/optimizer/proximal_hessian_sparse_minimize.md): Minimize using Hessian-informed proximal gradient descent.

[`proximal_hessian_sparse_one_step(...)`](../tfp/optimizer/proximal_hessian_sparse_one_step.md): One step of (the outer loop of) the minimization algorithm.

