# Oryx

Oryx is a library for probabilistic programming and deep learning built on top
of Jax. The approach is to expose a set of function transformations that compose
and integrate with JAX's existing transformations (e.g. `jit`, `grad`, and
`vmap`).

## Installation

You can install Oryx via `pip`:

```bash
$ pip install oryx
```

## Documentation and Examples

Documentation is available
[on the Oryx website](https://www.tensorflow.org/probability/oryx/api_docs/python/oryx).

### Guides

*   [A tour of Oryx](https://www.tensorflow.org/probability/oryx/notebooks/a_tour_of_oryx)
*   [Probabilistic programming](https://www.tensorflow.org/probability/oryx/notebooks/probabilistic_programming)

## Development

To develop and modify Oryx, you need to install
[`poetry`](https://python-poetry.org/), a tool for Python packaging and
dependency management.

To install the development dependencies of Oryx, you can run

```bash
$ poetry install
```

and to enter a virtual environment for testing or debugging, you can run:

```bash
$ poetry shell
```

### Running tests

Oryx uses [Bazel](https://bazel.build/) for building and testing. Once Bazel is
installed, you can run tests by executing:

```
$ poetry run bazel test //oryx/...
```
