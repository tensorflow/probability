# AutoBNN

This library contains code to specify BNNs that correspond to various useful GP
kernels and assemble them into models using operators such as Addition,
Multiplication and Changepoint.

It is based on the ideas in the following papers:

* Lassi Meronen, Martin Trapp, Arno Solin. _Periodic Activation Functions
Induce Stationarity_. NeurIPS 2021.

* Tim Pearce, Russell Tsuchida, Mohamed Zaki, Alexandra Brintrup, Andy Neely.
_Expressive Priors in Bayesian Neural Networks: Kernel Combinations and
Periodic Functions_. UAI 2019.

* Feras A. Saad, Brian J. Patton, Matthew D. Hoffman, Rif A. Saurous,
Vikash K. Mansinghka.  _Sequential Monte Carlo Learning for Time Series
Structure Discovery_. ICML 2023.


## Installation

AutoBNN can be installed with pip

```
pip install autobnn
```

or it can be installed by source by following [these instructions]
(https://github.com/tensorflow/probability?tab=readme-ov-file#installing-from-source).
