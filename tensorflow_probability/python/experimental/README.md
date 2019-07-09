# Experimental APIs

This directory contains potentially useful code which is under active
development and with plans to eventually migrate to TFP proper. All code in
`tfp.experimental` is of production quality, i.e., idiomatically consistent,
well tested, and extensively documented. `tfp.experimental` code relaxes the
TFP non-experimental contract in two regards:
1. `tfp.experimental` has no API stability guarantee. The public footprint of
   `tfp.experimental` code may change without notice or warning.
2. Code outside `tfp.experimental` cannot depend on code within
   `tfp.experimental`.

You are welcome to try any of this out (and tell us how well it works for you!).
