# TensorFlow Probability Examples

This directory contains standalone usage examples of TensorFlow Probability API.

## Running Examples

You can run an example by executing the following command from anywhere in the
TensorFlow Probability source directory:

```bash
bazel run //tensorflow_probability/examples:cifar10_bnn -- ${FLAGS}
```

Replace `cifar10_bnn` with the example you're interested in and `FLAGS` with
whatever flags you wish to pass to the example.

## Running Examples on Google Cloud AI Platform

Some examples are adapted and tested to be readily runnable on
[AI Platform Training](https://cloud.google.com/ai-platform/).

You can find them in the [ml-on-gcp repository](https://github.com/GoogleCloudPlatform/ml-on-gcp/tree/master/example_zoo/tensorflow/probability).
