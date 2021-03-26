import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


def inv_f(self, y, N=20):
    # inverse with Newton iteration
    x = tf.ones(y.shape)
    for _ in range(N):
        x = x - (self.residual_fraction * x + (1. - self.residual_fraction) * tf.math.softplus(x) - y) / (
            self.df(x))
    return x

def df(self, x):
    # derivative of activation
    return self.residual_fraction + (1. - self.residual_fraction) * tf.math.sigmoid(x) * (1. - tf.math.sigmoid(x))


@test_util.test_all_tf_execution_regimes
class HighwayFlowTests(test_util.TestCase):

    def testBijector(self):
        width = 5
        bijector = tfp.experimental.bijectors.build_highway_flow_layer(width)
        self.assertStartsWith(bijector.name, 'highway_flow')
        x = tf.random.uniform(0., 10., (width,))
        self.assertAllClose(x, self.evaluate(bijector.inverse(tf.identity(bijector.forward(x)))))

    #def testThatInverseActivationGradientIsCorrect(self):


if __name__ == '__main__':
    tf.test.main()