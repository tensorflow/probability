import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions

@test_util.test_all_tf_execution_regimes
class HighwayFlowTests(test_util.TestCase):

    def testBijector(self):
        width = 5
        bijector = tfp.experimental.bijectors.build_highway_flow_layer(width)
        self.assertStartsWith(bijector.name, 'highway_flow')
        x = np.random.uniform(0., 10., (width,))
        self.assertAllClose(x, self.evaluate(bijector.inverse(tf.identity(bijector.forward(x)))))

if __name__ == '__main__':
  tf.test.main()