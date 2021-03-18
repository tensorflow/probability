import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions

@test_util.test_all_tf_execution_regimes
class HighwayFlowTests(test_util.TestCase):
    pass

if __name__ == '__main__':
  tf.test.main()