
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, BatchNormalization, Activation
from tensorflow.python.keras.layers import Dropout, AveragePooling2D, Flatten
from tensorflow.python.keras.models import Model

from tensorflow_probability.python.layers import Convolution2DFlipout as Conv2DFlip
from tensorflow_probability.python.layers import DenseFlipout
from tensorflow_probability.python.layers.util import default_mean_field_normal_fn

class BayesResNet(object):
    def __init__(self, input_dim, 
                 num_classes=10, 
                 bn_axis = 3,
                 vmean=-9, 
                 vstd=0.1, 
                 vconstraint=0.2):
        self.save_path = None
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.bn_axis = bn_axis
        
        self.qW = default_mean_field_normal_fn(
                untransformed_scale_initializer=tf.random_normal_initializer(
                        mean=vmean, stddev=vstd),
                untransformed_scale_constraint=lambda t: tf.clip_by_value(t, -1000, tf.log(vconstraint)))
        

    def build_model(self):
        filters   = [64, 128, 256, 512]
        kernels   = [3,3,3,3]
        strides   = [1,2,2,2]

        image = Input(shape=self.input_dim, dtype='float32')
        with tf.name_scope('prob_layer'):
            x = Conv2DFlip(64, 3, strides=1, padding='same', 
                           kernel_posterior_fn=self.qW)(image)

        for i in range(len(kernels)):
            x = self._resnet_block(x, filters[i], kernels[i], strides[i])

        x = BatchNormalization(axis=self.bn_axis)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(4, 1)(x)
        x = Flatten()(x)
        
        with tf.name_scope('prob_layer'):
            x = DenseFlipout(self.num_classes,
                             kernel_posterior_fn=self.qW)(x)
            
        self.model = Model(inputs=image, outputs=x, name='bayesian_resnet')
        return self.model
        
    def _resnet_block(self, x, filters, kernel, stride):
        out = BatchNormalization(axis=self.bn_axis)(x)
        out = Activation('relu')(out)
        
        if stride != 1 or filters != x.shape[1]:
            shortcut = self._projection_shortcut(out, filters, stride)
        else:
            shortcut = x
        
        with tf.name_scope('prob_layer'):
            out = Conv2DFlip(filters, kernel, strides=stride, 
                          padding='same',
                          kernel_posterior_fn=self.qW)(out)
        out = BatchNormalization(axis=self.bn_axis)(out)
        out = Activation('relu')(out)
        
        with tf.name_scope('prob_layer'):
            out = Conv2DFlip(filters, kernel, strides=1, 
                         padding='same',
                         kernel_posterior_fn=self.qW)(out)
        out = layers.add([out, shortcut])
        return out

    def _projection_shortcut(self, x, out_filters, stride):
        with tf.name_scope('prob_layer'):
            out = Conv2DFlip(out_filters, 1, strides=stride, padding='valid',
                          kernel_posterior_fn=self.qW)(x)
        return out

