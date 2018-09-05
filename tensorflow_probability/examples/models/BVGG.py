
import tensorflow as tf
from tensorflow.python.keras.layers import Input, BatchNormalization, Activation
from tensorflow.python.keras.layers import Dropout, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Model

from tensorflow_probability.python.layers import Convolution2DFlipout as Conv2DFlip
from tensorflow_probability.python.layers import DenseFlipout
from tensorflow_probability.python.layers.util import default_mean_field_normal_fn

class BVGG(object):
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
        filters   = [64, 128, 256, 512, 512]
        kernels   = [3,3,3,3,3]
        strides   = [2,2,2,2,2]

        image = Input(shape=self.input_dim, dtype='float32')
        
        x = image
        for i in range(len(kernels)):
            x = self._vggconv_block(x, filters[i], kernels[i], strides[i])

        x = Flatten()(x)
        x = DenseFlipout(self.num_classes, 
                         kernel_posterior_fn=self.qW)(x)
        self.model = Model(inputs=image, outputs=x, name='resnet')
        return self.model
    
    def _vggconv_block(self, x, filters, kernel, stride):
        out = Conv2DFlip(filters, kernel, padding='same', 
                         kernel_posterior_fn=self.qW)(x)
        out = BatchNormalization(axis=self.bn_axis)(out)
        out = Activation('relu')(out)

        out = Conv2DFlip(filters, kernel, padding='same', 
                         kernel_posterior_fn=self.qW)(out)
        out = BatchNormalization(axis=self.bn_axis)(out)
        out = Activation('relu')(out)
        
        out = MaxPooling2D(pool_size=(2,2), strides=stride)(out)
        return out
    
    