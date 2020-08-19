'''

Micro-Net: A unified model for segmentation of various objects in microscopy images

    https://www.sciencedirect.com/science/article/pii/S1361841518300628

'''

import keras
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras import backend as K

# Paper uses tanh instead of relu. This can be changed to relu.
ACTIVATION = 'relu'

class Group1Block:
    def __init__(self, filters, kernel_size, batch_normalization):
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_normalization = batch_normalization
        
    def __call__(self, layer, resize):
        B = Conv2D(self.filters, self.kernel_size, use_bias=False if self.batch_normalization else True)(layer)
        B = BatchNormalization()(B) if self.batch_normalization else B
        B = Activation(ACTIVATION)(B)
        B = Conv2D(self.filters, self.kernel_size, activation=ACTIVATION)(B)
        B = MaxPooling2D(pool_size=(2,2))(B)
        inter = Conv2D(self.filters, self.kernel_size, use_bias=False if self.batch_normalization else True)(resize)
        inter = BatchNormalization()(inter) if self.batch_normalization else inter
        inter = Activation(ACTIVATION)(inter)
        inter = Conv2D(self.filters, self.kernel_size, activation=ACTIVATION)(inter)
        B = concatenate([inter, B], axis=-1)
        return B
        
class Group2:
    def __init__(self, filters, kernel_size):
        self.filters = filters
        self.kernel_size = kernel_size
        
    def __call__(self, layer):
        out = Conv2D(self.filters, self.kernel_size, activation=ACTIVATION)(layer)
        out = Conv2D(self.filters, self.kernel_size, activation=ACTIVATION)(out)
        return out
        
class Group3Block:
    def __init__(self, filters, kernel_size):
        self.filters = filters
        self.kernel_size = kernel_size
        
    def __call__(self, layer, B):
        out = Conv2DTranspose(self.filters, (2,2), strides=(2,2), padding='same')(layer)
        out = Conv2D(self.filters, self.kernel_size, activation=ACTIVATION)(out)
        out = Conv2D(self.filters, self.kernel_size, activation=ACTIVATION)(out)
        out = Conv2DTranspose(self.filters, (5,5))(out)
        inter = Conv2DTranspose(self.filters, (5,5))(B)
        out = concatenate([out, inter], axis=-1)
        out = Conv2D(self.filters, self.kernel_size, activation=ACTIVATION, padding='same')(out)
        return out
    
class Group4Block:
    def __init__(self, filters, kernel_size, dropout_rate, output_ch, strides, name):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.output_ch = output_ch
        self.strides = strides
        self.name = name
        
    def __call__(self, layer):
        out = Conv2DTranspose(self.filters, self.kernel_size, strides=(self.strides, self.strides), padding='same')(layer)
        out = Conv2D(self.filters, self.kernel_size, activation=ACTIVATION)(out)
        aux = Dropout(self.dropout_rate)(out)
        aux = Conv2D(self.output_ch, self.kernel_size, activation='softmax', name=self.name)(aux)
        return out, aux
        
def micronet252(filters, dropout_rate, output_ch, batch_normalization):
    kernel_size=(3,3)
    size=252
    
    with tf.device('/gpu:0'):
        inputs = Input((size,size,3), name='input')
        size = (size-4)//2+4
        resize1 = Input((size,size,3), name='resize1')
        size = (size-8)//2+4
        resize2 = Input((size,size,3), name='resize2')
        size = (size-8)//2+4
        resize3 = Input((size,size,3), name='resize3')
        size = (size-8)//2+4
        resize4 = Input((size,size,3), name='resize4')

        B1 = Group1Block(filters, kernel_size, batch_normalization)(inputs, resize1)
        B2 = Group1Block(2*filters, kernel_size, batch_normalization)(B1, resize2)
        B3 = Group1Block(4*filters, kernel_size, batch_normalization)(B2, resize3)
        B4 = Group1Block(8*filters, kernel_size, batch_normalization)(B3, resize4)
        B6 = Group2(32*filters, kernel_size)(B4)
        B7 = Group3Block(16*filters, kernel_size)(B6, B4)
        B8 = Group3Block(8*filters, kernel_size)(B7, B3)
        B9 = Group3Block(4*filters, kernel_size)(B8, B2)
        
    with tf.device('/gpu:1'):
        B10 = Group3Block(2*filters, kernel_size)(B9, B1)
        out1, aux1 =  Group4Block(filters, kernel_size, dropout_rate, output_ch, 2, 'aux1')(B10)
        out2, aux2 =  Group4Block(2*filters, kernel_size, dropout_rate, output_ch, 4, 'aux2')(B9)
        out3, aux3 =  Group4Block(4*filters, kernel_size, dropout_rate, output_ch, 8, 'aux3')(B8)
        out = concatenate([out1, out2, out3], axis=-1)
        out = Dropout(dropout_rate)(out)
        out = Conv2D(output_ch, kernel_size, activation='softmax', name='mask')(out)

    model = Model(inputs=[inputs, resize1, resize2, resize3, resize4], outputs = [out, aux1, aux2, aux3])  
    
    return model

