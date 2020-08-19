'''

U-Net: Convolutional Networks for Biomedical Image Segmentation

    https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

DCAN: Deep Contour-Aware Networks for Accurate Gland Segmentation

    https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Chen_DCAN_Deep_Contour-Aware_CVPR_2016_paper.html

'''

import keras
import tensorflow as tf
from keras.layers import *
from keras.models import *

KERNEL_INITIALIZER = 'glorot_uniform'

class UNetUnit:
    def __init__(self, filters, kernel_size, dropout_rate):
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
    def __call__(self, layer):
        out = Conv2D(self.filters, self.kernel_size, padding='same', activation='relu', kernel_initializer=KERNEL_INITIALIZER)(layer)
        if self.dropout_rate > 0:
            out = Dropout(self.dropout_rate)(out)
        out = Conv2D(self.filters, self.kernel_size, padding='same', activation='relu', kernel_initializer=KERNEL_INITIALIZER)(out)
        return out
    
class UNetUp:
    def __call__(self, left, down):
        out = UpSampling2D(size=(2,2))(down)
        if left.shape[1]!=out.shape[1] or left.shape[2]!=out.shape[2]:
            total_padding = (left.shape[1].value-out.shape[1].value)
            padding_1 = (total_padding//2, total_padding-total_padding//2)
            total_padding = (left.shape[2].value-out.shape[2].value)
            padding_2 = (total_padding//2, total_padding-total_padding//2)
            padding = (padding_1, padding_2)
            out = ZeroPadding2D(padding=padding)(out)
        out = concatenate([left, out], axis=-1)
        return out

def fourier_net(input_shape, kernel_size, filters, dropout_rate, output_ch, N):
    with tf.device('/gpu:0'):
        inputs = Input(input_shape, name='input')
        
        conv1 = UNetUnit(filters, kernel_size, dropout_rate=dropout_rate)(inputs)
        pool = MaxPooling2D(pool_size=(2,2))(conv1)
        conv2 = UNetUnit(2*filters, kernel_size, dropout_rate=dropout_rate)(pool)
        pool = MaxPooling2D(pool_size=(2,2))(conv2)
        conv3 = UNetUnit(4*filters, kernel_size, dropout_rate=dropout_rate)(pool)
        pool = MaxPooling2D(pool_size=(2,2))(conv3)
        conv4 = UNetUnit(8*filters, kernel_size, dropout_rate=dropout_rate)(pool)
        pool = MaxPooling2D(pool_size=(2,2))(conv4)
        conv5 = UNetUnit(16*filters, kernel_size, dropout_rate=dropout_rate)(pool)
        
        transpose = UNetUp()(conv4, conv5)
        conv = UNetUnit(8*filters, kernel_size, dropout_rate=dropout_rate)(transpose)
        transpose = UNetUp()(conv3, conv)
        conv = UNetUnit(4*filters, kernel_size, dropout_rate=dropout_rate)(transpose)
        transpose = UNetUp()(conv2, conv)
        
    with tf.device('/gpu:1'):
        conv = UNetUnit(2*filters, kernel_size, dropout_rate=dropout_rate)(transpose)
        transpose = UNetUp()(conv1, conv)
        conv = UNetUnit(filters, kernel_size, dropout_rate=dropout_rate)(transpose)
        out1 = Conv2D(output_ch, (1,1), activation='softmax', name='mask')(conv)
        
        transpose = UNetUp()(conv4, conv5)
        conv = UNetUnit(8*filters, kernel_size, dropout_rate=dropout_rate)(transpose)
        transpose = UNetUp()(conv3, conv)
        conv = UNetUnit(4*filters, kernel_size, dropout_rate=dropout_rate)(transpose)
        transpose = UNetUp()(conv2, conv)
        conv = UNetUnit(2*filters, kernel_size, dropout_rate=dropout_rate)(transpose)
        transpose = UNetUp()(conv1, conv)
        conv = UNetUnit(filters, kernel_size, dropout_rate=dropout_rate)(transpose)
        out2 = Conv2D(N, (1,1), name='fourier')(conv)
        
    model = Model(inputs=inputs, outputs=[out1, out2])
    return model

