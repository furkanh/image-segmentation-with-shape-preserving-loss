'''

The Importance of Skip Connections in Biomedical Image Segmentation

    https://link.springer.com/chapter/10.1007/978-3-319-46976-8_19

'''

import keras
import tensorflow as tf
from keras.layers import *
from keras.models import *

BN = False

class Bottleneck:
    def __init__(self, filters, kernel_size, dropout_rate, n, color):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.n = n
        self.color = color
        
    def generate_bottleneck(self, layer):
        out = BatchNormalization()(layer) if BN else layer
        out = Activation('relu')(out)
        out = Conv2D(self.filters//4, (1,1), padding='same')(out)
        out = BatchNormalization()(out) if BN else out
        out = Activation('relu')(out)
        out = Conv2D(self.filters//4, self.kernel_size, padding='same')(out)
        out = BatchNormalization()(out) if BN else out
        out = Activation('relu')(out)
        out = Conv2D(self.filters, (1,1), padding='same')(out)
        out = Dropout(self.dropout_rate)(out)
        out = add([out, Conv2D(self.filters, (1,1), padding='same')(layer)])
        return out
    
    def generate_bottleneck_upsample(self, layer):
        out = BatchNormalization()(layer) if BN else layer
        out = Activation('relu')(out)
        out = Conv2D(self.filters//4, (1,1), padding='same')(out)
        out = BatchNormalization()(out) if BN else out
        out = Activation('relu')(out)
        out = Conv2D(self.filters//4, self.kernel_size, padding='same')(out)
        out = BatchNormalization()(out) if BN else out
        out = Activation('relu')(out)
        out = UpSampling2D(size=(2,2))(out)
        out = Conv2D(self.filters, (1,1), padding='same')(out)
        out = Dropout(self.dropout_rate)(out)
        layer = UpSampling2D(size=(2,2))(layer)
        layer = Conv2D(self.filters, (1,1), padding='same')(layer)
        out = add([out, layer])
        return out
        
    def generate_bottleneck_downsample(self, layer):
        out = BatchNormalization()(layer) if BN else layer
        out = Activation('relu')(out)
        out = Conv2D(self.filters//4, (1,1), padding='same', strides=2)(out)
        out = BatchNormalization()(out) if BN else out
        out = Activation('relu')(out)
        out = Conv2D(self.filters//4, self.kernel_size, padding='same')(out)
        out = BatchNormalization()(out) if BN else out
        out = Activation('relu')(out)
        out = Conv2D(self.filters, (1,1), padding='same')(out)
        out = Dropout(self.dropout_rate)(out)
        out = add([out, Conv2D(self.filters, (1,1), padding='same', strides=2)(layer)])
        return out
    
    def __call__(self, layer):
        if self.n==1 and self.color=='blue':
            return self.generate_bottleneck_downsample(layer)
        elif self.n==1 and self.color=='yellow':
            return self.generate_bottleneck_upsample(layer)
        
        out = self.generate_bottleneck(layer)
        for i in range(self.n-2):
            out = self.generate_bottleneck(out)

        if self.color=='blue':
            out = self.generate_bottleneck_downsample(layer)
        elif self.color=='yellow':
            out = self.generate_bottleneck_upsample(layer)
        else:
            out = self.generate_bottleneck(out)
        return out

class SimpleBlock:
    def __init__(self, filters, kernel_size, dropout_rate, color):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.color = color
        
    def generate_downsample(self, layer):
        out = BatchNormalization()(layer) if BN else layer
        out = Activation('relu')(out)
        out = MaxPooling2D(pool_size=(2,2))(out)
        out = Conv2D(self.filters, self.kernel_size, padding='same')(out)
        out = Dropout(self.dropout_rate)(out)
        layer = Conv2D(self.filters, (1,1), padding='same')(layer)
        layer = MaxPooling2D(pool_size=(2,2))(layer)
        out = add([out, layer])
        return out
    
    def generate_upsample(self, layer):
        out = BatchNormalization()(layer) if BN else layer
        out = Activation('relu')(out)
        out = Conv2D(self.filters, self.kernel_size, padding='same')(out)
        out = UpSampling2D(size=(2,2))(out)
        out = Dropout(self.dropout_rate)(out)
        layer = UpSampling2D(size=(2,2))(layer)
        layer = Conv2D(self.filters, (1,1), padding='same')(layer)
        out = add([out, layer])
        return out
        
    def generate_simple_block(self, layer):
        out = BatchNormalization()(layer) if BN else layer
        out = Activation('relu')(out)
        out = Conv2D(self.filters, self.kernel_size, padding='same')(out)
        out = Dropout(self.dropout_rate)(out)
        out = add([out, layer])
        return out
    
    def __call__(self, layer):
        if self.color=='blue':
            return self.generate_downsample(layer)
        elif self.color=='yellow':
            return self.generate_upsample(layer)
        else:
            return self.generate_simple_block(layer)
    
def resnet(input_shape, kernel_size, filters, dropout_rate, output_ch):
    with tf.device('/gpu:0'):
        inputs = Input(input_shape, name='input')
        
        down1 = Conv2D(filters, kernel_size, padding='same')(inputs)
        down2 = SimpleBlock(filters, kernel_size, dropout_rate, 'blue')(down1)
        down3 = Bottleneck(4*filters, kernel_size, dropout_rate, 3, 'blue')(down2)
        down4 = Bottleneck(8*filters, kernel_size, dropout_rate, 8, 'blue')(down3)
        down5 = Bottleneck(16*filters, kernel_size, dropout_rate, 10, 'blue')(down4)
        
        across = Bottleneck(32*filters, kernel_size, dropout_rate, 3, 'white')(down5)
        
    with tf.device('/gpu:1'):
        
        up1 = add([down5, Conv2D(16*filters, (1,1), padding='same')(across)])
        up1 = Bottleneck(16*filters, kernel_size, dropout_rate, 10, 'yellow')(up1)
        up2 = add([down4, Conv2D(8*filters, (1,1), padding='same')(up1)])
        up2 = Bottleneck(8*filters, kernel_size, dropout_rate, 8, 'yellow')(up2)
        up3 = add([down3, Conv2D(4*filters, (1,1), padding='same')(up2)])
        up3 = Bottleneck(4*filters, kernel_size, dropout_rate, 3, 'yellow')(up3)
        up4 = add([down2, Conv2D(filters, (1,1), padding='same')(up3)])
        up4 = SimpleBlock(filters, kernel_size, dropout_rate, 'yellow')(up4)
        up5 = add([down1, Conv2D(filters, (1,1), padding='same')(up4)])
        up5 = Conv2D(filters, kernel_size, padding='same')(up5)
        
        out = Conv2D(output_ch, (1,1), activation='softmax', name='mask')(up5)
        
    model = Model(inputs=inputs, outputs=out)
    return model