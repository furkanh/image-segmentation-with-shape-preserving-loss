'''

Nuclei Segmentation in Histopathological Images Using Two-Stage Learning

    https://link.springer.com/chapter/10.1007/978-3-030-32239-7_78

'''

import keras
import tensorflow as tf
from keras.layers import *
from keras.models import *
    
def two_stage_dla_unet(
    input_shape,
    kernel_size,
    filters,
    output_ch
):
    
    with tf.device('/gpu:0'):
        inputs = Input(input_shape, name='input')
        
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(inputs)
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(conv1)

        pool = MaxPooling2D(pool_size=(2,2))(conv1)
        conv2 = Conv2D(2*filters, kernel_size, padding='same', activation='relu')(pool)
        conv2 = Conv2D(2*filters, kernel_size, padding='same', activation='relu')(conv2)

        pool = MaxPooling2D(pool_size=(2,2))(conv2)
        conv3 = Conv2D(4*filters, kernel_size, padding='same', activation='relu')(pool)
        conv3 = Conv2D(4*filters, kernel_size, padding='same', activation='relu')(conv3)

        pool = MaxPooling2D(pool_size=(2,2))(conv3)
        conv4 = Conv2D(8*filters, kernel_size, padding='same', activation='relu')(pool)
        conv4 = Conv2D(8*filters, kernel_size, padding='same', activation='relu')(conv4)

        pool = MaxPooling2D(pool_size=(2,2))(conv4)
        conv5 = Conv2D(16*filters, kernel_size, padding='same', activation='relu')(pool)
        conv5 = Conv2D(16*filters, kernel_size, padding='same', activation='relu')(conv5)

        deconv = Conv2DTranspose(filters, kernel_size, strides=(2,2), padding='same')(conv2)
        concat = concatenate([conv1, deconv], axis=-1)
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(concat)
    
        deconv = Conv2DTranspose(2*filters, kernel_size, strides=(2,2), padding='same')(conv3)
        concat = concatenate([conv2, deconv], axis=-1)
        conv2 = Conv2D(2*filters, kernel_size, padding='same')(concat)

        deconv = Conv2DTranspose(4*filters, kernel_size, strides=(2,2), padding='same')(conv4)
        concat = concatenate([conv3, deconv], axis=-1)
        conv3 = Conv2D(4*filters, kernel_size, padding='same')(concat)
        
    with tf.device('/gpu:1'):
        deconv = Conv2DTranspose(8*filters, kernel_size, strides=(2,2), padding='same')(conv5)
        concat = concatenate([conv4, deconv], axis=-1)
        conv4 = Conv2D(8*filters, kernel_size, padding='same')(concat)
        conv4 = Conv2D(8*filters, kernel_size, padding='same')(conv4)
        
        deconv = Conv2DTranspose(filters, kernel_size, strides=(2,2), padding='same')(conv2)
        concat = concatenate([conv1, deconv], axis=-1)
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(concat)

        deconv = Conv2DTranspose(2*filters, kernel_size, strides=(2,2), padding='same')(conv3)
        concat = concatenate([conv2, deconv], axis=-1)
        conv2 = Conv2D(2*filters, kernel_size, padding='same', activation='relu')(concat)

        deconv = Conv2DTranspose(4*filters, kernel_size, strides=(2,2), padding='same')(conv4)
        concat = concatenate([conv3, deconv], axis=-1)
        conv3 = Conv2D(4*filters, kernel_size, padding='same', activation='relu')(concat)
        conv3 = Conv2D(4*filters, kernel_size, padding='same', activation='relu')(conv3)

        deconv = Conv2DTranspose(filters, kernel_size, strides=(2,2), padding='same')(conv2)
        concat = concatenate([conv1, deconv], axis=-1)
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(concat)

        deconv = Conv2DTranspose(2*filters, kernel_size, strides=(2,2), padding='same')(conv3)
        concat = concatenate([conv2, deconv], axis=-1)
        conv2 = Conv2D(2*filters, kernel_size, padding='same', activation='relu')(concat)
        conv2 = Conv2D(2*filters, kernel_size, padding='same', activation='relu')(conv2)
        
        deconv = Conv2DTranspose(filters, kernel_size, strides=(2,2), padding='same')(conv2)
        concat = concatenate([conv1, deconv], axis=-1)
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(concat)
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(conv1)
        
    with tf.device('/gpu:2'):
        out1 = Conv2D(3, (1,1), padding='same', activation='softmax', name='mask_bnd')(conv1)

        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(conv1)
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(conv1)

        pool = MaxPooling2D(pool_size=(2,2))(conv1)
        conv2 = Conv2D(2*filters, kernel_size, padding='same', activation='relu')(pool)
        conv2 = Conv2D(2*filters, kernel_size, padding='same', activation='relu')(conv2)

        pool = MaxPooling2D(pool_size=(2,2))(conv2)
        conv3 = Conv2D(4*filters, kernel_size, padding='same', activation='relu')(pool)
        conv3 = Conv2D(4*filters, kernel_size, padding='same', activation='relu')(conv3)

        deconv = Conv2DTranspose(filters, kernel_size, strides=(2,2), padding='same')(conv2)
        concat = concatenate([conv1, deconv], axis=-1)
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(concat)

        deconv = Conv2DTranspose(2*filters, kernel_size, strides=(2,2), padding='same')(conv3)
        concat = concatenate([conv2, deconv], axis=-1)
        conv2 = Conv2D(2*filters, kernel_size, padding='same', activation='relu')(concat)
        conv2 = Conv2D(2*filters, kernel_size, padding='same', activation='relu')(conv2)

        deconv = Conv2DTranspose(filters, kernel_size, strides=(2,2), padding='same')(conv2)
        concat = concatenate([conv1, deconv], axis=-1)
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(concat)
        conv1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(conv1)
        out2 = Conv2D(output_ch, (1,1), padding='same', activation='softmax', name='mask')(conv1)
    
    model = Model(inputs=inputs, outputs=[out1, out2])
    
    return model
