import sys
from keras import backend as K
from keras.optimizers import *
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras
import warnings

from src.generator_functions import *
from src.losses import *
from src.operations import *
from src.datasets import *
from src.loggers import *

from src.models.micronet252 import *

def read_sys():
    assert len(sys.argv)==9 or len(sys.argv)==5
    num_gpu = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    dataset_name = sys.argv[3]
    if len(sys.argv)==9:
        img_shape = tuple([int(i) for i in sys.argv[4].split(',')])
        model_shape = tuple([int(i) for i in sys.argv[5].split(',')])
        train_stride = tuple([int(i) for i in sys.argv[6].split(',')])
        bnd_dilate = int(sys.argv[7])
        model_name = sys.argv[8]
        input_ch, output_ch, _, _, _, _, _ = DATASET_STATS[dataset_name]
    else:
        input_ch, output_ch, img_shape, model_shape, train_stride, test_stride, bnd_dilate = DATASET_STATS[dataset_name]
        model_name = sys.argv[4]
    return num_gpu, batch_size, dataset_name, img_shape, model_shape, train_stride, bnd_dilate, model_name, input_ch, output_ch

def main():
    '''
    python train_micronet252_wce.py [num_gpu] [batch_size] [dataset_name] [img_shape] [model_shape] [stride] [bnd_dilate] [model_name]
    or
    python train_micronet252_wce.py [num_gpu] [batch_size] [dataset_name] [model_name]
    '''
    num_gpu, batch_size, dataset_name, img_shape, model_shape, train_stride, bnd_dilate, model_name, input_ch, output_ch = read_sys()
    
    # changable parameters
    num_of_features = 64
    dropout_rate = 0.5
    optimizer = Adam(1e-4)
    patience = 100
    epochs = 1000
    batch_normalization = True
    
    allocate_GPU(num_gpu)
    model = micronet252(num_of_features, dropout_rate, output_ch, batch_normalization)
    model.summary()
    
    train_generator = get_dataset(dataset_name, 'train', img_shape, model_shape, train_stride, batch_size, True, bnd_dilate)
    validation_generator = get_dataset(dataset_name, 'validation', img_shape, model_shape, train_stride, batch_size, False, bnd_dilate)
    
    p = 0.5
    angle = 360
    horizontal_flip = RandomHorizontalFlip(p)
    vertical_flip = RandomVerticalFlip(p)
    gaussian_blur = RandomGaussianBlur(p)
    median_blur = RandomMedianBlur(p)
    rotation = RandomRotation(angle)
#     hue = RandomHue()
#     saturation = RandomSaturation()
#     brightness = RandomBrightness()
#     contrast = RandomContrast()
    
    weight_map = WeightMapGenerator()
    train_nearest_cell = NearestCellGenerator()
    null_map = NullMapGenerator('mask_nearest_cell_map')
    micronet_generator = MicroNet252Generator()
    train_normalization = Normalize()
    validation_normalization = Normalize()
    aux_generator = Micronet252AuxGenerator()
    
    weight_map.initialize(train_generator)
    train_nearest_cell.initialize(train_generator)
    
    train_generator = LambdaGenerator(
        train_generator,
        [
            weight_map,
            train_nearest_cell,
            horizontal_flip,
            vertical_flip,
            rotation,
            gaussian_blur,
            median_blur,
#             hue,
#             saturation,
#             brightness,
#             contrast,
            train_normalization,
            micronet_generator,
            aux_generator
        ]
    )
    validation_generator = LambdaGenerator(
        validation_generator,
        [
            validation_normalization,
            micronet_generator,
            null_map,
            aux_generator
        ]
    )
    
    mask_weight_map = Input(shape=model_shape, name='mask_nearest_cell_map')
    aux_weight_map = Input(shape=model_shape, name='aux_weight_map')
    model = Model(inputs=model.input + [mask_weight_map, aux_weight_map], outputs=model.output)
    mask_loss = weighted_crossentropy(mask_weight_map)
    aux_loss = weighted_crossentropy(aux_weight_map)
    
    model.compile(optimizer=optimizer, loss={'mask': mask_loss, 'aux1': aux_loss, 'aux2': aux_loss, 'aux3': aux_loss}, metrics=['categorical_accuracy'])
    
    logger = ModelDetailsLogger(DATASET_PATHS[dataset_name], model_name)
    logger.log([
        ('generator', train_generator),
        ('number of features', num_of_features),
        ('dropout rate', dropout_rate),
        ('optimizer', optimizer),
        ('patience', patience),
        ('batch normalization', batch_normalization)
    ])
    
    lr_scheduler = LearningRateScheduler(lambda epoch : 1.0e-5 if epoch>125 else 1.0e-4, verbose=1)
    checkpoint = ModelCheckpoint(
        os.path.join(DATASET_PATHS[dataset_name], 'models', '{}-temp.h5'.format(model_name)),
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True,
        mode='auto',
        period=1
    )
    aux_checkpoint = AuxModelCheckpoint(os.path.join(DATASET_PATHS[dataset_name], 'models', '{}-temp.h5'.format(model_name)))
    early_stopping = EarlyStopping(patience=patience, verbose=1, monitor='val_loss')
    log_dir = os.path.join(DATASET_PATHS[dataset_name], 'output', 'tensorboard')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, model_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=False)
    callbacks = [checkpoint, aux_checkpoint, early_stopping, tensorboard, lr_scheduler]
    train(model, train_generator, validation_generator, epochs, callbacks)
    
if __name__=='__main__':
    main()