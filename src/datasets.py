from src.data_generators import *
import pickle
import os

DATASET_STATS = {
    # (input_ch, output_ch, img_shape, model_shape, train_stride, test_stride, bnd_dilate)
    'cama' : (3, 2, None, (512,512), (256,256), (128,128), 9),
    'nucseg' : (3, 2, None, (768,1024), (1,1), (1,1), 5),
    'gland' : (3, 2, None, (480,640), (1,1), (1,1), 5),
    'decathlon5' : (2, 3, None, (320,320), (1,1), (1,1), 1),
    'kato' : (3, 3, None, (252,252), (128,128), (64,64), 5),
    'kato_unet' : (3, 3, None, (512,512), (256,256), (128,128), 5),
    'monusac' : (4, 5, None, (252,252), (252,252), (128,128), 3),
    'consep' : (3, 5, None, (252,252), (128,128), (64,64), 1),
    'consep_inst' : (3, 2, None, (1000,1000), (1,1), (1,1), 1),
    'nuclei' : (3, 2, None, (256,256), (128,128), (64,64), 5)
}

DATASET_PATHS = {
    'cama' : '/home/furkanh/Projects/media/CAMA_MDA_MB',
    'nucseg' : '/home/furkanh/Projects/media/NucleusSegData',
    'gland' : '/home/furkanh/Projects/media/GlandSegmentation',
    'decathlon5' : '/home/furkanh/Projects/media/Decathlon/task5',
    'kato' : '/home/furkanh/Projects/media/kato',
    'kato_unet' : '/home/furkanh/Projects/media/kato',
    'monusac' : '/home/furkanh/Projects/media/MoNuSAC',
    'consep' : '/home/furkanh/Projects/media/CoNSeP',
    'consep_inst' : '/home/furkanh/Projects/media/CoNSeP_Instance',
    'nuclei' : '/home/furkanh/Projects/media/NucleiSegm'
}

DATASETS = {
    'cama' : CAMA_MDA_MB,
    'nucseg' : NucSegGenerator,
    'gland' : GlandDataset,
    'decathlon5' : DecathlonTask5,
    'kato' : KatoGenerator,
    'kato_unet' : KatoGenerator,
    'monusac' : MoNuSAC,
    'consep' : CoNSeP,
    'consep_inst' : CoNSeP_Instance,
    'nuclei': NucleiSegm
}

def load_generator(path, data_type, img_shape, bnd_dilate):
    file_path = os.path.join(path, 'generators/{}_{}_{}.pickle'.format(data_type, img_shape, bnd_dilate))
    if os.path.exists(file_path):
        file = open(file_path, 'rb')
        obj = pickle.load(file)
        file.close()
        return obj
    return None

def save_generator(path, generator):
    data_type = generator.data_type
    img_shape = generator.img_shape
    bnd_dilate = generator.bnd_dilate
    if not os.path.exists(os.path.join(path, 'generators')):
        os.mkdir(os.path.join(path, 'generators'))
    file_path = os.path.join(path, 'generators/{}_{}_{}.pickle'.format(data_type, img_shape, bnd_dilate))
    file = open(file_path, 'wb')
    pickle.dump(generator, file)
    file.close()

def get_dataset(
    dataset_name,
    data_type,
    img_shape,
    model_shape,
    stride,
    batch_size,
    shuffle,
    bnd_dilate
):
    assert dataset_name in DATASET_PATHS
    if os.path.exists(os.path.join(DATASET_PATHS[dataset_name], 'generators/{}_{}_{}.pickle'.format(data_type, img_shape, bnd_dilate))):
        generator = load_generator(DATASET_PATHS[dataset_name], data_type, img_shape, bnd_dilate)
        generator.model_shape = model_shape
        generator.stride = stride
        generator.batch_size = batch_size
        generator.shuffle = shuffle
        generator.initialize_data()
        return generator
    dataset = DATASETS[dataset_name]
    try:
        generator = dataset(DATASET_PATHS[dataset_name], data_type, model_shape, stride, batch_size, shuffle, bnd_dilate, convert_on_init=False, img_shape=img_shape)
    except:
        print('Generating Dataset')
        generator = dataset(DATASET_PATHS[dataset_name], data_type, model_shape, stride, batch_size, shuffle, bnd_dilate, convert_on_init=True, img_shape=img_shape)
    save_generator(DATASET_PATHS[dataset_name], generator)
    return generator
