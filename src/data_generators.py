import keras
import skimage
import os
import cv2
import scipy.io
import numpy as np
import random
from keras import backend as K
from keras.models import Model
from keras.callbacks import *
import tensorflow as tf
from src.losses import *
import warnings
import nibabel

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class LambdaGenerator(keras.utils.Sequence):
    '''
    Given a list of LambdaGeneratorFunction, this class calls these
    LambdaGeneratorFunction classes in the given order.
    
    Attributes:
        generator (SemanticSegmentationDataset)
        func_list (list) : list of LambdaGeneratorFunction
    '''
    def __init__(self, generator, func_list):
        self.generator = generator
        self.func_list = func_list
        self.use_default_train = True
        for func in func_list:
            self.use_default_train = self.use_default_train and func.use_default_train
        
    def on_epoch_end(self):
        self.generator.on_epoch_end()
    
    def __len__(self):
        return self.generator.__len__()
        
    def __getitem__(self, index):
        x, y = self.generator.__getitem__(index)
        for func in self.func_list:
            x, y = func(self.generator, x, y)
        return x, y
    
    def get_dataset_as_batch(self):
        x, y = self.generator.get_dataset_as_batch()
        for func in self.func_list:
            x, y = func(self.generator, x, y)
        return x, y
    
    def __repr__(self):
        string = '\n'
        string += str(self.generator)
        for func in self.func_list:
            string += '\n'
            string += str(func)
        string += '\n'
        return string

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class SegmentationDataset(keras.utils.Sequence):
    '''
    Generates x and y for training and testing the network.
    x['input'], y['mask'], y['bnd'], y['stride_stats'] are available.
    
    y['stride_stats'] includes image cutting information path, h and w.
    
    Attributes:
        epoch (int): Shows epoch number during training
        
    Args:
        path (str): dataset path
        data_type (str): train, validation or test (can include different test names)
        img_shape (tuple): (H,W)
        model_shape (tuple): (H,W)
        stride (tuple): (H,W)
        batch_size (int):
        shuffle (boolean): shuffles dataset if True
        bnd_dilate (int): size of the structuring element
            which is used to dilate boundaries
        data (list): list of tuples (path, h, w)
        convert_on_init (boolean): convert function will be called in initalization
        remove_bnd (boolean): boundaries are subtracted from the mask
    '''
    def __init__(
        self,
        path,
        data_type,
        model_shape,
        stride,
        batch_size,
        shuffle,
        bnd_dilate,
        remove_bnd,
        convert_on_init=False,
        img_shape=None
    ):
        
        self.data_type = data_type
        self.path = path
        self.bnd_dilate = bnd_dilate
        self.img_shape = img_shape
        self.stride = stride
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epoch = 0
        self.model_shape = model_shape
        self.remove_bnd = remove_bnd
        
        if not os.path.exists(os.path.join(self.path, 'models')):
            os.mkdir(os.path.join(self.path, 'models'))
        if not os.path.exists(os.path.join(self.path, 'output')):
            os.mkdir(os.path.join(self.path, 'output'))
        
        if convert_on_init:
            self.convert()
        
        self.initialize()
        
    def __getitem__(self, index):
        data = self.data[index*self.batch_size:(index+1)*self.batch_size]
        x, y = {}, {}
        shape = (len(data),)+self.model_shape
        key = list(self.global_data.keys())[0]
        ch = self.global_data[key]['x'].shape[-1]
        x['input'] = np.zeros(shape+(ch,), dtype=np.float32)
        y['mask'] = np.zeros(shape+(self.num_of_classes+1,) , dtype=np.float32)
        y['bnd'] = np.zeros(shape+(2,), dtype=np.float32)
        y['stride_stats'] = []
        for i, (path, h, w) in enumerate(data):
            y['stride_stats'].append((path, h, w))
            img = self.global_data[path]['x'][h:h+self.model_shape[0],w:w+self.model_shape[1]]
            ann_mask = self.global_data[path]['y_mask'][h:h+self.model_shape[0],w:w+self.model_shape[1]]
            ann_bnd = self.global_data[path]['y_bnd'][h:h+self.model_shape[0],w:w+self.model_shape[1]]
            # assign images
            x['input'][i,:img.shape[0],:img.shape[1]] = np.float32(img)
            y['mask'][i,:ann_mask.shape[0],:ann_mask.shape[1]] = np.float32(ann_mask)
            y['bnd'][i,:ann_bnd.shape[0],:ann_mask.shape[1]] = np.float32(ann_bnd)
        return x, y
        
    def convert(self):
        '''
        Implement this function to convert dataset to required format.
        Each data point (cell image and annotation) should be saved into a .mat file.
        This mat file should have 2 keys 'x', 'y'.
        'x' is the image of shape (H,W,C) where C is the number of channels
        'y' is the annotation of shape (H,W,C) where C is the number of classes
        '''
        pass
        
    def initialize_global_data(self):
        folder_name = self.data_type
        self.global_data = {}
        self.num_of_classes = None
        assert os.path.exists(os.path.join(self.path, folder_name))
        for file in sorted(os.listdir(os.path.join(self.path, folder_name))):
            if file.endswith('.mat'):
                path = os.path.join(self.path, folder_name, file)
                self.global_data[path] = {}
                if self.num_of_classes is None:
                    mat = scipy.io.loadmat(path)
                    self.num_of_classes = int(mat['y'].shape[-1])
            
    def read_global_data(self):
        for path in self.global_data:
            mat = scipy.io.loadmat(path)
            self.global_data[path]['x'] = mat['x']
            y = mat['y']
            if len(y.shape)==2:
                y = y[...,np.newaxis]
            self.global_data[path]['y'] = y
            
    def get_boundaries(self, cell):
        cell = cell.astype(np.uint8)
        contour_img = np.zeros_like(cell)
        _, contours, _ = cv2.findContours(cell.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            for pixel in contour:
                contour_img[pixel[0][1], pixel[0][0]] = 1
        if self.bnd_dilate>1:
            strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.bnd_dilate, self.bnd_dilate))
            dilated = cv2.dilate(contour_img, strel)
            return np.int32(dilated)
        return np.int32(contour_img)
    
    def resize(self, data):
        if self.img_shape is not None:
            resized = cv2.resize(data, (self.img_shape[1], self.img_shape[0]), interpolation=cv2.INTER_NEAREST)
            if len(resized.shape)==2:
                resized = resized[...,np.newaxis]
            return resized
        return data
    
    def preprocess_global_data(self):
        for path in self.global_data:
            self.global_data[path]['x'] = self.resize(self.global_data[path]['x'])
            mask = self.resize(self.global_data[path]['y'].copy())
            bnd = np.zeros(mask.shape[:-1]+(1,), dtype=np.int32)
            for i in range(mask.shape[-1]):
                channel = np.int32(mask[:,:,i])
                for label in range(1, channel.max()+1):
                    cell = np.int32(channel==label)
                    bnd[:,:,0] += self.get_boundaries(cell)*(bnd[:,:,0]==0)
            bnd = np.concatenate((bnd.sum(axis=-1, keepdims=True)==0, bnd), -1)
            mask = np.int32(mask>0)
            for i in range(1,mask.shape[-1]):
                for j in range(i):
                    mask[:,:,i] = mask[:,:,i]*(mask[:,:,j]==0)
            if self.remove_bnd:
                for i in range(mask.shape[-1]):
                    mask[:,:,i] = mask[:,:,i]*bnd[:,:,0]
            mask = np.concatenate((mask.sum(axis=-1, keepdims=True)==0, mask), -1)
            self.global_data[path]['y_mask'] = np.float32(mask)
            self.global_data[path]['y_bnd'] = np.float32(bnd)
            
    def initialize(self):
        self.initialize_global_data()
        self.read_global_data()
        self.preprocess_global_data()
        self.initialize_data()
        self.on_epoch_end()
        
    def initialize_data(self):
        self.data = []
        for path in self.global_data:
            height = self.global_data[path]['x'].shape[0]
            width = self.global_data[path]['x'].shape[1]
            num_of_height = ((height-self.model_shape[0])//self.stride[0]+1)
            num_of_width = ((width-self.model_shape[1])//self.stride[1]+1)
            num_of_cuts = num_of_height*num_of_width
            h_fit = (height-self.model_shape[0])%self.stride[0]!=0
            w_fit = (width-self.model_shape[1])%self.stride[1]!=0
            for i in range(num_of_cuts):
                h = self.stride[0]*(i//num_of_width)
                w = self.stride[1]*(i%num_of_width)
                self.data.append((path, h, w))
            if h_fit:
                for i in range(num_of_width):
                    w = self.stride[1]*(i%num_of_width)
                    h = max(0, height-self.model_shape[0])
                    self.data.append((path, h, w))
            if w_fit:
                for i in range(num_of_height):
                    h = self.stride[0]*(i%num_of_height)
                    w = max(0, width-self.model_shape[1])
                    self.data.append((path, h, w))
            if h_fit and w_fit:
                w = max(0, width-self.model_shape[1])
                h = max(0, height-self.model_shape[0])
                self.data.append((path, h, w))
        if self.shuffle == True:
            random.shuffle(self.data)
                    
    def on_epoch_end(self):
        self.epoch += 1
        if self.shuffle == True:
            random.shuffle(self.data)
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def get_dataset_as_batch(self):
        temp_batch_size = self.batch_size
        self.batch_size = len(self.data)
        x, y = self.__getitem__(0)
        self.batch_size = temp_batch_size
        return x, y
    
    def __repr__(self):
        string = 'model shape: {}\n'.format(self.model_shape)
        string += 'image shape: {}\n'.format(self.img_shape)
        string += 'stride: {}\n'.format(self.stride)
        string += 'boundary dilate: {}\n'.format(self.bnd_dilate)
        string += 'batch size: {}\n'.format(self.batch_size)
        string += '\n'
        return string
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class CellDataset(SegmentationDataset):
    def __init__(
        self,
        path,
        data_type,
        model_shape,
        stride,
        batch_size,
        shuffle,
        bnd_dilate,
        convert_on_init=False,
        img_shape=None
    ):
        super(CellDataset, self).__init__(path, data_type, model_shape, stride, batch_size, shuffle, bnd_dilate, True, convert_on_init=convert_on_init, img_shape=img_shape)

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class CTMRDataset(SegmentationDataset):
    def __init__(
        self,
        path,
        data_type,
        model_shape,
        stride,
        batch_size,
        shuffle,
        bnd_dilate,
        convert_on_init=False,
        img_shape=None
    ):
        super(CTMRDataset, self).__init__(path, data_type, model_shape, stride, batch_size, shuffle, bnd_dilate, False, convert_on_init=convert_on_init, img_shape=img_shape)

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class DecathlonTask5(CTMRDataset):
    '''
    https://decathlon.grand-challenge.org/
    '''
    def convert_indexes(self, indexes, prefix):
        j = 1
        for index in indexes:
            path = 'src/imagesTr/prostate_{}.nii.gz'.format('0'+str(index) if index<10 else index)
            path = os.path.join(self.path, path)
            images = nibabel.load(path).get_fdata()
            path = 'src/labelsTr/prostate_{}.nii.gz'.format('0'+str(index) if index<10 else index)
            path = os.path.join(self.path, path)
            labels = nibabel.load(path).get_fdata()
            if labels.shape[:-1]==(320,320):
                for i in range(labels.shape[-1]):
                    mat = {}
                    img = images[:,:,i,:]
                    label = labels[:,:,i]
                    mat['x'] = np.float32(img)
                    mat['y'] = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
                    _, connected_components = cv2.connectedComponents(np.uint8(label==1), connectivity=4)
                    mat['y'][:,:,0] = np.float32(connected_components)
                    _, connected_components = cv2.connectedComponents(np.uint8(label==2), connectivity=4)
                    mat['y'][:,:,1] = np.float32(connected_components)
                    scipy.io.savemat(os.path.join(self.path, '{}{}.mat'.format(prefix, j)), mat)
                    j += 1
    
    def convert(self):
        num = 48
        test_num = int(0.2*num)
        validation_num = int(0.2*(num-test_num))
        indexes = np.array([0,1,2,4,6,7,10,13,14,16,17,18,20,21,24,25,28,29,31,32,34,35,37,38,39,40,41,42,43,44,46,47])
        np.random.seed(42)
        np.random.shuffle(indexes)
        test_indexes = indexes[:test_num]
        validation_indexes = indexes[test_num:test_num+validation_num]
        train_indexes = indexes[test_num+validation_num:]
        if not os.path.exists(os.path.join(self.path, 'train')):
            os.mkdir(os.path.join(self.path, 'train'))
        if not os.path.exists(os.path.join(self.path, 'validation')):
            os.mkdir(os.path.join(self.path, 'validation'))
        if not os.path.exists(os.path.join(self.path, 'test')):
            os.mkdir(os.path.join(self.path, 'test'))
        self.convert_indexes(train_indexes, 'train/tr_')
        self.convert_indexes(validation_indexes, 'validation/vl_')
        self.convert_indexes(test_indexes, 'test/ts_')
            
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class GlandDataset(CellDataset):
    def convert_indexes(self, num, gold_prefix, prefix):
        j = 1
        for i in range(1, num+1):
            mat = {}
            gold_path = os.path.join(self.path, 'src/gold/{}{}.gold'.format(gold_prefix, i))
            file = open(gold_path, 'r')
            line = next(file).split()
            gold = [[int(pixel) for pixel in line.split()] for line in file]
            file.close()
            gold = np.array(gold)
            mat['y'] = np.float32(gold[...,np.newaxis])
            img_path = os.path.join(self.path, 'src/images/{}{}.jpg'.format(gold_prefix, i))
            mat['x'] = np.float32(skimage.io.imread(img_path))
            scipy.io.savemat(os.path.join(self.path, prefix+str(j)+'.mat'), mat)
            j += 1

    def convert(self):
        if not os.path.exists(os.path.join(self.path, 'train')):
            os.mkdir(os.path.join(self.path, 'train'))
        if not os.path.exists(os.path.join(self.path, 'test')):
            os.mkdir(os.path.join(self.path, 'test'))
        if not os.path.exists(os.path.join(self.path, 'validation')):
            os.mkdir(os.path.join(self.path, 'validation'))
        self.convert_indexes(80, 'gtr', 'train/tr_')
        self.convert_indexes(100, 'gts', 'test/ts_')
        self.convert_indexes(20, 'gval', 'validation/vl_')
        
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
            
class CAMA_MDA_MB(CellDataset):
    '''
    Generates CAMA_MDA_MB Dataset
    '''
    def convert_names(self, names, prefix):
        j = 1
        for name in names:
            mat = {}
            ann_file_name = 'src/gold/' + name + '_gs.mat'
            img_file_name = 'src/images/' + name + '.jpg'
            mat['x'] = np.float32(skimage.io.imread(os.path.join(self.path, img_file_name)))
            ann_mat = scipy.io.loadmat(os.path.join(self.path, ann_file_name))
            ann = ann_mat['gold']
            mat['y'] = np.float32(ann[...,np.newaxis])
            scipy.io.savemat(os.path.join(self.path, prefix+str(j)+'.mat'), mat)
            j += 1
        
    def convert(self):
        valid_names = [
            'MDA_MB_453_100610_20x_03',
            'CamaI_020710_20x_02'
        ]
        train_names = [
            'CamaI_040710_20x_02',
            'CamaI_050710_20x_02',
            'MDA_MB_453_090610_20x_03',
            'MDA_MB_453_100610_20x_03',
            'MDA_MB_453_120610_20x_04'
        ]
        cama_names = [
            'CamaI_010710_20x_02',
            'CamaI_010710_20x_04',
            'CamaI_020710_20x_03',
            'CamaI_030710_20x_04',
            'CamaI_040710_20x_04',
            'CamaI_050710_20x_04'
        ]
        mda453_names = [
            'MDA_MB_453_090610_20x_01',
            'MDA_MB_453_090610_20x_02',
            'MDA_MB_453_090610_20x_04',
            'MDA_MB_453_110610_20x_05'
        ]
        mda468_names = [
            'MDA_MB_468_090610_20x_02',
            'MDA_MB_468_090610_20x_04',
            'MDA_MB_468_100610_20x_01',
            'MDA_MB_468_100610_20x_03',
            'MDA_MB_468_100610_20x_05',
            'MDA_MB_468_110610_20x_04',
            'MDA_MB_468_110610_20x_05',
            'MDA_MB_468_120610_20x_03'
        ]
        if not os.path.exists(os.path.join(self.path, 'train')):
            os.mkdir(os.path.join(self.path, 'train'))
        if not os.path.exists(os.path.join(self.path, 'cama')):
            os.mkdir(os.path.join(self.path, 'cama'))
        if not os.path.exists(os.path.join(self.path, 'mda453')):
            os.mkdir(os.path.join(self.path, 'mda453'))
        if not os.path.exists(os.path.join(self.path, 'mda468')):
            os.mkdir(os.path.join(self.path, 'mda468'))
        if not os.path.exists(os.path.join(self.path, 'validation')):
            os.mkdir(os.path.join(self.path, 'validation'))
        self.convert_names(valid_names, 'validation/vl_')
        self.convert_names(train_names, 'train/tr_')
        self.convert_names(cama_names, 'cama/cama_')
        self.convert_names(mda453_names, 'mda453/mda453_')
        self.convert_names(mda468_names, 'mda468/mda468_')

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class NucSegGenerator(CellDataset):
    '''
    Generates Nucleus Segmentation Dataset
    
    Nucleus Segmentation Dataset is available at:
        https://link.springer.com/chapter/10.1007/978-3-319-66182-7_24
    '''
    def convert_indexes(self, indexes, prefix, new_prefix):
        j = 1
        for i in indexes:
            mat = {}
            img_file_name = prefix + str(i) + '.jpg'
            ann_file_name = prefix + '_ann' + str(i)
            mat['x'] = skimage.io.imread(os.path.join(self.path, img_file_name)).astype(np.float32)
            file = open(os.path.join(self.path, ann_file_name))
            ann = np.asarray([[int(pixel) for pixel in line.split()] for line in file])
            mat['y'] = ann[...,np.newaxis]
            scipy.io.savemat(os.path.join(self.path, new_prefix+str(j)+'.mat'), mat)
            j += 1
        
    def convert(self):
        if not os.path.exists(os.path.join(self.path, 'train')):
            os.mkdir(os.path.join(self.path, 'train'))
        if not os.path.exists(os.path.join(self.path, 'huh7')):
            os.mkdir(os.path.join(self.path, 'huh7'))
        if not os.path.exists(os.path.join(self.path, 'hepg2')):
            os.mkdir(os.path.join(self.path, 'hepg2'))
        if not os.path.exists(os.path.join(self.path, 'validation')):
            os.mkdir(os.path.join(self.path, 'validation'))
        # convert train and validation
        indexes = np.asarray([i for i in range(0,10)])
        img_indexes = np.asarray([i for i in range(1,11)])
        num = 8
        train_indexes = img_indexes[indexes[:num]]
        validation_indexes = img_indexes[indexes[num:]]
        self.convert_indexes(train_indexes, 'TrainingSet/tr', 'train/tr_')
        self.convert_indexes(validation_indexes, 'TrainingSet/tr', 'validation/vl_')
        #convert hepg2
        hepg2_indexes = np.asarray([i for i in range(1,17)])
        self.convert_indexes(hepg2_indexes, 'HepG2TestSet/hepg2_ts', 'hepg2/hepg2_')
        #convert huh7
        huh7_indexes = np.asarray([i for i in range(1,12)])
        self.convert_indexes(huh7_indexes, 'Huh7TestSet/huh7_ts', 'huh7/huh7_')

       ###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class KatoGenerator(CellDataset):
    '''
    Generates Kato Segmentation Dataset
    
    We have divided the dataset as follows:
    
    train : [1,2,7,8,10,14,15,16,19,21,25,26,27,28,29,33,35,36,37,38]
    validation : [12,22,34,41]
    test : [3,4,5,6,9,11,13,17,18,20,23,24,30,31,32,39,40,42,43,44]
    
    KATO3-SEGMENTATION

    This dataset contains 1952 cells taken from 44 microscopic images. These are
    taken from KATO-3 gastric carcinoma cell line. The objective lens is 40x. 

    The images are divided into training and test sets. The index of the images:
    tr_ind = [   2    7   10   16   19   23   28   32   36   41   ];
    ts_ind = [   1    3    4    5    6    8    9   11   12   13   ...
            14   15   17   18   20   21   22   24   25   26   ...
                27   29   30   31   33   34   35   37   38   39   ...
                40   42   43   44   ];
    The distribution of these datasets are:
    Training:   10 images,   473 cells,   447 live,   26 apoptotic
    Test:       34 images,  1479 cells,  1390 live,   89 apoptotic
    Total:      44 images,  1952 cells,  1837 live,  115 apoptotic

    The kato3-gold-standard file contains the gold standard in the following format:
    the first line --> number of cells
    after the first line, each line -->   image-no   centX   centY   label
    label 1 --> live
    label 2 --> apoptotic

    The marked folder contains the images on which the centroids of the gold 
    standard are shown. Here red indicates live and blue indicates apoptotic cells.


    Note that when you work in Matlab
    imread('kato1.jpg')   gives    1024x1360x3    matrix and
    centX can be in between 0 and 1023 and
    centY can be in between 0 and 1359
    '''
    def get_class(self, ann, img_num):
        file = open(os.path.join(self.path,'src/kato3-gold-standard'), 'r')
        num = int(file.readline())
        for i in range(num):
            line = file.readline().split()
            line_img_num = int(line[0])
            x, y = int(line[1]), int(line[2])
            cell_info = int(line[3])
            if line_img_num==img_num and ann[x,y]>0:
                file.close()
                return cell_info
        file.close()
        return 1
        
    def convert(self):
        if not os.path.exists(os.path.join(self.path, 'train')):
            os.mkdir(os.path.join(self.path, 'train'))
        if not os.path.exists(os.path.join(self.path, 'test')):
            os.mkdir(os.path.join(self.path, 'test'))
        if not os.path.exists(os.path.join(self.path, 'validation')):
            os.mkdir(os.path.join(self.path, 'validation'))
        train_indexes = [1,2,7,8,10,14,15,16,19,21,25,26,27,28,29,33,35,36,37,38]
        validation_indexes = [12,22,34,41]
        test_indexes = [3,4,5,6,9,11,13,17,18,20,23,24,30,31,32,39,40,42,43,44]
        ann_path = os.path.join(self.path, 'src/gold-mat-files/gold')
        img_path = os.path.join(self.path, 'src/images/kato')
        train_index = 1
        validation_index = 1
        test_index = 1
        for i in range(1, 45):
            mat = {}
            mat['x'] = skimage.io.imread(img_path+str(i)+'.jpg').astype(np.float32)
            ann = scipy.io.loadmat(ann_path+str(i)+'.mat')
            ann = ann['gold'+str(i)].astype(np.int32)
            labels = np.unique(ann)
            labels = labels[labels>0].astype(np.int32)
            mat['y'] = np.zeros(ann.shape+(2,), dtype=np.float32)
            comps = [1,1]
            for j, label in enumerate(labels):
                y = (ann==int(label)).astype(np.float32)
                c = self.get_class(y, i)
                mat['y'][:,:,c-1] += y.astype(np.float32)*comps[c-1]
                comps[c-1] += 1
            if i in train_indexes:
                scipy.io.savemat(os.path.join(self.path,'train/tr_'+str(train_index)+'.mat'), mat)
                train_index += 1
            elif i in validation_indexes:
                scipy.io.savemat(os.path.join(self.path,'validation/vl_'+str(validation_index)+'.mat'), mat)
                validation_index += 1
            elif i in test_indexes:
                scipy.io.savemat(os.path.join(self.path,'test/ts_'+str(test_index)+'.mat'), mat)
                test_index += 1
            
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class MoNuSAC(CellDataset):
    def convert(self):
        if not os.path.exists(os.path.join(self.path, 'train')):
            os.mkdir(os.path.join(self.path, 'train'))
        if not os.path.exists(os.path.join(self.path, 'test')):
            os.mkdir(os.path.join(self.path, 'test'))
        if not os.path.exists(os.path.join(self.path, 'validation')):
            os.mkdir(os.path.join(self.path, 'validation'))
        path = os.path.join(self.path, 'src', 'MoNuSAC_masks')
        num = sum([len(os.listdir(os.path.join(path, dir_))) for dir_ in os.listdir(path)])
        indexes = np.arange(1,num+1)
        np.random.seed(42)
        np.random.shuffle(indexes)
        test = int(0.2*num)
        validation = int(0.2*0.8*num)
        test_indexes = indexes[:test]
        validation_indexes = indexes[test:test+validation]
        train_indexes = indexes[test+validation:]
        path = os.path.join(self.path, 'src/MoNuSAC_masks')
        train_index = 1
        test_index = 1
        validation_index = 1
        index = 1
        for dir_outer in os.listdir(path):
            for dir_inner in os.listdir(os.path.join(path, dir_outer)):
                img_path = os.path.join(self.path, 'src', 'MoNuSAC_images_and_annotations', dir_outer, dir_inner+'.tif')
                img = skimage.io.imread(img_path)
                ann = np.zeros(img.shape)
                ann_path = os.path.join(path, dir_outer, dir_inner, 'Epithelial')
                if os.path.exists(ann_path) and len(os.listdir(ann_path))>0:
                    ann_path = os.path.join(ann_path, os.listdir(ann_path)[0])
                    ann[:,:,0] = skimage.io.imread(ann_path)
                ann_path = os.path.join(path, dir_outer, dir_inner, 'Lymphocyte')
                if os.path.exists(ann_path) and len(os.listdir(ann_path))>0:
                    ann_path = os.path.join(ann_path, os.listdir(ann_path)[0])
                    ann[:,:,1] = skimage.io.imread(ann_path)
                ann_path = os.path.join(path, dir_outer, dir_inner, 'Macrophage')
                if os.path.exists(ann_path) and len(os.listdir(ann_path))>0:
                    ann_path = os.path.join(ann_path, os.listdir(ann_path)[0])
                    ann[:,:,2] = skimage.io.imread(ann_path)
                ann_path = os.path.join(path, dir_outer, dir_inner, 'Neutrophil')
                if os.path.exists(ann_path) and len(os.listdir(ann_path))>0:
                    ann_path = os.path.join(ann_path, os.listdir(ann_path)[0])
                    ann[:,:,3] = skimage.io.imread(ann_path)
                mat = {}
                mat['x'] = img.astype(np.float32)
                mat['y'] = ann.astype(np.float32)
                if index in train_indexes:
                    scipy.io.savemat(os.path.join(self.path,'train/tr_'+str(train_index)+'.mat'), mat)
                    train_index += 1
                elif index in validation_indexes:
                    scipy.io.savemat(os.path.join(self.path,'validation/vl_'+str(validation_index)+'.mat'), mat)
                    validation_index += 1
                elif index in test_indexes:
                    scipy.io.savemat(os.path.join(self.path,'test/ts_'+str(test_index)+'.mat'), mat)
                    test_index += 1
                index += 1
                    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class CoNSeP(CellDataset):
    '''
    https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/
    '''
    def convert_helper(self, folder, prefix, save_prefix, save_folder, indexes):
        i = 1
        for index in indexes:
            convert_c = lambda x : 1 if x==1 else (2 if x==2 else (3 if x==3 or x==4 else 4))
            mat = {}
            img = skimage.io.imread(os.path.join(self.path,'src',folder,'Images','{}_{}.png'.format(prefix, index)))[:,:,:3]
            ann = np.zeros(img.shape[:-1]+(4,))
            num = np.ones((4,), dtype=np.int32)
            mat = scipy.io.loadmat(os.path.join(self.path,'src',folder,'Labels','{}_{}.mat'.format(prefix, index)))
            for instance in range(len(mat['inst_type'])):
                c = convert_c(int(mat['inst_type'][instance]))
                ann[:,:,c-1] += (mat['inst_map']==(instance+1))*num[c-1]
                num[c-1] += 1
            mat['x'] = np.float32(img)
            mat['y'] = np.float32(ann)
            scipy.io.savemat(os.path.join(self.path,save_folder,'{}_{}.mat'.format(save_prefix, i)), mat)
            i += 1
    
    def convert(self):
        if not os.path.exists(os.path.join(self.path, 'train')):
            os.mkdir(os.path.join(self.path, 'train'))
        if not os.path.exists(os.path.join(self.path, 'test')):
            os.mkdir(os.path.join(self.path, 'test'))
        if not os.path.exists(os.path.join(self.path, 'validation')):
            os.mkdir(os.path.join(self.path, 'validation'))
        train_indexes = np.arange(1,28)
        test_indexes = np.arange(1,15)
        val_num = int(0.2*len(train_indexes))
        np.random.seed(4242)
        np.random.shuffle(train_indexes)
        validation_indexes = train_indexes[:val_num]
        train_indexes = train_indexes[val_num:]
        self.convert_helper('Train', 'train', 'tr', 'train', train_indexes)
        self.convert_helper('Train', 'train', 'vl', 'validation', validation_indexes)
        self.convert_helper('Test', 'test', 'ts', 'test', test_indexes)
        
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class CoNSeP_Instance(CellDataset):
    '''
    https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/
    '''
    def convert_helper(self, folder, prefix, save_prefix, save_folder, indexes):
        i = 1
        for index in indexes:
            convert_c = lambda x : 1
            mat = {}
            img = skimage.io.imread(os.path.join(self.path,'src',folder,'Images','{}_{}.png'.format(prefix, index)))[:,:,:3]
            ann = np.zeros(img.shape[:-1]+(1,))
            num = np.ones((1,), dtype=np.int32)
            mat = scipy.io.loadmat(os.path.join(self.path,'src',folder,'Labels','{}_{}.mat'.format(prefix, index)))
            for instance in range(len(mat['inst_type'])):
                c = convert_c(int(mat['inst_type'][instance]))
                ann[:,:,c-1] += (mat['inst_map']==(instance+1))*num[c-1]
                num[c-1] += 1
            mat['x'] = np.float32(img)
            mat['y'] = np.float32(ann)
            scipy.io.savemat(os.path.join(self.path,save_folder,'{}_{}.mat'.format(save_prefix, i)), mat)
            i += 1
    
    def convert(self):
        if not os.path.exists(os.path.join(self.path, 'train')):
            os.mkdir(os.path.join(self.path, 'train'))
        if not os.path.exists(os.path.join(self.path, 'test')):
            os.mkdir(os.path.join(self.path, 'test'))
        if not os.path.exists(os.path.join(self.path, 'validation')):
            os.mkdir(os.path.join(self.path, 'validation'))
        train_indexes = np.arange(1,28)
        test_indexes = np.arange(1,15)
        val_num = int(0.2*len(train_indexes))
        np.random.seed(4242)
        np.random.shuffle(train_indexes)
        validation_indexes = train_indexes[:val_num]
        train_indexes = train_indexes[val_num:]
        self.convert_helper('Train', 'train', 'tr', 'train', train_indexes)
        self.convert_helper('Train', 'train', 'vl', 'validation', validation_indexes)
        self.convert_helper('Test', 'test', 'ts', 'test', test_indexes)
        
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class NucleiSegm(CellDataset):
    '''
    http://www.andrewjanowczyk.com/deep-learning/
    '''
    def convert(self):
        if not os.path.exists(os.path.join(self.path, 'train')):
            os.mkdir(os.path.join(self.path, 'train'))
        if not os.path.exists(os.path.join(self.path, 'test')):
            os.mkdir(os.path.join(self.path, 'test'))
        if not os.path.exists(os.path.join(self.path, 'validation')):
            os.mkdir(os.path.join(self.path, 'validation'))
        patients = []
        for file_name in os.listdir(os.path.join(self.path, 'src')):
            if file_name.endswith('.png') or file_name.endswith('.tif'):
                patient = file_name.split('_')[0]
                if patient not in patients:
                    patients.append(patient)
        patients = np.array(patients)
        np.random.seed(42)
        np.random.shuffle(patients)
        test_num = int(0.2*len(patients))
        valid_num = int(0.2*int(0.2*len(patients)))
        test_patients = patients[:test_num]
        valid_patients = patients[test_num:test_num+valid_num]
        train_patients = patients[test_num+valid_num:]
        patients_dict = {}
        tr_num = 1
        vl_num = 1
        ts_num = 1
        for patient in patients:
            patients_dict[patient] = []
        for file_name in os.listdir(os.path.join(self.path, 'src')):
            if file_name.endswith('.png') or file_name.endswith('.tif'):
                img = file_name.split('_')[2]
                patient = file_name.split('_')[0]
                if img not in patients_dict[patient]:
                    patients_dict[patient].append(img)
        for patient in patients:
            for identifier in patients_dict[patient]:
                img_path = os.path.join(self.path, 'src', '{}_500_{}_original.tif'.format(patient, identifier))
                ann_path = os.path.join(self.path, 'src', '{}_500_{}_mask.png'.format(patient, identifier))
                img = skimage.io.imread(img_path)
                ann = skimage.io.imread(ann_path)
                mat = {}
                _, ann = cv2.connectedComponents(np.uint8(ann>0))
                ann = ann[...,np.newaxis]
                mat['x'] = np.float32(img[:,:,:3])
                mat['y'] = np.float32(ann)
                if patient in train_patients:
                    scipy.io.savemat(os.path.join(self.path, 'train', 'tr_{}.mat'.format(tr_num)), mat)
                    tr_num += 1
                elif patient in test_patients:
                    scipy.io.savemat(os.path.join(self.path, 'test', 'ts_{}.mat'.format(ts_num)), mat)
                    ts_num += 1
                elif patient in valid_patients:
                    scipy.io.savemat(os.path.join(self.path, 'validation', 'vl_{}.mat'.format(vl_num)), mat)
                    vl_num += 1
                else:
                    raise Exception('./src reading failed')
                    
                    