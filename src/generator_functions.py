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

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class LambdaGeneratorFunction:
    '''
    CellDataset class generates x and y dictonaries.
    Some models require different keys in x and y.
    Extend this class to add the reqiured keys to given x and y dictonaries.
    
    Args:
        generator (CellDataset):
        
    Dependencies:
        
    Dependency Breaker:
        
    '''
    def __init__(self):
        self.use_default_train = True
    
    def initialize(self, generator):
        pass
        
    def __call__(self, generator, x, y):
        pass
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class TwoStageDLAUNetGenerator(LambdaGeneratorFunction):
    '''
    Dependencies:
        
    '''
    def __call__(self, generator, x, y):
        mask_bnd = np.zeros(y['mask'].shape[:-1]+(3,), dtype=np.float32)
        mask_bnd[:,:,:,1] = np.float32(np.sum(y['mask'][:,:,:,1:], axis=-1)>0)
        mask_bnd[:,:,:,2] = y['bnd'][:,:,:,1]
        mask_bnd[:,:,:,0] = np.float32((mask_bnd[:,:,:,1]+mask_bnd[:,:,:,2])==0)
        y['mask_bnd'] = np.float32(mask_bnd)
        return x, y
    
###########################################################################################################################################
###########################################################################################################################################

class MicroNet508Generator(LambdaGeneratorFunction):
    '''
    Dependencies:
    '''
    def __call__(self, generator, x, y):
        size = 508
        size = (size-4)//2+4
        x['resize1'] = np.zeros((x['input'].shape[0],size,size,3), dtype=np.float32)
        size = (size-8)//2+4
        x['resize2'] = np.zeros((x['input'].shape[0],size,size,3), dtype=np.float32)
        size = (size-8)//2+4
        x['resize3'] = np.zeros((x['input'].shape[0],size,size,3), dtype=np.float32)
        size = (size-8)//2+4
        x['resize4'] = np.zeros((x['input'].shape[0],size,size,3), dtype=np.float32)
        for i in range(x['input'].shape[0]):
            x['resize1'][i] = cv2.resize(x['input'][i], (x['resize1'].shape[2], x['resize1'].shape[1]), interpolation=cv2.INTER_CUBIC)
            x['resize2'][i] = cv2.resize(x['input'][i], (x['resize2'].shape[2], x['resize2'].shape[1]), interpolation=cv2.INTER_CUBIC)
            x['resize3'][i] = cv2.resize(x['input'][i], (x['resize3'].shape[2], x['resize3'].shape[1]), interpolation=cv2.INTER_CUBIC)
            x['resize4'][i] = cv2.resize(x['input'][i], (x['resize4'].shape[2], x['resize4'].shape[1]), interpolation=cv2.INTER_CUBIC)
        x['aux_weight_map'] = x['mask_weight_map']/generator.epoch
        y['aux1'] = y['mask']
        y['aux2'] = y['mask']
        y['aux3'] = y['mask']
        return x, y
    
    
###########################################################################################################################################
###########################################################################################################################################

class Micronet252AuxGenerator(LambdaGeneratorFunction):
    '''
    Dependencies:
        WeightMapGenerator
    '''
    def __call__(self, generator, x, y):
        if 'mask_nearest_cell_map' in x:
            x['aux_weight_map'] = x['mask_nearest_cell_map']/generator.epoch
        elif 'mask_fourier_map' in x:
            x['aux_weight_map'] = x['mask_fourier_map']/generator.epoch
        else:
            x['aux_weight_map'] = x['mask_weight_map']/generator.epoch
        y['aux1'] = y['mask']
        y['aux2'] = y['mask']
        y['aux3'] = y['mask']
        return x, y
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class MicroNet252Generator(LambdaGeneratorFunction):
    def __call__(self, generator, x, y):
        size = 252
        size = (size-4)//2+4
        x['resize1'] = np.zeros((x['input'].shape[0],size,size,3), dtype=np.float32)
        size = (size-8)//2+4
        x['resize2'] = np.zeros((x['input'].shape[0],size,size,3), dtype=np.float32)
        size = (size-8)//2+4
        x['resize3'] = np.zeros((x['input'].shape[0],size,size,3), dtype=np.float32)
        size = (size-8)//2+4
        x['resize4'] = np.zeros((x['input'].shape[0],size,size,3), dtype=np.float32)
        for i in range(x['input'].shape[0]):
            x['resize1'][i] = cv2.resize(x['input'][i], (x['resize1'].shape[2], x['resize1'].shape[1]), interpolation=cv2.INTER_CUBIC)
            x['resize2'][i] = cv2.resize(x['input'][i], (x['resize2'].shape[2], x['resize2'].shape[1]), interpolation=cv2.INTER_CUBIC)
            x['resize3'][i] = cv2.resize(x['input'][i], (x['resize3'].shape[2], x['resize3'].shape[1]), interpolation=cv2.INTER_CUBIC)
            x['resize4'][i] = cv2.resize(x['input'][i], (x['resize4'].shape[2], x['resize4'].shape[1]), interpolation=cv2.INTER_CUBIC)
        return x, y

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class FourierGenerator(LambdaGeneratorFunction):
    def __init__(self, model, fourier_descriptors, map_name='mask_fourier_map'):
        self.model = model
        self.fourier_descriptors = fourier_descriptors
        self.use_default_train = False
        self.map_name = map_name
        
    def initialize(self, train_generator):
        _, y = train_generator.get_dataset_as_batch()
        y_mask_train = y['mask']
        self.fourier_descriptors.fit(np.float32(y_mask_train))
            
    def __call__(self, generator, x, y):
        if isinstance(self.model.output, list):
            y_pred = self.model.predict(x)
            y_pred_mask = y_pred[0]
        else:
            y_pred_mask = self.model.predict(x)
        x[self.map_name] = self.fourier_descriptors.calculate_weight_map(y['mask'], y_pred_mask)
        return x, y
    
    def __repr__(self):
        return self.fourier_descriptors.__repr__()
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class NullMapGenerator(LambdaGeneratorFunction):
    def __init__(self, map_name, value=1):
        super().__init__()
        self.map_name = map_name
        self.value = value
        
    def __call__(self, generator, x, y):
        x[self.map_name] = np.ones(y['mask'].shape[:-1])*self.value
        return x, y
    
    def __repr__(self):
        return f'null map: {self.map_name} value={self.value}'
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class FourierGeneratorFullImage(LambdaGeneratorFunction):
    def __init__(self, model, fourier_descriptors, map_name='mask_fourier_map'):
        self.model = model
        self.fourier_descriptors = fourier_descriptors
        self.use_default_train = False
        self.map_name = map_name
        
    def initialize(self, train_generator):
        y = []
        for path in train_generator.global_data:
            y.append(train_generator.global_data[path]['y_mask'])
        y = np.array(y)
        self.fourier_descriptors.fit(np.float32(y))
            
    def __call__(self, generator, x, y):
        if isinstance(self.model.output, list):
            y_pred = self.model.predict(x)
            y_pred_mask = y_pred[0]
        else:
            y_pred_mask = self.model.predict(x)
        x[self.map_name] = self.fourier_descriptors.calculate_weight_map(y['mask'], y_pred_mask)
        return x, y
    
    def __repr__(self):
        return self.fourier_descriptors.__repr__()
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class FourierNetGenerator(LambdaGeneratorFunction):
    def __init__(self, N, map_name='fourier'):
        super(FourierNetGenerator, self).__init__()
        self.N = N
        self.map_name = map_name
        
    def calculate_center(self, contour):
        x, y, num = 0, 0, 0
        for pixel in contour:
            num += 1
            x += pixel[0][1]
            y += pixel[0][0]
        return x/num, y/num

    def calculate_fourier_coefficients(self, n, l, delta):
        a, b = 0, 0
        L = l[-1]
        for i in range(len(l)):
            if delta[i]>0:
                a += delta[i]*np.sin((2*np.pi*n*l[i])/L)
                b += delta[i]*np.cos((2*np.pi*n*l[i])/L)
        a = -a/(n*np.pi)
        b = b/(n*np.pi)
        return np.sqrt(a*a+b*b)
        
    def calculate_fourier_descriptors(self, contour):
        center = self.calculate_center(contour)
        delta = []
        l = []
        for i in range(1, len(contour)+1):
            point1 = (contour[i-1][0][1], contour[i-1][0][0])
            point2 = (contour[i%len(contour)][0][1], contour[i%len(contour)][0][0])
            d1 = np.sqrt((point1[0]-center[0])**2+(point1[1]-center[1])**2)
            d2 = np.sqrt((point2[0]-center[0])**2+(point2[1]-center[1])**2)
            delta.append(d1-d2)
            d3 = np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
            l.append(d3)
        for i in range(1, len(l)):
            l[i] += l[i-1]
        A = []
        for i in range(1, self.N+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
        
    def generate_fourier_map(self, y):
        map_ = img = np.zeros(y.shape[:-1]+(self.N,))
        mask = np.uint8(np.argmax(y, axis=-1)>0)
        _, conn_comp = cv2.connectedComponents(mask, connectivity=8)
        labels = np.unique(conn_comp)
        labels = labels[labels>0]
        for label in labels:
            comp = np.uint8(conn_comp==label)
            _, contours, _ = cv2.findContours(np.uint8(comp).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = contours[0]
            fd = self.calculate_fourier_descriptors(contour)
            self.max_.append(fd)
            map_[comp>0] = fd
        return map_
        
    def initialize(self, generator):
        self.max_ = []
        for path in generator.global_data:
            y = generator.global_data[path]['y_mask']
            generator.global_data[path]['fourier_map'] = self.generate_fourier_map(y)
        self.max_ = np.array(self.max_)
        self.max_ = self.max_.max(axis=0)
            
    def __call__(self, generator, x, y):
        y[self.map_name] = np.zeros(x['input'].shape[:-1]+(self.N,), dtype=np.float32)
        stride_stats_list = y['stride_stats']
        for i, stride_stats in enumerate(stride_stats_list):
            path, h, w = stride_stats
            fourier_map = np.zeros(x['input'][i].shape[:-1]+(self.N,))
            map_ = generator.global_data[path]['fourier_map'][h:h+generator.model_shape[0],w:w+generator.model_shape[1]]
            fourier_map[:map_.shape[0],:map_.shape[1]] = map_
            y[self.map_name][i] = np.float32(fourier_map/self.max_)
        return x, y
    
    def __repr__(self):
        return 'FourierNetGenerator'

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class NearestCellGenerator(LambdaGeneratorFunction):
    def __init__(self, map_name='mask_nearest_cell_map'):
        super(NearestCellGenerator, self).__init__()
        self.map_name = map_name
    
    def initialize(self, generator):
        self.calculate_nearest_cell_maps(generator)
        
    def calculate_nearest_cell_maps(self, generator):
        for path in generator.global_data:
            dist = calculate_distance_to_nearest_cell(generator.global_data[path]['y_mask'], sigma=5, w_0=10)
            generator.global_data[path]['mask_nearest_cell_map'] = np.float32(dist)
            
    def __call__(self, generator, x, y):
        x[self.map_name] = np.zeros(x['input'].shape[:-1], dtype=np.float32)
        stride_stats_list = y['stride_stats']
        for i, stride_stats in enumerate(stride_stats_list):
            path, h, w = stride_stats
            mask_nearest_cell_map = np.zeros_like(x['mask_weight_map'][i])
            map_ = generator.global_data[path]['mask_nearest_cell_map'][h:h+generator.model_shape[0],w:w+generator.model_shape[1]]
            mask_nearest_cell_map[:map_.shape[0],:map_.shape[1]] = map_
            x[self.map_name][i] = np.float32(mask_nearest_cell_map+x['mask_weight_map'][i])
        return x, y
    
    def __repr__(self):
        return 'distance to nearest cell weights'
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class WeightMapGenerator(LambdaGeneratorFunction):
    def initialize(self, generator):
        mask_weights = np.zeros((generator.num_of_classes+1,))
        bnd_weights = np.zeros((2,))
        for _, y in generator:
            for ann_mask, ann_bnd in zip(y['mask'], y['bnd']):
                mask_weights += ann_mask.sum(axis=(0,1))
                bnd_weights += ann_bnd.sum(axis=(0,1))
        total_mask = mask_weights.sum()
        total_bnd = bnd_weights.sum()
        self.mask_weights = total_mask-mask_weights
        self.bnd_weights = total_bnd-bnd_weights
#         self.mask_weights = total_mask/(len(mask_weights)*mask_weights)
#         self.bnd_weights = total_bnd/(len(bnd_weights)*bnd_weights)
        self.mask_weights = self.mask_weights/self.mask_weights.sum()
        self.bnd_weights = self.bnd_weights/self.bnd_weights.sum()
        warnings.warn('mask weights are {}'.format(self.mask_weights))
        warnings.warn('bnd weights are {}'.format(self.bnd_weights))
            
    def __call__(self, generator, x, y):
        x['mask_weight_map'] = np.float32(np.sum(self.mask_weights*y['mask'], axis=-1))
        x['bnd_weight_map'] = np.float32(np.sum(self.bnd_weights*y['bnd'], axis=-1))
        return x, y
    
    def __repr__(self):
        return 'class weights'

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class Standardize(LambdaGeneratorFunction):
    def __call__(self, generator, x, y):
        a = x['input'] - np.mean(x['input'], axis=(1,2), keepdims=True)
        b = np.std(x['input'], axis=(1,2), keepdims=True)
        x['input'] = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        return x, y
    
    def __repr__(self):
        return 'standardize'
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
     
class Normalize(LambdaGeneratorFunction):
    def __call__(self, generator, x, y):
        x['input'] = x['input']/255
        return x, y
    
    def __repr__(self):
        return 'normalize'
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class GlobalStandardization(LambdaGeneratorFunction):
    def initialize(self, generator):
        self.mean = {}
        self.std = {}
        for key in generator.global_data:
            self.mean[key] = np.mean(generator.global_data[key]['x'], axis=(0,1), keepdims=True)
            self.std[key] = np.std(generator.global_data[key]['x'], axis=(0,1), keepdims=True)
        
    def __call__(self, generator, x, y):
        for i in range(len(y['stride_stats'])):
            key = y['stride_stats'][i][0]
            a = x['input'][i]-self.mean[key]
            b = self.std[key]
            x['input'][i] = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        return x, y
    
    def __repr__(self):
        return 'global standardization'
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class RandomGaussianBlur(LambdaGeneratorFunction):
    def __init__(self, p):
        super(RandomGaussianBlur, self).__init__()
        self.p = p
    
    def __call__(self, generator, x, y):
        for i in range(x['input'].shape[0]):
            if random.random()<self.p:
                sx, sy = np.random.randint(1, 3, size=(2,))
                sx = sx * 2 + 1
                sy = sy * 2 + 1
                x['input'][i] = np.reshape(cv2.GaussianBlur(x['input'][i].copy(), (sx,sy), sigmaX=0, sigmaY=0,
                                           borderType=cv2.BORDER_REPLICATE), x['input'][i].shape)
        return x, y
    
    def __repr__(self):
        return 'Random Gaussian Blur'
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class RandomMedianBlur(LambdaGeneratorFunction):
    def __init__(self, p):
        super(RandomMedianBlur, self).__init__()
        self.p = p
    
    def __call__(self, generator, x, y):
        for i in range(x['input'].shape[0]):
            if random.random()<self.p:
                s = np.random.randint(1, 3)
                s = s*2 + 1
                x['input'][i] = cv2.medianBlur(x['input'][i].copy(), s)
        return x, y
    
    def __repr__(self):
        return 'Random Median Blur'

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class RandomHorizontalFlip(LambdaGeneratorFunction):
    def __init__(self, p):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p
    
    def __call__(self, generator, x, y):
        for i in range(x['input'].shape[0]):
            if random.random()<self.p:
                x['input'][i] = x['input'][i,::-1,:]
                y['mask'][i] = y['mask'][i,::-1,:]
                y['bnd'][i] = y['bnd'][i,::-1,:]
                if 'mask_nearest_cell_map' in x:
                    x['mask_nearest_cell_map'][i] = x['mask_nearest_cell_map'][i,::-1,:]
        return x, y
    
    def __repr__(self):
        return 'Random Horiontal Flip'
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class RandomVerticalFlip(LambdaGeneratorFunction):
    def __init__(self, p):
        super(RandomVerticalFlip, self).__init__()
        self.p = p
    
    def __call__(self, generator, x, y):
        for i in range(x['input'].shape[0]):
            if random.random()<self.p:
                x['input'][i] = x['input'][i,:,::-1]
                y['mask'][i] = y['mask'][i,:,::-1]
                y['bnd'][i] = y['bnd'][i,:,::-1]
                if 'mask_nearest_cell_map' in x:
                    x['mask_nearest_cell_map'][i] = x['mask_nearest_cell_map'][i,:,::-1]
        return x, y
    
    def __repr__(self):
        return 'Random Vertical Flip'
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class RandomRotation(LambdaGeneratorFunction):
    def __init__(self, angle):
        super(RandomRotation, self).__init__()
        self.angle = angle
    
    def __call__(self, generator, x, y):
        for i in range(x['input'].shape[0]):
            rotation = random.random()*2*self.angle-self.angle
            cX, cY = x['input'].shape[2]//2, x['input'].shape[1]//2
            w, h = x['input'].shape[2], x['input'].shape[1]
            M = cv2.getRotationMatrix2D((cX, cY), rotation, 1.0)
            x['input'][i] = cv2.warpAffine(x['input'][i].copy(), M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            y['mask'][i] = cv2.warpAffine(y['mask'][i], M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            y['bnd'][i] = cv2.warpAffine(y['bnd'][i], M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            if 'mask_nearest_cell_map' in x:
                x['mask_nearest_cell_map'][i] = cv2.warpAffine(x['mask_nearest_cell_map'][i], M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        return x, y
    
    def __repr__(self):
        return 'Random Rotation'
    

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class RandomHue(LambdaGeneratorFunction):
    def __init__(self, range_=(-8,8)):
        super(RandomHue, self).__init__()
        self.range_ = range_
    
    def __call__(self, generator, x, y):
        for i in range(x['input'].shape[0]):
            img = x['input'][i].copy()
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # https://docs.opencv.org/3.2.0/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv
            hue = np.random.randint(self.range_[0], self.range_[1])
            if hsv.dtype.itemsize == 1:
                # OpenCV uses 0-179 for 8-bit images
                hsv[..., 0] = (hsv[..., 0] + hue) % 180
            else:
                # OpenCV uses 0-360 for floating point images
                hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            x['input'][i] = img
        return x, y
    
    def __repr__(self):
        return 'Random Hue Adjust'
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class RandomBrightness(LambdaGeneratorFunction):
    def __init__(self, delta=26):
        super(RandomBrightness, self).__init__()
        self.delta = delta
    
    def __call__(self, generator, x, y):
        for i in range(x['input'].shape[0]):
            v = np.random.randint(-self.delta, self.delta)
            img = x['input'][i].copy()
            img += v
            img = np.clip(img, 0, 255)
            x['input'][i] = img
        return x, y
    
    def __repr__(self):
        return 'Random Brightness'
    

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class RandomContrast(LambdaGeneratorFunction):
    def __init__(self, range_=(0.75, 1.25)):
        super(RandomContrast, self).__init__()
        self.range_ = range_
    
    def __call__(self, generator, x, y):
        for i in range(x['input'].shape[0]):
            r = random.uniform(self.range_[0], self.range_[1])
            img = x['input'][i].copy()
            grey = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2GRAY)
            mean = np.mean(grey)
            img = img * r + mean * (1 - r)
            img = np.clip(img, 0, 255)
            x['input'][i] = img
        return x, y
    
    def __repr__(self):
        return 'Random Contrast'
    
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    
class RandomSaturation(LambdaGeneratorFunction):
    def __init__(self, alpha=0.2):
        super(RandomSaturation, self).__init__()
        self.alpha = alpha
    
    def __call__(self, generator, x, y):
        for i in range(x['input'].shape[0]):
            v = 1 + random.uniform(-self.alpha, self.alpha)
            img = x['input'][i].copy()
            grey = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2GRAY)
            ret = img * v + (grey * (1 - v))[:, :, np.newaxis]
            img = np.clip(img, 0, 255)
            x['input'][i] = img
        return x, y
    
    def __repr__(self):
        return 'Random Saturation'
    
