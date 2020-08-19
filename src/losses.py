from keras import backend as K
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.covariance import *
import keras
import warnings

class FourierDescriptors:
    def __init__(self, K, N, modes, descriptor, covariance_types, epsilon, max_distance):
        self.K = K
        self.N = N
        self.modes = modes
        self.descriptor = descriptor
        self.covariance_types = covariance_types
        self.epsilon = epsilon
        self.max_distance = max_distance
        
    def __repr__(self):
        return 'N={}, epsilon={}, K={}, max_distance={}, mode={}, descriptor={}, cov={}'.format(self.N, self.epsilon, self.K, self.max_distance, self.modes, self.descriptor, self.covariance_types)
        
    def initialize(self, num_of_classes):
        self.gaussian_mixtures = []
        for k, covariance_type in zip(self.K, self.covariance_types):
            self.gaussian_mixtures.append(GaussianMixture(n_components=k, n_init=100, max_iter=1000, covariance_type=covariance_type, init_params='kmeans', random_state=42))
        self.mean_distance = []
        self.std_distance = []
        
    def calculate_descriptors(self, y_true):
        y_true = np.argmax(y_true, axis=-1)
        num_of_classes = y_true.max()
        descriptors = []
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_true==c)
            descriptors_c = []
            for y in y_c:
                _, contours, _ = cv2.findContours(y.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    descriptors_c.append(self.calculate_fourier_descriptors(contour, c-1))
            descriptors.append(np.array(descriptors_c))
        return descriptors
    
    def fit(self, y_true):
        '''
        Given y_true, calculates mean and covariance for all classes.
        
        Args:
            y_true (np.float32, [N,H,W,C]) : one hot encoded annotation
            
        Returns:
            None
        '''
        descriptors = self.calculate_descriptors(y_true)
        y_true = np.argmax(y_true, axis=-1)
        num_of_classes = y_true.max()
        self.initialize(num_of_classes)
        for c in range(num_of_classes):
            x = np.array(descriptors[c])
            self.gaussian_mixtures[c].fit(x)
            pred = self.gaussian_mixtures[c].predict(x)
            mean_distance = []
            std_distance = []
            for k in range(self.K[c]):
                x_class = x[pred==k]
                distances = []
                for i in range(x_class.shape[0]):
                    m = x_class[i]-self.gaussian_mixtures[c].means_[k]
                    if self.covariance_types[c]=='spherical':
                        precision = np.eye(self.N[c])*self.gaussian_mixtures[c].precisions_[k]
                    elif self.covariance_types[c]=='diag':
                        precision = np.diag(self.gaussian_mixtures[c].precisions_[k])
                    else:
                        precision = self.gaussian_mixtures[c].precisions_[k]
                    d = np.dot(m, precision)
                    d = np.dot(d, m)
                    d = np.sqrt(d)
                    distances.append(d)
                distances = np.array(distances)
                mean_distance.append(distances.mean())
                std_distance.append(distances.std())
            self.mean_distance.append(mean_distance)
            self.std_distance.append(std_distance)
            
    def get_losses(self, y_pred):
        losses = []
        y_pred = np.argmax(y_pred, axis=-1)
        num_of_classes = y_pred.max()
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_pred==c)
            for j in range(y_c.shape[0]):
                _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
                for i in range(1, conn_comp.max()+1):
                    comp = np.uint8(conn_comp==i)
                    distance = self.calculate_comp_distance(comp, c-1)
                    losses.append(distance)
        return np.array(losses)
    
    def get_losses_c(self, y_pred, c):
        losses = []
        y_pred = np.argmax(y_pred, axis=-1)
        num_of_classes = y_pred.max()
        y_c = np.uint8(y_pred==c)
        for j in range(y_c.shape[0]):
            _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
            for i in range(1, conn_comp.max()+1):
                comp = np.uint8(conn_comp==i)
                distance = self.calculate_comp_distance(comp, c-1)
                losses.append(distance)
        return np.array(losses)
    
    def calculate_weight_map(self, y_true, y_pred):
        '''
        Given y_pred, calculates weight map using fourier descriptors.

        Args:
            y_true (np.float32, [N,H,W,C]) : one hot encoded annotation
            y_pred (np.float32, [N,H,W,C]) : probabilities
        Returns:
            weight_map (np.float32, [N,H,W]) : 
        '''
        weight_map = np.zeros(y_pred.shape[:-1], dtype=np.float32)
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        fn_map = np.float32((y_true>0)*(y_pred!=y_true))
        num_of_classes = y_pred.max()
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_pred==c)
            for j in range(y_c.shape[0]):
                _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
                for i in range(1, conn_comp.max()+1):
                    comp = np.uint8(conn_comp==i)
                    distance = self.calculate_comp_distance(comp, c-1)
                    weight_map[j] += np.float32(distance*comp)
        max_map = weight_map.max(axis=(1,2), keepdims=True)
        max_map[max_map==0] = self.max_distance
        weight_map += fn_map*max_map
        weight_map[weight_map<self.epsilon] = self.epsilon
        return weight_map
    
    def calculate_vector_distance(self, vector, c):
        k = self.gaussian_mixtures[c].predict(vector[np.newaxis])[0]
        if self.covariance_types[c]=='spherical':
            precision = np.eye(self.N[c])*self.gaussian_mixtures[c].precisions_[k]
        elif self.covariance_types[c]=='diag':
            precision = np.diag(self.gaussian_mixtures[c].precisions_[k])
        else:
            precision = self.gaussian_mixtures[c].precisions_[k]
        m = vector-self.gaussian_mixtures[c].means_[k]
        d = np.dot(m, precision)
        d = np.dot(d, m)
        d = np.sqrt(d)
        d = max(0, (d-self.mean_distance[c][k])/(self.std_distance[c][k]+1e-20))
        d = min(self.max_distance, d)
        return d
            
    def calculate_comp_distance(self, comp, c):
        distance = 0
        _, contours, _ = cv2.findContours(np.uint8(comp).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x = self.calculate_fourier_descriptors(contour, c)
            temp_dist = self.calculate_vector_distance(x, c)
            if temp_dist > distance:
                distance = temp_dist
        return distance
    
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
        if self.descriptor=='harmonic_amplitude':
            return np.sqrt(a*a+b*b)
        elif self.descriptor=='phase_angle':
            return np.arctan(b/(a+K.epsilon()))
            
    def calculate_fourier_descriptors_center(self, contour, c):
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
        for i in range(1, self.N[c]+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
    
    def calculate_phi(self, point1, point2, point3):
        line1 = (point2[0]-point1[0], point2[1]-point1[1])
        line2 = (point3[0]-point2[0], point3[1]-point2[1])
        phi1 = np.arccos(line1[0]/np.sqrt(line1[0]**2+line1[1]**2+1e-20))
        phi2 = np.arccos(line2[0]/np.sqrt(line2[0]**2+line2[1]**2+1e-20))
        res = phi2-phi1
        return res
    
    def calculate_fourier_descriptors_angle(self, contour, c):
        delta = []
        l = []
        for i in range(2, len(contour)+2):
            point1 = (contour[i-2][0][1], contour[i-2][0][0])
            point2 = (contour[(i-1)%len(contour)][0][1], contour[(i-1)%len(contour)][0][0])
            point3 = (contour[i%len(contour)][0][1], contour[i%len(contour)][0][0])
            delta.append(self.calculate_phi(point1, point2, point3))
            l.append(np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2))
        for i in range(1, len(l)):
            l[i] += l[i-1]
        A = []
        for i in range(1, self.N[c]+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
    
    def calculate_fourier_descriptors(self, contour, c):
        if self.modes[c]=='center':
            return self.calculate_fourier_descriptors_center(contour, c)
        elif self.modes[c]=='angle':
            return self.calculate_fourier_descriptors_angle(contour, c)
        
class BayesianFourierDescriptors:
    def __init__(self, K, N, modes, descriptor, covariance_types, epsilon, max_distance):
        self.K = K
        self.N = N
        self.modes = modes
        self.descriptor = descriptor
        self.covariance_types = covariance_types
        self.epsilon = epsilon
        self.max_distance = max_distance
        
    def __repr__(self):
        return 'N={}, epsilon={}, K={}, max_distance={}, mode={}, descriptor={}, cov={}'.format(self.N, self.epsilon, self.K, self.max_distance, self.modes, self.descriptor, self.covariance_types)
        
    def initialize(self, num_of_classes):
        self.gaussian_mixtures = []
        for k, covariance_type in zip(self.K, self.covariance_types):
            self.gaussian_mixtures.append(BayesianGaussianMixture(n_components=k, n_init=100, max_iter=1000, covariance_type=covariance_type, init_params='kmeans', random_state=42))
        self.mean_distance = []
        self.std_distance = []
        
    def calculate_descriptors(self, y_true):
        y_true = np.argmax(y_true, axis=-1)
        num_of_classes = y_true.max()
        descriptors = []
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_true==c)
            descriptors_c = []
            for y in y_c:
                _, contours, _ = cv2.findContours(y.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    descriptors_c.append(self.calculate_fourier_descriptors(contour, c-1))
            descriptors.append(np.array(descriptors_c))
        return descriptors
    
    def fit(self, y_true):
        '''
        Given y_true, calculates mean and covariance for all classes.
        
        Args:
            y_true (np.float32, [N,H,W,C]) : one hot encoded annotation
            
        Returns:
            None
        '''
        descriptors = self.calculate_descriptors(y_true)
        y_true = np.argmax(y_true, axis=-1)
        num_of_classes = y_true.max()
        self.initialize(num_of_classes)
        for c in range(num_of_classes):
            x = np.array(descriptors[c])
            self.gaussian_mixtures[c].fit(x)
            pred = self.gaussian_mixtures[c].predict(x)
            mean_distance = []
            std_distance = []
            for k in range(self.K[c]):
                x_class = x[pred==k]
                distances = []
                for i in range(x_class.shape[0]):
                    m = x_class[i]-self.gaussian_mixtures[c].means_[k]
                    if self.covariance_types[c]=='spherical':
                        precision = np.eye(self.N[c])*self.gaussian_mixtures[c].precisions_[k]
                    elif self.covariance_types[c]=='diag':
                        precision = np.diag(self.gaussian_mixtures[c].precisions_[k])
                    else:
                        precision = self.gaussian_mixtures[c].precisions_[k]
                    d = np.dot(m, precision)
                    d = np.dot(d, m)
                    d = np.sqrt(d)
                    distances.append(d)
                distances = np.array(distances)
                mean_distance.append(distances.mean())
                std_distance.append(distances.std())
            self.mean_distance.append(mean_distance)
            self.std_distance.append(std_distance)
            
    def get_losses(self, y_pred):
        losses = []
        y_pred = np.argmax(y_pred, axis=-1)
        num_of_classes = y_pred.max()
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_pred==c)
            for j in range(y_c.shape[0]):
                _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
                for i in range(1, conn_comp.max()+1):
                    comp = np.uint8(conn_comp==i)
                    distance = self.calculate_comp_distance(comp, c-1)
                    losses.append(distance)
        return np.array(losses)
    
    def get_losses_c(self, y_pred, c):
        losses = []
        y_pred = np.argmax(y_pred, axis=-1)
        num_of_classes = y_pred.max()
        y_c = np.uint8(y_pred==c)
        for j in range(y_c.shape[0]):
            _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
            for i in range(1, conn_comp.max()+1):
                comp = np.uint8(conn_comp==i)
                distance = self.calculate_comp_distance(comp, c-1)
                losses.append(distance)
        return np.array(losses)
    
    def calculate_weight_map(self, y_true, y_pred):
        '''
        Given y_pred, calculates weight map using fourier descriptors.

        Args:
            y_true (np.float32, [N,H,W,C]) : one hot encoded annotation
            y_pred (np.float32, [N,H,W,C]) : probabilities
        Returns:
            weight_map (np.float32, [N,H,W]) : 
        '''
        weight_map = np.zeros(y_pred.shape[:-1], dtype=np.float32)
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        fn_map = np.float32((y_true>0)*(y_pred!=y_true))
        num_of_classes = y_pred.max()
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_pred==c)
            for j in range(y_c.shape[0]):
                _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
                for i in range(1, conn_comp.max()+1):
                    comp = np.uint8(conn_comp==i)
                    distance = self.calculate_comp_distance(comp, c-1)
                    weight_map[j] += np.float32(distance*comp)
        max_map = weight_map.max(axis=(1,2), keepdims=True)
        max_map[max_map==0] = self.max_distance
        weight_map += fn_map*max_map
        weight_map[weight_map<self.epsilon] = self.epsilon
        return weight_map
    
    def calculate_vector_distance(self, vector, c):
        k = self.gaussian_mixtures[c].predict(vector[np.newaxis])[0]
        if self.covariance_types[c]=='spherical':
            precision = np.eye(self.N[c])*self.gaussian_mixtures[c].precisions_[k]
        elif self.covariance_types[c]=='diag':
            precision = np.diag(self.gaussian_mixtures[c].precisions_[k])
        else:
            precision = self.gaussian_mixtures[c].precisions_[k]
        m = vector-self.gaussian_mixtures[c].means_[k]
        d = np.dot(m, precision)
        d = np.dot(d, m)
        d = np.sqrt(d)
        d = max(0, (d-self.mean_distance[c][k])/(self.std_distance[c][k]))
        d = min(self.max_distance, d)
        return d
            
    def calculate_comp_distance(self, comp, c):
        distance = 0
        _, contours, _ = cv2.findContours(np.uint8(comp).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x = self.calculate_fourier_descriptors(contour, c)
            temp_dist = self.calculate_vector_distance(x, c)
            if temp_dist > distance:
                distance = temp_dist
        return distance
    
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
        if self.descriptor=='harmonic_amplitude':
            return np.sqrt(a*a+b*b)
        elif self.descriptor=='phase_angle':
            return np.arctan(b/(a+K.epsilon()))
            
    def calculate_fourier_descriptors_center(self, contour, c):
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
        for i in range(1, self.N[c]+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
    
    def calculate_phi(self, point1, point2, point3):
        line1 = (point2[0]-point1[0], point2[1]-point1[1])
        line2 = (point3[0]-point2[0], point3[1]-point2[1])
        phi1 = np.arccos(line1[0]/np.sqrt(line1[0]**2+line1[1]**2+1e-20))
        phi2 = np.arccos(line2[0]/np.sqrt(line2[0]**2+line2[1]**2+1e-20))
        res = phi2-phi1
        return res
    
    def calculate_fourier_descriptors_angle(self, contour, c):
        delta = []
        l = []
        for i in range(2, len(contour)+2):
            point1 = (contour[i-2][0][1], contour[i-2][0][0])
            point2 = (contour[(i-1)%len(contour)][0][1], contour[(i-1)%len(contour)][0][0])
            point3 = (contour[i%len(contour)][0][1], contour[i%len(contour)][0][0])
            delta.append(self.calculate_phi(point1, point2, point3))
            l.append(np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2))
        for i in range(1, len(l)):
            l[i] += l[i-1]
        A = []
        for i in range(1, self.N[c]+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
    
    def calculate_fourier_descriptors(self, contour, c):
        if self.modes[c]=='center':
            return self.calculate_fourier_descriptors_center(contour, c)
        elif self.modes[c]=='angle':
            return self.calculate_fourier_descriptors_angle(contour, c)
        
        
class NearestFourierDescriptors:
    def __init__(self, N, modes, descriptor, epsilon, max_distance, d):
        assert descriptor=='harmonic_amplitude' or descriptor=='phase_angle'
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.modes = modes
        self.descriptor = descriptor
        self.N = N
        self.d = d
        
    def __repr__(self):
        return 'Nearest Fourier N={}, epsilon={}, max_distance={}, mode={}, descriptor={}'.format(self.N, self.epsilon, self.max_distance, self.modes, self.descriptor)
        
    def calculate_descriptors(self, y_true):
        y_true = np.argmax(y_true, axis=-1)
        num_of_classes = y_true.max()
        descriptors = []
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_true==c)
            descriptors_c = []
            for y in y_c:
                _, contours, _ = cv2.findContours(y.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    descriptors_c.append(self.calculate_fourier_descriptors(contour, c-1))
            descriptors.append(np.array(descriptors_c))
        return descriptors
    
    def fit(self, y_true):
        '''
        Given y_true, calculates mean and covariance for all classes.
        
        Args:
            y_true (np.float32, [N,H,W,C]) : one hot encoded annotation
            
        Returns:
            None
        '''
        self.train_descriptors = self.calculate_descriptors(y_true)
            
    def get_losses(self, y_pred):
        losses = []
        y_pred = np.argmax(y_pred, axis=-1)
        num_of_classes = y_pred.max()
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_pred==c)
            for j in range(y_c.shape[0]):
                _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
                for i in range(1, conn_comp.max()+1):
                    comp = np.uint8(conn_comp==i)
                    distance = self.calculate_comp_distance(comp, c-1)
                    losses.append(distance)
        return np.array(losses)
    
    def get_losses_c(self, y_pred, c):
        losses = []
        y_pred = np.argmax(y_pred, axis=-1)
        num_of_classes = y_pred.max()
        y_c = np.uint8(y_pred==c)
        for j in range(y_c.shape[0]):
            _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
            for i in range(1, conn_comp.max()+1):
                comp = np.uint8(conn_comp==i)
                distance = self.calculate_comp_distance(comp, c-1)
                losses.append(distance)
        return np.array(losses)
    
    def calculate_weight_map(self, y_true, y_pred):
        '''
        Given y_pred, calculates weight map using fourier descriptors.

        Args:
            y_true (np.float32, [N,H,W,C]) : one hot encoded annotation
            y_pred (np.float32, [N,H,W,C]) : probabilities
        Returns:
            weight_map (np.float32, [N,H,W]) : 
        '''
        weight_map = np.zeros(y_pred.shape[:-1], dtype=np.float32)
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        fn_map = np.float32((y_true>0)*(y_pred!=y_true))
        num_of_classes = y_pred.max()
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_pred==c)
            for j in range(y_c.shape[0]):
                _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
                for i in range(1, conn_comp.max()+1):
                    comp = np.uint8(conn_comp==i)
                    distance = self.calculate_comp_distance(comp, c-1)
                    weight_map[j] += np.float32(distance*comp)
        max_map = weight_map.max(axis=(1,2), keepdims=True)
        max_map[max_map==0] = self.max_distance
        weight_map += fn_map*max_map
        weight_map[weight_map<self.epsilon] = self.epsilon
        return weight_map
    
    def calculate_vector_distance(self, vector, c):
        distances = np.sqrt(np.sum(np.square(self.train_descriptors[c]-vector), axis=-1))
        return max(0, distances.min()-self.d)
            
    def calculate_comp_distance(self, comp, c):
        '''
        Given a component, calculates the mahalonobis distance
        to the average shape of class c
        
        Mahalonobis_Distance(x) = sqrt( (x-mean)^T covariance^-1 (x-mean) )
        
        Args:
            comp (np.uint8, [H,W]) : only one connected component in the image
            c (int) : the distance will be calculated using this class
            
        Returns:
            distance (float) : mahalonobis distance
        '''
        distance = 0
        _, contours, _ = cv2.findContours(np.uint8(comp).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x = self.calculate_fourier_descriptors(contour, c)
            temp_dist = self.calculate_vector_distance(x, c)
            if temp_dist > distance:
                distance = temp_dist
        return distance
    
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
        if self.descriptor=='harmonic_amplitude':
            return np.sqrt(a*a+b*b)
        elif self.descriptor=='phase_angle':
            return np.arctan(b/(a+K.epsilon()))
            
    def calculate_fourier_descriptors_center(self, contour, c):
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
        for i in range(1, self.N[c]+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
    
    def calculate_phi(self, point1, point2, point3):
        line1 = (point2[0]-point1[0], point2[1]-point1[1])
        line2 = (point3[0]-point2[0], point3[1]-point2[1])
        phi1 = np.arccos(line1[0]/np.sqrt(line1[0]**2+line1[1]**2+1e-20))
        phi2 = np.arccos(line2[0]/np.sqrt(line2[0]**2+line2[1]**2+1e-20))
        res = phi2-phi1
        return res
    
    def calculate_fourier_descriptors_angle(self, contour, c):
        delta = []
        l = []
        for i in range(2, len(contour)+2):
            point1 = (contour[i-2][0][1], contour[i-2][0][0])
            point2 = (contour[(i-1)%len(contour)][0][1], contour[(i-1)%len(contour)][0][0])
            point3 = (contour[i%len(contour)][0][1], contour[i%len(contour)][0][0])
            delta.append(self.calculate_phi(point1, point2, point3))
            l.append(np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2))
        for i in range(1, len(l)):
            l[i] += l[i-1]
        A = []
        for i in range(1, self.N[c]+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
    
    def calculate_fourier_descriptors(self, contour, c):
        if self.modes[c]=='center':
            return self.calculate_fourier_descriptors_center(contour, c)
        elif self.modes[c]=='angle':
            return self.calculate_fourier_descriptors_angle(contour, c)
        
        

class RobustCovarianceFourierDescriptors:
    def __init__(self, K=1, epsilon=5.0, max_distance=50, mode='center', descriptor='harmonic_amplitude', N=None):
        assert mode=='center' or mode=='angle'
        assert descriptor=='harmonic_amplitude' or descriptor=='phase_angle'
        self.K = K
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.mode = mode
        self.descriptor = descriptor
        self.N = N
        
    def __repr__(self):
        return 'N={}, epsilon={}, K={}, max_distance={}, mode={}, descriptor={}'.format(self.N, self.epsilon, self.K, self.max_distance, self.mode, self.descriptor)
        
    def initialize(self, num_of_classes):
        if not isinstance(self.K, list):
            K = self.K
            self.K = []
            for i in range(num_of_classes):
                self.K.append(K)
        self.gaussian_mixtures = []
        self.covs = []
        for k in self.K:
            self.gaussian_mixtures.append(GaussianMixture(n_components=k, n_init=100, tol=1e-20, max_iter=1000, covariance_type='full', init_params='random'))
            self.covs.append([])
            for i in range(k):
                self.covs[-1].append(MinCovDet())
        self.mean_distance = []
        self.std_distance = []
        
    def calculate_descriptors(self, y_true):
        y_true = np.argmax(y_true, axis=-1)
        num_of_classes = y_true.max()
        descriptors = []
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_true==c)
            descriptors_c = []
            for y in y_c:
                _, contours, _ = cv2.findContours(y.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    descriptors_c.append(self.calculate_fourier_descriptors(contour, c-1))
            descriptors.append(np.array(descriptors_c))
        return descriptors
    
    def fit(self, y_true_train, y_true_validation):
        avg_dists = []
        if self.N is None:
            num_of_classes = y_true_train.shape[-1]-1
            n_values = [10,20,30,40,50,60,70,80,90,100]
            mean_lists = []
            for i in range(num_of_classes):
                mean_lists.append([])
            for n in n_values:
                self.N = []
                for i in range(num_of_classes):
                    self.N.append(n)
                self.fit_y(y_true_train)
                descriptors_list = self.calculate_descriptors(y_true_validation)
                for i, descriptors in enumerate(descriptors_list):
                    dist = []
                    for vec in descriptors:
                        dist.append(self.calculate_vector_distance(vec, i))
                    dist = np.array(dist)
                    mean_lists[i].append(dist.mean())
            self.N = []
            for i in range(num_of_classes):
                avg_dists.append(mean_lists[i][np.argmin(mean_lists[i])])
                self.N.append(n_values[np.argmin(mean_lists[i])])
        self.fit_y(y_true_train)
        warnings.warn('N values are {}'.format(self.N))
        return avg_dists
    
    def fit_y(self, y_true):
        '''
        Given y_true, calculates mean and covariance for all classes.
        
        Args:
            y_true (np.float32, [N,H,W,C]) : one hot encoded annotation
            
        Returns:
            None
        '''
        descriptors = self.calculate_descriptors(y_true)
        y_true = np.argmax(y_true, axis=-1)
        num_of_classes = y_true.max()
        self.initialize(num_of_classes)
        for c in range(num_of_classes):
            x = np.array(descriptors[c])
            self.gaussian_mixtures[c].fit(x)
            pred = self.gaussian_mixtures[c].predict(x)
            mean_distance = []
            std_distance = []
            for k in range(self.K[c]):
                x_class = x[pred==k]
                self.covs[c][k].fit(x_class)
                distances = []
                for i in range(x_class.shape[0]):
                    d = self.covs[c][k].mahalanobis(x_class[i])
                    distances.append(d)
                distances = np.array(distances)
                mean_distance.append(distances.mean())
                std_distance.append(distances.std())
            self.mean_distance.append(mean_distance)
            self.std_distance.append(std_distance)
            
    def get_losses(self, y_pred):
        losses = []
        y_pred = np.argmax(y_pred, axis=-1)
        num_of_classes = y_pred.max()
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_pred==c)
            for j in range(y_c.shape[0]):
                _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
                for i in range(1, conn_comp.max()+1):
                    comp = np.uint8(conn_comp==i)
                    distance = self.calculate_comp_distance(comp, c-1)
                    losses.append(distance)
        return np.array(losses)
    
    def calculate_weight_map(self, y_true, y_pred):
        '''
        Given y_pred, calculates weight map using fourier descriptors.

        Args:
            y_true (np.float32, [N,H,W,C]) : one hot encoded annotation
            y_pred (np.float32, [N,H,W,C]) : probabilities
        Returns:
            weight_map (np.float32, [N,H,W]) : 
        '''
        weight_map = np.zeros(y_pred.shape[:-1], dtype=np.float32)
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        fn_map = np.float32((y_true!=0)*(y_pred==0))
        num_of_classes = y_pred.max()
        for c in range(1, num_of_classes+1):
            y_c = np.uint8(y_pred==c)
            for j in range(y_c.shape[0]):
                _, conn_comp = cv2.connectedComponents(np.uint8(y_c[j]), connectivity=4)
                for i in range(1, conn_comp.max()+1):
                    comp = np.uint8(conn_comp==i)
                    distance = self.calculate_comp_distance(comp, c-1)
                    weight_map[j] += np.float32(distance*comp)
        max_map = weight_map.max(axis=(1,2), keepdims=True)
#         max_map[max_map==0] = self.max_distance
        weight_map += fn_map*max_map
        weight_map[weight_map<self.epsilon] = self.epsilon
        return weight_map
    
    def calculate_vector_distance(self, vector, c):
        k = self.gaussian_mixtures[c].predict(vector[np.newaxis])[0]
        d = self.covs[c][k].mahalanobis(vector)
        d = max(0, (d-self.mean_distance[c][k])/self.std_distance[c][k])
        d = min(self.max_distance, d)
        return d
            
    def calculate_comp_distance(self, comp, c):
        '''
        Given a component, calculates the mahalonobis distance
        to the average shape of class c
        
        Mahalonobis_Distance(x) = sqrt( (x-mean)^T covariance^-1 (x-mean) )
        
        Args:
            comp (np.uint8, [H,W]) : only one connected component in the image
            c (int) : the distance will be calculated using this class
            
        Returns:
            distance (float) : mahalonobis distance
        '''
        distance = 0
        _, contours, _ = cv2.findContours(np.uint8(comp).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x = self.calculate_fourier_descriptors(contour, c)
            temp_dist = self.calculate_vector_distance(x, c)
            if temp_dist > distance:
                distance = temp_dist
        return distance
    
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
        if self.descriptor=='harmonic_amplitude':
            return np.sqrt(a*a+b*b)
        elif self.descriptor=='phase_angle':
            return np.arctan(b/(a+K.epsilon()))
            
    def calculate_fourier_descriptors_center(self, contour, c):
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
        for i in range(1, self.N[c]+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
    
    def calculate_phi(self, point1, point2, point3):
        line1 = (point2[0]-point1[0], point2[1]-point1[1])
        line2 = (point3[0]-point2[0], point3[1]-point2[1])
        phi1 = np.arccos(line1[0]/np.sqrt(line1[0]**2+line1[1]**2+1e-20))
        phi2 = np.arccos(line2[0]/np.sqrt(line2[0]**2+line2[1]**2+1e-20))
        res = phi2-phi1
        return res
    
    def calculate_fourier_descriptors_angle(self, contour, c):
        delta = []
        l = []
        for i in range(2, len(contour)+2):
            point1 = (contour[i-2][0][1], contour[i-2][0][0])
            point2 = (contour[(i-1)%len(contour)][0][1], contour[(i-1)%len(contour)][0][0])
            point3 = (contour[i%len(contour)][0][1], contour[i%len(contour)][0][0])
            delta.append(self.calculate_phi(point1, point2, point3))
            l.append(np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2))
        for i in range(1, len(l)):
            l[i] += l[i-1]
        A = []
        for i in range(1, self.N[c]+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
    
    def calculate_fourier_descriptors(self, contour, c):
        if self.mode=='center':
            return self.calculate_fourier_descriptors_center(contour, c)
        elif self.mode=='angle':
            return self.calculate_fourier_descriptors_angle(contour, c)
    
def calculate_distance_to_nearest_cell(y_true, sigma=5, w_0=10):
    '''
    Given a y_true it returns distance to nearest cell weight map.
    
    Args:
        y_true (np.ndarray) : (H,W,C) C is the number of classes
        
    Returns:
        weight_map (np.ndarray) : (H,W) distance to nearest cell weight map
    '''
    y_true = np.argmax(y_true, axis=-1)
    background = y_true==0
    _, connected_components = cv2.connectedComponents(np.uint8(y_true), connectivity=4)
    distance_transforms = np.zeros(y_true.shape+(connected_components.max(),))
    labels = np.unique(connected_components)
    labels = labels[labels>0]
    for label in labels:
        component = np.uint8(connected_components!=label)
        dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        distance_transforms[:,:,label-1] = dist
    distance_transforms = np.sort(distance_transforms, axis=-1)
    d1 = distance_transforms[:,:,0]
    d2 = distance_transforms[:,:,1]
    weight_map = w_0*np.exp(-((d1+d2)**2)/(2*(sigma**2)))*background
    return np.float32(weight_map)

def calculate_betti_number(img):
    '''
    Given an image, it returns the betti number (number of connected components)
    of the image.
    
    Args:
        img (np.ndarray) : (H,W)
    
    Returns:
        betti_number (int) : number of connected components
    '''
    _, connected_components = cv2.connectedComponents(img.astype(np.uint8), connectivity=4)
    return connected_components.max()

def calculate_betti_loss_weights_for_img(y_true, y_pred, classes=[]):
    '''
    Given a y_true and y_pred it calculates the Betti weight.
    
    Args:
        y_true (np.ndarray) : (H,W,C) C is the number of classes
        y_pred (np.ndarray) : (H,W,C) C is the number of classes
        
    Returns:
        betti_weights (np.ndarray) : (H,W) betti weight for each pixel at each class
    '''
    if len(classes)==0:
        for i in range(y_true.shape[-1]):
            classes.append(i)
    betti_weights = np.zeros(y_true.shape[:-1], dtype=np.float32)
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    for class_ in classes:
        true = (y_true==class_)
        pred = (y_pred==class_)
        simplicial_complex = ((true+pred)>0).astype(np.float32)
        betti_number = calculate_betti_number(simplicial_complex)
        only_y_true = true*(1.-pred)
        only_y_pred = true*(1.-pred)
        xor = only_y_true + only_y_pred
        set_of_simplicies = (xor>0).astype(np.float32)
        _, set_of_simplicies = cv2.connectedComponents(set_of_simplicies.astype(np.uint8), connectivity=4)
        set_of_simplicies = set_of_simplicies.astype(np.uint8)
        labels = np.unique(set_of_simplicies)
        labels = labels[labels>0]
        for label in labels:
            component = (set_of_simplicies==label).astype(np.float32)
            new_betti_number = calculate_betti_number(simplicial_complex-component)
            betti_weight = abs(betti_number-new_betti_number)
            betti_weights += (betti_weight*component).astype(np.float32)
    return betti_weights

def calculate_betti_loss_weights(y_true, y_pred, classes=[]):
    '''
    Given a y_true and y_pred it calculates the Betti weight.
    
    Betti weight is defined as the absolute change in betti number
    of a simplicial complex by removing/degluing a simplex from
    the simplicial complex of merged pixel space (y_true or y_pred).
    
    Args:
        y_true (np.ndarray) : (N,H,W,C) C is the number of classes
        y_pred (np.ndarray) : (N,H,W,C) C is the number of classes
        
    Returns:
        betti_weights (np.ndarray) : (N,H,W) betti weight for each pixel at each class
    '''
    betti_weights = np.zeros(y_true.shape[:-1], dtype=np.float32)
    for i in range(y_true.shape[0]):
        betti_weights[i] = calculate_betti_loss_weights_for_img(y_true[i,:,:,:], y_pred[i,:,:,:], classes=classes).astype(np.float32)
    return betti_weights

def compile_for_sigma_loss(model, custom):
    """
    Uses Uncertainity to losses.
    
    The paper is available at:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    
    """
    sigma = K.variable(0.)
    model.layers[-1].trainable_weights.append(sigma)
    def loss(y_true, y_pred):
        return K.in_train_phase(custom(y_true, y_pred)*K.exp(-2*sigma)+sigma, custom(y_true, y_pred))
    return loss, sigma

def generalized_dice_loss(y_true, y_pred):
    w = 1./(K.square(K.sum(y_true, axis=(1,2), keepdims=True))+K.epsilon())
    numerator = K.sum(w*K.sum(y_true*y_pred, axis=(1,2), keepdims=True), axis=-1)
    denominator = K.sum(w*K.sum(y_true+y_pred, axis=(1,2), keepdims=True), axis=-1)
    return 1.-(2.*numerator)/denominator

def focal_loss(y_true, y_pred):
    gamma = 2
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    ce = -y_true * K.log(y_pred)
    loss = K.pow(1 - y_pred, gamma)*ce
    return loss

def weighted_crossentropy(weight_map):
    def loss(y_true, y_pred):
        ce = keras.losses.categorical_crossentropy(y_true, y_pred)
        return ce*weight_map
    return loss

def weighted_mse(weight_map):
    def loss(y_true, y_pred):
        ce = keras.losses.mean_squared_error(y_true, y_pred)
        return ce*weight_map
    return loss

class Contour:
    def generate_from_contour(self, contour):
        self.points = []
        for pixel in contour:
            self.points.append((pixel[0][1], pixel[0][0]))
    
    def generate_line(self, p1, p2):
        '''
        DDA Line Drawing Algorithm
        
        p1 and p2 are tuples of coordinates e.g. (x,y)
        '''
        self.points = []
        x1, y1 = p1
        x2, y2 = p2
        dx = x2-x1
        dy = y2-y1
        if abs(dx)>abs(dy):
            steps = abs(dx)
        else:
            steps = abs(dy)
        if steps>0:
            x_inc = dx/float(steps)
            y_inc = dy/float(steps)
            x = x1
            y = y1
            self.points.append((x,y))
            for i in range(steps):
                x = x + x_inc
                y = y + y_inc
                self.points.append((int(x),int(y)))
        else:
            self.points.append((int(x1), int(y1)))
            
    def derivative(self, p):
        if p in self.points:
            return 1
        else:
            return 0
        
    def derivative_at(self, p, l):
        if p in self.points[:l+1]:
            return 1
        else:
            return 0
        
    def __len__(self):
        return len(self.points)
    
class FourierComponent:
    def __init__(self, contour, N):
        self.outer_contour = Contour()
        self.outer_contour.generate_from_contour(contour)
        self.inner_contours = []
        x, y, num = 0, 0, 0
        for pixel in self.outer_contour.points:
            num += 1
            x += pixel[0]
            y += pixel[1]
        x = int(x/num)
        y = int(y/num)
        center = (x, y)
        for pixel in self.outer_contour.points:
            contour = Contour()
            contour.generate_line(pixel, center)
            self.inner_contours.append(contour)
        self.N = N
        self.calculate_fourier_descriptors()
    
    def derivative_a_b(self, p, n):
        da = 0
        db = 0
        L = len(self.outer_contour)
        for k in range(1, len(self.outer_contour)+1):
            if p in self.outer_contour.points:
                mult = (2*np.pi*n*(L*self.outer_contour.derivative_at(p,k)-k*self.outer_contour.derivative_at(p,L)))/(L*L)
                inner = (2*np.pi*n*k)/L
                sin = np.sin(inner)
                cos = np.cos(inner)
                d_sin = cos*mult
                d_cos = -sin*mult
                delta = (len(self.inner_contours[k-1])-len(self.inner_contours[k%len(self.outer_contour)]))
                a_first = d_sin*delta
                b_first = d_cos*delta
                d_delta = self.inner_contours[k-1].derivative(p)-self.inner_contours[k%len(self.outer_contour)].derivative(p)
                a_second = sin*d_delta
                b_second = cos*d_delta
                da += a_first+a_second
                db += b_first+b_second
        da = -da/(n*np.pi)
        db = db/(n*np.pi)
        return da, db
    
    def derivative(self, p, n):
        da, db = self.derivative_a_b(p, n)
        a = self.a[n-1]
        b = self.b[n-1]
        d = (2*a*da+2*b*db)/(2*np.sqrt(a*a+b*b+K.epsilon()))
        return d
        
    def calculate_fourier_coefficients(self, n, l, delta):
        a, b = 0, 0
        L = l[-1]
        for i in range(len(l)):
            if delta[i]>0:
                a += delta[i]*np.sin((2*np.pi*n*l[i])/L)
                b += delta[i]*np.cos((2*np.pi*n*l[i])/L)
        a = -a/(n*np.pi)
        b = b/(n*np.pi)
        return a, b
            
    def calculate_fourier_descriptors(self):
        delta = []
        l = []
        for i in range(1, len(self.outer_contour)+1):
            d1 = len(self.inner_contours[i-1])
            d2 = len(self.inner_contours[i%len(self.outer_contour)])
            delta.append(d1-d2)
            l.append(1)
        for i in range(1, len(l)):
            l[i] += l[i-1]
        self.a = []
        self.b = []
        self.A = []
        for i in range(1, self.N+1):
            a_i, b_i = self.calculate_fourier_coefficients(i, l, delta)
            self.a.append(a_i)
            self.b.append(b_i)
            self.A.append(np.sqrt(a_i*a_i+b_i*b_i))
        self.A = np.array(self.A)
        
    def get_all_points(self):
        points = set()
        for point in self.outer_contour.points:
            points.add(point)
        for contour in self.inner_contours:
            for point in contour.points:
                points.add(point)
        return list(points)

class DifferentiableFourierDescriptors:
    def __init__(self, N=5):
        self.N = N
        self.gaussian_mixture = GaussianMixture(n_components=1, n_init=100, tol=1e-20, max_iter=1000, covariance_type='full', init_params='random')
        
    def calculate_fourier_comps(self, y_true):
        y_true = np.argmax(y_true, axis=-1)
        fourier_comps = []
        for y in y_true:
            _, contours, _ = cv2.findContours(np.uint8(y).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                fourier_comp = FourierComponent(contour, self.N)
                fourier_comp.calculate_fourier_descriptors()
                fourier_comps.append(fourier_comp)
        return fourier_comps
        
    def fit(self, y_true):
        '''
        Args:
            y_true (np.float32, [N,H,W,C]) : one hot encoded annotation
            
        Returns:
            None
        '''
        fourier_comps = self.calculate_fourier_comps(y_true)
        x = np.zeros((len(fourier_comps), self.N))
        for i in range(len(fourier_comps)):
            x[i] = fourier_comps[i].A
        self.gaussian_mixture.fit(x)
            
    def calculate_weight_map(self, y_pred):
        '''
        Given y_pred, calculates weight map using fourier descriptors.

        Args:
            y_pred (np.float32, [N,H,W]) : probabilities
        Returns:
            weight_map (np.float32, [N,H,W]) : 
        '''
        y_pred = np.argmax(y_pred, axis=-1)
        weight_map = np.zeros(y_pred.shape, dtype=np.float32)
        dy = np.zeros((y_pred.shape[0], self.N, weight_map.shape[1]*weight_map.shape[2]))
        precision = self.gaussian_mixture.precisions_[0]
        precision = precision+precision.T
        for j in range(y_pred.shape[0]):
            _, contours, _ = cv2.findContours(np.uint8(y_pred[j]).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                d_P_i = np.zeros((self.N, weight_map.shape[1], weight_map.shape[2]))
                fourier_comp = FourierComponent(contour, self.N)
                fourier_comp.calculate_fourier_descriptors()
                for point in fourier_comp.outer_contour.points:
                    for n in range(self.N):
                        d_P_i[n,point[0], point[1]] += fourier_comp.derivative(point, n+1)
                d_P_i = d_P_i.reshape((self.N, -1))
                m = fourier_comp.A-self.gaussian_mixture.means_[0]
                d = np.dot(m, precision)
                d = np.dot(d, d_P_i)
                d = d.reshape((weight_map.shape[1], weight_map.shape[2]))
                weight_map[j,:,:] += d
        return weight_map