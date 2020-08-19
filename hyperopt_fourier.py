from hyperopt import fmin, tpe, hp
from src.datasets import *
from src.losses import *
import sys
import numpy as np
from multiprocessing import Pool

def main(dataset, c):
    try:
        def fn(space):
            input_ch, output_ch, img_shape, model_shape, train_stride, test_stride, bnd_dilate = DATASET_STATS[dataset]
            train_generator = get_dataset(dataset, 'train', img_shape, model_shape, train_stride, 1, False, bnd_dilate)
            validation_generator = get_dataset(dataset, 'validation', img_shape, model_shape, train_stride, 1, False, bnd_dilate)
            _, y = train_generator.get_dataset_as_batch()
            y = y['mask']
            _, y_val = validation_generator.get_dataset_as_batch()
            y_val = y_val['mask']
            N = [1,1,1,1]
            K = [1,1,1,1]
            modes = ['center','center','center','center']
            covariance_types = ['full','full','full','full']
            N[c-1] = (space['n']+1)
            K[c-1] = space['k']+1
            covariance_types[c-1] = space['cov']
            modes[c-1] = space['mode']
            fd = FourierDescriptors(K, N, modes, 'harmonic_amplitude', covariance_types, 1.0, 1000)
            fd.fit(y)
            loss_val = fd.get_losses_c(y_val, c).mean()
            return loss_val

        space = {
            'mode' : hp.choice('mode', ['center', 'angle']),
            'cov' : hp.choice('cov', ['full', 'diag', 'spherical']),
            'n' : hp.randint('n', 99),
            'k' : hp.randint('k', 9)
        }

        best = fmin(
            fn=fn,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            rstate=np.random.RandomState(42)
        )
        
        file = open('./fourier_hyperopt', 'a+')
        file.write(f'{dataset} {c} {best}\n')
        file.close()
    except:
        print(f'FAIL {dataset} {c}')
    
if __name__=='__main__':
    args = []
    
    args.append(['cama', 1])
    
    p = Pool(processes=len(args))
    p.starmap(main, args)