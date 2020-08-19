import scipy.io
from src.datasets import *
from src.metrics import *
import glob
import cv2
import sys
from multiprocessing import Pool

def main(model_name, c, dataset_name, threshold):
    def loss_fn_helper(erode, dilate, area, c):
        y_pred = []
        y_true = []
        y_pred_bnd = []

        for path in glob.glob(os.path.join(DATASET_PATHS[dataset_name], 'output', model_name, '{}_*.mat'.format('train'))):
            mat = scipy.io.loadmat(path)
            y_pred.append(mat['y_pred'])
            y_true.append(mat['y_true'])
            if 'y_pred_bnd' in mat:
                y_pred_bnd.append(mat['y_pred_bnd'])
                
        for path in glob.glob(os.path.join(DATASET_PATHS[dataset_name], 'output', model_name, '{}_*.mat'.format('validation'))):
            mat = scipy.io.loadmat(path)
            y_pred.append(mat['y_pred'])
            y_true.append(mat['y_true'])
            if 'y_pred_bnd' in mat:
                y_pred_bnd.append(mat['y_pred_bnd'])
        
        y_pred_c = []
        y_true_c = []
        for i in range(len(y_pred)):
            pred = np.int32(y_pred[i][:,:,1]>threshold)
            if len(y_pred_bnd)>0:
                pred_bnd = np.int32(np.argmax(y_pred_bnd[i], axis=-1))
                pred = np.int32(pred*(1-pred_bnd))
            true = y_true[i][:,:,c-1]
            if erode>0:
                strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
                pred = cv2.erode(np.uint8(pred), strel)
            _, connected_components = cv2.connectedComponents(np.uint8(pred), connectivity=4)
            if erode>0:
                connected_components = dilate_connected_components(connected_components, erode)
            if dilate>0:
                connected_components = dilate_connected_components(connected_components, dilate)
            connected_components = area_open(connected_components, area)
            y_pred_c.append(connected_components)
            y_true_c.append(true)
        iou = IntersectionOverUnion()
        intersection_over_union = iou.calculate(y_true_c, y_pred_c).mean()
#         object_based_metrics = ObjectBasedMetrics()
#         tp, fp, fn = object_based_metrics.calculate(y_true_c, y_pred_c)
#         precision = tp/(tp+fp)
#         recall = tp/(tp+fn)
#         f_score = (2*precision*recall)/(precision+recall)
        return -(intersection_over_union)

    def loss_fn(erode, dilate, area, c):
        loss = loss_fn_helper(erode, dilate, area, c)
        return loss

    erodes = [0, 3, 5, 7, 9]
    dilates = [3, 5, 7, 9, 11]
    areas = [0, 50, 100, 250, 500]

    min_loss = None
    min_erode = None
    min_dilate = None
    min_area = None
    for erode in erodes:
        for dilate in dilates:
            for area in areas:
                try:
                    loss = loss_fn(erode, dilate, area, c)
                    if min_loss is None or loss < min_loss:
                        min_loss = loss
                        min_erode = erode
                        min_dilate = dilate
                        min_area = area
                        print(dataset_name, model_name, c, min_loss)
                except Exception:
                    print('fail')

    file = open(f'./post_hyperopt_{dataset_name}', 'a+')
    file.write(f'{dataset_name} - {c} - {model_name}: (th) {threshold} (e) {min_erode} (d) {min_dilate} (a) {min_area} (l) {min_loss}\n')
    file.close()
    
if __name__=='__main__':
    args = []
    
    for th in np.arange(0.1, 1.0, 0.1):
        args.append(['unet_wce_3', 1, 'nucseg', th])
        
    for th in np.arange(0.1, 1.0, 0.1):
        args.append(['unet_fourier_11', 1, 'nucseg', th])
    
    p = Pool(processes=len(args))
    p.starmap(main, args)
    