import scipy.io
from src.datasets import *
from src.metrics import *
import glob
import cv2
import sys
from multiprocessing import Pool

def main(model_name, data_type, c, dataset_name, erode, dilate, area, threshold):
    try:
        y_pred = []
        y_true = []
        y_pred_bnd = []

        for path in glob.glob(os.path.join(DATASET_PATHS[dataset_name], 'output', model_name, '{}_*.mat'.format(data_type))):
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

        iou = IntersectionOverUnion().calculate(y_true_c, y_pred_c).mean()
        tp, fp, fn = ObjectBasedMetrics().calculate(y_true_c, y_pred_c)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f_score = (2*precision*recall)/(precision+recall)
        hausdorff = Hausdorff().calculate(y_true_c, y_pred_c)
        dice = ObjectDice().calculate(y_true_c, y_pred_c)

        file = open('./results', 'a+')
        file.write(f'{dataset_name} - {data_type} - {c} - {model_name} - {threshold} - {100*precision}, {100*recall}, {100*f_score}, {hausdorff}, {100*dice}, {100*iou}\n')
        file.close()
    except:
        print("FAIL")
    
if __name__=='__main__':
    args = []
    
#     args.append(['unet_fourier_11', 'huh7', 1, 'nucseg', 9, 3, 250, 0.1])
#     args.append(['unet_fourier_11', 'huh7', 1, 'nucseg', 9, 3, 250, 0.2])
#     args.append(['unet_fourier_11', 'huh7', 1, 'nucseg', 9, 3, 250, 0.3])
#     args.append(['unet_fourier_11', 'huh7', 1, 'nucseg', 9, 5, 250, 0.4])
#     args.append(['unet_fourier_11', 'huh7', 1, 'nucseg', 7, 7, 250, 0.5])
#     args.append(['unet_fourier_11', 'huh7', 1, 'nucseg', 0, 9, 250, 0.6])
#     args.append(['unet_fourier_11', 'huh7', 1, 'nucseg', 0, 9, 250, 0.7])
#     args.append(['unet_fourier_11', 'huh7', 1, 'nucseg', 3, 11, 250, 0.8])
#     args.append(['unet_fourier_11', 'huh7', 1, 'nucseg', 0, 11, 250, 0.9])
    
#     args.append(['unet_fourier_11', 'hepg2', 1, 'nucseg', 9, 3, 250, 0.1])
#     args.append(['unet_fourier_11', 'hepg2', 1, 'nucseg', 9, 3, 250, 0.2])
#     args.append(['unet_fourier_11', 'hepg2', 1, 'nucseg', 9, 3, 250, 0.3])
#     args.append(['unet_fourier_11', 'hepg2', 1, 'nucseg', 9, 5, 250, 0.4])
#     args.append(['unet_fourier_11', 'hepg2', 1, 'nucseg', 7, 7, 250, 0.5])
#     args.append(['unet_fourier_11', 'hepg2', 1, 'nucseg', 0, 9, 250, 0.6])
#     args.append(['unet_fourier_11', 'hepg2', 1, 'nucseg', 0, 9, 250, 0.7])
#     args.append(['unet_fourier_11', 'hepg2', 1, 'nucseg', 3, 11, 250, 0.8])
#     args.append(['unet_fourier_11', 'hepg2', 1, 'nucseg', 0, 11, 250, 0.9])
    
#     args.append(['unet_wce_3', 'huh7', 1, 'nucseg', 9, 3, 250, 0.1])
    args.append(['unet_wce_3', 'huh7', 1, 'nucseg', 9, 3, 250, 0.2])
#     args.append(['unet_wce_3', 'huh7', 1, 'nucseg', 9, 3, 250, 0.3])
#     args.append(['unet_wce_3', 'huh7', 1, 'nucseg', 9, 5, 250, 0.4])
#     args.append(['unet_wce_3', 'huh7', 1, 'nucseg', 7, 5, 250, 0.5])
#     args.append(['unet_wce_3', 'huh7', 1, 'nucseg', 7, 7, 500, 0.6])
#     args.append(['unet_wce_3', 'huh7', 1, 'nucseg', 0, 7, 250, 0.7])
#     args.append(['unet_wce_3', 'huh7', 1, 'nucseg', 0, 9, 500, 0.8])
#     args.append(['unet_wce_3', 'huh7', 1, 'nucseg', 0, 11, 500, 0.9])
    
#     args.append(['unet_wce_3', 'hepg2', 1, 'nucseg', 9, 3, 250, 0.1])
    args.append(['unet_wce_3', 'hepg2', 1, 'nucseg', 9, 3, 250, 0.2])
#     args.append(['unet_wce_3', 'hepg2', 1, 'nucseg', 9, 3, 250, 0.3])
#     args.append(['unet_wce_3', 'hepg2', 1, 'nucseg', 9, 5, 250, 0.4])
#     args.append(['unet_wce_3', 'hepg2', 1, 'nucseg', 7, 5, 250, 0.5])
#     args.append(['unet_wce_3', 'hepg2', 1, 'nucseg', 7, 7, 500, 0.6])
#     args.append(['unet_wce_3', 'hepg2', 1, 'nucseg', 0, 7, 250, 0.7])
#     args.append(['unet_wce_3', 'hepg2', 1, 'nucseg', 0, 9, 500, 0.8])
#     args.append(['unet_wce_3', 'hepg2', 1, 'nucseg', 0, 11, 500, 0.9])
    
    p = Pool(processes=len(args))
    p.starmap(main, args)
    