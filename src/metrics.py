from src import c_module
import numpy as np
import cv2

def merge_connected_components(args):
    connected_components = np.zeros_like(args[0])
    num = 1
    for arg in args:
        labels = np.unique(arg)
        labels = labels[labels>0]
        for label in labels:
            component = (arg==label)*num
            connected_components += np.int32(component*((connected_components==0)*1))
            num += 1
    return connected_components

def dilate_connected_components(connected_components, dilate):
    '''
    Given a connected components image, dilates all connected components

    Args:
        connected_components (np.ndarray) : (H,W)
        dilate (int) : size of the structuring element

    Returns:
        result (np.ndarray) : (H,W) dilated components
    '''
    if dilate>0:
        result = np.zeros_like(connected_components)
        labels = np.unique(connected_components)
        labels = labels[labels>0]
        for label in labels:
            component = (connected_components==label).astype(np.int32)
            strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
            component = cv2.dilate(component.astype(np.uint8), strel)
            result += ((component*(1.-(result>0)))*label).astype(np.int32)
        return np.int32(result)
    else:
        return connected_components
    
def area_open(connected_components, area):
    '''
    Given a connected components image, removes small areas
    and returns the resulting connected components

    Args:
        connected_components (np.ndarray) : (H,W)

    Returns:
        result (np.ndarray) : (H,W)
    '''
    if area>0:
        result = np.zeros_like(connected_components)
        i = 1
        labels = np.unique(connected_components)
        labels = labels[labels>0]
        for label in labels:
            component = (connected_components==label).astype(np.int32)
            if component.sum()>=area:
                result += component*i
                i += 1
        return result.astype(np.int32)
    else:
        return connected_components

class Metric:
    def calculate(self, y_true, y_pred):
        '''
        y_true [N][H,W] : connected components
        y_pred [N][H,W] : connected components
        '''
        pass
    
    def findOverlaps(self, first, second):
        '''
        Args:
            first (np.ndarray) : (H,W) connected components
            second (np.ndarray) : (H,W) connected components

        Returns:
            overlaps (np.ndarray) : (H,W) i,j in overlaps shows the area of
                overlap between ith and jth components in first and second respectively
                H: number of components in first
                W: number of components in second
        '''
        dx = second.shape[0]
        dy = second.shape[1]

        firstNo = int(first.max())
        secondNo = int(second.max())

        overlaps = np.zeros((firstNo, secondNo))

        for i in range(dx):
            for j in range(dy):
                if second[i,j]>0 and first[i,j]>0:
                    overlaps[int(first[i,j])-1, int(second[i,j])-1] += 1

        return overlaps

    def findAreas(self, cmap):
        '''
        Args:
            cmap (np.ndarray) : (H,W) connected components

        Returns:
            componentAreas (np.ndarray) : area of each component
                (H,) H: number of components in cmap
        '''
        dx = cmap.shape[0]
        dy = cmap.shape[1]
        componentNo = int(cmap.max())
        componentAreas = np.zeros((componentNo,))
        for i in range(dx):
            for j in range(dy):
                if cmap[i,j]>0:
                    componentAreas[int(cmap[i,j])-1] += 1
        return componentAreas

    def findMaximallyOverlapObjects(self, first, second):
        '''
        Args:
            first (np.ndarray) : (H,W) connected components
            second (np.ndarray) : (H,W) connected components

        Returns:
            maximallyOverlapObjects (np.ndarray) : (H,) H: number of connected components in first
            overlappingAreas (np.ndarray) : (H,) H: number of connected components in first
        '''
        overlaps = self.findOverlaps(first, second)
        firstNo, secondNo = overlaps.shape[0], overlaps.shape[1]
        maximallyOverlapObjects = np.ones((firstNo,))*(-1)
        overlappingAreas = np.zeros((firstNo,))
        for i in range(firstNo):
            for j in range(secondNo):
                if overlaps[i,j]>0:
                    if maximallyOverlapObjects[i]==-1 or overlappingAreas[i]<overlaps[i,j]:
                        maximallyOverlapObjects[i] = j
                        overlappingAreas[i] = overlaps[i,j]
        return maximallyOverlapObjects, overlappingAreas
    
class ObjectDice(Metric):
    def calculateObjectDice(self, segm, gold, allSegmAreas, allGoldAreas):
        goldAreas = self.findAreas(gold)
        goldNo = goldAreas.shape[0]

        segmAreas = self.findAreas(segm)
        segmNo = segmAreas.shape[0]

        gold4segm, overlaps4segm = self.findMaximallyOverlapObjects(segm, gold)
        segm4gold, overlaps4gold = self.findMaximallyOverlapObjects(gold, segm)

        goldRatios = goldAreas / allGoldAreas
        segmRatios = segmAreas / allSegmAreas

        dice = 0
        for i in range(segmNo):
            if gold4segm[i]>=0:
                curr = (2*overlaps4segm[i]) / (segmAreas[i] + goldAreas[int(gold4segm[i])])
                dice = dice + segmRatios[i]*curr

        for i in range(goldNo):
            if segm4gold[i]>=0:
                curr = (2*overlaps4gold[i]) / (goldAreas[i] + segmAreas[int(segm4gold[i])])
                dice = dice + goldRatios[i]*curr
        return dice/2.
    
    def calculate(self, y_true, y_pred):
        '''
        y_true [N][H,W] : connected components
        y_pred [N][H,W] : connected components
        '''
        dice = 0
        allSegmAreas = 0
        allGoldAreas = 0
        for i in range(len(y_pred)):
            allSegmAreas += (y_pred[i]>0).sum()
            allGoldAreas += (y_true[i]>0).sum()
        for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
            dice += self.calculateObjectDice(segm, gold, allSegmAreas, allGoldAreas)
        return dice
    
class Hausdorff(Metric):
    def calculate(self, y_true, y_pred):
        '''
        y_true [N][H,W] : connected components
        y_pred [N][H,W] : connected components
        '''
        allSegmAreas = 0
        allGoldAreas = 0
        for i in range(len(y_pred)):
            allSegmAreas += (y_pred[i]>0).sum()
            allGoldAreas += (y_true[i]>0).sum()
        hausdorff = 0
        for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
            if segm.sum()>0 and gold.sum()>0:
                hausdorff += c_module.hausdorff_distance(np.int32(segm).copy(), np.int32(gold).copy(), float(allSegmAreas), float(allGoldAreas))
        return hausdorff
    
class ObjectBasedMetrics(Metric):
    def calculate4Detection(self, segm, gold):
        '''
        Args:
            segm (np.ndarray) : (H,W) connected components of segmentation
            gold (np.ndarray) : (H,W) connected components of annotation

        Returns:
            TP (int) : true positives
            FP (int) : false positives
            FN (int) : false negatives
        '''
        maximallyOverlapObjects, overlappingAreas = self.findMaximallyOverlapObjects(segm, gold)
        goldAreas = self.findAreas(gold)
        segmentedNo = overlappingAreas.shape[0]
        goldNo = goldAreas.shape[0]
        TP, FP, FN = 0, 0, 0
        for i in range(segmentedNo):
            if int(maximallyOverlapObjects[i])>=0 and (overlappingAreas[i]/goldAreas[int(maximallyOverlapObjects[i])]) >= 0.5:
                TP += 1
            else:
                FP += 1
                maximallyOverlapObjects[i] = -1

        for i in range(goldNo):
            found = 0
            for j in range(segmentedNo):
                if int(maximallyOverlapObjects[j]) == i:
                    found = 1
                    break
            if found==0:
                FN += 1
        return TP, FP, FN
    
    def calculate(self, y_true, y_pred):
        '''
        y_true [N][H,W] : connected components
        y_pred [N][H,W] : connected components
        '''
        TP, FP, FN = 0, 0, 0
        for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
            tp, fp, fn = self.calculate4Detection(segm, gold)
            TP += tp
            FP += fp
            FN += fn
        return TP, FP, FN
    
class IntersectionOverUnion(Metric):
    def __init__(self, low=0.5, high=0.95, step=0.05):
        self.thresholds = np.arange(low, high, step)
        self.thresholds = self.thresholds[np.newaxis]
    
    def calculate(self, y_true, y_pred):
        '''
        y_true [N][H,W] : connected components
        y_pred [N][H,W] : connected components
        '''
        TP, FP, FN = 0, 0, 0
        for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
            tp, fp, fn = c_module.intersection_over_union(np.int32(segm).copy(), np.int32(gold).copy(), np.float64(self.thresholds))
            TP += tp
            FP += fp
            FN += fn
        iou = TP/(FP+TP+FN)
        return iou
        
# class Metrics:
#     def __init__(self, metrics=['precision', 'recall', 'fscore', 'hausdorff', 'dice', 'IoU']):
#         self.metrics = metrics
    
#     def calculate_metrics(self, y_true, y_pred, num_of_neighbors, neighborhood, mode, erode, dilate, area):
#         '''
#         Returns precision, recall, fscore, hausdorff, dice for each channel

#         Args:
#             y_true [N][H,W,C] : connected components
#             y_pred [N][H,W] : predicted classes

#         Returns:
#             result (list of dict) : len(result) is C, [(precision, recall, fscore, hausdorff, dice), ...]
#             num_of_objects (list of int) : len(num_of_objects) is C
#         '''
#         for i in range(len(y_true)):
#             y_true[i] = np.int32(y_true[i])
#             y_pred[i] = np.int32(y_pred[i])
#         y_true, y_pred, num_of_objects = self.fit_transform(y_true, y_pred, num_of_neighbors, neighborhood, mode, erode, dilate, area)
#         result = []
#         for c in range(y_true[0].shape[-1]):
#             y_true_ = []
#             y_pred_ = []
#             for i in range(len(y_true)):
#                 y_true_.append(y_true[i][:,:,c])
#                 y_pred_.append(y_pred[i][:,:,c])
#             result.append(self.calculate_object_metrics_for_channel(y_true_, y_pred_))
#         return result, num_of_objects

#     def fit_transform(
#         self, y_true, y_pred,
#         num_of_neighbors, neighborhood, mode,
#         erode, dilate, area
#     ):
#         y_pred = self.connected_components(y_pred, erode=erode, dilate=dilate, area=area)
#         for i in range(len(y_true)):
#             if y_true[i].shape[:-1] != y_pred[i].shape[:-1]:
#                 resized = cv2.resize(y_pred[i], (y_true[i].shape[1], y_true[i].shape[0]), interpolation=cv2.INTER_NEAREST)
#                 resized = resized[...,np.newaxis] if y_pred[i].shape[-1]==1 else resized
#                 y_pred[i] = resized
#         object_list = self.calculate_object_list(y_true, num_of_neighbors=num_of_neighbors, neighborhood=2, mode=mode)
#         y_true_new, y_pred_new = self.keep_objects(y_true, y_pred, object_list)
#         num_of_objects = []
#         for i in range(len(object_list)):
#             num_of_objects.append(0)
#             for j in range(len(object_list[i])):
#                 num_of_objects[i] += len(object_list[i][j])
#         return y_true_new, y_pred_new, num_of_objects
    
# #     def findOverlaps(self, first, second):
# #         '''
# #         Args:
# #             first (np.ndarray) : (H,W) connected components
# #             second (np.ndarray) : (H,W) connected components

# #         Returns:
# #             overlaps (np.ndarray) : (H,W) i,j in overlaps shows the area of
# #                 overlap between ith and jth components in first and second respectively
# #                 H: number of components in first
# #                 W: number of components in second
# #         '''
# #         dx = second.shape[0]
# #         dy = second.shape[1]

# #         firstNo = int(first.max())
# #         secondNo = int(second.max())

# #         overlaps = np.zeros((firstNo, secondNo))

# #         for i in range(dx):
# #             for j in range(dy):
# #                 if second[i,j]>0 and first[i,j]>0:
# #                     overlaps[int(first[i,j])-1, int(second[i,j])-1] += 1

# #         return overlaps

# #     def findAreas(self, cmap):
# #         '''
# #         Args:
# #             cmap (np.ndarray) : (H,W) connected components

# #         Returns:
# #             componentAreas (np.ndarray) : area of each component
# #                 (H,) H: number of components in cmap
# #         '''
# #         dx = cmap.shape[0]
# #         dy = cmap.shape[1]
# #         componentNo = int(cmap.max())
# #         componentAreas = np.zeros((componentNo,))
# #         for i in range(dx):
# #             for j in range(dy):
# #                 if cmap[i,j]>0:
# #                     componentAreas[int(cmap[i,j])-1] += 1
# #         return componentAreas

# #     def findMaximallyOverlapObjects(self, first, second):
# #         '''
# #         Args:
# #             first (np.ndarray) : (H,W) connected components
# #             second (np.ndarray) : (H,W) connected components

# #         Returns:
# #             maximallyOverlapObjects (np.ndarray) : (H,) H: number of connected components in first
# #             overlappingAreas (np.ndarray) : (H,) H: number of connected components in first
# #         '''
# #         overlaps = self.findOverlaps(first, second)
# #         firstNo, secondNo = overlaps.shape[0], overlaps.shape[1]
# #         maximallyOverlapObjects = np.ones((firstNo,))*(-1)
# #         overlappingAreas = np.zeros((firstNo,))
# #         for i in range(firstNo):
# #             for j in range(secondNo):
# #                 if overlaps[i,j]>0:
# #                     if maximallyOverlapObjects[i]==-1 or overlappingAreas[i]<overlaps[i,j]:
# #                         maximallyOverlapObjects[i] = j
# #                         overlappingAreas[i] = overlaps[i,j]
# #         return maximallyOverlapObjects, overlappingAreas

# #     def calculate4Detection(self, segm, gold):
# #         '''
# #         Args:
# #             segm (np.ndarray) : (H,W) connected components of segmentation
# #             gold (np.ndarray) : (H,W) connected components of annotation

# #         Returns:
# #             TP (int) : true positives
# #             FP (int) : false positives
# #             FN (int) : false negatives
# #         '''
# #         maximallyOverlapObjects, overlappingAreas = self.findMaximallyOverlapObjects(segm, gold)
# #         goldAreas = self.findAreas(gold)
# #         segmentedNo = overlappingAreas.shape[0]
# #         goldNo = goldAreas.shape[0]
# #         TP, FP, FN = 0, 0, 0
# #         for i in range(segmentedNo):
# #             if int(maximallyOverlapObjects[i])>=0 and (overlappingAreas[i]/goldAreas[int(maximallyOverlapObjects[i])]) >= 0.5:
# #                 TP += 1
# #             else:
# #                 FP += 1
# #                 maximallyOverlapObjects[i] = -1

# #         for i in range(goldNo):
# #             found = 0
# #             for j in range(segmentedNo):
# #                 if int(maximallyOverlapObjects[j]) == i:
# #                     found = 1
# #                     break
# #             if found==0:
# #                 FN += 1
# #         return TP, FP, FN
    
# #     def calculateObjectDice(self, segm, gold, allSegmAreas, allGoldAreas):
# #         '''
# #         Args:
# #             segm (np.ndarray) : (H,W) connected components of segmentation
# #             gold (np.ndarray) : (H,W) connected components of annotation
# #             allSegmAreas (int) : total area of all segmentations
# #             allGoldAreas (int) : total area of all annotations

# #         Returns:
# #             dice (double) : object dice score between segm and gold
# #         '''
# #         goldAreas = self.findAreas(gold)
# #         goldNo = goldAreas.shape[0]

# #         segmAreas = self.findAreas(segm)
# #         segmNo = segmAreas.shape[0]

# #         gold4segm, overlaps4segm = self.findMaximallyOverlapObjects(segm, gold)
# #         segm4gold, overlaps4gold = self.findMaximallyOverlapObjects(gold, segm)

# #         goldRatios = goldAreas / allGoldAreas
# #         segmRatios = segmAreas / allSegmAreas

# #         dice = 0
# #         for i in range(segmNo):
# #             if gold4segm[i]>=0:
# #                 curr = (2*overlaps4segm[i]) / (segmAreas[i] + goldAreas[int(gold4segm[i])])
# #                 dice = dice + segmRatios[i]*curr

# #         for i in range(goldNo):
# #             if segm4gold[i]>=0:
# #                 curr = (2*overlaps4gold[i]) / (goldAreas[i] + segmAreas[int(segm4gold[i])])
# #                 dice = dice + goldRatios[i]*curr
# #         return dice/2.

# #     def calculate_tp_fp_fn(self, y_true, y_pred):
# #         TP, FP, FN = 0, 0, 0
# #         for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
# #             tp, fp, fn = self.calculate4Detection(segm, gold)
# #             TP += tp
# #             FP += fp
# #             FN += fn
# #         return TP, FP, FN

# #     def calculate_dice(self, y_true, y_pred):
# #         dice = 0
# #         allSegmAreas = 0
# #         allGoldAreas = 0
# #         for i in range(len(y_pred)):
# #             allSegmAreas += (y_pred[i]>0).sum()
# #             allGoldAreas += (y_true[i]>0).sum()
# #         for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
# #             dice += self.calculateObjectDice(segm, gold, allSegmAreas, allGoldAreas)
# #         return dice

# #     def calculate_hausdorff(self, y_true, y_pred):
# #         allSegmAreas = 0
# #         allGoldAreas = 0
# #         for i in range(len(y_pred)):
# #             allSegmAreas += (y_pred[i]>0).sum()
# #             allGoldAreas += (y_true[i]>0).sum()
# #         hausdorff = 0
# #         for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
# #             if segm.sum()>0 and gold.sum()>0:
# #                 hausdorff += c_module.hausdorff_distance(np.int32(segm).copy(), np.int32(gold).copy(), float(allSegmAreas), float(allGoldAreas))
# #         return hausdorff

# #     def calculate_iou(self, y_true, y_pred, thresholds):
# #         iou_TP, iou_FP, iou_FN = 0, 0, 0
# #         for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
# #             iou_tp, iou_fp, iou_fn = c_module.intersection_over_union(np.int32(segm).copy(), np.int32(gold).copy(), np.float64(thresholds))
# #             iou_TP += iou_tp
# #             iou_FP += iou_fp
# #             iou_FN += iou_fn
# #         iou = iou_TP/(iou_FP+iou_TP+iou_FN)
# #         return iou
    
#     def calculate_object_metrics_for_channel(self, y_true, y_pred):
#         '''
#         Returns metrics for images

#         Args:
#             y_true [N][H,W] : connected components
#             y_pred [N][H,W] : connected components

#         Returns:
#             result (tuple) : (precision, recall, fscore, hausdorff, dice)
#         '''
#         res = {}
#         metrics = self.metrics
#         if 'precision' in metrics or 'recall' in metrics or 'fscore' in metrics:
#             TP, FP, FN = self.calculate_tp_fp_fn(y_true, y_pred)
#             precision = TP / (TP+FP) if (TP+FP) > 0 else 0
#             res['precision'] = precision*100.
#             recall = TP / (TP+FN) if (TP+FN) > 0 else 0
#             res['recall'] = recall*100.
#             fscore = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
#             res['fscore'] = fscore*100.
#         if 'hausdorff' in metrics:
#             hausdorff = self.calculate_hausdorff(y_true, y_pred)
#             res['hausdorff'] = hausdorff
#         if 'dice' in metrics:
#             dice = self.calculate_dice(y_true, y_pred)
#             res['dice'] = dice*100.
#         if 'IoU' in metrics:
#             thresholds = np.arange(0.5,0.95,0.05)
#             thresholds = thresholds[np.newaxis]
#             iou = self.calculate_iou(y_true, y_pred, thresholds)
#             res['IoU'] = iou.mean()*100.
#             if 'IoU50' in metrics:
#                 res['IoU50'] = iou[0,0]*100
#             if 'IoU75' in metrics:
#                 res['IoU75'] = iou[0,5]*100
#             if 'IoU85' in metrics:
#                 res['IoU85'] = iou[0,7]*100
#             if 'IoU90' in metrics:
#                 res['IoU90'] = iou[0,8]*100
#         return res
    
#     def area_open(self, connected_components, area):
#         '''
#         Given a connected components image, removes small areas
#         and returns the resulting connected components

#         Args:
#             connected_components (np.ndarray) : (H,W)

#         Returns:
#             result (np.ndarray) : (H,W)
#         '''
#         result = np.zeros_like(connected_components)
#         i = 1
#         labels = np.unique(connected_components)
#         labels = labels[labels>0]
#         for label in labels:
#             component = (connected_components==label).astype(np.int32)
#             if component.sum()>=area:
#                 result += component*i
#                 i += 1
#         return result.astype(np.int32)

#     def dilate_connected_components(self, connected_components, dilate):
#         '''
#         Given a connected components image, dilates all connected components

#         Args:
#             connected_components (np.ndarray) : (H,W)
#             dilate (int) : size of the structuring element

#         Returns:
#             result (np.ndarray) : (H,W) dilated components
#         '''
#         result = np.zeros_like(connected_components)
#         labels = np.unique(connected_components)
#         labels = labels[labels>0]
#         for label in labels:
#             component = (connected_components==label).astype(np.int32)
#             strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
#             component = cv2.dilate(component.astype(np.uint8), strel)
#             result += ((component*(1.-(result>0)))*label).astype(np.int32)
#         return result.astype(np.int32)
        
#     def connected_components(self, y, erode=0, dilate=0, area=0):
#         '''
#         Returns connected components for each image.
#         Ignores the first channel as it belongs to background class.

#         Args:
#             y [N][H,W] : y_pred with predicted classes
#             area (int) : area open value
#             dilate (int) : connected components will be dilated this much
#             erode (int) : predicted classes will be eroded this much

#         Returns:
#             result [N][H,W,C] : connected components
#         '''
#         num = y[0].max()
#         for i in range(1, len(y)):
#             m = y[i].max()
#             if m > num:
#                 num = m
#         if num==0:
#             num = 1
#         result = []
#         for n in range(len(y)):
#             res = np.zeros(y[n].shape+(num,), dtype=np.int32)
#             for c in range(num):
#                 class_image = np.int32(y[n]==(c+1))
#                 if erode>0:
#                     strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
#                     class_image = cv2.erode(np.uint8(class_image), strel)
#                 _, connected_components = cv2.connectedComponents(np.uint8(class_image), connectivity=4)
#                 if erode>0:
#                     connected_components = self.dilate_connected_components(connected_components, erode)
#                 if dilate>0:
#                     connected_components = self.dilate_connected_components(connected_components, dilate)
#                 if area>0:
#                     connected_components = self.area_open(connected_components, area)
#                 res[:,:,c] = np.int32(connected_components)
#             result.append(res)
#         return result
    
#     def keep_objects(self, y_true, y_pred, object_list):
#         '''
#         Args:
#             y_true [N][H,W,C] : connected components
#             y_pred [N][H,W,C] : connected components
#             object_list (list) : list of objects in y_true to be kept

#         Returns:
#             y_true_new [N][H,W,C] : connected components
#             y_pred_new [N][H,W,C] : connected components
#         '''
#         y_true_new = []
#         y_pred_new = []
#         for j in range(len(y_true)):
#             y_true_img = np.zeros_like(y_true[j])
#             y_pred_img = np.zeros_like(y_pred[j])
#             for i in range(y_true[j].shape[-1]):
#                 label = 1
#                 for gold_label in object_list[i][j]:
#                     component = (y_true[j][:,:,i]==(gold_label+1))
#                     y_true_img[:,:,i] += component*label
#                     label += 1
#                 label = 1
#                 gold4segm, _ = self.findMaximallyOverlapObjects(y_pred[j][:,:,i], y_true[j][:,:,i])
#                 for k in range(len(gold4segm)):
#                     if int(gold4segm[k]) in object_list[i][j] or int(gold4segm[k])<0:
#                         component = (y_pred[j][:,:,i]==(k+1))
#                         y_pred_img[:,:,i] += component*label
#                         label += 1
#             y_true_new.append(y_true_img)
#             y_pred_new.append(y_pred_img)
#         return y_true_new, y_pred_new
    
#     def calculate_neighborhood_matrix(self, y, neighborhood=2):
#         '''
#         Args:
#             y (np.ndarray) : (H,W) connected components 
#             neighborhood (int) : area of neighborhood around a pixel

#         Returns:
#             neighborhood_matrix (np.ndarray) : (N,N)
#                 N is the number of connected components in y
#         '''
#         num = int(y.max())
#         neighborhood_matrix = np.zeros((num, num))
#         labels = np.unique(y)
#         labels = labels[labels>0]
#         for label in labels:
#             component = (y==label).astype(np.uint8)
#             _, contours, _ = cv2.findContours(component.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#             for contour in contours:
#                 for pixel in contour:
#                     pixel_neighborhood = y[pixel[0][1]-neighborhood:pixel[0][1]+neighborhood+1,
#                                            pixel[0][0]-neighborhood:pixel[0][0]+neighborhood+1]
#                     components_in_neighborhood = np.unique(pixel_neighborhood)
#                     components_in_neighborhood = components_in_neighborhood[components_in_neighborhood>0]
#                     for component_in_neighborhood in components_in_neighborhood:
#                         neighborhood_matrix[int(label-1),int(component_in_neighborhood-1)] = 1
#                         neighborhood_matrix[int(component_in_neighborhood-1),int(label-1)] = 1
#         return neighborhood_matrix
        
#     def calculate_object_list(self, y_true, num_of_neighbors=-1, neighborhood=2, mode='gt'):
#         '''
#         Args:
#             y_true [N][H,W,C] : connected components
#             num (int) : number of neighbors
#             neighborhood (int) : area of neighborhood calculating the neighborhood matrix
#             mode (string) : 
#                 'eq' get objects whose neighbors are equal to num.
#                 'ls' get objects whose neighbors are less than num.
#                 'gt' get objects whose neighbors are greater than num.

#         Returns:
#             objectList (list) : objectList[i][j] gives object list for ith channel of jth image
#         '''
#         assert mode=='eq' or mode=='ls' or mode=='gt'
#         objectList = []
#         for i in range(y_true[0].shape[-1]):
#             ls1 = []
#             for j in range(len(y_true)):
#                 ls2 = []
#                 neighborhood_matrix = self.calculate_neighborhood_matrix(y_true[j][:,:,i], neighborhood=neighborhood)
#                 neighborhood_matrix = neighborhood_matrix.sum(axis=1)-1
#                 for k in range(neighborhood_matrix.shape[0]):
#                     if mode=='eq' and neighborhood_matrix[k]==num_of_neighbors:
#                         ls2.append(k)
#                     elif mode=='ls' and neighborhood_matrix[k]<num_of_neighbors:
#                         ls2.append(k)
#                     elif mode=='gt' and neighborhood_matrix[k]>num_of_neighbors:
#                         ls2.append(k)
#                 ls1.append(ls2)
#             objectList.append(ls1)
#         return objectList
        
        
        
        
# def findOverlaps(first, second):
#     '''
#     Args:
#         first (np.ndarray) : (H,W) connected components
#         second (np.ndarray) : (H,W) connected components
        
#     Returns:
#         overlaps (np.ndarray) : (H,W) i,j in overlaps shows the area of
#             overlap between ith and jth components in first and second respectively
#             H: number of components in first
#             W: number of components in second
#     '''
#     dx = second.shape[0]
#     dy = second.shape[1]
    
#     firstNo = first.max()
#     secondNo = second.max()
    
#     overlaps = np.zeros((firstNo, secondNo))
    
#     for i in range(dx):
#         for j in range(dy):
#             if second[i,j]>0 and first[i,j]>0:
#                 overlaps[first[i,j]-1, second[i,j]-1] += 1
                
#     return overlaps

# def findAreas(cmap):
#     '''
#     Args:
#         cmap (np.ndarray) : (H,W) connected components
        
#     Returns:
#         componentAreas (np.ndarray) : area of each component
#             (H,) H: number of components in cmap
#     '''
#     dx = cmap.shape[0]
#     dy = cmap.shape[1]
#     componentNo = cmap.max()
#     componentAreas = np.zeros((componentNo,))
#     for i in range(dx):
#         for j in range(dy):
#             if cmap[i,j]>0:
#                 componentAreas[int(cmap[i,j])-1] += 1
#     return componentAreas

# def findMaximallyOverlapObjects(first, second):
#     '''
#     Args:
#         first (np.ndarray) : (H,W) connected components
#         second (np.ndarray) : (H,W) connected components
        
#     Returns:
#         maximallyOverlapObjects (np.ndarray) : (H,) H: number of connected components in first
#         overlappingAreas (np.ndarray) : (H,) H: number of connected components in first
#     '''
#     overlaps = findOverlaps(first, second)
#     firstNo, secondNo = overlaps.shape[0], overlaps.shape[1]
#     maximallyOverlapObjects = np.ones((firstNo,))*(-1)
#     overlappingAreas = np.zeros((firstNo,))
#     for i in range(firstNo):
#         for j in range(secondNo):
#             if overlaps[i,j]>0:
#                 if maximallyOverlapObjects[i]==-1 or overlappingAreas[i]<overlaps[i,j]:
#                     maximallyOverlapObjects[i] = j
#                     overlappingAreas[i] = overlaps[i,j]
#     return maximallyOverlapObjects, overlappingAreas
    
# def calculate4Detection(segm, gold):
#     '''
#     Args:
#         segm (np.ndarray) : (H,W) connected components of segmentation
#         gold (np.ndarray) : (H,W) connected components of annotation
        
#     Returns:
#         TP (int) : true positives
#         FP (int) : false positives
#         FN (int) : false negatives
#     '''
#     maximallyOverlapObjects, overlappingAreas = findMaximallyOverlapObjects(segm, gold)
#     goldAreas = findAreas(gold)
#     segmentedNo = overlappingAreas.shape[0]
#     goldNo = goldAreas.shape[0]
#     TP, FP, FN = 0, 0, 0
#     for i in range(segmentedNo):
#         if int(maximallyOverlapObjects[i])>=0 and (overlappingAreas[i]/goldAreas[int(maximallyOverlapObjects[i])]) >= 0.5:
#             TP += 1
#         else:
#             FP += 1
#             maximallyOverlapObjects[i] = -1
            
#     for i in range(goldNo):
#         found = 0
#         for j in range(segmentedNo):
#             if int(maximallyOverlapObjects[j]) == i:
#                 found = 1
#                 break
#         if found==0:
#             FN += 1
#     return TP, FP, FN

# def calculateObjectDice(segm, gold, allSegmAreas, allGoldAreas):
#     '''
#     Args:
#         segm (np.ndarray) : (H,W) connected components of segmentation
#         gold (np.ndarray) : (H,W) connected components of annotation
#         allSegmAreas (int) : total area of all segmentations
#         allGoldAreas (int) : total area of all annotations
        
#     Returns:
#         dice (double) : object dice score between segm and gold
#     '''
#     goldAreas = findAreas(gold)
#     goldNo = goldAreas.shape[0]
    
#     segmAreas = findAreas(segm)
#     segmNo = segmAreas.shape[0]
    
#     gold4segm, overlaps4segm = findMaximallyOverlapObjects(segm, gold)
#     segm4gold, overlaps4gold = findMaximallyOverlapObjects(gold, segm)
    
#     goldRatios = goldAreas / allGoldAreas
#     segmRatios = segmAreas / allSegmAreas
    
#     dice = 0
#     for i in range(segmNo):
#         if gold4segm[i]>=0:
#             curr = (2*overlaps4segm[i]) / (segmAreas[i] + goldAreas[int(gold4segm[i])])
#             dice = dice + segmRatios[i]*curr
            
#     for i in range(goldNo):
#         if segm4gold[i]>=0:
#             curr = (2*overlaps4gold[i]) / (goldAreas[i] + segmAreas[int(segm4gold[i])])
#             dice = dice + goldRatios[i]*curr
#     return dice/2.

# def calculate_tp_fp_fn(y_true, y_pred):
#     TP, FP, FN = 0, 0, 0
#     for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
#         tp, fp, fn = calculate4Detection(segm, gold)
#         TP += tp
#         FP += fp
#         FN += fn
#     return TP, FP, FN

# def calculate_dice(y_true, y_pred):
#     dice = 0
#     allSegmAreas = (y_pred>0).sum()
#     allGoldAreas = (y_true>0).sum()
#     for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
#         dice += calculateObjectDice(segm, gold, allSegmAreas, allGoldAreas)
#     return dice

# def calculate_hausdorff(y_true, y_pred):
#     allSegmAreas = (y_pred>0).sum()
#     allGoldAreas = (y_true>0).sum()
#     hausdorff = 0
#     for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
#         if segm.sum()>0 and gold.sum()>0:
#             hausdorff += c_module.hausdorff_distance(
#                 segm.astype(np.int32), gold.astype(np.int32), float(allSegmAreas), float(allGoldAreas))
#     return hausdorff
            
# def calculate_iou(y_true, y_pred, thresholds):
#     iou_TP, iou_FP, iou_FN = 0, 0, 0
#     for i, (segm, gold) in enumerate(zip(y_pred, y_true)):
#         iou_tp, iou_fp, iou_fn = c_module.intersection_over_union(segm.astype(np.int32), gold.astype(np.int32), np.float64(thresholds))
#         iou_TP += iou_tp
#         iou_FP += iou_fp
#         iou_FN += iou_fn
#     iou = iou_TP/(iou_FP+iou_TP+iou_FN)
#     return iou

# def calculate_object_metrics_for_channel(
#     y_true, y_pred,
#     metrics=['precision', 'recall', 'fscore', 'hausdorff', 'dice', 'IoU']
# ):
#     '''
#     Returns metrics for images
    
#     Args:
#         y_true (np.ndarray) : (N,H,W) connected components
#         y_pred (np.ndarray) : (N,H,W) connected components
        
#     Returns:
#         result (tuple) : (precision, recall, fscore, hausdorff, dice)
#     '''
#     res = {}
#     if 'precision' in metrics or 'recall' in metrics or 'fscore' in metrics:
#         TP, FP, FN = calculate_tp_fp_fn(y_true, y_pred)
#         precision = TP / (TP+FP)
#         res['precision'] = precision*100.
#         recall = TP / (TP+FN)
#         res['recall'] = recall*100.
#         fscore = (2*precision*recall)/(precision+recall)
#         res['fscore'] = fscore*100.
#     if 'hausdorff' in metrics:
#         hausdorff = calculate_hausdorff(y_true, y_pred)
#         res['hausdorff'] = hausdorff
#     if 'dice' in metrics:
#         dice = calculate_dice(y_true, y_pred)
#         res['dice'] = dice*100.
#     if 'IoU' in metrics:
#         thresholds = np.arange(0.5,0.95,0.05)
#         thresholds = thresholds[np.newaxis]
#         iou = calculate_iou(y_true, y_pred, thresholds)
#         res['IoU'] = iou.mean()*100.
#         if 'IoU50' in metrics:
#             res['IoU50'] = iou[0,0]*100
#         if 'IoU75' in metrics:
#             res['IoU75'] = iou[0,5]*100
#         if 'IoU85' in metrics:
#             res['IoU85'] = iou[0,7]*100
#         if 'IoU90' in metrics:
#             res['IoU90'] = iou[0,8]*100
#     return res

# def area_open(connected_components, area):
#     '''
#     Given a connected components image, removes small areas
#     and returns the resulting connected components
    
#     Args:
#         connected_components (np.ndarray) : (H,W)
        
#     Returns:
#         result (np.ndarray) : (H,W)
#     '''
#     result = np.zeros_like(connected_components)
#     i = 1
#     labels = np.unique(connected_components)
#     labels = labels[labels>0]
#     for label in labels:
#         component = (connected_components==label).astype(np.int32)
#         if component.sum()>=area:
#             result += component*i
#             i += 1
#     return result.astype(np.int32)

# def dilate_connected_components(connected_components, dilate):
#     '''
#     Given a connected components image, dilates all connected components
    
#     Args:
#         connected_components (np.ndarray) : (H,W)
#         dilate (int) : size of the structuring element
        
#     Returns:
#         result (np.ndarray) : (H,W) dilated components
#     '''
#     result = np.zeros_like(connected_components)
#     labels = np.unique(connected_components)
#     labels = labels[labels>0]
#     for label in labels:
#         component = (connected_components==label).astype(np.int32)
#         strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
#         component = cv2.dilate(component.astype(np.uint8), strel)
#         result += ((component*(1.-(result>0)))*label).astype(np.int32)
#     return result.astype(np.int32)

# def calculate_neighborhood_matrix(y, neighborhood=2):
#     '''
#     Args:
#         y (np.ndarray) : (H,W) connected components 
#         neighborhood (int) : area of neighborhood around a pixel
        
#     Returns:
#         neighborhood_matrix (np.ndarray) : (N,N)
#             N is the number of connected components in y
#     '''
#     num = y.max()
#     neighborhood_matrix = np.zeros((num, num))
#     labels = np.unique(y)
#     labels = labels[labels>0]
#     for label in labels:
#         component = (y==label).astype(np.uint8)
#         _, contours, _ = cv2.findContours(component.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#         for contour in contours:
#             for pixel in contour:
#                 pixel_neighborhood = y[pixel[0][1]-neighborhood:pixel[0][1]+neighborhood+1,
#                                        pixel[0][0]-neighborhood:pixel[0][0]+neighborhood+1]
#                 components_in_neighborhood = np.unique(pixel_neighborhood)
#                 components_in_neighborhood = components_in_neighborhood[components_in_neighborhood>0]
#                 for component_in_neighborhood in components_in_neighborhood:
#                     neighborhood_matrix[label-1,component_in_neighborhood-1] = 1
#                     neighborhood_matrix[component_in_neighborhood-1,label-1] = 1
#     return neighborhood_matrix

# def connected_components(y, erode=0, dilate=0, area=0):
#     '''
#     Returns connected components for each image.
#     Ignores the first channel as it belongs to background class.
    
#     Args:
#         y (np.ndarray) : (N,H,W) y_pred with predicted classes
#         area (int) : area open value
#         dilate (int) : connected components will be dilated this much
#         erode (int) : predicted classes will be eroded this much
    
#     Returns:
#         result (np.ndarray) : (N,H,W,C) connected components
#     '''
#     num = y.max()
#     if num==0:
#         num = 1
#     result = np.zeros(y.shape+(num,), dtype=np.int32)
#     for c in range(result.shape[-1]):
#         class_images = np.int32(y==(c+1))
#         for n in range(result.shape[0]):
#             class_image = class_images[n]
#             if erode>0:
#                 strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
#                 class_image = cv2.erode(np.uint8(class_image), strel)
#             _, connected_components = cv2.connectedComponents(np.uint8(class_image), connectivity=4)
#             if erode>0:
#                 connected_components = dilate_connected_components(connected_components, erode)
#             if dilate>0:
#                 connected_components = dilate_connected_components(connected_components, dilate)
#             if area>0:
#                 connected_components = area_open(connected_components, area)
#             result[n,:,:,c] = np.int32(connected_components)
#     return result

# def calculate_object_list(y_true, num_of_neighbors=-1, neighborhood=2, mode='gt'):
#     '''
#     Args:
#         y_true (np.ndarray) : (N,H,W,C) connected components
#         num (int) : number of neighbors
#         neighborhood (int) : area of neighborhood calculating the neighborhood matrix
#         mode (string) : 'eq' get objects whose neighbors are equal to num.
#             'ls' get objects whose neighbors are less than num.
#             'gt' get objects whose neighbors are greater than num.
        
#     Returns:
#         objectList (list) : objectList[i][j] gives object list for ith channel of jth image
#     '''
#     assert mode=='eq' or mode=='ls' or mode=='gt'
#     objectList = []
#     for i in range(y_true.shape[-1]):
#         ls1 = []
#         for j in range(y_true.shape[0]):
#             ls2 = []
#             neighborhood_matrix = calculate_neighborhood_matrix(y_true[j,:,:,i], neighborhood=neighborhood)
#             neighborhood_matrix = neighborhood_matrix.sum(axis=1)-1
#             for k in range(neighborhood_matrix.shape[0]):
#                 if mode=='eq' and neighborhood_matrix[k]==num_of_neighbors:
#                     ls2.append(k)
#                 elif mode=='ls' and neighborhood_matrix[k]<num_of_neighbors:
#                     ls2.append(k)
#                 elif mode=='gt' and neighborhood_matrix[k]>num_of_neighbors:
#                     ls2.append(k)
#             ls1.append(ls2)
#         objectList.append(ls1)
#     return objectList

# def keep_objects(y_true, y_pred, object_list):
#     '''
#     Args:
#         y_true (np.ndarray) : (N,H,W,C) connected components
#         y_pred (np.ndarray) : (N,H,W,C) connected components
#         object_list (list) : list of objects in y_true to be kept
        
#     Returns:
#         y_true_new (np.ndarray) : (N,H,W,C) connected components
#         y_pred_new (np.ndarray) : (N,H,W,C) connected components
#     '''
#     y_true_new = np.zeros_like(y_true)
#     y_pred_new = np.zeros_like(y_pred)
#     for i in range(y_true.shape[-1]):
#         for j in range(y_true.shape[0]):
#             label = 1
#             for gold_label in object_list[i][j]:
#                 component = (y_true[j,:,:,i]==(gold_label+1))
#                 y_true_new[j,:,:,i] += component*label
#                 label += 1
#             label = 1
#             gold4segm, _ = findMaximallyOverlapObjects(y_pred[j,:,:,i], y_true[j,:,:,i])
#             for k in range(len(gold4segm)):
#                 if int(gold4segm[k]) in object_list[i][j] or int(gold4segm[k])<0:
#                     component = (y_pred[j,:,:,i]==(k+1))
#                     y_pred_new[j,:,:,i] += component*label
#                     label += 1
#     return y_true_new.astype(np.int32), y_pred_new.astype(np.int32)

# def calculate_object_metrics(
#     y_true, y_pred,
#     metrics=['precision', 'recall', 'fscore', 'hausdorff', 'dice', 'IoU']
# ):
#     '''
#     Returns precision, recall, fscore, hausdorff, dice for each channel
    
#     Args:
#         y_true (np.ndarray) : (N,H,W,C) connected components
#         y_pred (np.ndarray) : (N,H,W,C) connected components
        
#     Returns:
#         result (list of dict) : len(result) is C, [(precision, recall, fscore, hausdorff, dice), ...]
#     '''
#     result = []
#     for c in range(y_true.shape[-1]):
#         result.append(calculate_object_metrics_for_channel(y_true[:,:,:,c], y_pred[:,:,:,c], metrics=metrics))
#     return result        
        