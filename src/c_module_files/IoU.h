#ifndef IOU_H
#define IOU_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"

MATRIXD findIntersectionOverUnions(MATRIX gold, MATRIX segm);
void calculateMetrics4IoU(MATRIXD IoU, double thr, int *TP, int *FP, int *FN);
void calculateMatchScores(MATRIXD IoU, MATRIXD thrs, MATRIX *TP, MATRIX *FP, MATRIX *FN);

#endif