#ifndef CV_H
#define CV_H

#include "matrix.h"

void getNeighborsOfMatrixD(MATRIXD *M, int x, int y, double *neighbors);
void getNeighborsOfMatrix(MATRIX *M, int x, int y, int *neighbors);
int getLeftNeighbor(MATRIX *M, int x, int y);
int getTopNeighbor(MATRIX *M, int x, int y);
MATRIX getCriticalPoints(MATRIXD *y_pred);
int partitionCriticalPoints(MATRIXD *y_pred, MATRIX *criticalPoints, int low, int high);
void quickSortCriticalPoints(MATRIXD *y_pred, MATRIX *criticalPoints, int low, int high);
void connectedComponents(MATRIX *M);
MATRIXD3 persistentHomology(MATRIX *y_true, MATRIXD *y_pred);
MATRIXD computePersistentDots(MATRIX *criticalPoints, MATRIXD *y_pred);
void addRandomNoise(MATRIX *criticalPoints, MATRIXD *y_pred);
void quickSortPersistentDots(MATRIXD *persistentDots, int low, int high);
int partitionPersistentDots(MATRIXD *persistentDots, int low, int high);

#endif