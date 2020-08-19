#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"
#include "IoU.h"

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
MATRIXD findIntersectionOverUnions(MATRIX gold, MATRIX segm){
    int i, j, gid, sid;
    int maxGold = maxMatrixEntry(gold);
    int maxSegm = maxMatrixEntry(segm);
    MATRIXD intersectionPixels = allocateMatrixD(maxGold + 1, maxSegm + 1);
    int *goldAreas = (int *)calloc(maxGold + 1, sizeof(int));
    int *segmAreas = (int *)calloc(maxSegm + 1, sizeof(int));
    double unionPixels;
    
    initializeMatrixD(&intersectionPixels, 0.0);
    for (i = 0; i <= maxGold; i++)
        goldAreas[i] = 0;
    for (i = 0; i <= maxSegm; i++)
        segmAreas[i] = 0;
    
    for (i = 0; i < gold.row; i++)
        for (j = 0; j < gold.column; j++){
            gid = gold.data[i][j];
            sid = segm.data[i][j];
            if (gid)
                goldAreas[gold.data[i][j]] += 1;
            if (sid)
                segmAreas[segm.data[i][j]] += 1;
            if (gid && sid)
                intersectionPixels.data[gid][sid] += 1;
        }

    for (i = 1; i <= maxGold; i++)
        for (j = 1; j <= maxSegm; j++){
            unionPixels = goldAreas[i] + segmAreas[j] - intersectionPixels.data[i][j];
            if (unionPixels > 0)
                intersectionPixels.data[i][j] /= unionPixels;
        }
    free(goldAreas);
    free(segmAreas);
    return intersectionPixels;      // now it is IoU pixels
}
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
void calculateMetrics4IoU(MATRIXD IoU, double thr, int *TP, int *FP, int *FN){
    int i, j, totalMatch;
    // IoU: gold x segm
    // TP: one-to-one match
    // FN: gold that matches 0 or > 1 segm
    // FP: segm that matches 0 or > 1 gold
    
    *TP = *FP = *FN = 0;
    for (i = 1; i < IoU.row; i++){
        totalMatch = 0;
        for (j = 1; j < IoU.column; j++)
            if (IoU.data[i][j] > thr)
                totalMatch++;
        if (totalMatch == 1)
            (*TP)++;
        else if (totalMatch == 0)
            (*FN)++;

      
    }
    for (j = 1; j < IoU.column; j++){
        totalMatch = 0;
        for (i = 1; i < IoU.row; i++)
            if (IoU.data[i][j] > thr)
                totalMatch++;
        if (totalMatch == 0)
            (*FP)++;
      
    }
}
void calculateMatchScores(MATRIXD IoU, MATRIXD thrs, MATRIX *TP, MATRIX *FP, MATRIX *FN){
    int i;
    
    *TP = allocateMatrix(1, thrs.column);
    *FP = allocateMatrix(1, thrs.column);
    *FN = allocateMatrix(1, thrs.column);
    for (i = 0; i < thrs.column; i++)
        calculateMetrics4IoU(IoU, thrs.data[0][i], &(TP->data[0][i]), &(FP->data[0][i]), &(FN->data[0][i]));
}

