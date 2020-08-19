#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"
#include "hausdorff.h"

ALL_OBJECTS initializeObjects(MATRIX cmap){
    ALL_OBJECTS A;
    int cid, i, j;
    
    A.no = maxMatrixEntry(cmap);
    A.obj = (OBJECT *) calloc(A.no + 1, sizeof(OBJECT));
    for (i = 0; i <= A.no; i++){
        A.obj[i].maxOverlapId = 0;
        A.obj[i].maxOverlapArea = 0;
        A.obj[i].area = 0;
    }
    for (i = 0; i < cmap.row; i++)
        for (j = 0; j < cmap.column; j++)
            if (cmap.data[i][j])
                A.obj[cmap.data[i][j]].area += 1;
    
    for (i = 1; i <= A.no; i++){
        A.obj[i].x = (int *) calloc(A.obj[i].area, sizeof(int));
        A.obj[i].y = (int *) calloc(A.obj[i].area, sizeof(int));
        A.obj[i].area = 0;
    }
    
    for (i = 0; i < cmap.row; i++)
        for (j = 0; j < cmap.column; j++)
            if (cmap.data[i][j]){
                cid = cmap.data[i][j];
                A.obj[cid].x[A.obj[cid].area] = i;
                A.obj[cid].y[A.obj[cid].area] = j;
                A.obj[cid].area += 1;
            }
    return A;
}
void freeObjects(ALL_OBJECTS A){
    int i;
    for (i = 1; i <= A.no; i++){
        free(A.obj[i].x);
        free(A.obj[i].y);
    }
    free(A.obj);
}
MATRIX findOverlaps(MATRIX segm, MATRIX gold, int segmNo, int goldNo){
    MATRIX overlaps = allocateMatrix(segmNo + 1, goldNo + 1);
    int i, j;
    
    initializeMatrix(&overlaps, 0);
    for (i = 0; i < segm.row; i++)
        for (j = 0; j < segm.column; j++)
            if (segm.data[i][j] && gold.data[i][j])
                overlaps.data[segm.data[i][j]][gold.data[i][j]] += 1;
    return overlaps;
}
void findMaximallyOverlapObjects(MATRIX segm, MATRIX gold, ALL_OBJECTS *segmObj, ALL_OBJECTS *goldObj){
    MATRIX overlaps = findOverlaps(segm, gold, segmObj->no, goldObj->no);
    int i, j;
    
    for (i = 1; i <= segmObj->no; i++){
        segmObj->obj[i].maxOverlapId = 0;
        segmObj->obj[i].maxOverlapArea = 0;
        for (j = 1; j <= goldObj->no; j++)
            if (overlaps.data[i][j]){
                if (segmObj->obj[i].maxOverlapId == 0 || segmObj->obj[i].maxOverlapArea < overlaps.data[i][j]){
                    segmObj->obj[i].maxOverlapId = j;
                    segmObj->obj[i].maxOverlapArea = overlaps.data[i][j];
                }
            }
    }
    for (i = 1; i <= goldObj->no; i++){
        goldObj->obj[i].maxOverlapId = 0;
        goldObj->obj[i].maxOverlapArea = 0;
        for (j = 1; j <= segmObj->no; j++)
            if (overlaps.data[j][i]){
                if (goldObj->obj[i].maxOverlapId == 0 || goldObj->obj[i].maxOverlapArea < overlaps.data[j][i]){
                    goldObj->obj[i].maxOverlapId = j;
                    goldObj->obj[i].maxOverlapArea = overlaps.data[j][i];
                }
            }
    }
    freeMatrix(overlaps);
}
/**********************************************************************************/
/**********************************************************************************/
/**********************************************************************************/
double checkClosePoints(int x, int y, MATRIX O2map, int O2id){
    if (O2map.data[x][y] == O2id)       return 0;
    
    if (x > 0                && O2map.data[x - 1][y] == O2id)       return 1;
    if (x < O2map.row - 1    && O2map.data[x + 1][y] == O2id)       return 1;
    if (y > 0                && O2map.data[x][y - 1] == O2id)       return 1;
    if (y < O2map.column - 1 && O2map.data[x][y + 1] == O2id)       return 1;
    
    if (x > 0                && y > 0                 && O2map.data[x - 1][y - 1] == O2id)      return 2;
    if (x > 0                && y < O2map.column - 1  && O2map.data[x - 1][y + 1] == O2id)      return 2;
    if (x < O2map.row - 1    && y > 0                 && O2map.data[x + 1][y - 1] == O2id)      return 2;
    if (x < O2map.row - 1    && y < O2map.column - 1  && O2map.data[x + 1][y + 1] == O2id)      return 2;
    
    
    if (x > 1                && O2map.data[x - 2][y] == O2id)       return 4;
    if (x < O2map.row - 2    && O2map.data[x + 2][y] == O2id)       return 4;
    if (y > 1                && O2map.data[x][y - 2] == O2id)       return 4;
    if (y < O2map.column - 2 && O2map.data[x][y + 2] == O2id)       return 4;
    
    return -1;
}
double calculateMaximumOfMinimumDistances(OBJECT O1, OBJECT O2, MATRIX O2map, int O2id){
    double maxDist = 0, minDist, d;
    int i, j;
    
    for (i = 0; i < O1.area; i++){
        minDist = checkClosePoints(O1.x[i], O1.y[i], O2map, O2id);
        if (minDist < -0.5){
            minDist = SQUARE(O1.x[i] - O2.x[0]) + SQUARE(O1.y[i] - O2.y[0]);
            for (j = 1; j < O2.area; j++){
                d = SQUARE(O1.x[i] - O2.x[j]) + SQUARE(O1.y[i] - O2.y[j]);
                if (d < minDist)
                    minDist = d;
            }
        }
        
        if (maxDist < minDist)
            maxDist = minDist;
    }
    return sqrt(maxDist);
}
double calculateHausdorff(OBJECT O1, MATRIX O1map, int O1id, OBJECT O2, MATRIX O2map, int O2id){
    double d1 = calculateMaximumOfMinimumDistances(O1, O2, O2map, O2id);
    double d2 = calculateMaximumOfMinimumDistances(O2, O1, O1map, O1id);
    
    if (d1 > d2)
        return d1;
    return d2;
}
double calculateForNonOverlapping(OBJECT O1, MATRIX O1map, int O1id, ALL_OBJECTS O2s, MATRIX O2map){
    double d, h = calculateHausdorff(O1, O1map, O1id, O2s.obj[1], O2map, 1);
    int i;
    
    for (i = 2; i <= O2s.no; i++){
        d = calculateHausdorff(O1, O1map, O1id, O2s.obj[i], O2map, i);
        if (d < h)
            h = d;
    }
    return h;
}
double calculateForAll(MATRIX segm, ALL_OBJECTS segmObj, double allSegmAreas, MATRIX gold, ALL_OBJECTS goldObj, double allGoldAreas){
    MATRIXD lookup = allocateMatrixD(segmObj.no + 1, goldObj.no + 1);
    double h1, h2, d;
    int i, j, oid;
    
    initializeMatrixD(&lookup, -1.0);
    /* to calculate the first term */
    h1 = 0.0;
    for (i = 1; i <= segmObj.no; i++){
        if (segmObj.obj[i].maxOverlapId){
            oid = segmObj.obj[i].maxOverlapId;
            d = calculateHausdorff(segmObj.obj[i], segm, i, goldObj.obj[oid], gold, oid);
            lookup.data[i][oid] = d;
        }
        else
            d = calculateForNonOverlapping(segmObj.obj[i], segm, i, goldObj, gold);
        h1 += d * segmObj.obj[i].area / allSegmAreas;
    }
    
    /* to calculate the second term */
    h2 = 0.0;
    for (i = 1; i <= goldObj.no; i++){
        if (goldObj.obj[i].maxOverlapId){
            oid = goldObj.obj[i].maxOverlapId;
            if (lookup.data[oid][i] > -0.5)
                d = lookup.data[oid][i];
            else
                d = calculateHausdorff(goldObj.obj[i], gold, i, segmObj.obj[oid], segm, oid);
        }
        else
            d = calculateForNonOverlapping(goldObj.obj[i], gold, i, segmObj, segm);
        h2 += d * goldObj.obj[i].area / allGoldAreas;
    }
    freeMatrixD(lookup);
    return h1/2 + h2/2;
}

double hausdorff_distance(const MATRIX segm, const MATRIX gold, double allSegmAreas, double allGoldAreas){
    ALL_OBJECTS goldObj = initializeObjects(gold);
    ALL_OBJECTS segmObj = initializeObjects(segm);
    double hausdorff;
    
    findMaximallyOverlapObjects(segm, gold, &segmObj, &goldObj);
    hausdorff = calculateForAll(segm, segmObj, allSegmAreas, gold, goldObj, allGoldAreas);
    
    freeObjects(goldObj);
    freeObjects(segmObj);
    
    return hausdorff;
}

