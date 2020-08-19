#ifndef HAUSDORFF_H
#define HAUSDORFF_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"

#define SQUARE(A) ((A) * (A))

struct TObject{
    int maxOverlapId;
    int maxOverlapArea;
    int area;
    int *x;
    int *y;
};
typedef struct TObject OBJECT;

struct TObjects{
    int no;
    OBJECT *obj;
};
typedef struct TObjects ALL_OBJECTS;

ALL_OBJECTS initializeObjects(MATRIX cmap);
void freeObjects(ALL_OBJECTS A);
MATRIX findOverlaps(MATRIX segm, MATRIX gold, int segmNo, int goldNo);
void findMaximallyOverlapObjects(MATRIX segm, MATRIX gold, ALL_OBJECTS *segmObj, ALL_OBJECTS *goldObj);
double checkClosePoints(int x, int y, MATRIX O2map, int O2id);
double calculateMaximumOfMinimumDistances(OBJECT O1, OBJECT O2, MATRIX O2map, int O2id);
double calculateHausdorff(OBJECT O1, MATRIX O1map, int O1id, OBJECT O2, MATRIX O2map, int O2id);
double calculateForNonOverlapping(OBJECT O1, MATRIX O1map, int O1id, ALL_OBJECTS O2s, MATRIX O2map);
double calculateForAll(MATRIX segm, ALL_OBJECTS segmObj, double allSegmAreas, MATRIX gold, ALL_OBJECTS goldObj, double allGoldAreas);
double hausdorff_distance(const MATRIX segm, const MATRIX gold, double allSegmAreas, double allGoldAreas);

#endif