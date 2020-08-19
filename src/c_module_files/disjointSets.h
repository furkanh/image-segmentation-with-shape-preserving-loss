#ifndef __DISJOINT_SETS_H
#define __DISJOINT_SETS_H

#include "matrix.h"

typedef struct TDisjointSets{
    int *arr;
    int size;
} DisjointSets;

void freeDisjointSets(DisjointSets *ds);
DisjointSets allocateDisjointSets(int maxSize);
int getRootOfDisjointSet(DisjointSets *disjointSets, int id);
void insertDisjointSets(DisjointSets *ds, int child, int parent);

#endif
