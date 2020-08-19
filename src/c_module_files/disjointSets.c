#include <stdio.h>
#include <stdlib.h>
#include "disjointSets.h"
#include "matrix.h"

void freeDisjointSets(DisjointSets *ds){
    free(ds->arr);
}

DisjointSets allocateDisjointSets(int maxSize){
    DisjointSets ds;
    ds.size = maxSize;
    ds.arr = (int *) malloc(maxSize*sizeof(int));
    for(int i = 0; i<maxSize; i++)
        ds.arr[i] = 0;
    return ds;
}

int getRootOfDisjointSet(DisjointSets *disjointSets, int id){
    int j = id;
    while(disjointSets->arr[j]>0)
        j = disjointSets->arr[j];
    return j;
}

void insertDisjointSets(DisjointSets *ds, int child, int parent){
    if(child < parent){
        int temp = child;
        child = parent;
        parent = temp;
    }
    if(ds->arr==NULL){
        ds->arr = (int *) malloc((child+1)*sizeof(int));
        ds->size = child+1;
    }
    else if(child >= ds->size){
        ds->arr = (int *) realloc(ds->arr, (child+1)*sizeof(int));
        for(int i=ds->size; i<(child+1); i++)
            ds->arr[i] = 0;
        ds->size = child+1;
    }
    int childRoot, parentRoot;
    childRoot = getRootOfDisjointSet(ds, child);
    parentRoot = getRootOfDisjointSet(ds, parent);
    if(childRoot > parentRoot){
        ds->arr[childRoot] = parentRoot;
    }
    else if(childRoot < parentRoot){
        ds->arr[parentRoot] = childRoot;
    }
}