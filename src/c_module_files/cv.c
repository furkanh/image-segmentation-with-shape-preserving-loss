#include "matrix.h"
#include "cv.h"
#include "disjointSets.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void getNeighborsOfMatrixD(MATRIXD *M, int x, int y, double *neighbors){
    /*
        Returns all neighbors in an array
        in the following order
         ______
        |0|1|2|
        |7|_|3|
        |6|5|4|
    */
    for(int i=0; i<8; i++)
        neighbors[i] = 0;
    if(x>0 && y>0) neighbors[0] = M->data[x-1][y-1];
    if(x>0) neighbors[1] = M->data[x-1][y];
    if(x>0 && y<M->column-1) neighbors[2] = M->data[x-1][y+1];
    if(y<M->column-1) neighbors[3] = M->data[x][y+1];
    if(x<M->row-1 && y<M->column-1) neighbors[4] = M->data[x+1][y+1];
    if(x<M->row-1) neighbors[5] = M->data[x+1][y];
    if(x<M->row-1 && y>0) neighbors[6] = M->data[x+1][y-1];
    if(y>0) neighbors[7] = M->data[x][y-1];
}

void getNeighborsOfMatrix(MATRIX *M, int x, int y, int *neighbors){
    /*
        Returns all neighbors in an array
        in the following order
         ______
        |0|1|2|
        |7|_|3|
        |6|5|4|
    */
    for(int i=0; i<8; i++)
        neighbors[i] = 0;
    if(x>0 && y>0) neighbors[0] = M->data[x-1][y-1];
    if(x>0) neighbors[1] = M->data[x-1][y];
    if(x>0 && y<M->column-1) neighbors[2] = M->data[x-1][y+1];
    if(y<M->column-1) neighbors[3] = M->data[x][y+1];
    if(x<M->row-1 && y<M->column-1) neighbors[4] = M->data[x+1][y+1];
    if(x<M->row-1) neighbors[5] = M->data[x+1][y];
    if(x<M->row-1 && y>0) neighbors[6] = M->data[x+1][y-1];
    if(y>0) neighbors[7] = M->data[x][y-1];
}

int getLeftNeighbor(MATRIX *M, int x, int y){
    if(y>0) return M->data[x][y-1];
    return 0;
}

int getTopNeighbor(MATRIX *M, int x, int y){
    if(x>0) return M->data[x-1][y];
    return 0;
}

MATRIX getCriticalPoints(MATRIXD *y_pred){
    int i, j;
    int numOfCriticalPoints = 0;
    double vd, hd, dd, ndd;
    MATRIX criticalPoints = allocateMatrix(0,0);
    for(i = 1; i<y_pred->row-1; i++){
        for(j = 1; j<y_pred->column-1; j++){
            // Calculate vertical, horizontal, diagonal derivatives of point (i,j)
            // WARNING sobel filter is used but can be changed
            double first = 1.0;
            double second = 2.0;
            vd = (first*y_pred->data[i-1][j-1]+second*y_pred->data[i-1][j]+first*y_pred->data[i-1][j+1]);
            vd -= (first*y_pred->data[i+1][j+1]+second*y_pred->data[i+1][j]+first*y_pred->data[i+1][j-1]);
            hd = (first*y_pred->data[i-1][j-1]+second*y_pred->data[i][j-1]+first*y_pred->data[i+1][j-1]);
            hd -= (first*y_pred->data[i-1][j+1]+second*y_pred->data[i][j+1]+first*y_pred->data[i+1][j+1]);
            dd = y_pred->data[i+1][j+1]-y_pred->data[i-1][j-1];
            ndd = y_pred->data[i+1][j-1]-y_pred->data[i-1][j+1];
            if(vd<0) vd = -vd;
            if(hd<0) hd = -hd;
            if(dd<0) dd = -dd;
            if(ndd<0) ndd = -ndd;
            if(vd<ZERO && hd<ZERO && dd<ZERO && ndd<ZERO){
                //(i,j) is a critical point
                numOfCriticalPoints++;
                if(criticalPoints.data==NULL)
                    criticalPoints = allocateMatrix(numOfCriticalPoints, 2);
                else
                    reallocateMatrix(&criticalPoints, numOfCriticalPoints, 2);
                criticalPoints.data[numOfCriticalPoints-1][0] = i;
                criticalPoints.data[numOfCriticalPoints-1][1] = j;
            }
        }
    }
    return criticalPoints;
}

int partitionCriticalPoints(MATRIXD *y_pred, MATRIX *criticalPoints, int low, int high){
    int high_x = criticalPoints->data[high][0];
    int high_y = criticalPoints->data[high][1];
    double pivot = y_pred->data[high_x][high_y];
    
    int i = low - 1;
    int j;
    for(j = low; j < high; j++){
        int j_x = criticalPoints->data[j][0];
        int j_y = criticalPoints->data[j][1];
        if(y_pred->data[j_x][j_y] < pivot){
            i++;
            criticalPoints->data[j][0] = criticalPoints->data[i][0];
            criticalPoints->data[j][1] = criticalPoints->data[i][1];
            criticalPoints->data[i][0] = j_x;
            criticalPoints->data[i][1] = j_y;
        }
    }
    int temp_x = criticalPoints->data[i+1][0];
    int temp_y = criticalPoints->data[i+1][1];
    criticalPoints->data[i+1][0] = criticalPoints->data[high][0];
    criticalPoints->data[i+1][1] = criticalPoints->data[high][1];
    criticalPoints->data[high][0] = temp_x;
    criticalPoints->data[high][1] = temp_y;
    return i + 1;
}

void quickSortCriticalPoints(MATRIXD *y_pred, MATRIX *criticalPoints, int low, int high){
    if(low < high){
        int mid = partitionCriticalPoints(y_pred, criticalPoints, low, high);
        quickSortCriticalPoints(y_pred, criticalPoints, low, mid-1);
        quickSortCriticalPoints(y_pred, criticalPoints, mid + 1, high);
    }
}

void connectedComponents(MATRIX *M){
    int i, j, root, left, top;
    int *rootIDs;
    DisjointSets ds = allocateDisjointSets(1);
    int idCounter = 0;
    for(i = 0; i < M->row; i++){
        for(j = 0; j < M->column; j++){
            if(M->data[i][j]>0){
                left = getLeftNeighbor(M, i, j);
                top = getTopNeighbor(M, i, j);
                if(left==0 && top==0){
                    idCounter++;
                    M->data[i][j] = idCounter;
                    insertDisjointSets(&ds, idCounter, idCounter);
                }
                else if(left==0 && top>0){
                    M->data[i][j] = top;
                    insertDisjointSets(&ds, top, top);
                }
                else if((left>0 && top==0) || left==top){
                    M->data[i][j] = left;
                    insertDisjointSets(&ds, left, left);
                }
                else{
                    M->data[i][j] = left;
                    insertDisjointSets(&ds, top, left);
                }
            }
        }
    }
    
    idCounter = 0;
    int size = maxMatrixEntry(*M)+1;
    rootIDs = (int *) malloc(size*sizeof(int));
    for(i = 0; i<size; i++)
        rootIDs[i] = 0;
    
    for(i = 0; i < M->row; i++){
        for(j = 0; j < M->column; j++){
            if(M->data[i][j]>0){
                root =  getRootOfDisjointSet(&ds, M->data[i][j]);
                if(rootIDs[root]==0){
                    idCounter++;
                    rootIDs[root] = idCounter;
                }
                M->data[i][j] = rootIDs[root];
            }
        }
    }
    
    free(rootIDs);
    freeDisjointSets(&ds);
}

MATRIXD computePersistentDots(MATRIX *criticalPoints, MATRIXD *y_pred){
    MATRIX connectedComponents;
    MATRIXD objects; // birth | death | expected_birth | expected_death
    int k, prevMaxId;
    
    connectedComponents = allocateMatrix(y_pred->row, y_pred->column);
    initializeMatrix(&connectedComponents, 0);
    objects = allocateMatrixD(0,0);
    prevMaxId = 0;
    
    for(k = criticalPoints->row-1; k>=0; k--){
        double alpha = y_pred->data[criticalPoints->data[k][0]][criticalPoints->data[k][1]];
        int i, j, root, left, top;
        int *rootIDs;
        DisjointSets ds = allocateDisjointSets(1);
        int idCounter = prevMaxId;
        for(i = 0; i < connectedComponents.row; i++){
            for(j = 0; j < connectedComponents.column; j++){
                if(y_pred->data[i][j]>=alpha){
                    left = getLeftNeighbor(&connectedComponents, i, j);
                    top = getTopNeighbor(&connectedComponents, i, j);
                    if(left==0 && top==0 && connectedComponents.data[i][j]==0){
                        idCounter++;
                        connectedComponents.data[i][j] = idCounter;
                        insertDisjointSets(&ds, idCounter, idCounter);
                    }
                    else if(left==0 && top>0){
                        if(connectedComponents.data[i][j]==0){
                            connectedComponents.data[i][j] = top;
                            insertDisjointSets(&ds, top, top);
                        }
                        else
                            insertDisjointSets(&ds, connectedComponents.data[i][j], top);
                    }
                    else if((left>0 && top==0) || left==top){
                        if(connectedComponents.data[i][j]==0){
                            connectedComponents.data[i][j] = left;
                            insertDisjointSets(&ds, left, left);
                        }
                        else
                            insertDisjointSets(&ds, connectedComponents.data[i][j], left);
                    }
                    else{
                        insertDisjointSets(&ds, top, left);
                        if(connectedComponents.data[i][j]==0){
                            connectedComponents.data[i][j] = left;
                            insertDisjointSets(&ds, left, left);
                        }
                        else
                            insertDisjointSets(&ds, connectedComponents.data[i][j], left);
                        
                    }
                }
            }
        }
        
        idCounter = prevMaxId;
        int size = maxMatrixEntry(connectedComponents)+1;
        rootIDs = (int *) malloc(size*sizeof(int));
        for(i = 0; i<size; i++)
            rootIDs[i] = 0;
        
        for(i = 0; i < connectedComponents.row; i++){
            for(j = 0; j < connectedComponents.column; j++){
                if(connectedComponents.data[i][j]>0){
                    root =  getRootOfDisjointSet(&ds, connectedComponents.data[i][j]);
                    if(rootIDs[root]==0){
                        if(root>prevMaxId){
                            idCounter++;
                            rootIDs[root] = idCounter;
                            // idCounter is born at alpha
                            if(objects.row<=idCounter){
                                reallocateMatrixD(&objects, idCounter+1, 4);
                            }
                            objects.data[idCounter][0] = alpha;
                        }
                        else{
                            rootIDs[root] = root;
                        }
                    }
                    connectedComponents.data[i][j] = rootIDs[root];
                }
            }
        }
        
        for(i = 0; i<ds.size; i++){
            if(i!=getRootOfDisjointSet(&ds, i) && i<=prevMaxId){
                // i dies at alpha
                objects.data[i][1] = alpha;
            }
        }
        
        prevMaxId = idCounter;
        
        free(rootIDs);
        freeDisjointSets(&ds);
        
    } // end alpha for
    
    freeMatrix(connectedComponents);
    
    //remove the first
    objects.data[0][0] = objects.data[objects.row-1][0];
    objects.data[0][1] = objects.data[objects.row-1][1];
    reallocateMatrixD(&objects, objects.row-1, 2);
    
    return objects;
}

void addRandomNoise(MATRIX *criticalPoints, MATRIXD *y_pred){
    for(int i = 0; i<criticalPoints->row; i++){
        int x, y;
        x = criticalPoints->data[i][0];
        y = criticalPoints->data[i][1];
        double epsilon = 0.00001;
        double randomValue = -(((double)rand()/RAND_MAX)*epsilon);
        y_pred->data[x][y] += randomValue;
    }
}

int partitionPersistentDots(MATRIXD *persistentDots, int low, int high){
    double birth = persistentDots->data[high][0];
    double death = persistentDots->data[high][1];
    double pivot = (1.0-birth)*(1.0-birth)+death*death; // distance to (1,0)
    
    int i = low - 1;
    int j;
    for(j = low; j < high; j++){
        double j_birth = persistentDots->data[j][0];
        double j_death = persistentDots->data[j][1];
        if(((1.0-j_birth)*(1.0-j_birth)+j_death*j_death) < pivot){
            i++;
            persistentDots->data[j][0] = persistentDots->data[i][0];
            persistentDots->data[j][1] = persistentDots->data[i][1];
            persistentDots->data[i][0] = j_birth;
            persistentDots->data[i][1] = j_death;
        }
    }
    double temp_birth = persistentDots->data[i+1][0];
    double temp_death = persistentDots->data[i+1][1];
    persistentDots->data[i+1][0] = persistentDots->data[high][0];
    persistentDots->data[i+1][1] = persistentDots->data[high][1];
    persistentDots->data[high][0] = temp_birth;
    persistentDots->data[high][1] = temp_death;
    return i + 1;
}

void quickSortPersistentDots(MATRIXD *persistentDots, int low, int high){
    if(low < high){
        int mid = partitionPersistentDots(persistentDots, low, high);
        quickSortPersistentDots(persistentDots, low, mid-1);
        quickSortPersistentDots(persistentDots, mid + 1, high);
    }
}

MATRIXD3 persistentHomology(MATRIX *y_true, MATRIXD *y_pred){
    /*
        Given a probability map and ground truth,
        this function calculates persistent homology
        and returns a 3D matrix[H][W][4] where each
        channel corresponds to b1, b2, d1, d2
        Then topological loss can be calculates as
        follows:
        
        L_topo = ||b1*y_pred-b2||^2+||d1*y_pred-d2||^2
    */
    srand(time(NULL));
    MATRIX criticalPoints;
    MATRIXD3 persistentHomologyMap; // bmap1 | bmap2 | dmap1 | dmap2
    MATRIXD persistentDots, truePersistentDots; // birth | death
    
    persistentHomologyMap = allocateMatrixD3(y_true->row, y_true->column, 4);
    initializeMatrixD3(&persistentHomologyMap, 0.0);
    
    criticalPoints = getCriticalPoints(y_pred); // x | y
    if(criticalPoints.data!=NULL){
        addRandomNoise(&criticalPoints, y_pred);
        quickSortCriticalPoints(y_pred, &criticalPoints, 0, criticalPoints.row-1);
        persistentDots = computePersistentDots(&criticalPoints, y_pred);
        if(persistentDots.data!=NULL){
            quickSortPersistentDots(&persistentDots, 0, persistentDots.row-1);
            connectedComponents(y_true);
            int num = maxMatrixEntry(*y_true);
            truePersistentDots = allocateMatrixD(persistentDots.row, persistentDots.column);
            for(int i = 0; i<num; i++){
                if(i>=persistentDots.row)
                    break;
                truePersistentDots.data[i][0] = 1.0;
                truePersistentDots.data[i][1] = 0.0;
            }
            for(int i=num; i<persistentDots.row; i++){
                double projection = (persistentDots.data[i][0]+persistentDots.data[i][1])/2.0;
                truePersistentDots.data[i][0] = projection;
                truePersistentDots.data[i][1] = projection;
            }
            for(int i=0; i<persistentDots.row; i++){
                int bx, by, dx, dy;
                bx = by = dx = dy = 0;
                for(int j=0; j<criticalPoints.row; j++){
                    int x, y;
                    x = criticalPoints.data[j][0];
                    y = criticalPoints.data[j][1];
                    if(y_pred->data[x][y]==persistentDots.data[i][0]){
                        bx = x;
                        by = y;
                    }
                    else if(y_pred->data[x][y]==persistentDots.data[i][1]){
                        dx = x;
                        dy = y;
                    }
                    if(bx!=0 && by!=0 && dx!=0 && dy!=0)
                        break;
                }
                if(bx!=0 && by!=0 && dx!=0 && dy!=0){
                    if(persistentHomologyMap.data[bx][by][0]>0 || persistentHomologyMap.data[dx][dy][2]>0){
                        persistentHomologyMap.data[bx][by][0] = -1.0;
                        persistentHomologyMap.data[bx][by][1] = -1.0;
                        persistentHomologyMap.data[dx][dy][2] = -1.0;
                        persistentHomologyMap.data[dx][dy][3] = -1.0;
                    }
                    else if(persistentHomologyMap.data[bx][by][0]<EPSILON && persistentHomologyMap.data[dx][dy][2]<EPSILON){
                        persistentHomologyMap.data[bx][by][0] = 1.0;
                        persistentHomologyMap.data[bx][by][1] = truePersistentDots.data[i][0];
                        persistentHomologyMap.data[dx][dy][2] = 1.0;
                        persistentHomologyMap.data[dx][dy][3] = truePersistentDots.data[i][1];
                    }
                }
            }
            for(int i=0; i<persistentHomologyMap.d1; i++)
                for(int j=0; j<persistentHomologyMap.d2; j++)
                    for(int k=0; k<persistentHomologyMap.d3; k++)
                        if(persistentHomologyMap.data[i][j][k]<0)
                            persistentHomologyMap.data[i][j][k] = 0.0;
            freeMatrixD(truePersistentDots);
        }
        freeMatrixD(persistentDots);
    }
    freeMatrix(criticalPoints);
    return persistentHomologyMap;
}