#include "Python.h"
#include "numpy/arrayobject.h"
#include "matrix.h"
#include "conversion.h"

MATRIX convertPyMatrix2Matrix(const PyArrayObject *pyMatrix){
    int row = (int) pyMatrix->dimensions[0];
    int column = (int) pyMatrix->dimensions[1];
    MATRIX M = allocateMatrix(row, column);
    int *matrix = (int *)pyMatrix->data;
    int count = 0;
    for (int i=0; i<row; i++){
        for(int j=0; j<column; j++){
            M.data[i][j] = matrix[count++];
        }
    }
    return M;
}

MATRIXD convertPyMatrix2MatrixD(const PyArrayObject *pyMatrix){
    int row = (int) pyMatrix->dimensions[0];
    int column = (int) pyMatrix->dimensions[1];
    MATRIXD M = allocateMatrixD(row, column);
    double *matrix = (double *)pyMatrix->data;
    int count = 0;
    for (int i=0; i<row; i++){
        for(int j=0; j<column; j++){
            M.data[i][j] = matrix[count++];
        }
    }
    return M;
}

PyArrayObject *convertMatrix2PyMatrix(const MATRIX *M){
    import_array();
    PyArrayObject *pyMatrix;
    npy_intp *dims = (npy_intp *) malloc(2*sizeof(npy_intp));
    dims[0] = M->row;
    dims[1] = M->column;
    int *data = (int *) malloc(M->row*M->column*sizeof(int));
    int count = 0;
    for(int i = 0; i<M->row; i++){
        for(int j = 0; j<M->column; j++){
            data[count++] = M->data[i][j];
        }
    }
    pyMatrix = (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NPY_INT32, (void *) data);
    return pyMatrix;
}

PyArrayObject *convertMatrixD2PyMatrix(const MATRIXD *M){
    import_array();
    PyArrayObject *pyMatrix;
    npy_intp *dims = (npy_intp *) malloc(2*sizeof(npy_intp));
    dims[0] = M->row;
    dims[1] = M->column;
    double *data = (double *) malloc(M->row*M->column*sizeof(double));
    int count = 0;
    for(int i = 0; i<M->row; i++){
        for(int j = 0; j<M->column; j++){
            data[count++] = M->data[i][j];
        }
    }
    pyMatrix = (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, (void *) data);
    return pyMatrix;
}

PyArrayObject *convertMatrixD32PyMatrix(const MATRIXD3 *M){
    import_array();
    PyArrayObject *pyMatrix;
    npy_intp *dims = (npy_intp *) malloc(3*sizeof(npy_intp));
    dims[0] = M->d1;
    dims[1] = M->d2;
    dims[2] = M->d3;
    double *data = (double *) malloc(((int)(M->d1*M->d2*M->d3))*sizeof(double));
    int count = 0;
    for(int i = 0; i<M->d1; i++){
        for(int j = 0; j<M->d2; j++){
            for(int k = 0; k<M->d3; k++){
                data[count++] = M->data[i][j][k];
            }
        }
    }
    pyMatrix = (PyArrayObject *) PyArray_SimpleNewFromData(3, dims, NPY_FLOAT64, (void *) data);
    return pyMatrix;
}