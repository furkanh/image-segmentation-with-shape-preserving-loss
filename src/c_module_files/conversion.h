#ifndef CONVERSION_H
#define CONVERSION_H

#include "Python.h"
#include "numpy/arrayobject.h"
#include "matrix.h"

MATRIX convertPyMatrix2Matrix(const PyArrayObject *pyMatrix);
MATRIXD convertPyMatrix2MatrixD(const PyArrayObject *pyMatrix);
PyArrayObject *convertMatrix2PyMatrix(const MATRIX *M);
PyArrayObject *convertMatrixD32PyMatrix(const MATRIXD3 *M);
PyArrayObject *convertMatrixD2PyMatrix(const MATRIXD *M);

#endif