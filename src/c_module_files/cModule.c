#include "Python.h"
#include "numpy/arrayobject.h"
#include "matrix.h"
#include "conversion.h"

#include "hausdorff.h"
#include "cv.h"

#include "disjointSets.h"
#include "IoU.h"

PyObject *wrapper_persistent_homology(PyObject *self, PyObject *args){
    PyArrayObject *pyTrue, *pyPred, *pyOut;
    MATRIX y_true;
    MATRIXD y_pred;
    MATRIXD3 persistentHomologyMap;
    
    if(!PyArg_ParseTuple(args, "OO", &pyTrue, &pyPred)){
        return NULL;
    }
    
    y_true = convertPyMatrix2Matrix(pyTrue);
    y_pred = convertPyMatrix2MatrixD(pyPred);
    
    persistentHomologyMap = persistentHomology(&y_true, &y_pred);
    pyOut = convertMatrixD32PyMatrix(&persistentHomologyMap);
    
    freeMatrix(y_true);
    freeMatrixD(y_pred);
    freeMatrixD3(persistentHomologyMap);
    
    return Py_BuildValue("O", pyOut);
}

PyObject *wrapper_hausdorff_distance(PyObject *self, PyObject *args){
    PyArrayObject *pyGold, *pySegm;
    MATRIX gold, segm;
    double allSegmAreas, allGoldAreas, hausdorff;
    
    if(!PyArg_ParseTuple(args, "OOdd", &pySegm, &pyGold, &allSegmAreas, &allGoldAreas)){
        return NULL;
    }
    
    Py_INCREF(pyGold);
    Py_INCREF(pySegm);
    
    segm = convertPyMatrix2Matrix(pySegm);
    gold = convertPyMatrix2Matrix(pyGold);
    
    Py_DECREF(pyGold);
    Py_DECREF(pySegm);
    
    hausdorff = hausdorff_distance(segm, gold, allSegmAreas, allGoldAreas);
    
    freeMatrix(segm);
    freeMatrix(gold);
    
    return Py_BuildValue("d", hausdorff);
}

PyObject *wrapper_connected_components(PyObject *self, PyObject *args){
    PyArrayObject *pySegm, *pyOut;
    MATRIX segm;
    
    if(!PyArg_ParseTuple(args, "O", &pySegm)){
        return NULL;
    }
    
    Py_INCREF(pySegm);
    
    segm = convertPyMatrix2Matrix(pySegm);
    
    Py_DECREF(pySegm);
    
    connectedComponents(&segm);
    
    pyOut = convertMatrix2PyMatrix(&segm);
    
    freeMatrix(segm);
    
    return Py_BuildValue("O", pyOut);
}

PyObject *wrapper_IoU(PyObject *self, PyObject *args){
    PyArrayObject *pySegm, *pyGold, *pyThrs, *pyTP, *pyFP, *pyFN;
    MATRIX segm, gold;
    MATRIXD thrs;
    
    if(!PyArg_ParseTuple(args, "OOO", &pySegm, &pyGold, &pyThrs)){
        return NULL;
    }
    
    Py_INCREF(pySegm);
    Py_INCREF(pyGold);
    Py_INCREF(pyThrs);
    
    segm = convertPyMatrix2Matrix(pySegm);
    gold = convertPyMatrix2Matrix(pyGold);
    thrs = convertPyMatrix2MatrixD(pyThrs);
    
    Py_DECREF(pyThrs);
    Py_DECREF(pyGold);
    Py_DECREF(pySegm);
    
    MATRIXD IoU = findIntersectionOverUnions(gold, segm);
    MATRIX TP, FP, FN;
    
    calculateMatchScores(IoU, thrs, &TP, &FP, &FN);
    
    pyTP = convertMatrix2PyMatrix(&TP);
    pyFP = convertMatrix2PyMatrix(&FP);
    pyFN = convertMatrix2PyMatrix(&FN);
    
    freeMatrix(TP);
    freeMatrix(FP);
    freeMatrix(FN);
    freeMatrix(gold);
    freeMatrix(segm);
    freeMatrixD(IoU);
    freeMatrixD(thrs);
    
    return Py_BuildValue("OOO", pyTP, pyFP, pyFN);
}

static PyMethodDef c_methods[] = {
    {"hausdorff_distance", wrapper_hausdorff_distance, METH_VARARGS, "Calculates hausdorff distance between gold and segm"},
    {"connected_components", wrapper_connected_components, METH_VARARGS, "Returns connected components of given segmentation"},
    {"persistent_homology", wrapper_persistent_homology, METH_VARARGS, "Returns persisten homogology of two images"},
    {"intersection_over_union", wrapper_IoU, METH_VARARGS, "Returns IoU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef c_module = {
    PyModuleDef_HEAD_INIT,
    "c_module",
    "dosctring for c module",
    -1,
    c_methods
};

PyMODINIT_FUNC PyInit_c_module(void){
    return PyModule_Create(&c_module);
}

