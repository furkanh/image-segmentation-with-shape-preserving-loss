#ifndef __matrix_h
#define __matrix_h

#define ZERO 0.00000000000000000000001
#define EPSILON 0.00000000000000000000001

/*******************************************************/
/**************** 2D matrix of integers ****************/
/*******************************************************/
struct TMatrix{
	long row;
	long column;
	int **data;
};
typedef struct TMatrix  MATRIX;

MATRIX allocateMatrix(long row, long column);
void reallocateMatrix(MATRIX *M, long row, long column);
void freeMatrix(MATRIX M);
void initializeMatrix(MATRIX *M, int c);
void copyMatrix(MATRIX *A, MATRIX B);

void displayMatrix(MATRIX M);
void writeMatrixIntoFile(MATRIX M, char *filename, int headerFlag);

int maxMatrixEntry(MATRIX M);
int minMatrixEntry(MATRIX M);
/*******************************************************/
/**************** 2D matrix of doubles *****************/
/*******************************************************/
struct TMatrixD{
	long row;
	long column;
	double **data;
};
typedef struct TMatrixD MATRIXD;

MATRIXD allocateMatrixD(long row, long column);
void freeMatrixD(MATRIXD M);
void initializeMatrixD(MATRIXD *M, double c);
void copyMatrixD(MATRIXD *A, MATRIXD B);
void reallocateMatrixD(MATRIXD *M, long row, long column);

void displayMatrixD(MATRIXD M, int precision);
void writeMatrixDIntoFile(MATRIXD M, char *filename, int precision, int headerFlag);

double maxMatrixDEntry(MATRIXD M);
double minMatrixDEntry(MATRIXD M);
double maxMatrixDColumnEntry(MATRIXD M, long whichColumn);
double minMatrixDColumnEntry(MATRIXD M, long whichColumn);

MATRIXD computeMeanMatrixD(MATRIXD A);
MATRIXD computeCovarianceMatrixD(MATRIXD A);
MATRIXD computeCorrelationMatrixD(MATRIXD A);

MATRIXD multiplyMatrixD(MATRIXD A, MATRIXD B);
/*******************************************************/
/**************** 3D matrix of integers ****************/
/*******************************************************/
struct TMatrix3{
	long d1, d2, d3;
	int ***data;
};
typedef struct TMatrix3  MATRIX3;

MATRIX3 allocateMatrix3(long d1, long d2, long d3);
void freeMatrix3(MATRIX3 M);
void initializeMatrix3(MATRIX3 *M, int c);
void copyMatrix3(MATRIX3 *dest, MATRIX3 src);
/*******************************************************/
/**************** 3D matrix of doubles *****************/
/*******************************************************/
struct TMatrixD3{
	long d1, d2, d3;
	double ***data;
};
typedef struct TMatrixD3 MATRIXD3;

MATRIXD3 allocateMatrixD3(long d1, long d2, long d3);
void freeMatrixD3(MATRIXD3 M);
void initializeMatrixD3(MATRIXD3 *M, double c);
/*******************************************************/
/*******************************************************/
/*******************************************************/

#endif