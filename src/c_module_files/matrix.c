#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

/*******************************************************/
/**************** 2D matrix of integers ****************/
/*******************************************************/
MATRIX allocateMatrix(long row, long column){
	MATRIX M;
	long i;
	
	if (row <= 0 || column <= 0){
		M.row  = M.column = 0;
		M.data = NULL;
	}
	else {
		M.row = row;
		M.column = column;
		M.data = (int **) malloc(row * sizeof(int*));
		for (i = 0; i < row; i++)
			M.data[i] = (int *) malloc(column * sizeof(int));
	}
	return M;
}
void reallocateMatrix(MATRIX *M, long row, long column){
	int i, j, minrow, mincol;
	MATRIX tmp = allocateMatrix(M->row,M->column);
    
	copyMatrix(&tmp,*M);
	freeMatrix(*M);
    
	*M = allocateMatrix(row,column);
	initializeMatrix(M,0);
    
	if (tmp.row > row)			minrow = row;
	else						minrow = tmp.row;
	
	if (tmp.column > column)	mincol = column;
	else						mincol = tmp.column;
	
	for (i = 0; i < minrow; i++)
		for (j = 0; j < mincol; j++)
			M->data[i][j] = tmp.data[i][j];
	freeMatrix(tmp);
}
void freeMatrix(MATRIX M){
	long i;
	
	if (M.data == NULL)
		return;
	
	for (i = 0; i < M.row; i++)
		free(M.data[i]);
	free(M.data);
}
void initializeMatrix(MATRIX *M, int c){
	long i, j;
	for (i = 0; i < M->row; i++)
		for (j = 0; j < M->column; j++)
			M->data[i][j] = c;
}
void copyMatrix(MATRIX *A, MATRIX B){
	long i, j;
	if (A->row != B.row || A->column != B.column){
		printf("\nError: Matrix dimensions mismatch in copy operation\n");
		exit(1);
	}
	for (i = 0; i < B.row; i++)
		for (j = 0; j < B.column; j++)
			A->data[i][j] = B.data[i][j];
}
void displayMatrix(MATRIX M){
    long i, j;
    
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			printf("%d ",M.data[i][j]);
		printf("\n");
	}
}
void writeMatrixIntoFile(MATRIX M, char *filename, int headerFlag){
    long i, j;
    FILE *id = fopen(filename,"w");
    
	if (headerFlag)
		fprintf(id,"%ld\t%ld\n",M.row,M.column);
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			fprintf(id,"%d ",M.data[i][j]);
		fprintf(id,"\n");
	}
	fclose(id);
}
int maxMatrixEntry(MATRIX M){
	int maxEntry;
	long i, j;
    
	maxEntry = M.data[0][0];
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			if (maxEntry < M.data[i][j])
				maxEntry = M.data[i][j];
	return maxEntry;
}
int minMatrixEntry(MATRIX M){
	int minEntry;
	long i, j;
    
	minEntry = M.data[0][0];
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			if (minEntry > M.data[i][j])
				minEntry = M.data[i][j];
	return minEntry;
}
/*******************************************************/
/**************** 2D matrix of doubles *****************/
/*******************************************************/
void reallocateMatrixD(MATRIXD *M, long row, long column){
	int i, j, minrow, mincol;
	MATRIXD tmp = allocateMatrixD(M->row,M->column);
    
	copyMatrixD(&tmp,*M);
	freeMatrixD(*M);
    
	*M = allocateMatrixD(row,column);
	initializeMatrixD(M,0.0);
    
	if (tmp.row > row)			minrow = row;
	else						minrow = tmp.row;
	
	if (tmp.column > column)	mincol = column;
	else						mincol = tmp.column;
	
	for (i = 0; i < minrow; i++)
		for (j = 0; j < mincol; j++)
			M->data[i][j] = tmp.data[i][j];
	freeMatrixD(tmp);
}
MATRIXD allocateMatrixD(long row, long column){
	MATRIXD M;
	long i;
    
	if (row <= 0 || column <= 0){
		M.row  = M.column = 0;
		M.data = NULL;
	}
	else {
		M.row = row;
		M.column = column;
		M.data = (double **) malloc(row * sizeof(double*));
		for (i = 0; i < row; i++)
			M.data[i] = (double *) malloc(column * sizeof(double));
	}
	return M;
}
void freeMatrixD(MATRIXD M){
	long i;
	
	if (M.data == NULL)
		return;
	
	for (i = 0; i < M.row; i++)
		free(M.data[i]);
	free(M.data);
}
void initializeMatrixD(MATRIXD *M, double c){
	long i, j;
	for (i = 0; i < M->row; i++)
		for (j = 0; j < M->column; j++)
			M->data[i][j] = c;
}
void copyMatrixD(MATRIXD *A, MATRIXD B){
	long i, j;
	if (A->row != B.row || A->column != B.column){
		printf("\nError: Matrix dimensions mismatch in copy operation\n");
		exit(1);
	}
	for (i = 0; i < B.row; i++)
		for (j = 0; j < B.column; j++)
			A->data[i][j] = B.data[i][j];
}
void displayMatrixD(MATRIXD M, int precision){
    long i, j;
	char temp[100];
    
	sprintf(temp,"%c.%d%c ",37,precision,'f');
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			printf(temp,M.data[i][j]);
		printf("\n");
	}
}
void writeMatrixDIntoFile(MATRIXD M, char *filename, int precision, int headerFlag){
    long i, j;
	char temp[100];
    FILE *id = fopen(filename,"w");
    
	if (headerFlag)
		fprintf(id,"%ld\t%ld\n",M.row,M.column);
	sprintf(temp,"%c.%d%c ",37,precision,'f');
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			fprintf(id,temp,M.data[i][j]);
		fprintf(id,"\n");
	}
	fclose(id);
}
double maxMatrixDEntry(MATRIXD M){
    double maxEntry = M.data[0][0];
	long i, j;
    
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			if (maxEntry < M.data[i][j])
				maxEntry = M.data[i][j];
	return maxEntry;
}
double minMatrixDEntry(MATRIXD M){
    double minEntry = M.data[0][0];
	long i, j;
    
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			if (minEntry > M.data[i][j])
				minEntry = M.data[i][j];
	return minEntry;
}
double maxMatrixDColumnEntry(MATRIXD M, long whichColumn){
	double maxEntry;
	long i, j;
    
	maxEntry = M.data[0][whichColumn];
	for (i = 0; i < M.row; i++)
		if (maxEntry < M.data[i][whichColumn])
			maxEntry = M.data[i][whichColumn];
	return maxEntry;
}
double minMatrixDColumnEntry(MATRIXD M, long whichColumn){
	double minEntry;
	long i, j;
    
	minEntry = M.data[0][whichColumn];
	for (i = 0; i < M.row; i++)
		if (minEntry > M.data[i][whichColumn])
			minEntry = M.data[i][whichColumn];
	return minEntry;
}
MATRIXD computeMeanMatrixD(MATRIXD A){
	long i, j;
	MATRIXD M = allocateMatrixD(1,A.column);
    
	initializeMatrixD(&M,0.0);
	for (i = 0; i < A.row; i++)
		for (j = 0; j < A.column; j++)
			M.data[0][j] += A.data[i][j];
	for (j = 0; j < M.column; j++)
        M.data[0][j] /= A.row;
	return M;
}
MATRIXD computeCovarianceMatrixD(MATRIXD A){
	long i, j, t;
	MATRIXD M = computeMeanMatrixD(A);
	MATRIXD S = allocateMatrixD(A.column,A.column);
    
	initializeMatrixD(&S,0.0);
	for (i = 0; i < S.row; i++)
		for (j = 0; j < S.column; j++)
			for (t = 0; t < A.row; t++)
				S.data[i][j] += (A.data[t][i] - M.data[0][i]) * (A.data[t][j] - M.data[0][j]);
	for (i = 0; i < S.row; i++)
		for (j = 0; j < S.column; j++)
			S.data[i][j] /= A.row - 1;
	freeMatrixD(M);
	return S;
}
MATRIXD computeCorrelationMatrixD(MATRIXD A){
	long i, j;
	MATRIXD S = computeCovarianceMatrixD(A);
	MATRIXD R = allocateMatrixD(S.row,S.column);
	for (i = 0; i < S.row; i++)
		for (j = 0; j < S.column; j++)
			if (fabs(sqrt(S.data[i][i]) * sqrt(S.data[j][j])) > ZERO)
				R.data[i][j] = S.data[i][j] / (sqrt(S.data[i][i]) * sqrt(S.data[j][j]));
			else
				R.data[i][j] = 0;
	freeMatrixD(S);
	return R;
}
MATRIXD multiplyMatrixD(MATRIXD A, MATRIXD B){
	MATRIXD result;
	long i, j, k;
    
	if (A.column != B.row){
		printf("\nError: Matrix dimensions do not match in matrix multiplication\n");
		exit(1);
	}
	result = allocateMatrixD(A.row,B.column);
	for (i = 0; i < A.row; i++)
		for (j = 0; j < B.column; j++){
			result.data[i][j] = 0.0;
			for (k = 0; k < A.column; k++)
				result.data[i][j] += A.data[i][k] * B.data[k][j];
		}
	return result;
}
/*******************************************************/
/**************** 3D matrix of integers ****************/
/*******************************************************/
MATRIX3 allocateMatrix3(long d1, long d2, long d3){
	MATRIX3 M;
	long i, j;
	
	if (d1 <= 0 || d2 <= 0 || d3 <= 0){
		M.d1 = M.d2 = M.d3 = 0;
		M.data = NULL;
	}
	else {
		M.d1 = d1;
		M.d2 = d2;
		M.d3 = d3;
		M.data = (int ***) calloc(d1, sizeof(int **));
		for (i = 0; i < d1; i++){
			M.data[i] = (int **) calloc(d2, sizeof(int *));
			for (j = 0; j < d2; j++)
				M.data[i][j] = (int *) calloc(d3, sizeof(int));
		}
	}
	return M;
}
void freeMatrix3(MATRIX3 M){
	long i, j;
	
	if (M.data == NULL)
		return;
	
	for (i = 0; i < M.d1; i++){
		for (j = 0; j < M.d2; j++)
			free(M.data[i][j]);
		free(M.data[i]);
	}
	free(M.data);
}
void initializeMatrix3(MATRIX3 *M, int c){
	long i, j, k;
	
	for (i = 0; i < M->d1; i++)
		for (j = 0; j < M->d2; j++)
			for (k = 0; k < M->d3; k++)
				M->data[i][j][k] = c;
}
void copyMatrix3(MATRIX3 *dest, MATRIX3 src){
    long i, j, k;
	
	for (i = 0; i < src.d1; i++)
		for (j = 0; j < src.d2; j++)
			for (k = 0; k < src.d3; k++)
				dest->data[i][j][k] = src.data[i][j][k];
}
/*******************************************************/
/**************** 3D matrix of doubles *****************/
/*******************************************************/
MATRIXD3 allocateMatrixD3(long d1, long d2, long d3){
	MATRIXD3 M;
	long i, j;
	
	if (d1 <= 0 || d2 <= 0 || d3 <= 0){
		M.d1 = M.d2 = M.d3 = 0;
		M.data = NULL;
	}
	else {
		M.d1 = d1;
		M.d2 = d2;
		M.d3 = d3;
		M.data = (double ***) calloc(d1, sizeof(double **));
		for (i = 0; i < d1; i++){
			M.data[i] = (double **) calloc(d2, sizeof(double *));
			for (j = 0; j < d2; j++)
				M.data[i][j] = (double *) calloc(d3, sizeof(double));
		}
	}
	return M;
}
void freeMatrixD3(MATRIXD3 M){
	long i, j;
	
	if (M.data == NULL)
		return;
	
	for (i = 0; i < M.d1; i++){
		for (j = 0; j < M.d2; j++)
			free(M.data[i][j]);
		free(M.data[i]);
	}
	free(M.data);
}
void initializeMatrixD3(MATRIXD3 *M, double c){
	long i, j, k;
	
	for (i = 0; i < M->d1; i++)
		for (j = 0; j < M->d2; j++)
			for (k = 0; k < M->d3; k++)
				M->data[i][j][k] = c;
}
/*******************************************************/
/*******************************************************/
/*******************************************************/
