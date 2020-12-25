#include <curand.h>
#include <conio.h>
#include <iostream>
#include <cublas_v2.h>
#include "MatMul.h"

void MatMul(const double *A, const double *B, double *C ,const int m, const int k, const int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
	
	
	int lda=m,ldb=k,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	cublasDestroy(handle);
	}
	
	
