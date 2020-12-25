#include <curand.h>
#include <conio.h>
#include <iostream>
#include <cublas_v2.h>
#include "MatAdd.h"

void MatAdd(const double *A, double *C ,const int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
	
	
	
	const double alf = 1;
	const double *alpha = &alf;
	
	cublasDaxpy(handle, n,  alpha, A, 1,  C, 1);
	cublasDestroy(handle);
	}
	
	

