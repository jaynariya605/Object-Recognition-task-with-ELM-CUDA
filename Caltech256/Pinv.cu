#include <curand.h>
#include <conio.h>
#include <iostream>
#include <cublas_v2.h>
#include "Pinv.h"



// using cublas matrix multiplication
 void ATA(cublasHandle_t &handle,const double *A, double *C ,const int m, const int k) {
    

	
	int lda=m,ldb=m,ldc=k;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	 //create cuda stream
	
	cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, k, m, alpha, A, lda, A, ldb, beta, C, ldc);

 }
 
 
  void AATA(cublasHandle_t &handle,const double *A,const double *B, double *C ,const int k, const int m) {
    

	
	int lda=k,ldb=m,ldc=k;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	 //create cuda stream
	
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, m, k, alpha, A, lda, B, ldb, beta, C, ldc);

 }

void gpu_inverse(cublasHandle_t &handle, double *A,double *C,const int m){//Reference :https://stackoverflow.com/questions/37731103/cublas-matrix-inverse-much-slower-than-matlab
double** adL;
double** adC;
double* dL;
double* dC;
int* dLUPivots;
int* dLUInfo;

cudaMalloc(&adL, sizeof(double*));
cudaMalloc(&adC, sizeof(double*));
cudaMalloc(&dL,  m * m * sizeof(double));
cudaMalloc(&dC,  m * m * sizeof(double));
cudaMalloc(&dLUPivots, m * sizeof(int));
cudaMalloc(&dLUInfo, sizeof(int));
cudaMemcpy(dL, A, m * m * sizeof(double), cudaMemcpyDeviceToDevice);
cudaMemcpy(adL, &dL, sizeof(double*), cudaMemcpyHostToDevice);
cudaMemcpy(adC, &dC, sizeof(double*), cudaMemcpyHostToDevice);

cublasDgetrfBatched(handle, m, adL, m, dLUPivots, dLUInfo, 1);
cudaDeviceSynchronize();
cublasDgetriBatched(handle, m, (const double **)adL, m, dLUPivots, adC, m, dLUInfo, 1);
cudaDeviceSynchronize();
cudaMemcpy(C, dC, m * m * sizeof(double), cudaMemcpyDeviceToDevice);
cudaFree(adC);
cudaFree(adL);
cudaFree(dC);
cudaFree(dL);
cudaFree(dLUInfo);
cudaFree(dLUPivots);
}


void Pinv(double *d_A, double *d_B,const int m , const int k){


double *d_C;
int CSize = k*k;
cudaMalloc((void**)&d_C, sizeof(double)*CSize) ;
cublasHandle_t handle;
cublasCreate(&handle);
	ATA(handle,d_A, d_C, m, k);
	
	gpu_inverse(handle,d_C,d_C,  k);
	
	AATA(handle,d_C, d_A,d_B, k, m);
	
	cudaFree(d_A);
	cudaFree(d_C);
	cublasDestroy(handle);
	
	}
	





	
	
	
	
	
	
	