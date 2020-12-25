#include <curand.h>
#include <conio.h>
#include <iostream>
#include <cublas_v2.h>
#include "Pinv.h"
#include "RandomGPU.h"
#include "ReadCSV.h"
#include "MatMul.h"
#include "MatAdd.h"
#include "Matrixprint.h"



int main(void)
{
float *training;
float *Res;
float *Htraining;
float *HRes;


Htraining	= (float *)malloc(5	* 4	* sizeof(float));
HRes	= (float *)malloc(4	* 5	* sizeof(float));
cudaMalloc((void**)&training, sizeof(float)*5*4) ;
cudaMalloc((void**)&Res, sizeof(float)*4*5) ;

RandomGPU(training,5,4);
cudaMemcpy(Htraining	, training		, 5	 * 4	* sizeof(float), cudaMemcpyDeviceToHost);
Pinv(training,Res,5,4);

cudaMemcpy(HRes, Res		, 5	 * 4	* sizeof(float), cudaMemcpyDeviceToHost);

Matrixprint(Htraining,5,4);
Matrixprint(HRes , 4,5);


}
	
	
	
	
	
	
	
	