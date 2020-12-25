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


__global__ void matrixRandomBalance(double *a)
{
	unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;
	
	a[x] = a[x]*2.0 -1.0;
	
}

__global__ void set(double *a)
{
	unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;
	
	a[x] = 1.0;
	
}




__device__  __forceinline__ double  sigmoid (double a)
{
    return 1.0 / (1.0 + exp (-a));
	//return tanh(a);
}

__global__ void activation(double *a)
{
unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;
	
	a[x] = sigmoid(a[x]);
}







void ELM(double *training, double *train_lable, double *testing,double *test_lable, int NumberofHiddenNeurons, int NumberofTrainingData, int NumberofTestingData, int NumberofInputNeurons)
{

double *DtempH, *D ;

DtempH	= (double *)malloc(NumberofTrainingData	* 1	* sizeof(double));
D	= (double *)malloc(1	* NumberofTrainingData	* sizeof(double));

double *InputWeight, *BiasofHiddenNeurons, *ind, *BiasMatrix, *tempH , *H, *OutputWeight, *Y;
cudaMalloc((void**)&InputWeight, sizeof(double)*NumberofHiddenNeurons*NumberofInputNeurons) ;
cudaMalloc((void**)&BiasofHiddenNeurons, sizeof(double)*NumberofHiddenNeurons) ;
cudaMalloc((void**)&ind, sizeof(double)*NumberofTrainingData) ;
cudaMalloc((void**)&BiasMatrix, sizeof(double)*NumberofHiddenNeurons*NumberofTrainingData) ;
cudaMalloc((void**)&tempH, sizeof(double)*NumberofHiddenNeurons*NumberofTrainingData) ;
cudaMalloc((void**)&H, sizeof(double)*NumberofTrainingData*NumberofHiddenNeurons) ;
cudaMalloc((void**)&OutputWeight, sizeof(double)*NumberofTrainingData*NumberofHiddenNeurons) ;
cudaMalloc((void**)&Y, sizeof(double)*NumberofTrainingData) ;




RandomGPU(InputWeight,NumberofHiddenNeurons,NumberofInputNeurons);
matrixRandomBalance<<<NumberofHiddenNeurons,NumberofInputNeurons>>>(InputWeight);
RandomGPU(BiasofHiddenNeurons,NumberofHiddenNeurons,1);

set<<<1,NumberofTrainingData>>>(ind);

MatMul(BiasofHiddenNeurons,ind,BiasMatrix,NumberofHiddenNeurons , 1 , NumberofTrainingData);
cudaFree(ind);
MatMul(InputWeight,training,tempH,NumberofHiddenNeurons,NumberofInputNeurons,NumberofTrainingData);
cudaFree(training);
MatAdd(BiasMatrix,tempH,NumberofHiddenNeurons*NumberofTrainingData);
cudaFree(BiasMatrix);
activation<<<NumberofHiddenNeurons,NumberofTrainingData>>>(tempH);
Pinv(tempH,H,NumberofHiddenNeurons,NumberofTrainingData);
MatMul(train_lable,H,OutputWeight,1,NumberofTrainingData,NumberofHiddenNeurons);
cudaFree(H);
MatMul(OutputWeight,tempH,Y,1,NumberofHiddenNeurons,NumberofTrainingData);
cudaFree(tempH);

//=========================================================================================

cudaMemcpy(D	, Y		, 1 * NumberofTrainingData	* sizeof(double), cudaMemcpyDeviceToHost);



double m = D[0];
double mi = D[0];
for(int i = 0;i<NumberofTrainingData;i++){
m  = fmax(D[i],m);
mi  = fmin(D[i],mi);
}
for(int i = 0;i<NumberofTrainingData;i++){
D[i] =  (D[i]-mi)/(m-mi);
}


cudaMemcpy(DtempH	, train_lable		, NumberofTrainingData	 * 1	* sizeof(double), cudaMemcpyDeviceToHost);

double m1 = DtempH[0];
double mi1 = DtempH[0];
for(int i = 0;i<NumberofTrainingData;i++){
m1  = fmax(DtempH[i],m1);
mi1  = fmin(DtempH[i],mi1);
}
for(int i = 0;i<NumberofTrainingData;i++){
DtempH[i] =  (DtempH[i]-mi1)/(m1-mi1);
}
double miss = 0;
for(int i = 0;i<NumberofTrainingData;i++){
if ((DtempH[i]-D[i])>0.5){miss++;}
}

double Acc;
Acc = (NumberofTrainingData-miss)/NumberofTrainingData;

 
printf("Training Accuracy is : %lf \n",Acc);

//=====================================================================================================================
double *tempH_test,*H_test,*Y_test;
cudaMalloc((void**)&tempH_test, sizeof(double)*NumberofHiddenNeurons*NumberofTestingData) ;
cudaMalloc((void**)&ind, sizeof(double)*NumberofTestingData) ;
cudaMalloc((void**)&BiasMatrix, sizeof(double)*NumberofHiddenNeurons*NumberofTestingData) ;
cudaMalloc((void**)&H_test, sizeof(double)*NumberofTestingData*NumberofHiddenNeurons) ;
cudaMalloc((void**)&Y_test, sizeof(double)*NumberofTestingData) ;

set<<<1,NumberofTestingData>>>(ind);

MatMul(BiasofHiddenNeurons,ind,BiasMatrix,NumberofHiddenNeurons , 1 , NumberofTestingData);
cudaFree(ind);
cudaFree(BiasofHiddenNeurons);
MatMul(InputWeight,testing,tempH_test,NumberofHiddenNeurons,NumberofInputNeurons,NumberofTestingData);
cudaFree(testing);
cudaFree(InputWeight);
MatAdd(BiasMatrix,tempH_test,NumberofHiddenNeurons*NumberofTestingData);
cudaFree(BiasMatrix);
activation<<<NumberofHiddenNeurons,NumberofTestingData>>>(tempH_test);
MatMul(OutputWeight,tempH_test,Y_test,1,NumberofHiddenNeurons,NumberofTestingData);
cudaFree(tempH_test);

double *DtempH_test, *D_test ;

DtempH_test	= (double *)malloc(NumberofTestingData	* 1	* sizeof(double));
D_test	= (double *)malloc(1	* NumberofTestingData	* sizeof(double));

//=========================================================================================

cudaMemcpy(D_test	, Y_test		, 1 * NumberofTestingData	* sizeof(double), cudaMemcpyDeviceToHost);



double m_test = D_test[0];
double mi_test = D_test[0];
for(int i = 0;i<NumberofTestingData;i++){
m_test  = fmax(D[i],m_test);
mi_test  = fmin(D[i],mi_test);
}
for(int i = 0;i<NumberofTestingData;i++){
D_test[i] =  (D_test[i]-mi_test)/(m_test-mi_test);
}


cudaMemcpy(DtempH_test	, test_lable		, NumberofTestingData	 * 1	* sizeof(double), cudaMemcpyDeviceToHost);

double m1_test = DtempH_test[0];
double mi1_test = DtempH_test[0];
for(int i = 0;i<NumberofTestingData;i++){
m1_test  = fmax(DtempH_test[i],m1_test);
mi1_test  = fmin(DtempH_test[i],mi1_test);
}
for(int i = 0;i<NumberofTestingData;i++){
DtempH_test[i] =  (DtempH_test[i]-mi1_test)/(m1_test-mi1_test);
}
double miss_test = 0;
for(int i = 0;i<NumberofTestingData;i++){
if ((DtempH_test[i]-D_test[i])>0.5){miss_test++;}
}

double TAcc;
TAcc = (NumberofTestingData-miss_test)/NumberofTestingData;

 
printf("Testing Accuracy is : %lf \n",TAcc);




}






	
int main(void)
{
double *training;
double *t;
double *Mat;
double *Res;

double *testing;
double *t_test;
double *Mat_test;
double *Res_test;



Mat	= (double *)malloc(100	* 4	* sizeof(double));
Res	= (double *)malloc(1	* 100	* sizeof(double));
Mat_test	= (double *)malloc(50	* 4	* sizeof(double));
Res_test	= (double *)malloc(1	* 50	* sizeof(double));

cudaMalloc((void**)&training, sizeof(double)*100*4) ;
cudaMalloc((void**)&t, sizeof(double)*1*100) ;
cudaMalloc((void**)&testing, sizeof(double)*50*4) ;
cudaMalloc((void**)&t_test, sizeof(double)*1*50) ;


ReadCSV(Mat,"train_X.csv");
ReadCSV(Res,"train_Y.csv");
ReadCSV(Mat_test,"test_X.csv");
ReadCSV(Res_test,"test_Y.csv");


cudaMemcpy(training	, Mat		, 100	 * 4* sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(t	, Res		, 100	 * 1* sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(testing	, Mat_test		, 50	 * 4* sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(t_test	, Res_test		, 50	 * 1* sizeof(double), cudaMemcpyHostToDevice);


ELM(training,t,testing,t_test,1000,100,50,4);



}
	
	
	
	
	
	
	