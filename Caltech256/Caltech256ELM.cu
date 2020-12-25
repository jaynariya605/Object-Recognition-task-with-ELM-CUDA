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


__global__ void matrixRandomBalance(double *s)
{
	unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;
	
	s[x] = s[x]*2.0 -1.0;
	
}

__global__ void set(double *s)
{
	unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;
	
	s[x] = 1.0;
	
}




__device__  __forceinline__ double  sigmoid (double s)
{
    return 1.0 / (1.0 + exp (-s));
	//return tanh(s);
}

__global__ void activation(double *s)
{
unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;
	
	s[x] = sigmoid(s[x]);
}







void ELM(double *training, double *train_lable, double *testing,double *test_lable, int NumberofHiddenNeurons, int NumberofTrainingData, int NumberofTestingData, int NumberofInputNeurons)
{

double *DtempH, *D ;

DtempH	= (double *)malloc(NumberofTrainingData	* 1	* sizeof(double));
D	= (double *)malloc(1	* NumberofTrainingData	* sizeof(double));

double *InputWeight, *BiasofHiddenNeurons, *ind, *BiasMatrix, *tempH , *H, *OutputWeight, *Y;// Variable declaration and memory allocation
cudaMalloc((void**)&InputWeight, sizeof(double)*NumberofHiddenNeurons*NumberofInputNeurons) ;
cudaMalloc((void**)&BiasofHiddenNeurons, sizeof(double)*NumberofHiddenNeurons) ;
cudaMalloc((void**)&ind, sizeof(double)*NumberofTrainingData) ;
cudaMalloc((void**)&BiasMatrix, sizeof(double)*NumberofHiddenNeurons*NumberofTrainingData) ;
cudaMalloc((void**)&tempH, sizeof(double)*NumberofHiddenNeurons*NumberofTrainingData) ;
cudaMalloc((void**)&H, sizeof(double)*NumberofTrainingData*NumberofHiddenNeurons) ;
cudaMalloc((void**)&OutputWeight, sizeof(double)*NumberofTrainingData*NumberofHiddenNeurons) ;
cudaMalloc((void**)&Y, sizeof(double)*NumberofTrainingData) ;




RandomGPU(InputWeight,NumberofHiddenNeurons,NumberofInputNeurons);
matrixRandomBalance<<<NumberofInputNeurons,NumberofHiddenNeurons>>>(InputWeight); // InputWeight = randome(NumberofHiddenNeurons,NumberofInputNeurons)*2-1
RandomGPU(BiasofHiddenNeurons,NumberofHiddenNeurons,1);// BiasofHiddenNeurons = randome(NumberofHiddenNeurons,1)

set<<<NumberofTrainingData,1>>>(ind); //ind = ones(1,NumberofTrainingData)

MatMul(BiasofHiddenNeurons,ind,BiasMatrix,NumberofHiddenNeurons , 1 , NumberofTrainingData); //BiasMatrix = BiasofHiddenNeurons(:ind);
cudaFree(ind);
MatMul(InputWeight,training,tempH,NumberofHiddenNeurons,NumberofInputNeurons,NumberofTrainingData); // tempH = InputWeight* training
cudaFree(training);
MatAdd(BiasMatrix,tempH,NumberofHiddenNeurons*NumberofTrainingData); // tempH = tempH + BiasMatrix
cudaFree(BiasMatrix);
activation<<<NumberofTrainingData,NumberofHiddenNeurons>>>(tempH); // tempH = 1/1+exp(-tempH)
Pinv(tempH,H,NumberofHiddenNeurons,NumberofTrainingData);  //H = Pinv(tempH)
MatMul(train_lable,H,OutputWeight,1,NumberofTrainingData,NumberofHiddenNeurons); //OutputWeight = Pinv(tempH) * train_lable
cudaFree(H);
MatMul(OutputWeight,tempH,Y,1,NumberofHiddenNeurons,NumberofTrainingData); // Y = OutputWeight* tempH
cudaFree(tempH);

//==============================Prediction and Accuracy calculation===========================================

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

//========================================Testing======================================================
double *tempH_test,*Y_test;
cudaMalloc((void**)&tempH_test, sizeof(double)*NumberofHiddenNeurons*NumberofTestingData) ;
MatMul(InputWeight,testing,tempH_test,NumberofHiddenNeurons,NumberofInputNeurons,NumberofTestingData); // tempH_test = InputWeight*testing
cudaFree(testing);
cudaFree(InputWeight);



cudaMalloc((void**)&ind, sizeof(double)*NumberofTestingData) ; 
cudaMalloc((void**)&BiasMatrix, sizeof(double)*NumberofHiddenNeurons*NumberofTestingData) ;



set<<<NumberofTestingData,1>>>(ind);  // ind = ones(1,NumberofTestingData)


MatMul(BiasofHiddenNeurons,ind,BiasMatrix,NumberofHiddenNeurons , 1 , NumberofTestingData);  // BiasMatrix = BiasofHiddenNeurons(:ind)

cudaFree(ind);
cudaFree(BiasofHiddenNeurons);

MatAdd(BiasMatrix,tempH_test,NumberofHiddenNeurons*NumberofTestingData); // tempH_test = tempH_test + BiasMatrix
cudaFree(BiasMatrix);
activation<<<NumberofTestingData,NumberofHiddenNeurons>>>(tempH_test); // tempH_test = 1/1+exp(-tempH_test)
cudaMalloc((void**)&Y_test, sizeof(double)*NumberofTestingData) ; 
MatMul(OutputWeight,tempH_test,Y_test,1,NumberofHiddenNeurons,NumberofTestingData); // Y_test = tempH_test* OutputWeight
cudaFree(tempH_test);

double *DtempH_test, *D_test ;

DtempH_test	= (double *)malloc(NumberofTestingData	* 1	* sizeof(double));
D_test	= (double *)malloc(1	* NumberofTestingData	* sizeof(double));

//=============================Testing Accuracy===========================================

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



Mat	= (double *)malloc(7710	* 5048	* sizeof(double));
Res	= (double *)malloc(1	* 7710	* sizeof(double));
Mat_test	= (double *)malloc(22897	* 5048	* sizeof(double));
Res_test	= (double *)malloc(1	* 22897	* sizeof(double));

cudaMalloc((void**)&training, sizeof(double)*7710*5048) ;
cudaMalloc((void**)&t, sizeof(double)*1*7710) ;
cudaMalloc((void**)&testing, sizeof(double)*22897*5048) ;
cudaMalloc((void**)&t_test, sizeof(double)*1*22897) ;


ReadCSV(Mat,"Cnew_F_train.csv");
ReadCSV(Res,"Ctrain_labels.csv");
ReadCSV(Mat_test,"Cnew_F_test.csv");
ReadCSV(Res_test,"Ctest_labels.csv");


cudaMemcpy(training	, Mat		, 7710	 * 5048* sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(t	, Res		, 7710	 * 1* sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(testing	, Mat_test		, 22897	 * 5048* sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(t_test	, Res_test		, 22897	 * 1* sizeof(double), cudaMemcpyHostToDevice);


ELM(training,t,testing,t_test,10000,7710,22897,5048);



}
	
	
	
	
	
	
	