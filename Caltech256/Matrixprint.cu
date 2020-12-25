#include <curand.h>
#include <conio.h>
#include <iostream>
#include <cublas_v2.h>
#include "MatAdd.h"
#include "Matrixprint.h"

void Matrixprint(double *a, int m, int n)
{
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			printf("%f ",a[i+j*m]);
		}
		printf("\n");
	}
}