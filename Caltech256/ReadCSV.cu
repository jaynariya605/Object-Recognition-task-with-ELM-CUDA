#include <curand.h>
#include <conio.h>
#include <iostream>
#include <cublas_v2.h>
#include "ReadCSV.h"

void ReadCSV(double *a, const char *fname)
{
	FILE *f;

	int n=0;

	f = fopen(fname, "r");
	if (f == NULL) {
		printf("Failed to open file\n");
	}

	while (fscanf(f, "%lf", &a[n++]) == 1) {
		fscanf(f, ",");
	}
	
	fclose(f);
}


	
	
	
	