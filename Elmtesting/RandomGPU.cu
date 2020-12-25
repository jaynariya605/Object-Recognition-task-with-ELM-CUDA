#include <curand.h>
#include <conio.h>
#include <iostream>
#include <cublas_v2.h>
#include "RandomGPU.h"


// Reference : https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void RandomGPU(double *A, int nr_rows_A, int nr_cols_A)
{
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniformDouble(prng, A, nr_rows_A * nr_cols_A);
}