#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
//Code written by Alan Fleming

//CONSTANTS
#define MATRIXSIZE 2048
#define BLOCKSIZE 1024

void cpuHistogram(int* input, int* histogram, int size) {
	for(int i = 0; i < size; i++) {
		histrogram[input[i]]++;
	}
}

__global__ void histogram(int* input, int* histogram, int size) {
	//get starting index for thread
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	//calculate stride
	int stride = blockDim.x * gridDim.x;
	//preform histogram calculation
	while( i < size) {
		atomicAdd( &(histogram[input[i]], 1));
		i += stride;
	}
}	


