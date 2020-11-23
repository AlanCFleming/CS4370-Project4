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

//currently does not work for block sizes smaller than 256
__global__ void sharedHistogram(int* input, int* histogram, int size) {
	//initialize shared memory for the block
	__shared__ int privateHistogram[256];

	if(threadIdx.x < 256) privateHistogram[threadIdx.x] = 0;
	__syncthreads();

	//get starting index for thread
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	//calculate stride
	int stride = blockDim.x * gridDim.x;
	//preform histogram calculation
	while( i < size) {
		atomicAdd( &(privateHistogram[input[i]], 1));
		i += stride;
	}

	//ensure all threads have finished their additions
	__syncthreads();

	//add private histogram to public histogram
	if(threadIdx.x < 256) {
		atomixAdd( &(histogram[threadIdx.x]), privateHistogram[threadIdx.x]);
	}
}
