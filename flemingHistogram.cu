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


int main() {
	int *input = (int *)malloc(sizeof(int) * MATRIXSIZE); //allocate space for array
	int *cpuResult = (int *)malloc(sizeof(int) * 256); //allocate space for cpu output array
	int *basicGPUResult = (int *)malloc(sizeof(int) * 256); //allocate space for gpu output array using global memory
	int *sharedGPUResult = (int *)malloc(sizeof(int) * 256); //allocate space for gpu output array using shared memory

	//intialize the input array
	int init = 1325;
	for(int i=0; i < MATRIXSIZE; i++){
		init= 3125 * init % 65537;
		input[i]= init % 256;
	}
	//clear the output arrays to ensure proper adding
	for(int i = 0; i < 256; i++) {
		cpuResult[i] = 0;
		basicGPUResult[i] = 0;
		sharedGPUResult[i] = 0;
	}

	//Test CPU
	//Get start time
	clock_t t1 = clock();
	//Calculate reduction
	
	cpuHistogram(input, cpuResult, MATRIXSIZE);
	
	//Get stop time
	clock_t t2 = clock();
	//Calculate runtime
	float cpuTime= (float(t2-t1)/CLOCKS_PER_SEC*1000);

	//Allocate memory on GPU compution. dev_b is used to store the results of the first pass of reduction
	int *dev_input, *dev_basicGPU, *dev_sharedGPU;
	cudaMalloc((void **)(&dev_input), MATRIXSIZE *sizeof(int));
	cudaMalloc((void **)(&dev_basicGPU), 256 *sizeof(int));
	cudaMalloc((void **)(&dev_sharedGPU), 256 *sizeof(int));

	//copy memory to gpu
	cudaMemcpy(dev_input, input, MATRIXSIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_basicGPU, basicGPUResult, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sharedGPU, sharedGPUResult, 256 * sizeof(int), cudaMemcpyHostToDevice);

	//calculate dimentions for gpu
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(ceil(double(MATRIXSIZE)/dimBlock.x));

	return 0;
}
