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
	int *dev_input, *dev_basicGPUResult, *dev_sharedGPUResult;
	cudaMalloc((void **)(&dev_input), MATRIXSIZE *sizeof(int));
	cudaMalloc((void **)(&dev_basicGPUResult), 256 *sizeof(int));
	cudaMalloc((void **)(&dev_sharedGPUResult), 256 *sizeof(int));

	//copy memory to gpu
	cudaMemcpy(dev_input, input, MATRIXSIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_basicGPUResult, basicGPUResult, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sharedGPUResult, sharedGPUResult, 256 * sizeof(int), cudaMemcpyHostToDevice);

	//calculate dimentions for gpu
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(ceil(double(MATRIXSIZE)/dimBlock.x));

	//~~WITHOUT SHARED MEMORY~~
	//Set up cuda events for recording runtime
	cudaEvent_t start,stop;
	float basicGPUTime; 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
					
	//calculate histogram without shared memory
	histogram<<<dimGrid, dimBlock>>>(dev_input, dev_basicGPUResult, MATRIXSIZE);
						
	//calculate runtime 
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&basicGPUTime,start,stop);

	//destroy cuda events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copy sum from gpu
	cudaMemcpy(basicGPUResult, dev_basicGPUResult, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	//print speedup
	printf("--WITHOUT SHARED MEMORY--\nCPU Runtime: %f\nGpu Runtime: %f\nSpeedup: %f\n", (double)cpuTime, (double)basicGPUTime, double(cpuTime / basicGPUTime));

	//verify results
	bool valid = true;
	for(int i = 0; i < MATRIXSIZE; i++) {	
		if(cpuResult[i] != basicGPUResult[i]) {
			valid = false;
			break;
		}
	}
	if(valid) {
		printf("TEST PASSED\n");
	} else {
		printf("TEST FAILED\n");
	}

	//~~WITH SHARED MEMORY~~
	//Set up cuda events for recording runtime
	cudaEvent_t start,stop;
	float sharedGPUTime; 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
					
	//calculate histogram with shared memory
	histogram<<<dimGrid, dimBlock>>>(dev_input, dev_sharedGPUResult, MATRIXSIZE);
						
	//calculate runtime 
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&sharedGPUTime,start,stop);

	//destroy cuda events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copy sum from gpu
	cudaMemcpy(sharedGPUResult, dev_sharedGPUResult, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	//print speedup
	printf("--WITH SHARED MEMORY--\nCPU Runtime: %f\nGpu Runtime: %f\nSpeedup: %f\n", (double)cpuTime, (double)sharedGPUTime, double(cpuTime / sharedGPUTime));

	//verify results
	bool valid = true;
	for(int i = 0; i < MATRIXSIZE; i++) {	
		if(cpuResult[i] != sharedGPUResult[i]) {
			valid = false;
			break;
		}
	}
	if(valid) {
		printf("TEST PASSED\n");
	} else {
		printf("TEST FAILED\n");
	}
	

	//free up memory before returning
	free(input);
	free(cpuResult);
	free(basicGPUResult);
	free(sharedGPUResult);
	cudaFree(dev_input);
	cudaFree(dev_basicGPUResult);
	cudaFree(dev_sharedGPUResult);

	return 0;
}
