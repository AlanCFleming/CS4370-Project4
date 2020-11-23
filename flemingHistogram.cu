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


