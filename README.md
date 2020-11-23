# CS4370-Project4
This is a cuda program that covers histograms for class.

## Editing BLOCKSIZE and MATRIXSIZE
* A define statement for MATRIXSIZE can be found on line 8 of the .cu file
* A define statement for BLOCKSIZE can be found on line 9 of the .cu file


## Compiling
nvcc was used to compile these programs. This will create an executable program.
* Command for compiling sum reduction: nvcc flemingHistogram.cu -o histogram

## Running
These programs can be run directly from the command line.
* Command for parallel sum reduction: {path}/histogram
