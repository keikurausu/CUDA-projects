// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... +
// lst[n-1]}

#include <wb.h>
#define THREADS 256
#define BLOCK_SIZE 256 //@@ You can change this

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

/*performs the first order list scan*/
__global__ void scan(float *input, float *output, int len, float *intermediate) {
	//@@ Modify the body of this function to complete the functionality of
	//@@ the scan on the device
	//@@ You may need multiple kernel calls; write your kernels before this
	//@@ function and call them from here
	__shared__ float partialSum[2 * BLOCK_SIZE];
	int tx = threadIdx.x;
	int start = 2 * blockIdx.x * blockDim.x;

	//@@ Load a segment of the input vector into shared memory
	//each thread loads 2 elements into shared memory. Put the identity (0 for sum) in if outside of boundary
	if (start + tx >= len && start + tx + BLOCK_SIZE >= len)
	{
		partialSum[tx] = 0;
		partialSum[BLOCK_SIZE + tx] = 0;
	}
	else if (start + tx + BLOCK_SIZE >= len)
	{
		partialSum[tx] = input[start + tx];
		partialSum[BLOCK_SIZE + tx] = 0;
	}
	else
	{
		partialSum[tx] = input[start + tx];
		partialSum[BLOCK_SIZE + tx] = input[start + BLOCK_SIZE + tx];
	}
	__syncthreads();

	/*reduction phase*/
	for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
	{
		int index = (tx + 1)*stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
		{
			partialSum[index] += partialSum[index - stride];
		}
		__syncthreads();
	}

	/*post reduction reverse phase*/
	for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
	{
		__syncthreads();
		int index = (tx + 1)*stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE)
		{
			partialSum[index + stride] += partialSum[index];
		}
	}
	__syncthreads();

	if (blockIdx.x*blockDim.x * 2 + tx < len)
	{
		output[blockIdx.x*blockDim.x * 2 + tx] = partialSum[tx];
		if (blockIdx.x*blockDim.x * 2 + BLOCK_SIZE + tx < len)
		{
			output[blockIdx.x*blockDim.x * 2 + BLOCK_SIZE + tx] = partialSum[tx + BLOCK_SIZE];
			if (tx == BLOCK_SIZE - 1)
			{
				/*put total of this scan in intermediate at corresponding index*/
				intermediate[blockIdx.x] = partialSum[tx + BLOCK_SIZE];
				//printf("%d        ", (int)intermediate[blockIdx.x]);
			}
		}
	}


}
/*performs the second order list scan*/
__global__ void scan2(float *input, float *output) {
	//@@ Modify the body of this function to complete the functionality of
	//@@ the scan on the device
	//@@ You may need multiple kernel calls; write your kernels before this
	//@@ function and call them from here
	__shared__ float partialSum[2 * BLOCK_SIZE];
	int tx = threadIdx.x;

	//@@ Load a segment of the input vector into shared memory
	//each thread loads 2 elements into shared memory.
	partialSum[tx] = input[tx];
	partialSum[BLOCK_SIZE + tx] = input[BLOCK_SIZE + tx];
	__syncthreads();

	/*reduction phase*/
	for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
	{
		int index = (tx + 1)*stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
		{
			partialSum[index] += partialSum[index - stride];
		}
		__syncthreads();
	}

	/*post reduction reverse phase*/
	for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
	{
		__syncthreads();
		int index = (tx + 1)*stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE)
		{
			partialSum[index + stride] += partialSum[index];
		}
	}
	__syncthreads();

	output[tx] = partialSum[tx];
	output[BLOCK_SIZE + tx] = partialSum[BLOCK_SIZE + tx];

}

/*add together input and output and put result in output*/
__global__ void add_scans(float *input, float *output, int len)
{
	int tx = threadIdx.x;
	int start = 2 * blockIdx.x * blockDim.x;
	if (blockIdx.x > 0 && start + tx < len)
	{
		if (start + tx < len)
		{
			output[start + tx] += input[blockIdx.x-1];
			//printf("%d ", (int)input[blockIdx.x-1]);
		}
		if (start + tx + BLOCK_SIZE < len)
		{
			output[start + tx + BLOCK_SIZE] += input[blockIdx.x-1];
			//printf("%d ", (int)input[blockIdx.x-1]);
		}
	}
}

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostInput;  // The input 1D list
	float *hostOutput; // The output list
	float *Intermediate;
	float *deviceInput;
	float *deviceOutput;
	float *IntermediateOutput;
	int numElements; // number of elements in the list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
	hostOutput = (float *)malloc(numElements * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ", numElements);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
	wbCheck(cudaMalloc((void **)&Intermediate, 2 * BLOCK_SIZE * sizeof(float)));
	wbCheck(cudaMalloc((void **)&IntermediateOutput, 2 * BLOCK_SIZE * sizeof(float)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Clearing output memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
	wbCheck(cudaMemset(Intermediate, 0, 2 * BLOCK_SIZE * sizeof(float)));
	wbCheck(cudaMemset(IntermediateOutput, 0, 2 * BLOCK_SIZE * sizeof(float)));
	wbTime_stop(GPU, "Clearing output memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
		cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here
	dim3 grid((numElements) / 2 * BLOCK_SIZE, 1, 1);
	dim3 threads(THREADS, 1, 1);

	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Modify this to complete the functionality of the scan
	//@@ on the deivce
	scan << <grid, threads >> >(deviceInput, deviceOutput, numElements, Intermediate);
	cudaDeviceSynchronize();
	scan2 << <1, threads >> >(Intermediate, IntermediateOutput); //only need one block here.
	cudaDeviceSynchronize();
	add_scans << <grid, threads >> >(IntermediateOutput, deviceOutput, numElements); //add the 2 arrays returned by the previous 2 kernels.

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));

	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	//cudaFree(deviceInput);
	//cudaFree(deviceOutput);
	//cudaFree(Intermediate);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, numElements);

	//free(hostInput);
	free(hostOutput);

	return 0;
}
