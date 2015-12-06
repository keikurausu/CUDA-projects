#include    <wb.h>

#define wbCheck(stmt) do {                                         \
        cudaError_t err = stmt;                                    \
        if (err != cudaSuccess) {                                  \
            wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err)); \
            wbLog(ERROR, "Failed to run stmt ", #stmt);            \
            return -1;                                             \
        }                                                          \
    } while(0)


//@@ Define any useful program-wide constants here
#define KERNEL_SIZE 3
#define KERNEL_RADIUS 1
#define TILE_SIZE 8
#define THREADS TILE_SIZE + KERNEL_SIZE - 1

//@@ Define constant memory for device kernel here
__constant__ float Mc[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];

__global__ void conv3d(float *A, float *B,
	const int z_size, const int y_size, const int x_size) {
	//@@ Insert kernel code here
	//set up shared memory
	__shared__ float ds_Input[THREADS][THREADS][THREADS];

	//set up variable for block and thread indexes
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	//set up output index   since both input and output are same size, there is no need for an input index variable
	int y_o = by*TILE_SIZE + ty - KERNEL_RADIUS;
	int x_o = bx*TILE_SIZE + tx - KERNEL_RADIUS;
	int z_o = bz*TILE_SIZE + tz - KERNEL_RADIUS;
	float output = 0.0;
	//read in input to shared memory
	if ((x_o >= 0) && (x_o < x_size) && (y_o >= 0) && (y_o < y_size) && (z_o >= 0) && (z_o < z_size))
	{
		ds_Input[tz][ty][tx] = A[z_o*y_size*x_size + y_o*x_size + x_o];
	}
	else
	{
		ds_Input[tz][ty][tx] = 0.0;
	}
	__syncthreads();
	x_o++;
	y_o++;
	z_o++;
	//perform the computations
	if (tx < TILE_SIZE && ty < TILE_SIZE && tz < TILE_SIZE)
	{
		for (int i = 0; i < KERNEL_SIZE; i++)
		{
			for (int j = 0; j < KERNEL_SIZE; j++)
			{
				for (int k = 0; k < KERNEL_SIZE; k++)
				{
					if ((x_o >= 0) && (x_o < x_size) && (y_o >= 0) && (y_o < y_size) && (z_o >= 0) && (z_o < z_size))
					{
						output += Mc[i][j][k] * ds_Input[i + tz][j + ty][k + tx];
					}
				}
			}
		}
		//write to output
		if(z_o<z_size && y_o < y_size && x_o < x_size)
			B[(z_o)*y_size*x_size + (y_o)*x_size + x_o] = output;
	}
	__syncthreads();
}







int main(int argc, char* argv[]) {
	wbArg_t args;
	int z_size;
	int y_size;
	int x_size;
	int inputLength, kernelLength;
	float * hostInput;
	float * hostKernel;
	float * hostOutput;
	float * deviceInput;
	float * deviceOutput;

	args = wbArg_read(argc, argv);

	// Import data
	hostInput = (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostKernel = (float*)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
	hostOutput = (float*)malloc(inputLength * sizeof(float));

	// First three elements are the input dimensions  
	z_size = hostInput[0];
	y_size = hostInput[1];
	x_size = hostInput[2];
	wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
	assert(z_size * y_size * x_size == inputLength - 3);
	assert(kernelLength == 27);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//@@ Allocate GPU memory here
	// Recall that inputLength is 3 elements longer than the input data
	// because the first  three elements were the dimensions
	wbCheck(cudaMalloc(&deviceInput, z_size*y_size*x_size*sizeof(float)));
	wbCheck(cudaMalloc(&deviceOutput, z_size*y_size*x_size*sizeof(float)));
	wbTime_stop(GPU, "Doing GPU memory allocation");


	wbTime_start(Copy, "Copying data to the GPU");
	//@@ Copy input and kernel to GPU here
	// Recall that the first three elements of hostInput are dimensions and do
	// not need to be copied to the gpu
	wbCheck(cudaMemcpy(deviceInput, &hostInput[3], z_size*y_size*x_size*sizeof(float), cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpyToSymbol(Mc, hostKernel, KERNEL_SIZE*KERNEL_SIZE*KERNEL_SIZE*sizeof(float)));
	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ Initialize grid and block dimensions here
	dim3 grid((x_size - 1) / TILE_SIZE + 1, (y_size - 1) / TILE_SIZE + 1, (z_size - 1) / TILE_SIZE + 1);
	dim3 threads(THREADS, THREADS, THREADS);
	//@@ Launch the GPU kernel here
	conv3d << <grid, threads >> >(deviceInput, deviceOutput, z_size, y_size, x_size);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");


	wbTime_start(Copy, "Copying data from the GPU");
	//@@ Copy the device memory back to the host here
	// Recall that the first three elements of the output are the dimensions
	// and should not be set here (they are set below)
	wbCheck(cudaMemcpy(&hostOutput[3], deviceOutput, x_size*y_size*z_size*sizeof(float), cudaMemcpyDeviceToHost));
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	// Set the output dimensions for correctness checking
	hostOutput[0] = z_size;
	hostOutput[1] = y_size;
	hostOutput[2] = x_size;
	wbSolution(args, hostOutput, inputLength);
	// Free device memory
	cudaFree(deviceInput);
	cudaFree(deviceOutput);

	// Free host memory
	free(hostInput);
	free(hostOutput);
	//cudaFreeHost(hostOutput); alternate way of freeing?
	return 0;
}

