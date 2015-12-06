#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Compute C = A * B
template<int TILE_WIDTH>
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int Row = by*blockDim.y + ty;
	int Col = bx*blockDim.x + tx;
	float Cvalue = 0;
	
	for(int t = 0; t<(numBRows-1)/TILE_WIDTH+1; t++)
	{
		if(Row < numARows && (t*TILE_WIDTH + tx < numAColumns))
		{
			ds_A[ty][tx] = A[Row*numBRows + t*TILE_WIDTH + tx];
		}
		else
		{
			ds_A[ty][tx] = 0.0;
		}
		
		if(Col < numBColumns && (t*TILE_WIDTH + ty < numBRows))
		{
			ds_B[ty][tx] = B[(t*TILE_WIDTH + ty)*numBColumns + Col];
		}
		else
		{
			ds_B[ty][tx] = 0.0;
		}

		__syncthreads();
		
		for(int i = 0; i<TILE_WIDTH; i++)
		{
			Cvalue += ds_A[ty][i] * ds_B[i][tx];	
		}
		__syncthreads();
	}
	if(Row < numCRows && Col < numCColumns)
	{
		C[Row*numCColumns + Col] = Cvalue;
	}
	
	
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);
	
	

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  wbCheck(cudaMallocHost(&hostC, (numCRows*numCColumns*sizeof(float))));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc(&deviceA, (numARows*numAColumns*sizeof(float))));
  wbCheck(cudaMalloc(&deviceB, (numBRows*numBColumns*sizeof(float))));
  wbCheck(cudaMalloc(&deviceC, (numCRows*numCColumns*sizeof(float))));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, (numARows*numAColumns*sizeof(float)), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, (numBRows*numBColumns*sizeof(float)), cudaMemcpyHostToDevice));

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 grid(((numCColumns - 1) / 32 + 1), ((numCRows - 1) / 32 + 1));
  dim3 threads(32, 32);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<32> <<<grid, threads >>>(deviceA, deviceB, deviceC,
	  numARows, numAColumns,
	  numBRows, numBColumns,
	  numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, (numCRows*numCColumns*sizeof(float)), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
	
  free(hostB);
	
  cudaFreeHost(hostC);
	printf("Hello");

  return 0;
}
