// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define THREADS 256
#define BLOCK_SIZE 256

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)


//@@ insert code here

/* casts the input to unsigned char while also converting to grayscale*/
__global__ void FloatToUnsignedChar(float* input, unsigned char* uchar, unsigned char* grayImage, int height, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x*blockDim.x + tx; //column index
	int row = blockIdx.y*blockDim.y + ty; //row index
	int idx = row*width + col; //compute current position --similar to first few mps
	unsigned char r, g, b;
	//extract r, g, and b values while simulateneously casting to unsigned char
	if (row < height && col < width)
	{
		uchar[3 * idx] = r = (unsigned char)(255 * input[3 * idx]);
		uchar[3 * idx + 1] = g = (unsigned char)(255 * input[3 * idx + 1]);
		uchar[3 * idx + 2] = b = (unsigned char)(255 * input[3 * idx + 2]);

		//convert to grayscale
		grayImage[idx] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
	}
}


__global__ void ComputeHistogram(unsigned char* grayImage, int* histogram, int size)
{
	//set up shared memory
	__shared__ int histogram_private[HISTOGRAM_LENGTH];
	int tx = threadIdx.x;
	int i = blockIdx.x*blockDim.x + tx;
	
	if (tx < HISTOGRAM_LENGTH)
	{
		histogram_private[tx] = 0;
	}
	__syncthreads();
	//compute private histogram
	int stride = blockDim.x*gridDim.x;
	while (i < size)
	{
		atomicAdd(&(histogram_private[grayImage[i]]), 1);
		i += stride;
	}
	__syncthreads();
	//add to the public histrogram
	if (tx < HISTOGRAM_LENGTH)
	{
		atomicAdd(&(histogram[tx]), histogram_private[tx]);
	}

}

/*modified scan operation code from mp5*/
__global__ void scan(int *histogram, float *cdf, int size) 
{
	__shared__ float partialSum[2*BLOCK_SIZE];
	int tx = threadIdx.x;

	//@@ Load a segment of the input vector into shared memory
	//each thread loads 2 elements into shared memory. Put the identity (0 for sum) in if outside of boundary
	if (tx < HISTOGRAM_LENGTH)
	{
		partialSum[tx] = float(float(histogram[tx]) / (size));  //compute probability
	}
	else
	{
		partialSum[tx] = 0;
		partialSum[tx+BLOCK_SIZE] = 0;
	}
	
	__syncthreads();
	//printf(" %d ", histogram[tx]);

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
	__syncthreads(); //partialSum now holds the cdf

	//write to output
	if (tx < HISTOGRAM_LENGTH)
	{
		cdf[tx] = partialSum[tx];
	}

}

/*compute and apply the histogram equalization then cast back to float*/
__global__ void histogram_equalization(unsigned char* uchar, float* cdf, float* output, int height, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x*blockDim.x + tx; //column index
	int row = blockIdx.y*blockDim.y + ty; //row index
	int idx = row*width + col; //compute current position --similar to first few mps
	float cdfmin = cdf[0]; //extract cdf min

	if (row < height && col < width)
	{
		//apply histogram equalization function
		uchar[3 * idx] = min(max(255 * (cdf[uchar[3 * idx]] - cdfmin) / (1 - cdfmin), 0.0), 255.0);
		uchar[3 * idx + 1] = min(max(255 * (cdf[uchar[3 * idx + 1]] - cdfmin) / (1 - cdfmin), 0.0), 255.0);
		uchar[3 * idx + 2] = min(max(255 * (cdf[uchar[3 * idx + 2]] - cdfmin) / (1 - cdfmin), 0.0), 255.0);

		//cast back to float
		output[3 * idx] = float(uchar[3 * idx] / 255.0);
		output[3 * idx + 1] = float(uchar[3 * idx + 1] / 255.0);
		output[3 * idx + 2] = float(uchar[3 * idx + 2] / 255.0);
	}
}

int main(int argc, char **argv) {
	wbArg_t args;
	int imageWidth;
	int imageHeight;
	int imageChannels;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	const char *inputImageFile;

	float* deviceInput;
	float* deviceOutput;
	unsigned char* deviceUchar;
	unsigned char* deviceGrayImage;
	int* deviceHistogram;
	float* deviceCDF;

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);

	wbTime_start(Generic, "Importing data and creating memory on host");
	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	hostInputImageData = wbImage_getData(inputImage); //added this
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	wbTime_stop(Generic, "Importing data and creating memory on host");
	hostOutputImageData = (float *)malloc(imageWidth*imageHeight*imageChannels * sizeof(float));

	wbCheck(cudaMalloc(&deviceInput, (imageWidth*imageHeight*imageChannels*sizeof(float))));
	wbCheck(cudaMalloc(&deviceOutput, (imageWidth*imageHeight*imageChannels*sizeof(float))));
	wbCheck(cudaMalloc(&deviceUchar, (imageWidth*imageHeight*imageChannels*sizeof(unsigned char))));
	wbCheck(cudaMalloc(&deviceGrayImage, (imageWidth*imageHeight*sizeof(unsigned char))));
	wbCheck(cudaMalloc(&deviceHistogram, (HISTOGRAM_LENGTH*sizeof(int))));
	wbCheck(cudaMalloc(&deviceCDF, (HISTOGRAM_LENGTH*sizeof(float))));

	wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(int))); //initialize histogram to 0
	wbCheck(cudaMemset(deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(float)));

	wbCheck(cudaMemcpy(deviceInput, hostInputImageData, (imageWidth*imageHeight*imageChannels*sizeof(float)), cudaMemcpyHostToDevice));
	
	//use square root of 256 for number of threads in each direction
	dim3 grid((imageWidth-1)/16 + 1, (imageHeight - 1) / 16 + 1, 1);
	dim3 threads(16, 16, 1);
	//cast to unsigned char and convert to grayscale
	FloatToUnsignedChar<<<grid, threads>>>(deviceInput, deviceUchar, deviceGrayImage, imageHeight, imageWidth);
	
	dim3 grid2((imageWidth*imageHeight - 1) / THREADS + 1, 1, 1);
	dim3 threads2(THREADS, 1, 1);
	//compute the histogram
	ComputeHistogram<<<grid2, threads2>>>(deviceGrayImage, deviceHistogram, imageHeight*imageWidth);
	
	dim3 grid3(1, 1, 1);
	//compute the cdf
	scan<<<grid3, threads2>>>(deviceHistogram, deviceCDF, imageHeight*imageWidth);
	//printf("size: %d", imageHeight*imageWidth);

	//find and apply the histogram equalization
	histogram_equalization<<<grid, threads >>>(deviceUchar, deviceCDF, deviceOutput, imageHeight, imageWidth);
	cudaError_t code = cudaGetLastError();
     if (code != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(code)); }
	wbCheck(cudaMemcpy(hostOutputImageData, deviceOutput, imageHeight*imageWidth*imageChannels*sizeof(float), cudaMemcpyDeviceToHost));
	
	
	
	wbImage_setData(outputImage, hostOutputImageData); //added this
	wbSolution(args, outputImage);


	//@@ insert code here
	//free memory
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	cudaFree(deviceHistogram);
	cudaFree(deviceCDF);
	cudaFree(deviceUchar);
	cudaFree(deviceGrayImage);

	free(hostInputImageData);
	free(hostOutputImageData);

	return 0;
}
