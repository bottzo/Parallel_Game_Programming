#include <Windows.h>
#include <cuda_runtime.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

__global__ void RGBToGreyScale2D(const uchar3* const colorPixels, unsigned char* greyPixels, int rowElements, int columnElements)
{
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (y < columnElements && x < rowElements)
	{
		int index = y * rowElements + x;
		greyPixels[index] = 0.299f * colorPixels[index].x + 0.587f * colorPixels[index].y + 0.114f * colorPixels[index].z;
	}
}

__global__ void RGBToGreyScale(const uchar3* const colorPixels, unsigned char* greyPixels, unsigned int numElements)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numElements)
		greyPixels[idx] = 0.299f * colorPixels[idx].x + 0.587f * colorPixels[idx].y + 0.114f * colorPixels[idx].z;
}

//cacheSize = blockDim.x * 3 * 4
__global__ void CachedRGBToGreyScale(const uchar4* const colorPixels, unsigned char* greyPixels, unsigned int numElements)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numElements)
	{
		extern __shared__ uchar4 cacheData[];
		const unsigned int uneededBlockCache = (blockDim.x * 1 / 4);
		if (threadIdx.x < blockDim.x - uneededBlockCache)
			cacheData[threadIdx.x] = colorPixels[index - uneededBlockCache * blockIdx.x];
		__syncthreads();
		const uchar3 currentTexel = ((uchar3*)cacheData)[threadIdx.x];
		greyPixels[index] = 0.299f * currentTexel.x + 0.587f * currentTexel.y + 0.114f * currentTexel.z;
	}
}

__global__ void NewRGBToGreyScale(const uchar4* const colorPixels, unsigned char* greyPixels, unsigned int numElements)
{
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numElements)
	{
		const uchar4 currentTexel = colorPixels[index];
		const unsigned int grayIdxOffset = index / 3;
		if (index % 3 == 0)
		{
			greyPixels[index + grayIdxOffset] = 0.299f * currentTexel.x + 0.587f * currentTexel.y + 0.114f * currentTexel.z;
			greyPixels[index + grayIdxOffset + 1] += 0.299f * currentTexel.w;
		}
		else if (index % 3 == 1)
		{
			greyPixels[index + grayIdxOffset] += 0.587f * currentTexel.x + 0.114f * currentTexel.y;
			greyPixels[index + grayIdxOffset + 1] += 0.299f * currentTexel.z + 0.587f * currentTexel.w;
		}
		else
		{
			greyPixels[index + grayIdxOffset] += 0.114f * currentTexel.x;
			greyPixels[index + grayIdxOffset + 1] = 0.299f * currentTexel.y + 0.587f * currentTexel.z + 0.114f * currentTexel.w;
		}
	}
}

__global__ void RGB12ToGreyScale(const uchar4* const colorPixels, unsigned char* greyPixels, unsigned int numThreads)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numThreads)
	{
		//unsigned char threadTexels[12];
		//for (int i = 0; i < 3; ++i)
		//	((uchar4*)threadTexels)[i] = ((uchar4*)colorPixels)[index*3 + i];
		//for (int i = 0; i < 4; ++i)
		//	greyPixels[index*4 + i] = 0.299f * threadTexels[i*3] + 0.587f * threadTexels[i*3+1] + 0.114f * threadTexels[i*3+2];
		
		//unsigned char threadTexels[12];
		//((uchar4*)threadTexels)[0] = ((uchar4*)colorPixels)[index * 3];
		//greyPixels[index * 4] = 0.299f * threadTexels[0] + 0.587f * threadTexels[1] + 0.114f * threadTexels[2];
		//((uchar4*)threadTexels)[1] = ((uchar4*)colorPixels)[index * 3 + 1];
		//greyPixels[index * 4 + 1] = 0.299f * threadTexels[3] + 0.587f * threadTexels[4] + 0.114f * threadTexels[5];
		//((uchar4*)threadTexels)[2] = ((uchar4*)colorPixels)[index * 3 + 2];
		//greyPixels[index * 4 + 2] = 0.299f * threadTexels[6] + 0.587f * threadTexels[7] + 0.114f * threadTexels[8];
		//greyPixels[index * 4 + 3] = 0.299f * threadTexels[9] + 0.587f * threadTexels[10] + 0.114f * threadTexels[11];

		//unsigned long long doubleBlock = ((unsigned long long*)colorPixels)[(index * 3)/2];
		//const uchar4 threadDataBlock1 = *((uchar4*)&doubleBlock);
		//const uchar4 threadDataBlock2 = *(((uchar4*)&doubleBlock)+1);
		const uchar4 threadDataBlock1 = colorPixels[index * 3];
		const uchar4 threadDataBlock2 = colorPixels[index * 3 + 1];
		const uchar4 threadDataBlock3 = colorPixels[index * 3 + 2];
		uchar4 result;
		result.x = 0.299f * threadDataBlock1.x + 0.587f * threadDataBlock1.y + 0.114f * threadDataBlock1.z;
		result.y = 0.299f * threadDataBlock1.w + 0.587f * threadDataBlock2.x + 0.114f * threadDataBlock2.y;
		result.z = 0.299f * threadDataBlock2.z + 0.587f * threadDataBlock2.w + 0.114f * threadDataBlock3.x;
		result.w = 0.299f * threadDataBlock3.y + 0.587f * threadDataBlock3.z + 0.114f * threadDataBlock3.w;
		((uchar4*)greyPixels)[index] = result;
	}
}

//cacheSize = blockDim.x*4*3
__global__ void CachedRGB12ToGreyScale(const uchar4* const colorPixels, uchar4* greyPixels, unsigned int numThreads)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numThreads)
	{
		const unsigned int offset = 2 * blockDim.x * blockIdx.x;
		extern __shared__ uchar4 shared[];
		shared[threadIdx.x] = colorPixels[index + offset];
		shared[threadIdx.x + blockDim.x] = colorPixels[index + offset + blockDim.x];
		shared[threadIdx.x + 2*blockDim.x] = colorPixels[index + offset + 2*blockDim.x];
		__syncthreads();
		const uchar4 threadDataBlock1 = shared[threadIdx.x * 3];
		const uchar4 threadDataBlock2 = shared[threadIdx.x * 3 + 1];
		const uchar4 threadDataBlock3 = shared[threadIdx.x * 3 + 2];
		uchar4 result;
		result.x = 0.299f * threadDataBlock1.x + 0.587f * threadDataBlock1.y + 0.114f * threadDataBlock1.z;
		result.y = 0.299f * threadDataBlock1.w + 0.587f * threadDataBlock2.x + 0.114f * threadDataBlock2.y;
		result.z = 0.299f * threadDataBlock2.z + 0.587f * threadDataBlock2.w + 0.114f * threadDataBlock3.x;
		result.w = 0.299f * threadDataBlock3.y + 0.587f * threadDataBlock3.z + 0.114f * threadDataBlock3.w;
		greyPixels[index] = result;
	}
}

__global__ void RGB16ToGreyScale(const uchar4* const colorPixels, unsigned char* greyPixels, unsigned int numThreads)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numThreads)
	{

		//uchar4 threadData[12];
		//uchar4 result[4];
		//for (int i = 0; i < 12; ++i)
		//	threadData[i] = colorPixels[index * 12 + i];
		//for (int i = 0; i < 4; ++i)
		//{
		//	result[i].x = 0.299f * threadData[i*3 + 0].x + 0.587f * threadData[i*3 + 0].y + 0.114f * threadData[i*3 + 0].z;
		//	result[i].y = 0.299f * threadData[i*3 + 0].w + 0.587f * threadData[i*3 + 1].x + 0.114f * threadData[i*3 + 1].y;
		//	result[i].z = 0.299f * threadData[i*3 + 1].z + 0.587f * threadData[i*3 + 1].w + 0.114f * threadData[i*3 + 2].x;
		//	result[i].w = 0.299f * threadData[i*3 + 2].y + 0.587f * threadData[i*3 + 2].z + 0.114f * threadData[i*3 + 2].w;
		//}
		//for (int i = 0; i < 4; ++i)
		//	((uchar4*)greyPixels)[index*4+i] = result[i];

		uchar4 threadData[12];
		uchar4 result[4];
		for (int i = 0; i < 6; ++i)
			((unsigned long long*)threadData)[i] = ((unsigned long long*)colorPixels)[index * 6 + i];
		for (int i = 0; i < 4; ++i)
		{
			result[i].x = 0.299f * threadData[i*3 + 0].x + 0.587f * threadData[i*3 + 0].y + 0.114f * threadData[i*3 + 0].z;
			result[i].y = 0.299f * threadData[i*3 + 0].w + 0.587f * threadData[i*3 + 1].x + 0.114f * threadData[i*3 + 1].y;
			result[i].z = 0.299f * threadData[i*3 + 1].z + 0.587f * threadData[i*3 + 1].w + 0.114f * threadData[i*3 + 2].x;
			result[i].w = 0.299f * threadData[i*3 + 2].y + 0.587f * threadData[i*3 + 2].z + 0.114f * threadData[i*3 + 2].w;
		}
		for (int i = 0; i < 2; ++i)
			((unsigned long long*)greyPixels)[index*2+i] = ((unsigned long long*)result)[i];
	}
}

__global__ void RGBASeparateIntoChannels(const uchar3* const colorPixels, unsigned char* redPixels, unsigned char* greenPixels, unsigned char* bluePixels, int dataSize)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < dataSize)
	{
		redPixels[idx] =   colorPixels[idx].x;
		greenPixels[idx] = colorPixels[idx].y;
		bluePixels[idx] =  colorPixels[idx].z;
	}
}
__global__ void RToGreyScale(const unsigned char* const rTexels, unsigned char* greyPixels, unsigned int numElements)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numElements)
	{
		greyPixels[index] += 0.299f * rTexels[index];
	}
}
__global__ void GToGreyScale(const unsigned char* const gTexels, unsigned char* greyPixels, unsigned int numElements)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numElements)
	{
		greyPixels[index] += 0.587f * gTexels[index];
	}
}
__global__ void BToGreyScale(const unsigned char* const bTexels, unsigned char* greyPixels, unsigned int numElements)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numElements)
	{
		greyPixels[index] += 0.114f * bTexels[index];
	}
}

__global__ void RGBSeparateIntoChannels(const uchar3* const colorPixels, unsigned char* redPixels, unsigned char* greenPixels, unsigned char* bluePixels, int dataSize)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < dataSize)
	{
		redPixels[idx] = colorPixels[idx].x;
		greenPixels[idx] = colorPixels[idx].y;
		bluePixels[idx] = colorPixels[idx].z;
	}
}

__global__ void RGB12SeparateIntoChannels(const uchar4* const colorPixels, uchar4* redPixels, uchar4* greenPixels, uchar4* bluePixels, int dataSize)
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < dataSize)
	{
		const uchar4 threadDataBlock1 = colorPixels[idx * 3];
		const uchar4 threadDataBlock2 = colorPixels[idx * 3 + 1];
		const uchar4 threadDataBlock3 = colorPixels[idx * 3 + 2];

		redPixels[idx] = {threadDataBlock1.x, threadDataBlock1.w, threadDataBlock2.z, threadDataBlock3.y};
		greenPixels[idx] = {threadDataBlock1.y, threadDataBlock2.x, threadDataBlock2.w, threadDataBlock3.z};
		bluePixels[idx] = {threadDataBlock1.z, threadDataBlock2.y, threadDataBlock3.x, threadDataBlock3.w};
	}
}

//cache size = 3* 4 * blockDim.x
__global__ void RGB12SeparateIntoChannelsCache(const uchar4* const colorPixels, uchar4* redPixels, uchar4* greenPixels, uchar4* bluePixels, int dataSize)
{

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ uchar4 shared[];
	if (idx < dataSize)
	{
		const unsigned int offset = 2 * blockDim.x * blockIdx.x;
		shared[threadIdx.x] = colorPixels[idx + offset];
		shared[threadIdx.x + blockDim.x] = colorPixels[idx + offset + blockDim.x];
		shared[threadIdx.x + 2*blockDim.x] = colorPixels[idx + offset + 2*blockDim.x ];
		__syncthreads();

		const uchar4 threadDataBlock1 = shared[threadIdx.x * 3];
		const uchar4 threadDataBlock2 = shared[threadIdx.x * 3 + 1];
		const uchar4 threadDataBlock3 = shared[threadIdx.x * 3 + 2];

		redPixels[idx] = { threadDataBlock1.x, threadDataBlock1.w, threadDataBlock2.z, threadDataBlock3.y };
		greenPixels[idx] = { threadDataBlock1.y, threadDataBlock2.x, threadDataBlock2.w, threadDataBlock3.z };
		bluePixels[idx] = { threadDataBlock1.z, threadDataBlock2.y, threadDataBlock3.x, threadDataBlock3.w };
	}
}


__global__ void ChannelsUniteIntoRGB(unsigned char* redPixels, unsigned char* greenPixels, unsigned char* bluePixels, uchar3* rgbPixels, int dataSize)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < dataSize)
	{
		rgbPixels[idx].x = redPixels[idx];
		rgbPixels[idx].y = greenPixels[idx];
		rgbPixels[idx].z = bluePixels[idx];
	}
}


__global__ void ChannelWeightedImageBlur(const unsigned char* const colorPixels, unsigned char* bluredPixels, int rowElements, int columnElements)
{
	const unsigned int cyrcleCount = 2;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int gidx = y * rowElements + x;
	const int cacheX = threadIdx.x + cyrcleCount;
	const int cacheY = threadIdx.y + cyrcleCount;
	const int cacheDimX = (blockDim.x + cyrcleCount * 2);
	const int cacheIdx = cacheY * cacheDimX + cacheX;
	//extern __shared__ to allocate the shared memory size at lunch time with the parameters of grid and block sizes <<<grid size,block size,shared bytes>>> kernel(...)
	extern __shared__ unsigned char cache[];
	//TODO: Mirar de loadejar la data de forma diferent (amb menys ifs i tot coalesced com es pugi)
	cache[cacheIdx] = colorPixels[gidx];
	if (threadIdx.y == 0)
	{
		//if (blockIdx.y != 0)
		for (int i = 1; i <= cyrcleCount; ++i)
			if ((gidx - rowElements * i) >= 0)
				cache[cacheIdx - cacheDimX * i] = colorPixels[gidx - rowElements * i];
			else
				cache[cacheIdx - cacheDimX * i] = colorPixels[gidx];
	}
	else if (threadIdx.y == (blockDim.y - 1))
	{
		//if (blockIdx.y != 0)
		for (int i = 1; i <= cyrcleCount; ++i)
			if ((gidx + rowElements * i) < (rowElements * columnElements))
				cache[cacheIdx + cacheDimX * i] = colorPixels[gidx + rowElements * i];
			else
				cache[cacheIdx + cacheDimX * i] = colorPixels[gidx];
	}

	if (threadIdx.x == 0)
	{
		//if (blockIdx.x != 0)
		for (int i = 1; i <= cyrcleCount; ++i)
			if ((gidx - i) >= 0)
				cache[cacheIdx - i] = colorPixels[gidx - i];
			else
				cache[cacheIdx - i] = colorPixels[gidx];
	}
	else if (threadIdx.x == (blockDim.x - 1))
	{
		//if (blockIdx.x != 0)
		for (int i = 1; i <= cyrcleCount; ++i)
			if ((gidx + i) < (rowElements * columnElements))
				cache[cacheIdx + i] = colorPixels[gidx + i];
			else
				cache[cacheIdx + i] = colorPixels[gidx];
	}
	__syncthreads();
	if (y < columnElements && x < rowElements)
	{
		//kernel used for gaussian blur
		//0.00  0.01  0.01  0.01  0.00
		//0.01  0.05  0.11  0.05  0.01
		//0.01  0.11  0.24  0.11  0.01
		//0.01  0.05  0.11  0.05  0.01
		//0.00  0.01  0.01  0.01  0.00
		//TODO: FER-HO COM A MULTIPLICACIO DE MATRIUS!!!!! (kernel * cache(a partir del cacheIdx - cacheDimX * 2 - 1 i fins a la size total del kernel))
		bluredPixels[gidx] = cache[cacheIdx - cacheDimX * 2 - 1] * 0.01 + cache[cacheIdx - cacheDimX * 2] * 0.01 + cache[cacheIdx - cacheDimX * 2 + 1] * 0.01 +
			cache[cacheIdx - cacheDimX - 2] * 0.01 + cache[cacheIdx - cacheDimX - 1] * 0.05 + cache[cacheIdx - cacheDimX] * 0.11 + cache[cacheIdx - cacheDimX + 1] * 0.05 + cache[cacheIdx - cacheDimX + 2] * 0.01 +
			cache[cacheIdx - 2] * 0.01 + cache[cacheIdx - 1] * 0.11 + cache[cacheIdx] * 0.24 + cache[cacheIdx + 1] * 0.11 + cache[cacheIdx + 2] * 0.01 +
			cache[cacheIdx + cacheDimX - 2] * 0.01 + cache[cacheIdx + cacheDimX - 1] * 0.05 + cache[cacheIdx + cacheDimX] * 0.11 + cache[cacheIdx + cacheDimX + 1] * 0.05 + cache[cacheIdx + cacheDimX + 2] * 0.01 +
			cache[cacheIdx + cacheDimX * 2 - 1] * 0.01 + cache[cacheIdx + cacheDimX * 2] * 0.01 + cache[cacheIdx + cacheDimX * 2 + 1] * 0.01;
	}
}

__global__ void ChannelWeightedImageBlur2(const unsigned char* const colorPixels, unsigned char* bluredPixels, int rowElements, int columnElements)
{
	const unsigned int cyrcleCount = 2;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int gidx = y * rowElements + x;
	const int cacheX = threadIdx.x + cyrcleCount;
	const int cacheY = threadIdx.y + cyrcleCount;
	const int cacheDimX = (blockDim.x + cyrcleCount * 2);
	const int cacheIdx = cacheY * cacheDimX + cacheX;
	//extern __shared__ to allocate the shared memory size at lunch time with the parameters of grid and block sizes <<<grid size,block size,shared bytes>>> kernel(...)
	extern __shared__ unsigned char cache[];
	cache[cacheIdx] = colorPixels[gidx];
	if (threadIdx.y == 0 && threadIdx.x == 0)
	{
		const unsigned int cache2UpIdx = cacheIdx - cacheDimX * 2;
		if (blockIdx.y == 0)
		{
			//(*(unsigned short*)(cache + cacheIdx - cache2UpIdx - 2)) = 0;
			for (unsigned int i = 0; i < cacheDimX; ++i)
				cache[cache2UpIdx + i] = colorPixels[gidx + i];
			for (unsigned int i = 0; i < cacheDimX; ++i)
				cache[cacheIdx - cacheDimX + i] = cache[cache2UpIdx + i];
		}
		else
		{
			for (unsigned int i = 0; i < cacheDimX * 2; ++i)
				cache[cache2UpIdx + i] = colorPixels[gidx - rowElements * 2 + i];
		}
	}
	else if (threadIdx.y == (blockDim.y - 1) && threadIdx.x == 0)
	{
		const unsigned int cacheDownOne = cacheIdx + cacheDimX;
		if (blockIdx.y == 0)
		{
			//(*(unsigned short*)(cache + cacheIdx - cache2UpIdx - 2)) = 0;
			for (unsigned int i = 0; i < cacheDimX; ++i)
				cache[cacheDownOne + i] = colorPixels[gidx + i];
			for (unsigned int i = 0; i < cacheDimX; ++i)
				cache[cacheIdx + cacheDimX + i] = cache[cacheDownOne + i];
		}
		else
		{
			for (unsigned int i = 0; i < cacheDimX * 2; ++i)
				cache[cacheDownOne + i] = colorPixels[gidx + rowElements + i];
		}
	}

	if (threadIdx.x == 0)
	{
		//if (blockIdx.x != 0)
		for (int i = 1; i <= cyrcleCount; ++i)
			if ((gidx - i) >= 0)
				cache[cacheIdx - i] = colorPixels[gidx - i];
			else
				cache[cacheIdx - i] = colorPixels[gidx];
	}
	else if (threadIdx.x == (blockDim.x - 1))
	{
		//if (blockIdx.x != 0)
		for (int i = 1; i <= cyrcleCount; ++i)
			if ((gidx + i) < (rowElements * columnElements))
				cache[cacheIdx + i] = colorPixels[gidx + i];
			else
				cache[cacheIdx + i] = colorPixels[gidx];
	}
	__syncthreads();
	if (y < columnElements && x < rowElements)
	{
		//kernel used for gaussian blur
		//0.00  0.01  0.01  0.01  0.00
		//0.01  0.05  0.11  0.05  0.01
		//0.01  0.11  0.24  0.11  0.01
		//0.01  0.05  0.11  0.05  0.01
		//0.00  0.01  0.01  0.01  0.00
		bluredPixels[gidx] = cache[cacheIdx - cacheDimX * 2 - 1] * 0.01 + cache[cacheIdx - cacheDimX * 2] * 0.01 + cache[cacheIdx - cacheDimX * 2 + 1] * 0.01 +
			cache[cacheIdx - cacheDimX - 2] * 0.01 + cache[cacheIdx - cacheDimX - 1] * 0.05 + cache[cacheIdx - cacheDimX] * 0.11 + cache[cacheIdx - cacheDimX + 1] * 0.05 + cache[cacheIdx - cacheDimX + 2] * 0.01 +
			cache[cacheIdx - 2] * 0.01 + cache[cacheIdx - 1] * 0.11 + cache[cacheIdx] * 0.24 + cache[cacheIdx + 1] * 0.11 + cache[cacheIdx + 2] * 0.01 +
			cache[cacheIdx + cacheDimX - 2] * 0.01 + cache[cacheIdx + cacheDimX - 1] * 0.05 + cache[cacheIdx + cacheDimX] * 0.11 + cache[cacheIdx + cacheDimX + 1] * 0.05 + cache[cacheIdx + cacheDimX + 2] * 0.01 +
			cache[cacheIdx + cacheDimX * 2 - 1] * 0.01 + cache[cacheIdx + cacheDimX * 2] * 0.01 + cache[cacheIdx + cacheDimX * 2 + 1] * 0.01;
	}
}


__global__ void ChannelWeightedImageBlur3(const unsigned char* const colorPixels, unsigned char* bluredPixels, int rowElements, int columnElements, float* kernel, const unsigned int kernelRadius)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int gidx = y * rowElements + x;
	const int cacheX = threadIdx.x + kernelRadius;
	const int cacheY = threadIdx.y + kernelRadius;
	const int cacheDimX = (blockDim.x + kernelRadius * 2);
	const int cacheDimY = (blockDim.y + kernelRadius * 2);
	const int cacheIdx = cacheY * cacheDimX + cacheX;
	const int goffset = (((cacheDimX * kernelRadius) / blockDim.x) * rowElements + kernelRadius);   //+ (4 * (threadIdx.y * blockDim.x + threadIdx.x));
	//extern __shared__ to allocate the shared memory size at lunch time with the parameters of grid and block sizes <<<grid size,block size,shared bytes>>> kernel(...)
	extern __shared__ unsigned char cache[];
	//if ((threadIdx.y * blockDim.x + threadIdx.x) < (cacheDimX * cacheDimY))
	//{
	//	if (gidx - goffset >= 0)
	//		cache[threadIdx.y * blockDim.x + threadIdx.x] = colorPixels[gidx - goffset];
	//	else
	//		//TODO: poder algun dels 4 no es 0 !!!
	//		//Intentar fer un clamp al last pixel color
	//		cache[threadIdx.y * blockDim.x + threadIdx.x] = 0;
	//}
	cache[cacheIdx] = colorPixels[gidx];
	if (threadIdx.y == 0)
	{
		//if (blockIdx.y != 0)
		for (int i = 1; i <= kernelRadius; ++i)
			if ((gidx - rowElements * i) >= 0)
				cache[cacheIdx - cacheDimX * i] = colorPixels[gidx - rowElements * i];
			else
				cache[cacheIdx - cacheDimX * i] = colorPixels[gidx];
	}
	else if (threadIdx.y == (blockDim.y - 1))
	{
		//if (blockIdx.y != 0)
		for (int i = 1; i <= kernelRadius; ++i)
			if ((gidx + rowElements * i) < (rowElements * columnElements))
				cache[cacheIdx + cacheDimX * i] = colorPixels[gidx + rowElements * i];
			else
				cache[cacheIdx + cacheDimX * i] = colorPixels[gidx];
	}
	
	if (threadIdx.x == 0)
	{
		//if (blockIdx.x != 0)
		for (int i = 1; i <= kernelRadius; ++i)
			if ((gidx - i) >= 0)
				cache[cacheIdx - i] = colorPixels[gidx - i];
			else
				cache[cacheIdx - i] = colorPixels[gidx];
	}
	else if (threadIdx.x == (blockDim.x - 1))
	{
		//if (blockIdx.x != 0)
		for (int i = 1; i <= kernelRadius; ++i)
			if ((gidx + i) < (rowElements * columnElements))
				cache[cacheIdx + i] = colorPixels[gidx + i];
			else
				cache[cacheIdx + i] = colorPixels[gidx];
	}
	__syncthreads();
	if (y < columnElements && x < rowElements)
	{
		const unsigned int kernelDiam = kernelRadius * 2 + 1;
		const unsigned int kernelSize = kernelDiam * kernelDiam;
		unsigned char result = 0;
		for (unsigned int i = 0; i < kernelSize; ++i)
			result += cache[cacheIdx + (-cacheDimX * (2 - i/kernelDiam)) - 2 + (i%kernelDiam)] * kernel[i];
		bluredPixels[gidx] = result;

		//bluredPixels[gidx] = cache[cacheIdx - cacheDimX * 2 - 1] * 0.01 + cache[cacheIdx - cacheDimX * 2] * 0.01 + cache[cacheIdx - cacheDimX * 2 + 1] * 0.01 +
		//	cache[cacheIdx - cacheDimX - 2] * 0.01 + cache[cacheIdx - cacheDimX - 1] * 0.05 + cache[cacheIdx - cacheDimX] * 0.11 + cache[cacheIdx - cacheDimX + 1] * 0.05 + cache[cacheIdx - cacheDimX + 2] * 0.01 +
		//	cache[cacheIdx - 2] * 0.01 + cache[cacheIdx - 1] * 0.11 + cache[cacheIdx] * 0.24 + cache[cacheIdx + 1] * 0.11 + cache[cacheIdx + 2] * 0.01 +
		//	cache[cacheIdx + cacheDimX - 2] * 0.01 + cache[cacheIdx + cacheDimX - 1] * 0.05 + cache[cacheIdx + cacheDimX] * 0.11 + cache[cacheIdx + cacheDimX + 1] * 0.05 + cache[cacheIdx + cacheDimX + 2] * 0.01 +
		//	cache[cacheIdx + cacheDimX * 2 - 1] * 0.01 + cache[cacheIdx + cacheDimX * 2] * 0.01 + cache[cacheIdx + cacheDimX * 2 + 1] * 0.01;
	}
}

__global__ void ChannelWeightedImageBlur4(const unsigned char* const colorPixels, unsigned char* bluredPixels, int rowElements, int columnElements, float* kernel, const unsigned int kernelRadius)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int gidx = y * rowElements + x;
	const int cacheX = threadIdx.x + kernelRadius;
	const int cacheY = threadIdx.y + kernelRadius;
	const int cacheDimX = (blockDim.x + kernelRadius * 2);
	//const int cacheDimY = (blockDim.y + kernelRadius * 2);
	const int cacheIdx = cacheY * cacheDimX + cacheX;
	//const int goffset = (((cacheDimX * kernelRadius) / blockDim.x) * rowElements + kernelRadius);   //+ (4 * (threadIdx.y * blockDim.x + threadIdx.x));
	const unsigned int blockIndex = threadIdx.x + blockDim.x * threadIdx.y;
	//extern __shared__ to allocate the shared memory size at lunch time with the parameters of grid and block sizes <<<grid size,block size,shared bytes>>> kernel(...)
	extern __shared__ uchar4 cachee[];
	//if ((threadIdx.y * blockDim.x + threadIdx.x) < (cacheDimX * cacheDimY))
	//{
	//	if (gidx - goffset >= 0)
	//		cache[threadIdx.y * blockDim.x + threadIdx.x] = colorPixels[gidx - goffset];
	//	else
	//		//TODO: poder algun dels 4 no es 0 !!!
	//		//Intentar fer un clamp al last pixel color
	//		cache[threadIdx.y * blockDim.x + threadIdx.x] = 0;
	//}
	if(blockIndex % 4 == 0)
		cachee[cacheIdx/4] = (*(uchar4*)(colorPixels + gidx));
	unsigned char* cache = ((unsigned char*)cachee);
	if (threadIdx.y == 0)
	{
		//if (blockIdx.y != 0)
		for (int i = 1; i <= kernelRadius; ++i)
			if ((gidx - rowElements * i) >= 0)
				cache[cacheIdx - cacheDimX * i] = colorPixels[gidx - rowElements * i];
			else
				cache[cacheIdx - cacheDimX * i] = colorPixels[gidx];
	}
	else if (threadIdx.y == (blockDim.y - 1))
	{
		//if (blockIdx.y != 0)
		for (int i = 1; i <= kernelRadius; ++i)
			if ((gidx + rowElements * i) < (rowElements * columnElements))
				cache[cacheIdx + cacheDimX * i] = colorPixels[gidx + rowElements * i];
			else
				cache[cacheIdx + cacheDimX * i] = colorPixels[gidx];
	}

	if (threadIdx.x == 0)
	{
		//if (blockIdx.x != 0)
		for (int i = 1; i <= kernelRadius; ++i)
			if ((gidx - i) >= 0)
				cache[cacheIdx - i] = colorPixels[gidx - i];
			else
				cache[cacheIdx - i] = colorPixels[gidx];
	}
	else if (threadIdx.x == (blockDim.x - 1))
	{
		//if (blockIdx.x != 0)
		for (int i = 1; i <= kernelRadius; ++i)
			if ((gidx + i) < (rowElements * columnElements))
				cache[cacheIdx + i] = colorPixels[gidx + i];
			else
				cache[cacheIdx + i] = colorPixels[gidx];
	}
	__syncthreads();
	if (y < columnElements && x < rowElements)
	{
		const unsigned int kernelDiam = kernelRadius * 2 + 1;
		const unsigned int kernelSize = kernelDiam * kernelDiam;
		unsigned char result = 0;
		for (unsigned int i = 0; i < kernelSize; ++i)
			result += cache[cacheIdx + (-cacheDimX * (2 - i / kernelDiam)) - 2 + (i % kernelDiam)] * kernel[i];
		bluredPixels[gidx] = result;

		//bluredPixels[gidx] = cache[cacheIdx - cacheDimX * 2 - 1] * 0.01 + cache[cacheIdx - cacheDimX * 2] * 0.01 + cache[cacheIdx - cacheDimX * 2 + 1] * 0.01 +
		//	cache[cacheIdx - cacheDimX - 2] * 0.01 + cache[cacheIdx - cacheDimX - 1] * 0.05 + cache[cacheIdx - cacheDimX] * 0.11 + cache[cacheIdx - cacheDimX + 1] * 0.05 + cache[cacheIdx - cacheDimX + 2] * 0.01 +
		//	cache[cacheIdx - 2] * 0.01 + cache[cacheIdx - 1] * 0.11 + cache[cacheIdx] * 0.24 + cache[cacheIdx + 1] * 0.11 + cache[cacheIdx + 2] * 0.01 +
		//	cache[cacheIdx + cacheDimX - 2] * 0.01 + cache[cacheIdx + cacheDimX - 1] * 0.05 + cache[cacheIdx + cacheDimX] * 0.11 + cache[cacheIdx + cacheDimX + 1] * 0.05 + cache[cacheIdx + cacheDimX + 2] * 0.01 +
		//	cache[cacheIdx + cacheDimX * 2 - 1] * 0.01 + cache[cacheIdx + cacheDimX * 2] * 0.01 + cache[cacheIdx + cacheDimX * 2 + 1] * 0.01;
	}
}

__global__ void ChannelUnweightedImageBlurQuad(const unsigned char* const colorPixels, unsigned char* bluredPixels, int rowElements, int columnElements, int quadDim)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int gidx = y * rowElements + x;
	const int cacheX = x - (blockIdx.x * blockDim.y) + quadDim;
	const int cacheY = y - (blockIdx.y * blockDim.y) + quadDim;
	const int cacheDimX = (blockDim.x + quadDim*2);
	const int cacheIdx = cacheY * cacheDimX + cacheX;
	//extern __shared__ to allocate the shared memory size at lunch time with the parameters of grid and block sizes <<<grid size,block size,shared bytes>>> kernel(...)
	extern __shared__ unsigned char cache[];
	cache[cacheIdx] = colorPixels[gidx];
	if (threadIdx.y == 0)
	{
		//if (blockIdx.y != 0)
			for(int i = 1; i <= quadDim; ++i)
				if((gidx - rowElements * i) >= 0)
					cache[cacheIdx - cacheDimX* i] = colorPixels[gidx - rowElements * i];
				else
					cache[cacheIdx - cacheDimX* i] = 255;
	}
	else if (threadIdx.y == (blockDim.y - 1))
	{
		//if (blockIdx.y != 0)
			for (int i = 1; i <= quadDim; ++i)
				if((gidx + rowElements * i) < (rowElements*columnElements))
					cache[cacheIdx + cacheDimX * i] = colorPixels[gidx + rowElements * i];
				else
					cache[cacheIdx + cacheDimX * i] = 255;
	}

	if (threadIdx.x == 0)
	{
		//if (blockIdx.x != 0)
			for (int i = 1; i <= quadDim; ++i)
				if((gidx - i) >= 0)
					cache[cacheIdx - i] = colorPixels[gidx - i];
				else
					cache[cacheIdx - i] = 255;
	}
	else if (threadIdx.x == (blockDim.x - 1))
	{
		//if (blockIdx.x != 0)
			for (int i = 1; i <= quadDim; ++i)
				if((gidx + i) < (rowElements * columnElements))
					cache[cacheIdx + i] = colorPixels[gidx + i];
				else
					cache[cacheIdx + i] = 255;
	}
	__syncthreads();
	if (y < columnElements && x < rowElements)
	{
		unsigned int numSums = 0;
		unsigned int sum = 0;
		for (int i = 1; i <= quadDim; ++i, numSums += 8)
		{
			sum += (cache[cacheIdx - i]  /* + cache[cacheIdx]*/ + cache[cacheIdx + i] +
				cache[cacheIdx - cacheDimX*i - 1] + cache[cacheIdx - cacheDimX*i] + cache[cacheIdx - cacheDimX*i + 1] +
				cache[cacheIdx + cacheDimX*i - 1] + cache[cacheIdx + cacheDimX*i] + cache[cacheIdx + cacheDimX*i + 1]);
		}
		bluredPixels[gidx] = (1.f / (float)numSums) * sum;
	}
}
__global__ void ChannelUnweightedImageBlurAvgQuad(const unsigned char* const colorPixels, unsigned char* bluredPixels, int rowElements, int columnElements, int cyrcleCount)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int gidx = y * rowElements + x;
	const int cacheX = threadIdx.x + cyrcleCount;
	const int cacheY = threadIdx.y + cyrcleCount;
	const int cacheDimX = (blockDim.x + cyrcleCount * 2);
	const int cacheIdx = cacheY * cacheDimX + cacheX;
	//extern __shared__ to allocate the shared memory size at lunch time with the parameters of grid and block sizes <<<grid size,block size,shared bytes>>> kernel(...)
	extern __shared__ unsigned char cache[];
	cache[cacheIdx] = colorPixels[gidx];
	if (threadIdx.y == 0)
	{
		//if (blockIdx.y != 0)
		for (int i = 1; i <= cyrcleCount; ++i)
			if ((gidx - rowElements * i) >= 0)
				cache[cacheIdx - cacheDimX * i] = colorPixels[gidx - rowElements * i];
			else
				cache[cacheIdx - cacheDimX * i] = colorPixels[gidx];
	}
	else if (threadIdx.y == (blockDim.y - 1))
	{
		//if (blockIdx.y != 0)
		for (int i = 1; i <= cyrcleCount; ++i)
			if ((gidx + rowElements * i) < (rowElements * columnElements))
				cache[cacheIdx + cacheDimX * i] = colorPixels[gidx + rowElements * i];
			else
				cache[cacheIdx + cacheDimX * i] = colorPixels[gidx];
	}

	if (threadIdx.x == 0)
	{
		//if (blockIdx.x != 0)
		for (int i = 1; i <= cyrcleCount; ++i)
			if ((gidx - i) >= 0)
				cache[cacheIdx - i] = colorPixels[gidx - i];
			else
				cache[cacheIdx - i] = colorPixels[gidx];
	}
	else if (threadIdx.x == (blockDim.x - 1))
	{
		//if (blockIdx.x != 0)
		for (int i = 1; i <= cyrcleCount; ++i)
			if ((gidx + i) < (rowElements * columnElements))
				cache[cacheIdx + i] = colorPixels[gidx + i];
			else
				cache[cacheIdx + i] = colorPixels[gidx];
	}
	__syncthreads();
	if (y < columnElements && x < rowElements)
	{
		unsigned int sum = 0;
		unsigned int i = 1;
		for (; i <= cyrcleCount; ++i)
		{
			sum += (cache[cacheIdx - i] /* + cache[cacheIdx]*/ + cache[cacheIdx + i] +
				cache[cacheIdx - cacheDimX * i - 1] + cache[cacheIdx - cacheDimX * i] + cache[cacheIdx - cacheDimX * i + 1] +
				cache[cacheIdx + cacheDimX * i - 1] + cache[cacheIdx + cacheDimX * i] + cache[cacheIdx + cacheDimX * i + 1]);
		}
		bluredPixels[gidx] = (float)sum / (float)(i * 8);
	}
}

int main(int argc, char* argv[])
{
	if (argc == 4)
	{
		if (!strcmp(argv[1], "b&w")) 
		{
			int width, height, channels;
			unsigned char* h_imgData = stbi_load(argv[2], &width, &height, &channels, 0);
			if (h_imgData == 0)
			{
				printf("Error loading the image from disk");
				return 0;
			}
			int size = width * height * channels;
			unsigned char* d_resultData;
			unsigned char* h_resultData = (unsigned char*)malloc(width*height);
			uchar3* d_imgData;
			//if (channels == 3)
			//{
			//	unsigned char* cH_imgData = (unsigned char*)malloc(width * height * 4);
			//	for (unsigned int i = 0; i < width * height; ++i)
			//	{
			//		((uchar4*)cH_imgData)[i].x = ((uchar3*)h_imgData)[i].x;
			//		((uchar4*)cH_imgData)[i].y = ((uchar3*)h_imgData)[i].y;
			//		((uchar4*)cH_imgData)[i].z = ((uchar3*)h_imgData)[i].z;
			//		((uchar4*)cH_imgData)[i].w = 1;
			//	}
			//	free(h_imgData);
			//	h_imgData = cH_imgData;
			//}
			
			
			//unsigned char* r_imgData, *g_imgData, *b_imgData;
			//cudaMalloc(&r_imgData, width * height);
			//cudaMalloc(&g_imgData, width * height);
			//cudaMalloc(&b_imgData, width * height);
			cudaMalloc(&d_resultData, size / channels);
			cudaMalloc(&d_imgData, width * height * 3);
			cudaMemcpy(d_imgData, h_imgData, width * height * 3, cudaMemcpyHostToDevice);
			//dim3 blocks(((width + 16 - 1) / 16), (height + 16 - 1) / 16, 1);
			//dim3 threads(16, 16, 1);
			//dim3 blocks(((width + 32 - 1) / 32), (height + 8 - 1) / 8, 1);
			//dim3 threads(32, 8, 1);
			dim3 blocks(((width*height + 256 - 1) / 256), 1, 1);
			dim3 threads(256, 1, 1);
			LARGE_INTEGER freq;
			LARGE_INTEGER first_count;
			LARGE_INTEGER second_count;
			QueryPerformanceFrequency(&freq);
			QueryPerformanceCounter(&first_count);
			//RGBToGreyScale2D <<< blocks, threads >>> (d_imgData, d_resultData, width, height);
			//RGBToGreyScale <<< blocks, threads >>> ((uchar3*)d_imgData, d_resultData, width*height);
			//const unsigned int sharedMemSize = (threads.x * 3 * 4);
			//CachedRGBToGreyScale <<< blocks, threads,  sharedMemSize>>> ((uchar4*)d_imgData, d_resultData, width*height);
			//blocks.x = (((width * height * 3) / 4) + 256 - 1) / 256;
			//NewRGBToGreyScale <<< blocks, threads >>> (d_imgData, d_resultData, (width * height * 3) / 4);
			blocks.x = (((width * height) / 4) + 256 - 1) / 256;
			CachedRGB12ToGreyScale <<< blocks, threads, threads.x*3*4 >>> ((uchar4*)d_imgData, (uchar4*)d_resultData, (width*height)/4);
			//blocks.x = (((width * height) / 16) + 256 - 1) / 256;
			//RGB16ToGreyScale <<< blocks, threads >>> (d_imgData, d_resultData, (width*height)/16);
			//RGBASeparateIntoChannels <<< blocks, threads, threads.x*3 >>> (d_imgData, r_imgData, g_imgData, b_imgData, width * height);
			//RToGreyScale <<< blocks, threads >>> (r_imgData, d_resultData, width*height);
			//GToGreyScale <<< blocks, threads >>> (g_imgData, d_resultData, width*height);
			//BToGreyScale <<< blocks, threads >>> (b_imgData, d_resultData, width*height);
			cudaDeviceSynchronize();
			QueryPerformanceCounter(&second_count);
			long long counts = second_count.QuadPart - first_count.QuadPart;
			double ms = 1000 * ((double)counts / (double)freq.QuadPart);
			printf("\n%f ms \n", ms);
			cudaMemcpy(h_resultData, d_resultData, width*height, cudaMemcpyDeviceToHost);
			if (strstr(argv[3], ".png") || strstr(argv[3], ".PNG"))
				stbi_write_png(argv[3], width, height, 1, h_resultData, width * 1);
			else if (strstr(argv[3], ".jpg") || strstr(argv[3], ".JPG"))
				stbi_write_jpg(argv[3], width, height, 1, h_resultData, 80);
			cudaFree(d_imgData);
			cudaFree(d_resultData);
			//cudaFree(r_imgData);
			//cudaFree(g_imgData);
			//cudaFree(b_imgData);
			free(h_resultData);
			stbi_image_free(h_imgData);
		}
		else if (!strcmp(argv[1], "bl"))
		{
			int width, height, channels;
			unsigned char* h_imgData = stbi_load(argv[2], &width, &height, &channels, 0);
			if (h_imgData == 0)
			{
				printf("Error loading the image from disk");
				return 0;
			}
			int size = width * height;
			unsigned char* d_sortedChannelsR; unsigned char* d_sortedChannelsG; unsigned char* d_sortedChannelsB;
			unsigned char* d_blurredChannelsR; unsigned char* d_blurredChannelsG; unsigned char* d_blurredChannelsB;
			cudaMalloc(&d_sortedChannelsR, size); cudaMalloc(&d_sortedChannelsG, size); cudaMalloc(&d_sortedChannelsB, size);
			cudaMalloc(&d_blurredChannelsR, size * channels); cudaMalloc(&d_blurredChannelsG, size * channels); cudaMalloc(&d_blurredChannelsB, size * channels);
			//unsigned char* dr_imgData, *dg_imgData, *db_imgData;
			//cudaMalloc(&dr_imgData, size);
			//cudaMalloc(&dg_imgData, size);
			//cudaMalloc(&db_imgData, size);
			//unsigned char* dr_blurredimgData, *dg_blurredimgData, *db_blurredimgData;
			//cudaMalloc(&dr_blurredimgData, size);
			//cudaMalloc(&dg_blurredimgData, size);
			//cudaMalloc(&db_blurredimgData, size);
			uchar3* d_imgData;
			cudaMalloc(&d_imgData, size * channels);
			cudaMemcpy(d_imgData, h_imgData, size * channels, cudaMemcpyHostToDevice);
			//dim3 blocks(((width + 16 - 1) / 16), (height + 16 - 1) / 16, 1);
			//dim3 threads(16, 16, 1);
			dim3 blocks(((width + 32 - 1) / 32), (height + 8 - 1) / 8, 1);
			dim3 threads(32, 8, 1);
			const int transBlocks = (size + 256 - 1) / 256;
			LARGE_INTEGER freq;
			LARGE_INTEGER first_count;
			LARGE_INTEGER second_count;
			QueryPerformanceFrequency(&freq);
			QueryPerformanceCounter(&first_count);
			//RGBSeparateIntoChannels <<< transBlocks, 256 >>> (d_imgData, d_sortedChannelsR, d_sortedChannelsG, d_sortedChannelsB, size);
			const unsigned int cBlocks = ((size / 4) + 256 - 1) / 256;
			dim3 cThreads(256, 1, 1);
			RGB12SeparateIntoChannelsCache <<< cBlocks, cThreads, 3*4*cThreads.x >>> ((uchar4*)d_imgData, (uchar4*)d_sortedChannelsR, (uchar4*)d_sortedChannelsG, (uchar4*)d_sortedChannelsB, size/4);
			//RGBSeparateIntoChannels <<< transBlocks, 256 >>> (d_imgData, dr_imgData, dg_imgData, db_imgData, size);
			//ChannelUnweightedImageBlur <<< blocks, threads, 18*18 >>> (d_sortedChannels, d_blurredChannels, width, height);
			//ChannelUnweightedImageBlur <<< blocks, threads, 18*18 >>> (d_sortedChannels + size, d_blurredChannels + size, width, height);
			//ChannelUnweightedImageBlur <<< blocks, threads, 18*18 >>> (d_sortedChannels + size * 2, d_blurredChannels + size * 2, width, height);
			const unsigned int kernelRadius = 2;
			float h_kernel[] = {
				//kernel used for gaussian blur
				0.00f,  0.01f,  0.01f,  0.01,  0.00f,
				0.01f,  0.05f,  0.11f,  0.05f,  0.01f,
				0.01f,  0.11f,  0.24f,  0.11f,  0.01f,
				0.01f,  0.05f,  0.11f,  0.05f,  0.01f,
				0.00f,  0.01f,  0.01f,  0.01f,  0.00f
			};
			//0.001f 0.224f 0.49f 0.001f 0.224f
			float* d_kernel;
			cudaMalloc(&d_kernel, _countof(h_kernel) * sizeof(float));
			cudaMemcpy(d_kernel, h_kernel, _countof(h_kernel) * sizeof(float), cudaMemcpyHostToDevice);
			//We are assuming threads.x == treads.y !!!!
			int sharedMemSize = (threads.x + kernelRadius) * (threads.y + kernelRadius);
			ChannelWeightedImageBlur4 <<< blocks, threads, sharedMemSize >>> (d_sortedChannelsR, d_blurredChannelsR, width, height, d_kernel, kernelRadius);
			ChannelWeightedImageBlur4 <<< blocks, threads, sharedMemSize >>> (d_sortedChannelsG, d_blurredChannelsG, width, height, d_kernel, kernelRadius);
			ChannelWeightedImageBlur4 << < blocks, threads, sharedMemSize >> > (d_sortedChannelsB, d_blurredChannelsB, width, height, d_kernel, kernelRadius);
			ChannelsUniteIntoRGB << < transBlocks, 256 >> > (d_blurredChannelsR, d_blurredChannelsG, d_blurredChannelsB, (uchar3*)d_imgData, size);
			//ChannelUnweightedImageBlurQuad <<< blocks, threads, sharedMemSize >>> (d_sortedChannels + size, d_blurredChannels + size, width, height, quadDim);
			//ChannelUnweightedImageBlurQuad <<< blocks, threads, sharedMemSize >>> (d_sortedChannels + size * 2, d_blurredChannels + size * 2, width, height, quadDim);
			//ChannelUnweightedImageBlurQuad <<< blocks, threads, sharedMemSize >>> (d_sortedChannels + size * 2, d_blurredChannels + size * 2, width, height, quadDim);
			//ChannelsUniteIntoRGB <<< transBlocks, 256 >>> (d_blurredChannels, d_blurredChannels + size, d_blurredChannels + size * 2, (uchar3*)d_imgData, size);
			cudaDeviceSynchronize();
			QueryPerformanceCounter(&second_count);
			cudaError_t error = cudaGetLastError();
			long long counts = second_count.QuadPart - first_count.QuadPart;
			double ms = 1000 * ((double)counts / (double)freq.QuadPart);
			printf("\n%f ms \n", ms);
			cudaMemcpy(h_imgData, d_imgData, size * channels, cudaMemcpyDeviceToHost);
			if (strstr(argv[3], ".png") || strstr(argv[3], ".PNG"))
				stbi_write_png(argv[3], width, height, channels, h_imgData, width * channels);
			else if (strstr(argv[3], ".jpg") || strstr(argv[3], ".JPG"))
				stbi_write_jpg(argv[3], width, height, channels, h_imgData, 80);
			cudaFree(d_imgData);
			cudaFree(d_sortedChannelsR); cudaFree(d_sortedChannelsG); cudaFree(d_sortedChannelsB);
			cudaFree(d_blurredChannelsR); cudaFree(d_blurredChannelsG); cudaFree(d_blurredChannelsB);
			//cudaFree(d_sortedChannels);
			//cudaFree(dr_imgData);
			//cudaFree(dg_imgData);
			//cudaFree(db_imgData);
			//cudaFree(dr_blurredimgData);
			//cudaFree(dg_blurredimgData);
			//cudaFree(db_blurredimgData);
			stbi_image_free(h_imgData);
		}
	}
	else
	{
		printf("First parameter is the desired effect (\"b&w\":blackAndWhite \"bl\":blur) second parameter is the source file and third parameter is the destination file");
	}
	return 0;
}