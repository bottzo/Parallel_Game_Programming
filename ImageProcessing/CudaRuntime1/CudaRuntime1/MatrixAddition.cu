#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Windows.h>

#include <stdio.h>
#include <stdlib.h>

#define N 8
#define BLOCKTHREADS 2

__global__ void MatrixAddition(float* d_a, float* d_b, float* d_c)
{
	int globalIndex = blockIdx.y * BLOCKTHREADS * N + blockIdx.x * BLOCKTHREADS + threadIdx.y * N + threadIdx.x;
	d_c[globalIndex] = d_a[globalIndex] + d_b[globalIndex];
}

void HostMatrixAddition(float* h_a, float* h_b, float* h_c)
{
	for (int i = 0; i < (N * N); ++i)
	{
		h_c[i] = h_a[i] + h_b[i];
	}
}

__global__ void MatrixMultiplication(float* d_a, float* d_b, float* d_c) 
{
	int globalIndex = blockIdx.y * BLOCKTHREADS * N + blockIdx.x * BLOCKTHREADS + threadIdx.y * N + threadIdx.x;
}

void HostMatrixMultiplication(float* h_a, float* h_b, float* h_c)
{
	//for (int i = 0; i < N; ++i)
	//{
	//	for ()
	//	{
	//
	//	}
	//}
}

//int main()
//{
//	LARGE_INTEGER freq;
//	LARGE_INTEGER first_count;
//	LARGE_INTEGER second_count;
//	QueryPerformanceFrequency(&freq);
//	QueryPerformanceCounter(&first_count);
//	float *h_a, *h_b, *h_c;
//	float *d_a, *d_b, *d_c;
//	int memsize = N * N * sizeof(float);
//
//	h_a = (float*)malloc(memsize);
//	h_b = (float*)malloc(memsize);
//	h_c = (float*)malloc(memsize);
//
//	cudaError error = cudaMalloc(&d_a, memsize);
//	if (error != cudaSuccess)
//	{
//		printf("Error allocating device memory");
//		return -1;
//	}
//	error = cudaMalloc(&d_b, memsize);
//	if (error != cudaSuccess)
//	{
//		printf("Error allocating device memory");
//		return -1;
//	}
//	error = cudaMalloc(&d_c, memsize);
//	if (error != cudaSuccess)
//	{
//		printf("Error allocating device memory");
//		return -1;
//	}
//
//	for (int i = 0; i < N * N; ++i)
//	{
//		h_a[i] = 1.0f;
//		h_b[i] = 1.0f;
//	}
//
//	error = cudaMemcpy(d_a, h_a, memsize, cudaMemcpyHostToDevice);
//	if (error != cudaSuccess)
//	{
//		printf("Error memcpy host to device");
//		return -1;
//	}
//	error = cudaMemcpy(d_b, h_b, memsize, cudaMemcpyHostToDevice);
//	if (error != cudaSuccess)
//	{
//		printf("Error memcpy host to device");
//		return -1;
//	}
//
//	dim3 block(N / BLOCKTHREADS, N / BLOCKTHREADS);
//	dim3 threads(BLOCKTHREADS, BLOCKTHREADS);
//	MatrixAddition<<< block, threads >>> (d_a, d_b, d_c);
//
//	error = cudaMemcpy(h_c, d_c, memsize, cudaMemcpyDeviceToHost);
//	if (error != cudaSuccess)
//	{
//		printf("Error memcpy device to host");
//		return -1;
//	}
//
//	//print the result
//	printf("The result is:\n");
//	/*for (int i = 0; i < N; ++i)
//	{
//		for (int j = 0; j < N; ++j)
//		{
//			printf("%f, ", h_c[i*N + j]);
//		}
//		printf("\n");
//	}*/
//	for (int i = 0; i < N*N; ++i)
//	{
//		if (i % N == 0)
//			printf("\n");
//		printf("%f, ", h_c[i]);
//	}
//
//	free(h_a);
//	free(h_b);
//	free(h_c);
//	cudaFree(d_a);
//	cudaFree(d_b);
//	cudaFree(d_c);
//
//	QueryPerformanceCounter(&second_count);
//	long long counts = second_count.QuadPart - first_count.QuadPart;
//	double ms = 1000 * ((double)counts / (double)freq.QuadPart);
//	printf("\n%f, \n", ms);
//
//	return 0;
//}