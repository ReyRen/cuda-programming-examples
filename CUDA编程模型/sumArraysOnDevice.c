#include <stdio.h>
#include <string.h>
#include <time.h>

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
	for(int idx = 0; idx < N; idx++) {
		C[idx] = A[idx] + B[idx];
	}
}

void initialData(float *ip, int size) {
	// seed
	time_t t;
	srand((unsigned int)time(&t));

	for(int i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xFF)/10.0f; // 直接取低8位(应该就是0~25.5的浮点)
	}
}

int main(int argc, char **argv) {
	int nElem = 1024;
	size_t nBytes = nElem * sizeof(float);


	float *h_A, *h_B, *h_C;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	h_C = (float *)malloc(nBytes);

	float *d_A, *d_B, *d_C;
	d_A = cudaMalloc((float**)&d_A, nBytes);
	d_B = cudaMalloc((float**)&d_A, nBytes);
	d_C = cudaMalloc((float**)&d_A, nBytes);

	initialData(h_A, nElem);
	initialData(h_B, nElem);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	/*
	 *	数据被转移到GPU全局内存后，主机端调用核函数在GPU上进行数组求和。
	 *	一旦内核被调用，控制权立刻被传回主机。这样的话，当核函数在GPU上运行时，
	 *	主机可以执行其他函数。内核和主机是异步的。
	 * */

	sumArraysOnHost(d_A, d_B, d_C, nElem);

	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	free(h_A);
	free(h_B);
	free(h_C);

	cudaFree(d_A)
	cudaFree(d_B)
	cudaFree(d_C)

	return 0;
}
