#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)
{
  const cudaError_t error = call;
  if(error != cudaSuccess)
  {
    printf("Error: %s:%d, ", __FILE__, __LINE__);
    printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));
    exit(1);
  }
}

void initialData(float *ip, int size) {
     // generate different seed for random number
     time_t t;
     srand((unsigned)time(&t));

     for(int i = 0; i < size; i++) {
             ip[i] = (float)(rand() & 0xFF)/10.0f; // 0-25.5
     }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
     for(int idx = 0; idx < N; idx++) {
             C[idx] = A[idx] + B[idx];
     }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
     int i = threadIdx.x;
     C[i] = A[idx] + B[idx];
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0])

    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    // set up data size of vectors
    int nElem = 32;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    dim3 block(nElem); // 表示每个block中有多少个线程
    dim3 grid(nElem/block.x); // 表示每个grid有多少个block
    // 向量大小为32，执行配置被放在一个块内，其中包含32个元素

    sumArraysOnGPU<<<grid,block>>>(d_A, d_B, d_C);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0
}