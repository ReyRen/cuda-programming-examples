#include <stdio.h>
#include <cuda_runtime.h>

__global__ void checkIndex(void) {
/*
  在核函数中，每个线程都输出自己的线程索引，块索引，块维度和网格维度
*/

           printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
                  "gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
                  blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
                  gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv) {
    // 定义总的数据量
    int nElem = 6;

    // define grid and block structure
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host side
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z); // 2 1 1
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z); // 3 1 1

    // check grid and block dimension from device side
    checkIndex <<<grid, block>>>();

    // reset device before you leave;
    cudaDeviceReset();

    return(0);

}