在编程和算法设计的过程中，最关心的应是在领域曾如何解析数据和函数，以便在并行环境中能正确高效的进行。当进入编程阶段，关注点转向如何组织并发
线程。CUDA提出线程层次结构抽象的概念，允许控制线程行为。


### CUDA编程结构

**主机**： CPU及其内存(主机内存)

**设备**: GPU及其内存(设备内存)

从6.0开始，NVIDIA提出Unified Memory的编程模型，连接了主机内存和设备内存，可以用单个指针访问CPU和GPU内存，无须彼此手动拷贝。

但是目前应学会的是如何为主机和设备分配内存空间以及如何在CPU和GPU之间拷贝共享数据。这种程序员管理模式控制下的内存和数据可以优化应用
程序并实现硬件系统利用率的最大化。

**kernel**： 核函数是代码在GPU上运行。

多数情况下，主机可以独立的对设备进行操作。内核一旦被启动，管理权立刻返回给主机，释放CPU来执行设备上运行的并行代码实现的额外的任务。
CUDA编程模式是异步的，因此GPU上进行运算的同时也可以主机和设备通讯。一个典型的CUDA程序包括并行代码（GPU）互补的串行代码（CPU）。

**一个典型的cuda程序**:

1. 数据从CPU内存拷贝到GPU内存
2. 调用核函数对存储在GPU内存中的数据进行操作
3. 将数据从GPU内存传回CPU内存

如实例:

[sumArraysOnHost](https://github.com/ReyRen/cuda-programming-examples/blob/master/CUDA%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B/sumArraysOnHost.c)

[sumArrayOnDevice](https://github.com/ReyRen/cuda-programming-examples/blob/master/CUDA%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B/sumArraysOnDevice.c)

### 线程管理

由一个内核启动所产生的所有线程统称为一个网格。同一网格中的所有线程共享相同的全局内存空间。一个网格由多个线程块构成，一个线程块包含一组线程，
同一线程块内的线程协作可以通过同步、共享内存实现。不同块内的线程不能协作。

线程依靠blockIdx.xyz(线程块在线程格内的索引)和threadIx.xyz(块内的线程索引)这两个坐标变量进行区分。

通常，一个线程格会被组织成线程块的二维数组形式，一个线程块会被组织成线程的三维数组形式。

如案例[检查网络和块的索引和维度](https://github.com/ReyRen/cuda-programming-examples/blob/master/CUDA%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B/sumArrayOnGPU-small-case.cu)

**从主机端和设备端访问网络/块变量**:

区分主机端和设备端的网格和块变量的访问是很重要的。例如，声明一个主机端的块变量，按照如下定义其坐标并对其进行访问：

block.x, block.y, block.z

在设备端，已经预定义了内置块变量的大小：

blockDim.x, blockDim.y, blockDim.z

总之，在启动内核之前就定义了主机端的网格和块变量，并从主机端通过由x, y, z三个字段决定的矢量结构来访问它们。当内核启动时，可以使用内核中预初始化的内置变量。

### 启动一个核函数

<<<>>>内是核函数的执行配置

加入有32个元素用于计算，每8个元素一个块，需要启动4个块

```
kernel_name <<<grid,block>>>(arg list);
kernel_name <<<4,8>>>(arg list);
grid维度表示有多少个block, block的维度表示有多少个thread
```
不同于C语言的函数调用，所有CUDA核函数的启动都是异步的。CUDA内核调用完成后，控制权立刻返回给CPU. 但一些CUDA运行时API在主机和设备之间是隐式同步的，比如cudaMemcpy。

### 编写核函数

核函数是在设备端执行的代码。在核函数中，需要为一个线程规定要进行的计算以及要进行的数据访问。当核函数被调用时，许多不同的CUDA线程并行执行同一个计算任务。
用`__global__`来定义核函数

### 如何理解threadIdx、 blockIdx、 blockDim、gridDim

**threadIdx(.x/.y/.z代表几维索引)**: 线程所在block中各个维度上的线程号

**blockIdx(.x/.y/.z代表几维索引)**: 块所在grid中各个维度上的块号

**blockDim(.x/.y/.z代表各个维度上的block大小)**: block的大小也就是block中线程的数量，blockDim.x表示块中x轴上的线程数量，blockDim.y表示y轴上的线程数量，blockDim.z表示z轴上的线程数量

**gridDim(.x/.y/.z代表各维度上grid的大小)**: grid的大小也就是grid中block的数量，gridDim.x表示grid中x轴上块的数量, gridDim.y表示grid中y轴上的数量..

程序和总所定义的
```
dim3 grid(a, b, c);
dim3 block(d, e, f);
```
表示的就是blockDim.x = d, blockDim.y = e, blockDim.z = f; grid.x = a, grid.y = b, grid.z = c;

所有的Idx的序列号都是从左向右的！！！！

来自官方的一些案例，如何计算threadIdx:

**1D grid of 1D blocks**:

```
dim3 block(N);
dim3 grid(M);

__device__
int getGlobalIdx_1D_1D() {
    return blockIdx.x*blockDim.x 
           + threadIdx.x;
}
```

**1D grid of 2D blocks**:

```
dim3 block(N1, N2);
dim3 grid(M);

__device__
int getGlobalIdx_1D_2D() {
    return blockIdx.x*blockDim.x*blockDim.y 
           + threadIdx.y*blockDim.x 
           + threadIdx.x;
}
```

**1D grid of 3D blocks**:
```
dim3 block(N1, N2, N3);
dim3 grid(M);

__device__
int getGlobalIdx_1D_3D() {
   return blockIdx.x*blockDim.x*blockDim.y*blockDim.z
          + threadIdx.z*blockDim.y*blockDim.x
          + threadIdx.y*blockDim.x
          + threadIdx.x; // 3d是从最底下的一层开始的
}
```

**2D grid of 1D blocks**:
```
dim3 block(N);
dim3 grid(M1, M2);

__device__
int getGlobalIdx_2D_1D() {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}
```

**2D grid of 2D blocks**:
```
dim3 block(N1, N2);
dim3 grid(M1, M2);

__device__
int getGlobalIdx_2D_2D() {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId*blockDim.x*blockDim.y
                   + threadIdx.y*blockDim.x
                   + threadIdx.x;
    return threadId;
}
```

**2D grid of 3D blocks**:
```
dim3 block(N1, N2, N3);
dim3 grid(M1, M2);

__device__
int getGlobalIdx_2D_3D() {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId*blockDim.x*blockDim.y*blockDim.z
                   + threadIdx.z*blockDim.x*blockDim.y
                   + threadIdx.y*blockDim.x
                   + threadIdx.x;
    return threadId;
}
```

**3D grid of 1D blocks**:
```
dim3 block(N);
dim3 grid(M1, M2, M3);

__device__
int getGlobalIdx_3D_1D() {
    int blockId = gridDim.x * gridDim.y * blockIdx.z
                  + blockIdx.y * gridDim.x
                  + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}
```

**3D grid of 2D blocks**:
```
dim3 block(N1, N2);
dim3 grid(M1, M2, M3);

__device__
int getGlobalIdx_3D_2D() {
    int blockId = gridDim.x * gridDim.y * blockIdx.z
                  + blockIdx.y * gridDim.x
                  + blockIdx.x;
    int threadId = blockId * blockDim.x * blockDim.y // 原则就是把上一个blockId的所有线程数先加起来，然后再层层加现在的
                   + threadIdx.y * blockDim.x
                   + threadIdx.x;
    return threadId;
}
```

**3D grid of 3D blocks**:
```
dim3 block(N1, N2, N3);
dim3 grid(M1, M2, M3);

__device__
int getGlobalIdx_3D_3D() {
    int blockId = gridDim.x * gridDim.y * blockIdx.z
                  + blockIdx.y * gridDim.x
                  + blockIdx.x;
    int threadId = blockId * blockDim.x * blockDim.y * blockDim.z
                   + blockDim.x * blockDim.y * threadIdx.z
                   + threadIdx.x * threadIdx.y
                   + threadIdx.x;
    return threadIdx;
}
```
