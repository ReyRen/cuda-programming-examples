### GPU架构

当启动一个内核网格时，它的线程块被分布在了可用的SM上执行。多个block可能分布在同一个SM上。

CUDA采用单指令多线程(SIMD)架构管理和执行线程。每32个线程为一组称为线程束(warp)。wrap中的所有线程同时执行相同的指令。
每个线程都有自己的指令地址计数器和寄存器状态，利用自身的数据执行当前的指令。每个SM都将分配给它的线程块分到包含32个线程
的wrap中，然后在可用的硬件资源上进行调度执行。

### SIMD vs SIMT

SIMD是单指令多数据。两者都是将相同的指令广播给多个执行单元来实现并行。

SIMD要求同一个向量中的所有元素要在一个统一的同步组中一起执行

SIMT允许属于统一wrap的多个线程独立执行，尽管一个wrap中的所有线程在相同的程序地址上同时开始执行，但是单独的线程仍有可能不同的行为。

SM和SP是物理概念，block、grid、wrap等是CUDA的软件概念。

一个block只能在一个SM上被调度。一旦block在一个SM上被调度，就会一直保存在该SM上直到执行完成。在同一时间，一个SM可以容纳多个线程块

关于wrap推荐一篇很精彩的文章[理解线程束执行的本质](https://face2ai.com/CUDA-F-3-2-%E7%90%86%E8%A7%A3%E7%BA%BF%E7%A8%8B%E6%9D%9F%E6%89%A7%E8%A1%8C%E7%9A%84%E6%9C%AC%E8%B4%A8-P1/)

SM是GPU架构的核心。寄存器和共享内存是SM中的稀缺资源。CUDA将这些资源分配到SM中的所有常驻线程里。因此，这些有限的稀缺资源限制了在SM上活跃的线程束的数量，
活跃的线程束数量对应于SM上的并行量。

### 动态并行

到目前为止，所有kernel都在host端调用，GPU的工作完全在CPU的控制下。CUDA Dynamic Parallelism允许GPU kernel在device端创建调用。Dynamic Parallelism使递归更容易实现
和理解，由于启动的配置可以由device上的thread在运行时决定，这也减少了host和device之间传递数据和执行控制。通过动态并行性，可以直到程序运行时才推迟确定在GPU上创建
有多少块和网格。

**待定中，这个我也在研究中。。。。。**

### nvcc简单描述

在编译CUDA代码时，我们需要向nvcc提供我们想为哪个显卡架构编译我们的代码。然而由于CUDA代码特殊的编译过程，nvcc为我们提供了`-arch`、`-code`、`-gencode`
三个不同的编译选项。nvcc采用了一种两阶段编译过程，cuda代码首先被编译为一种面向虚拟架构(virtual architecture)的ptx代码，然后在第二阶段中将ptx代码面向具体的实际架构
(real architecture)编译为实际可执行的二进制代码

不同的架构包含了不同的功能，架构代码号越高，包含的功能越多。在向nvcc指定架构时，我们所指定的实际架构代码必须兼容于所指定的虚拟架构代码。

**-arch选项**:

用于向nvcc指定在第一阶段中使用什么虚拟架构，可用的选项包括：

- compute_30, compute_32
- compute_35
- compute_50, compute_52, compute_53
- compute_60, compute_61, compute_62
- compute_70, compute_72
- compute_75
- …

**-code选项**:

用于向nvcc将什么代码放入最后的目标文件中。它可用于指定nvcc在第二阶段中使用什么实际架构：

- sm_30, sm_32
- sm_35
- sm_50, sm_52, sm_53
- sm_60, sm_61, sm_62
- sm_70, sm_72
- sm_75
- ...

**-arch配合-code**:

-arch和-code选项的组合可以指定nvcc在编译过程中使用的虚拟架构，以及目标文件中包含哪些虚拟架构代码及实际架构代码，比方说：
```
-arch=compute_20 -code=sm_20
```
nvcc将以compute_20为虚拟架构产生ptx代码，所产生的目标文件将包含面向sm_20实际架构的二进制代码。

**-gencode选项**:

在使用-arch和-code时，我们能够指定不同的实际架构，但是只能指定一种虚拟架构。有时候我们希望nvcc在编译过程中使用不同的虚拟架构，
并在目标文件中包含面向多种虚拟架构的ptx代码，以及面向多种实际架构的二进制代码，此时我们可以使用-gencode达成这一目标，比方说
```
-gencode=arch=compute_50,code=sm_50
-gencode=arch=compute_52,code=sm_52
-gencode=arch=compute_60,code=sm_60
-gencode=arch=compute_61,code=sm_61
-gencode=arch=compute_61,code=compute_61
```
目标文件将包含：

- 基于compute_50ptx代码产生的sm_50二进制代码
- 基于compute_52ptx代码产生的sm_52二进制代码
- 基于compute_60ptx代码产生的sm_60二进制代码
- 基于compute_61ptx代码产生的sm_61二进制代码
- compute_61ptx代码

参考官方：[virtual-arch-feature-list](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)

还有一篇将的很棒的文章[Matchign CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

### 理解线程束执行的本质

线程束时SM中基本的执行单位。当一个block的grid启动后，grid中的block分布在SM中。一旦线程被调度到一个SM上，block中的线程会进一步划分为wrap。一个wrap由32个连续
的线程组成，所有的线程按照单指令多线程(SIMT)的方式执行：也就是说，wrap中的所有的线程都执行相同的指令，每个线程在私有数据上操作。
wrap不会在不同的线程块之间分离。如果线程块的大小不是wrap大小的偶数倍，那么在最后的wrap里，有些线程就不会活跃。

总结一下，从逻辑角度来看，线程块时线程的集合，它们可以被组织为一二三维布局。从硬件角度来看，线程块时一维线程束的集合。在线程块中线程被组织成一维布局，每32个连续线程
组成一个wrap。
    
### 资源分配

每个SM都有32位的寄存器组，它存储在寄存器文件中，并且可以在线程中进行分配，同时固定数量的共享内存用来在线程块中进行分配。对于一个给定的内核，同时存在于同一个SM中的
线程块和线程束的数量取决于SM中可用的且内核所需的寄存器和共享内存数量。
每个SM中寄存器和共享内存的数量因设备拥有的不同计算能力而不同。如果每个SM没有足够的寄存器或共享内存去处理至少一个块，那么内核将无法启动。

为了隐藏由线程束阻塞造成的延迟，需要让大量的线程束保持活跃。(选定的线程束，阻塞的线程束，符合条件的线程束)
