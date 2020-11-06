#include <stdio.h>

// __global__ 告诉编译器这个函数会从CPU中调用，然后在GPU上执行
__global__ void helloFromGPU(void)
{
	printf("Hello world from GPU!\n");
}

int main(void) {
	printf("Hello world from CPU!\n");

//	<<<意为着主线程到设备端代码的调用。一个内核函数通过一组线程来执行，所有线程执行相同的代码。10个GPU线程被调用。
	helloFromGPU <<<1, 10>>>();
//	显示的释放和清空当前进程中与当前设备有关的所有资源
	cudaDeviceReset();

	return 0;
}

