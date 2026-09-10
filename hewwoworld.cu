#include <iostream>
#include <cuda_runtime.h>

__global__ void hewwoFromGPU()
{
    printf("Hewwo from GPU thread %d!\n", threadIdx.x);
}

int main()
{
    std::cout << "Hewwo from CPU!!\n";

    hewwoFromGPU<<<1, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}