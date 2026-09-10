#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int* a, const int* b, int* c, int n)
{
    //      floor    *   rooms per floor + which room
    // i is now global unique room number
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

void initializeVectorsCPU(std::vector<int>& a, std::vector<int>& b)
{
    for (int i = 0; i < a.size(); i++)
    {
        a[i] = i;
        b[i] = i * 10;
    }
}

int main()
{
    // CPU arrays
    const int n = 1000;
    size_t bytes = n * sizeof(int);

    std::vector<int> a_vec(n);
    std::vector<int> b_vec(n);
    std::vector<int> c_vec(n);

    initializeVectorsCPU(a_vec, b_vec);

    // GPU pointers
    int* d_a;
    int* d_b;
    int* d_c;

    // allocate GPU memory
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // copy A CPU → GPU
    cudaMemcpy(d_a, a_vec.data(), bytes, cudaMemcpyHostToDevice);

    // copy B CPU → GPU
    cudaMemcpy(d_b, b_vec.data(), bytes, cudaMemcpyHostToDevice);

    // launch kernel
    const int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // wait for GPU
    cudaDeviceSynchronize();

    // copy C GPU → CPU
    cudaMemcpy(c_vec.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // print C
    for (int i = 0; i < n; i++)
    {
        if (c_vec[i] != i * 11)
        {
            std::cout << "Houston we have a problem!" << std::endl;
            std::cout << c_vec[i] << std::endl;
            return -1;
        }
    }

    std::cout << "All results correct!! Yayyyyyyy" << std::endl;

    // free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}