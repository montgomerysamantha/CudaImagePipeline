#include <iostream>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

template <typename Func>
float timeCuda(Func func)
{
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    func();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedMs;
}

template <typename Func>
float timeCpu(Func func)
{
    auto cpuStart = std::chrono::high_resolution_clock::now();

    func();

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = cpuEnd - cpuStart;

    return static_cast<float>(duration.count());
}

__global__ void vectorAdd(const int* a, const int* b, int* c, int n)
{
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

void vectorAddCPU(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c)
{
    for (int i = 0; i < a.size(); i++)
    {
        c[i] = a[i] + b[i];
    }
}

float timeVectorAddCPU(const std::vector<int>& a, const std::vector<int>& b)
{
    std::vector<int> c_cpu(a.size());
    float cpuTime = timeCpu([&]()
    {
        vectorAddCPU(a, b, c_cpu);
    });

    return cpuTime;
}

int main()
{
    // CPU arrays
    const int n = 10'000'000;
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

    // copy A and B CPU → GPU
    // calc time taken
    float hostToDeviceTime = timeCuda([&]()
    {
        cudaMemcpy(d_a, a_vec.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_vec.data(), bytes, cudaMemcpyHostToDevice);

    });

    // launch kernel
    const int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float gpuTime = timeCuda([&]()
    {
        vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    });

    // wait for GPU
    cudaDeviceSynchronize();

    // copy C GPU → CPU, calc time taken
    float deviceToHostTime = timeCuda([&]()
    {
        cudaMemcpy(c_vec.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    });

    // free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    float cpuTime = timeVectorAddCPU(a_vec, b_vec);

    // check GPU calc results
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
    std::cout << "CPU Compute:        " << cpuTime << " ms\n";

    std::cout << "Host to Device Time:" << hostToDeviceTime << " ms\n";
    std::cout << "GPU Compute:        " << gpuTime << " ms\n";
    std::cout << "Device to Host Time:" << deviceToHostTime << " ms\n";
    std::cout << "GPU Total:          " << hostToDeviceTime + gpuTime + deviceToHostTime << " ms\n";


    return 0;
}