#include "benchmarks/benchmark_utils.hpp"
#include "core/cuda_check.hpp"
#include "core/device_image.hpp"
#include "filters/gaussian_blur.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

constexpr int RUNS = 20;

struct BenchmarkContext
{
    HostImage input;
    HostImage gpuOutput;
    HostImage cpuOutput;

    DeviceImage deviceInput;
    DeviceImage deviceOutput;

    cudaStream_t stream = nullptr;

    dim3 threads{16, 16};
    dim3 blocks{1, 1};
};

struct BenchmarkResults
{
    bool resultsMatch = false;

    int width = 0;
    int height = 0;
    int channels = 0;
    int runs = 0;

    float cpuMs = 0.0f;
    float naiveKernelMs = 0.0f;
    float sharedKernelMs = 0.0f;

    float uploadMs = 0.0f;
    float downloadMs = 0.0f;
    float gpuEndToEndMs = 0.0f;
};

__host__ __device__
void blurPixel(
    int x,
    int y,
    int width,
    int height,
    const unsigned char* input,
    unsigned char* output)
{
    int kernel[3][3] =
    {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    int pixel = y * width + x;
    int pixelIndex = pixel * 3;

    int rSum = 0;
    int gSum = 0;
    int bSum = 0;
    int weightSum = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int neighborX = x + dx;
            int neighborY = y + dy;

            if (neighborX >= 0 && neighborX < width &&
                neighborY >= 0 && neighborY < height)
            {
                int weight = kernel[dy + 1][dx + 1];

                int neighborPixel =
                    neighborY * width + neighborX;

                int neighborPixelIndex =
                    neighborPixel * 3;

                int r = input[neighborPixelIndex];
                int g = input[neighborPixelIndex + 1];
                int b = input[neighborPixelIndex + 2];

                rSum += r * weight;
                gSum += g * weight;
                bSum += b * weight;

                weightSum += weight;
            }
        }
    }

    output[pixelIndex] = rSum / weightSum;
    output[pixelIndex + 1] = gSum / weightSum;
    output[pixelIndex + 2] = bSum / weightSum;
}

__global__ void gaussianBlurGpuNaive(int width, int height, const unsigned char* input, unsigned char* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
       blurPixel(x, y, width, height, input, output);
    }
}

void gaussianBlurCpu(int width, int height, const unsigned char* input, unsigned char* output)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            blurPixel(x, y, width, height, input, output);
        }
    }
}

void setupBenchmark(
    BenchmarkContext& context,
    const std::string& imagePath)
{
    context.input = loadRgbImage(imagePath);

    context.gpuOutput.width = context.input.width;
    context.gpuOutput.height = context.input.height;
    context.gpuOutput.channels = context.input.channels;
    context.gpuOutput.pixels.resize(
        context.input.pixels.size()
    );

    context.cpuOutput.width = context.input.width;
    context.cpuOutput.height = context.input.height;
    context.cpuOutput.channels = context.input.channels;
    context.cpuOutput.pixels.resize(
        context.input.pixels.size()
    );

    context.deviceInput.allocate(
        context.input.width,
        context.input.height,
        context.input.channels
    );

    context.deviceOutput.allocate(
        context.input.width,
        context.input.height,
        context.input.channels
    );

    CUDA_CHECK(cudaStreamCreate(&context.stream));

    const HostImage& readOnlyInput = context.input;

    context.deviceInput.uploadAsync(
        readOnlyInput.view(),
        context.stream
    );

    context.blocks = dim3(
        (context.input.width + context.threads.x - 1)
            / context.threads.x,

        (context.input.height + context.threads.y - 1)
            / context.threads.y
    );

    // Make sure setup is complete before benchmarking begins.
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
}

void warmUpGpu(BenchmarkContext& context)
{
    gaussianBlurGpuNaive<<<
        context.blocks,
        context.threads,
        0,
        context.stream>>>(
            context.input.width,
            context.input.height,
            context.deviceInput.view().data,
            context.deviceOutput.view().data
        );

    CUDA_CHECK(cudaGetLastError());

    const DeviceImage& readOnlyInput =
        context.deviceInput;

    launchGaussianBlur(
        readOnlyInput.view(),
        context.deviceOutput.view(),
        context.stream
    );

    CUDA_CHECK(cudaStreamSynchronize(context.stream));
}

void warmUpCpu(BenchmarkContext& context)
{
    gaussianBlurCpu(
        context.input.width,
        context.input.height,
        context.input.pixels.data(),
        context.cpuOutput.pixels.data()
    );
}

void printResults(const BenchmarkResults& results)
{
    const std::size_t imageBytes =
        static_cast<std::size_t>(results.width) *
        results.height *
        results.channels;

    const float imageMiB =
        static_cast<float>(imageBytes) /
        (1024.0f * 1024.0f);

    const float transferMs =
        results.uploadMs + results.downloadMs;

    const float transferPercent =
        100.0f * transferMs / results.gpuEndToEndMs;

    const float kernelPercent =
        100.0f * results.sharedKernelMs /
        results.gpuEndToEndMs;

    const float overheadPercent =
        100.0f - transferPercent - kernelPercent;

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\nGaussian Blur Benchmark\n";
    std::cout << "=======================\n";
    std::cout << "Image:        "
              << results.width << " x "
              << results.height << " RGB\n";
    std::cout << "Image size:   " << imageMiB << " MiB\n";
    std::cout << "Runs:         " << results.runs << '\n';
    std::cout << "Correctness:  "
              << (results.resultsMatch ? "PASS" : "FAIL")
              << "\n\n";

    std::cout << "Compute-only\n";
    std::cout << "------------\n";
    std::cout << "CPU reference:       "
              << results.cpuMs << " ms\n";
    std::cout << "GPU naive kernel:     "
              << results.naiveKernelMs << " ms\n";
    std::cout << "GPU shared kernel:    "
              << results.sharedKernelMs << " ms\n\n";

    std::cout << "GPU end-to-end\n";
    std::cout << "--------------\n";
    std::cout << "Host to device:       "
              << results.uploadMs << " ms\n";
    std::cout << "Shared kernel:         "
              << results.sharedKernelMs << " ms\n";
    std::cout << "Device to host:       "
              << results.downloadMs << " ms\n";
    std::cout << "Measured total:        "
              << results.gpuEndToEndMs << " ms\n\n";

    std::cout << "Speedups\n";
    std::cout << "--------\n";
    std::cout << "Naive -> shared GPU:   "
              << results.naiveKernelMs /
                     results.sharedKernelMs
              << "x\n";
    std::cout << "CPU -> GPU compute:    "
              << results.cpuMs /
                     results.sharedKernelMs
              << "x\n";
    std::cout << "CPU -> GPU end-to-end: "
              << results.cpuMs /
                     results.gpuEndToEndMs
              << "x\n\n";

    std::cout << "End-to-end breakdown\n";
    std::cout << "--------------------\n";
    std::cout << "Transfers:            "
              << transferPercent << "%\n";
    std::cout << "Kernel:                "
              << kernelPercent << "%\n";
    std::cout << "Other overhead:        "
              << overheadPercent << "%\n";
}

int main()
{
    BenchmarkContext context;

    setupBenchmark(context, "lena.jpg");

    std::cout << "Image: "
              << context.input.width << "x"
              << context.input.height << '\n';

    warmUpCpu(context);
    warmUpGpu(context);

    const DeviceImage& readOnlyInput = context.deviceInput;

    const float cpuAverage =
    benchmark::averageCpu(RUNS, [&]()
    {
        gaussianBlurCpu(
            context.input.width,
            context.input.height,
            context.input.pixels.data(),
            context.cpuOutput.pixels.data()
        );
    });

    const float naiveAverage =
    benchmark::averageCuda(
        RUNS,
        context.stream,
        [&]()
    {
        gaussianBlurGpuNaive<<<
            context.blocks,
            context.threads,
            0,
            context.stream>>>(
                context.input.width,
                context.input.height,
                context.deviceInput.view().data,
                context.deviceOutput.view().data
            );

        CUDA_CHECK(cudaGetLastError());
    });

    const float sharedAverage =
    benchmark::averageCuda(
        RUNS,
        context.stream,
        [&]()
    {
        launchGaussianBlur(
            readOnlyInput.view(),
            context.deviceOutput.view(),
            context.stream
        );
    });

    context.deviceOutput.downloadAsync(context.gpuOutput.view(), context.stream);

    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    if (context.cpuOutput.pixels == context.gpuOutput.pixels)
    {
        std::cout << "Correctness: CPU and GPU results match\n";
    }
    else
    {
        std::cout << "Correctness: CPU and GPU results do not match\n";
    }

    const HostImage& readOnlyHostInput = context.input;
    const DeviceImage& readOnlyDeviceInput = context.deviceInput;

    const float gpuEndToEndAverage =
        benchmark::averageCpu(RUNS, [&]()
    {
        // CPU → GPU
        context.deviceInput.uploadAsync(
            readOnlyHostInput.view(),
            context.stream
        );

        // GPU computation
        launchGaussianBlur(
            readOnlyDeviceInput.view(),
            context.deviceOutput.view(),
            context.stream
        );

        // GPU → CPU
        context.deviceOutput.downloadAsync(
            context.gpuOutput.view(),
            context.stream
        );

        // Wait for the complete operation
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    });

    const float uploadAverage =
    benchmark::averageCpu(RUNS, [&]()
    {
        context.deviceInput.uploadAsync(
            readOnlyHostInput.view(),
            context.stream
        );

        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    });
    const DeviceImage& readOnlyDeviceOutput =
    context.deviceOutput;

    const float downloadAverage =
        benchmark::averageCpu(RUNS, [&]()
    {
        readOnlyDeviceOutput.downloadAsync(
            context.gpuOutput.view(),
            context.stream
        );

        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    });

    BenchmarkResults results;

    results.width = context.input.width;
    results.height = context.input.height;
    results.channels = context.input.channels;
    results.runs = RUNS;

    results.resultsMatch =
        context.cpuOutput.pixels ==
        context.gpuOutput.pixels;

    results.cpuMs = cpuAverage;
    results.naiveKernelMs = naiveAverage;
    results.sharedKernelMs = sharedAverage;
    results.uploadMs = uploadAverage;
    results.downloadMs = downloadAverage;
    results.gpuEndToEndMs = gpuEndToEndAverage;

    printResults(results);

    CUDA_CHECK(cudaStreamDestroy(context.stream));

    return 0;
}