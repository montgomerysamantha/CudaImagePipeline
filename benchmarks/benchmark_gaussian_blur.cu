#include "benchmarks/benchmark_utils.hpp"
#include "core/cuda_check.hpp"
#include "core/device_image.hpp"
#include "filters/gaussian_blur.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

constexpr int RUNS = 20;

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

void warmUpGpu(benchmark::BenchmarkContext& context)
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

void warmUpCpu(benchmark::BenchmarkContext& context)
{
    gaussianBlurCpu(
        context.input.width,
        context.input.height,
        context.input.pixels.data(),
        context.cpuOutput.pixels.data()
    );
}

int main()
{
    benchmark::BenchmarkContext context;

    benchmark::setupBenchmark(context, "lena.jpg");

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
        // call benchmark-only naive kernel
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

    const DeviceImage& readOnlyDeviceOutput = context.deviceOutput;

    const float downloadAverage =
        benchmark::averageCpu(RUNS, [&]()
    {
        readOnlyDeviceOutput.downloadAsync(
            context.gpuOutput.view(),
            context.stream
        );

        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    });

    benchmark::BenchmarkResults results;

    results.title = "Gaussian Blur Benchmark";
    results.width = context.input.width;
    results.height = context.input.height;
    results.channels = context.input.channels;
    results.runs = RUNS;
    results.resultsMatch =
        context.cpuOutput.pixels ==
        context.gpuOutput.pixels;

    results.computeTimings =
    {
        {"CPU reference", cpuAverage},
        {"GPU naive kernel", naiveAverage},
        {"GPU shared kernel", sharedAverage}
    };

    results.productionKernelLabel =
        "GPU shared kernel";

    results.productionKernelMs = sharedAverage;
    results.uploadMs = uploadAverage;
    results.downloadMs = downloadAverage;
    results.gpuEndToEndMs = gpuEndToEndAverage;

    results.speedups =
    {
        {
            "Naive to shared GPU",
            naiveAverage / sharedAverage
        },
        {
            "CPU to GPU compute",
            cpuAverage / sharedAverage
        },
        {
            "CPU to GPU end-to-end",
            cpuAverage / gpuEndToEndAverage
        }
    };

    benchmark::printResults(results);

    CUDA_CHECK(cudaStreamDestroy(context.stream));

    return 0;
}