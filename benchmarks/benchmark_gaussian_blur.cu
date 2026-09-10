#include "benchmarks/benchmark_utils.hpp"
#include "core/cuda_check.hpp"
#include "core/device_image.hpp"
#include "filters/gaussian_blur.hpp"
#include "reference/gaussian_blur_cpu.hpp"

#include <exception>
#include <iostream>
#include <string>

constexpr int RUNS = 20;

__device__ void blurPixelGlobalMemory(
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

__global__ void gaussianBlurGlobalMemoryKernel(
    int width,
    int height,
    const unsigned char* input,
    unsigned char* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
       blurPixelGlobalMemory(x, y, width, height, input, output);
    }
}

void warmUpGpu(benchmark::BenchmarkContext& context)
{
    gaussianBlurGlobalMemoryKernel<<<
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
    const HostImage& readOnlyInput = context.input;

    reference::gaussianBlur(
        readOnlyInput.view(),
        context.cpuOutput.view()
    );
}

int main(int argc, char** argv)
{
    try
    {
        const std::string inputPath =
            benchmark::getInputPath(argc, argv, "assets/lena.jpg");

        benchmark::BenchmarkContext context;
        benchmark::setupBenchmark(
            context,
            inputPath,
            benchmark::InputRequirement::Rgb
        );

        warmUpCpu(context);
        warmUpGpu(context);

        const DeviceImage& readOnlyInput = context.deviceInput;
        const HostImage& readOnlyHostInput = context.input;

        const float cpuAverage =
            benchmark::averageCpu(RUNS, [&]()
        {
            reference::gaussianBlur(
                readOnlyHostInput.view(),
                context.cpuOutput.view()
            );
        });

        const float globalMemoryAverage =
            benchmark::averageCuda(
                RUNS,
                context.stream,
                [&]()
        {
            gaussianBlurGlobalMemoryKernel<<<
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

        context.deviceOutput.downloadAsync(
            context.gpuOutput.view(),
            context.stream
        );
        CUDA_CHECK(cudaStreamSynchronize(context.stream));

        const bool resultsMatch =
            context.cpuOutput.pixels == context.gpuOutput.pixels;

        const float gpuEndToEndAverage =
            benchmark::averageGpuEndToEnd(
                RUNS,
                context,
                launchGaussianBlur
            );

        const float uploadAverage =
            benchmark::averageUpload(RUNS, context);

        const float downloadAverage =
            benchmark::averageDownload(RUNS, context);

        const std::string cpuOutputPath =
            benchmark::makeOutputPath(inputPath, "_gaussian_cpu");
        const std::string gpuOutputPath =
            benchmark::makeOutputPath(inputPath, "_gaussian_gpu");

        savePngImage(cpuOutputPath, context.cpuOutput);
        savePngImage(gpuOutputPath, context.gpuOutput);

        std::cout << "Saved CPU result to " << cpuOutputPath << '\n'
                  << "Saved GPU result to " << gpuOutputPath << '\n';

        benchmark::BenchmarkResults results;
        results.title = "Gaussian Blur Benchmark";
        results.inputPath = context.inputPath;
        results.inputPreparation =
            benchmark::describeInputPreparation(context);
        results.width = context.input.width;
        results.height = context.input.height;
        results.channels = context.input.channels;
        results.runs = RUNS;
        results.resultsMatch = resultsMatch;

        results.computeTimings =
        {
            {"CPU reference", cpuAverage},
            {"GPU global-memory kernel", globalMemoryAverage},
            {"GPU shared kernel", sharedAverage}
        };

        results.productionKernelLabel = "GPU shared kernel";
        results.productionKernelMs = sharedAverage;
        results.uploadMs = uploadAverage;
        results.downloadMs = downloadAverage;
        results.gpuEndToEndMs = gpuEndToEndAverage;

        results.speedups =
        {
            {"Global to shared GPU", globalMemoryAverage / sharedAverage},
            {"CPU to GPU compute", cpuAverage / sharedAverage},
            {"CPU to GPU end-to-end", cpuAverage / gpuEndToEndAverage}
        };

        benchmark::printResults(results);
        return resultsMatch ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Gaussian blur benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
