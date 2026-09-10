#include "core/image.hpp"
#include "benchmarks/benchmark_utils.hpp"
#include "core/cuda_check.hpp"
#include "core/device_image.hpp"
#include "filters/edge_detection.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

constexpr int RUNS = 20;

constexpr int sobelX[3][3] =
{
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

constexpr int sobelY[3][3] =
{
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

std::pair<int, int> getGxGyValues(
    int x,
    int y,
    int width,
    const unsigned char* input)
{
    int gx = 0;
    int gy = 0;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int neighborX = x + dx;
            const int neighborY = y + dy;

            const int neighborIndex =
                (neighborY * width + neighborX) * 3;

            const int gray = input[neighborIndex];

            gx += gray * sobelX[dy + 1][dx + 1];
            gy += gray * sobelY[dy + 1][dx + 1];
        }
    }

    return {gx, gy};
}

unsigned char calcEdge(int gx, int gy)
{
    const float magnitude =
        std::sqrt(static_cast<float>(gx * gx + gy * gy));

    const unsigned char edge =
        static_cast<unsigned char>(
            std::min(255.0f, magnitude)
        );

    return edge;
}

bool isBorder(int x, int y, int width, int height)
{
    const bool isBorderCoord =
                x == 0 ||
                y == 0 ||
                x == width - 1 ||
                y == height - 1;

    return isBorderCoord;
}

void writeRGB(
    const unsigned char r,
    const unsigned char g,
    const unsigned char b,
    unsigned char* output,
    const int outputIndex)
{
    output[outputIndex] = r;
    output[outputIndex + 1] = g;
    output[outputIndex + 2] = b;
}

void sobelCpu(
    int width,
    int height,
    const unsigned char* input,
    unsigned char* output)
{
    // Loop over every pixel.
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const int outputIndex = (y * width + x) * 3;

            if (isBorder(x, y, width, height))
            {
                writeRGB(0, 0, 0, output, outputIndex);
                continue;
            }

            const auto [gx, gy] =
                getGxGyValues(
                    x,
                    y,
                    width,
                    input
                );

            const unsigned char edge =
                calcEdge(gx, gy);

            writeRGB(edge, edge, edge, output, outputIndex);
        }
    }
}

void warmUpGpu(benchmark::BenchmarkContext& context)
{
    const DeviceImage& readOnlyInput =
        context.deviceInput;

    launchEdgeDetectionGlobalMemory(
        readOnlyInput.view(),
        context.deviceOutput.view(),
        context.stream
    );

    launchEdgeDetection(
        readOnlyInput.view(),
        context.deviceOutput.view(),
        context.stream
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(context.stream));
}

void warmUpCpu(benchmark::BenchmarkContext& context)
{
    sobelCpu(
        context.input.width,
        context.input.height,
        context.input.pixels.data(),
        context.cpuOutput.pixels.data()
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
            benchmark::InputRequirement::Grayscale,
            3,
            3
        );

        warmUpCpu(context);
        warmUpGpu(context);

        const DeviceImage& readOnlyInput = context.deviceInput;

        const float cpuAverage =
        benchmark::averageCpu(RUNS, [&]()
        {
            sobelCpu(
                context.input.width,
                context.input.height,
                context.input.pixels.data(),
                context.cpuOutput.pixels.data()
            );
        });

        const float globalMemAverage =
        benchmark::averageCuda(
            RUNS,
            context.stream,
            [&]()
        {
            launchEdgeDetectionGlobalMemory(
                readOnlyInput.view(),
                context.deviceOutput.view(),
                context.stream
            );
        });

        const float sharedMemAverage =
        benchmark::averageCuda(
            RUNS,
            context.stream,
            [&]()
        {
            launchEdgeDetection(
                readOnlyInput.view(),
                context.deviceOutput.view(),
                context.stream
            );
        });

        context.deviceOutput.downloadAsync(context.gpuOutput.view(), context.stream);

        CUDA_CHECK(cudaStreamSynchronize(context.stream));

        const bool resultsMatch =
            context.cpuOutput.pixels == context.gpuOutput.pixels;

        const float gpuSharedEndToEndAverage =
            benchmark::averageGpuEndToEnd(
                RUNS,
                context,
                launchEdgeDetection
            );

        const float uploadAverage =
            benchmark::averageUpload(RUNS, context);

        const float downloadAverage =
            benchmark::averageDownload(RUNS, context);

        const std::string cpuOutputPath =
            benchmark::makeOutputPath(inputPath, "_sobel_cpu");
        const std::string gpuOutputPath =
            benchmark::makeOutputPath(inputPath, "_sobel_gpu");

        savePngImage(cpuOutputPath, context.cpuOutput);
        savePngImage(gpuOutputPath, context.gpuOutput);

        std::cout << "Saved CPU Sobel result to " << cpuOutputPath << '\n'
                  << "Saved GPU Sobel result to " << gpuOutputPath << '\n';

        benchmark::BenchmarkResults results;

        results.title = "Sobel Edge Detection Benchmark";
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
            {"GPU global-memory sobel kernel", globalMemAverage},
            {"GPU shared-memory sobel kernel", sharedMemAverage}
        };

        results.productionKernelLabel =
            "GPU shared-memory sobel kernel";

        results.productionKernelMs = sharedMemAverage;
        results.uploadMs = uploadAverage;
        results.downloadMs = downloadAverage;
        results.gpuEndToEndMs = gpuSharedEndToEndAverage;

        results.speedups =
        {
            {
                "CPU to GPU compute",
                cpuAverage / sharedMemAverage
            },
            {
                "CPU to GPU end-to-end",
                cpuAverage / gpuSharedEndToEndAverage
            },
            {
                "Global to shared GPU memory",
                globalMemAverage / sharedMemAverage
            }
        };

        benchmark::printResults(results);
        return resultsMatch ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "Sobel failed: "
            << error.what()
            << '\n';

        return 1;
    }
}
