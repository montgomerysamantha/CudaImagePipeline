#pragma once

#include "core/cuda_check.hpp"
#include "core/device_image.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace benchmark
{

template <typename Function>
float timeCpu(Function function)
{
    const auto start = std::chrono::steady_clock::now();

    function();

    const auto stop = std::chrono::steady_clock::now();

    const std::chrono::duration<float, std::milli> elapsed =
        stop - start;

    return elapsed.count();
}

template <typename Function>
float timeCuda(cudaStream_t stream, Function function)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    function();

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsedMs,
        start,
        stop
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsedMs;
}

template <typename Function>
float averageCpu(int runs, Function function)
{
    if (runs <= 0)
    {
        throw std::invalid_argument(
            "Benchmark runs must be greater than zero"
        );
    }

    float totalMs = 0.0f;

    for (int run = 0; run < runs; ++run)
    {
        totalMs += timeCpu(function);
    }

    return totalMs / runs;
}

template <typename Function>
float averageCuda(
    int runs,
    cudaStream_t stream,
    Function function)
{
    if (runs <= 0)
    {
        throw std::invalid_argument(
            "Benchmark runs must be greater than zero"
        );
    }

    float totalMs = 0.0f;

    for (int run = 0; run < runs; ++run)
    {
        totalMs += timeCuda(stream, function);
    }

    return totalMs / runs;
}

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

struct NamedTiming
{
    std::string label;
    float milliseconds = 0.0f;
};

struct NamedSpeedup
{
    std::string label;
    float factor = 0.0f;
};

struct BenchmarkResults
{
    std::string title;

    bool resultsMatch = false;

    int width = 0;
    int height = 0;
    int channels = 0;
    int runs = 0;

    std::vector<NamedTiming> computeTimings;
    std::vector<NamedSpeedup> speedups;

    std::string productionKernelLabel;
    float productionKernelMs = 0.0f;

    float uploadMs = 0.0f;
    float downloadMs = 0.0f;
    float gpuEndToEndMs = 0.0f;
};

inline void setupBenchmark(
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

inline void printResults(
    const BenchmarkResults& results)
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

    float transferPercent = 0.0f;
    float kernelPercent = 0.0f;
    float overheadPercent = 0.0f;

    if (results.gpuEndToEndMs > 0.0f)
    {
        transferPercent =
            100.0f * transferMs /
            results.gpuEndToEndMs;

        kernelPercent =
            100.0f * results.productionKernelMs /
            results.gpuEndToEndMs;

        overheadPercent =
            100.0f -
            transferPercent -
            kernelPercent;
    }

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout << '\n'
              << results.title << '\n'
              << std::string(results.title.size(), '=')
              << '\n';

    std::cout << "Image:        "
              << results.width << " x "
              << results.height << " x "
              << results.channels << " channels\n";

    std::cout << "Image size:   "
              << imageMiB << " MiB\n";

    std::cout << "Runs:         "
              << results.runs << '\n';

    std::cout << "Correctness:  "
              << (results.resultsMatch ? "PASS" : "FAIL")
              << "\n\n";

    std::cout << "Compute-only\n";
    std::cout << "------------\n";

    for (const NamedTiming& timing :
         results.computeTimings)
    {
        std::cout
            << std::left
            << std::setw(30)
            << timing.label
            << std::right
            << std::setw(10)
            << timing.milliseconds
            << " ms\n";
    }

    std::cout << "\nGPU end-to-end\n";
    std::cout << "--------------\n";

    std::cout
        << std::left << std::setw(30)
        << "Host to device"
        << std::right << std::setw(10)
        << results.uploadMs << " ms\n";

    std::cout
        << std::left << std::setw(30)
        << results.productionKernelLabel
        << std::right << std::setw(10)
        << results.productionKernelMs << " ms\n";

    std::cout
        << std::left << std::setw(30)
        << "Device to host"
        << std::right << std::setw(10)
        << results.downloadMs << " ms\n";

    std::cout
        << std::left << std::setw(30)
        << "Measured total"
        << std::right << std::setw(10)
        << results.gpuEndToEndMs << " ms\n";

    std::cout << "\nSpeedups\n";
    std::cout << "--------\n";

    for (const NamedSpeedup& speedup :
         results.speedups)
    {
        std::cout
            << std::left
            << std::setw(30)
            << speedup.label
            << std::right
            << std::setw(10)
            << speedup.factor
            << "x\n";
    }

    std::cout << "\nEnd-to-end breakdown\n";
    std::cout << "--------------------\n";

    std::cout
        << std::left << std::setw(30)
        << "Transfers"
        << std::right << std::setw(10)
        << transferPercent << "%\n";

    std::cout
        << std::left << std::setw(30)
        << "Kernel"
        << std::right << std::setw(10)
        << kernelPercent << "%\n";

    std::cout
        << std::left << std::setw(30)
        << "Other overhead"
        << std::right << std::setw(10)
        << overheadPercent << "%\n";
}

} // namespace benchmark