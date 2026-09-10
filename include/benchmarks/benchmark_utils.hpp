#pragma once

#include "core/cuda_check.hpp"
#include "core/device_image.hpp"

#include <chrono>
#include <stdexcept>

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

} // namespace benchmark