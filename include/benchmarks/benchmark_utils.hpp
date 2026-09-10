#pragma once

#include "core/cuda_check.hpp"

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

} // namespace benchmark