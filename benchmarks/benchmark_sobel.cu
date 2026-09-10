#include "benchmarks/benchmark_utils.hpp"
#include "core/cuda_check.hpp"
#include "core/device_image.hpp"
#include "filters/edge_detection.hpp"
#include "reference/sobel_cpu.hpp"

#include <exception>
#include <iostream>
#include <string>

constexpr int RUNS = 20;

void warmUpGpu(benchmark::BenchmarkContext& context)
{
    const DeviceImage& readOnlyInput =
        context.deviceInput;

    launchEdgeDetection(
        readOnlyInput.view(),
        context.deviceOutput.view(),
        context.stream
    );

    launchEdgeDetectionSharedMemory(
        readOnlyInput.view(),
        context.deviceOutput.view(),
        context.stream
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(context.stream));
}

void warmUpCpu(benchmark::BenchmarkContext& context)
{
    const HostImage& readOnlyInput = context.input;

    reference::sobel(
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
            benchmark::InputRequirement::Grayscale,
            3,
            3
        );

        warmUpCpu(context);
        warmUpGpu(context);

        const DeviceImage& readOnlyInput = context.deviceInput;
        const HostImage& readOnlyHostInput = context.input;

        const float cpuAverage =
        benchmark::averageCpu(RUNS, [&]()
        {
            reference::sobel(
                readOnlyHostInput.view(),
                context.cpuOutput.view()
            );
        });

        const float globalMemAverage =
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

        context.deviceOutput.downloadAsync(
            context.gpuOutput.view(),
            context.stream
        );
        CUDA_CHECK(cudaStreamSynchronize(context.stream));

        const bool globalResultsMatch =
            context.cpuOutput.pixels == context.gpuOutput.pixels;

        const float sharedMemAverage =
        benchmark::averageCuda(
            RUNS,
            context.stream,
            [&]()
        {
            launchEdgeDetectionSharedMemory(
                readOnlyInput.view(),
                context.deviceOutput.view(),
                context.stream
            );
        });

        context.deviceOutput.downloadAsync(context.gpuOutput.view(), context.stream);

        CUDA_CHECK(cudaStreamSynchronize(context.stream));

        const bool sharedResultsMatch =
            context.cpuOutput.pixels == context.gpuOutput.pixels;

        const bool resultsMatch =
            globalResultsMatch && sharedResultsMatch;

        const float gpuEndToEndAverage =
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
            "GPU global-memory sobel kernel";

        results.productionKernelMs = globalMemAverage;
        results.uploadMs = uploadAverage;
        results.downloadMs = downloadAverage;
        results.gpuEndToEndMs = gpuEndToEndAverage;

        results.speedups =
        {
            {
                "CPU to GPU compute",
                cpuAverage / globalMemAverage
            },
            {
                "CPU to GPU end-to-end",
                cpuAverage / gpuEndToEndAverage
            },
            {
                "Shared-memory speedup",
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
