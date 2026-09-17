#include "benchmarks/benchmark_utils.hpp"
#include "core/cuda_check.hpp"
#include "core/device_image.hpp"
#include "filters/sharpen.hpp"
#include "reference/sharpen_cpu.hpp"

#include <exception>
#include <iostream>
#include <string>

constexpr int RUNS = 20;
constexpr float STRENGTH = 1.0f;

// Adapt the strength-taking functions to the shared benchmark helper interface.
void runCpu(ConstImageView input, ImageView output)
{
    reference::launchSharpenCPU(input, output, STRENGTH);
}

void runGpu(ConstImageView input, ImageView output, cudaStream_t stream)
{
    launchSharpen(input, output, STRENGTH, stream);
}

void warmUpCpu(benchmark::BenchmarkContext& context)
{
    const HostImage& readOnlyInput = context.input;

    runCpu(
        readOnlyInput.view(),
        context.cpuOutput.view()
    );
}

void warmUpGpu(benchmark::BenchmarkContext& context)
{
    const DeviceImage& readOnlyInput = context.deviceInput;

    runGpu(
        readOnlyInput.view(),
        context.deviceOutput.view(),
        context.stream
    );

    CUDA_CHECK(cudaStreamSynchronize(context.stream));
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

        for (int warmup = 0; warmup < 3; warmup++)
        {
            warmUpCpu(context);
            warmUpGpu(context);
        }
        context.deviceOutput.downloadAsync(context.gpuOutput.view(), context.stream);
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        if (context.cpuOutput.pixels != context.gpuOutput.pixels)
        {
            throw std::runtime_error("CPU/GPU mismatch before timing");
        }
        std::cout << "Strength: " << STRENGTH << "; warm-up runs: 3\n";

        const DeviceImage& readOnlyInput = context.deviceInput;
        const HostImage& readOnlyHostInput = context.input;

        const float cpuAverage =
            benchmark::averageCpu(RUNS, [&]()
        {
            runCpu(
                readOnlyHostInput.view(),
                context.cpuOutput.view()
            );
        });

        const float gpuAverage =
            benchmark::averageCuda(
                RUNS,
                context.stream,
                [&]()
        {
            runGpu(
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

        bool resultsMatch =
            context.cpuOutput.pixels == context.gpuOutput.pixels;

        const float gpuEndToEndAverage =
            benchmark::averageGpuEndToEnd(
                RUNS,
                context,
                runGpu
            );

        const float uploadAverage =
            benchmark::averageUpload(RUNS, context);

        const float downloadAverage =
            benchmark::averageDownload(RUNS, context);

        const std::string cpuOutputPath =
            benchmark::makeOutputPath(inputPath, "_sharpen_cpu");
        const std::string gpuOutputPath =
            benchmark::makeOutputPath(inputPath, "_sharpen_gpu");

        // Validate again after the timed end-to-end and transfer runs.
        resultsMatch = resultsMatch && context.cpuOutput.pixels == context.gpuOutput.pixels;
        if (!resultsMatch)
        {
            throw std::runtime_error("CPU/GPU mismatch after timing");
        }

        savePngImage(cpuOutputPath, context.cpuOutput);
        savePngImage(gpuOutputPath, context.gpuOutput);

        std::cout << "Saved CPU sharpen result to " << cpuOutputPath << '\n'
                  << "Saved GPU sharpen result to " << gpuOutputPath << '\n';

        benchmark::BenchmarkResults results;
        results.title = "Sharpen Benchmark";
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
            {"GPU sharpen kernel", gpuAverage}
        };

        results.productionKernelLabel = "GPU sharpen kernel";
        results.productionKernelMs = gpuAverage;
        results.uploadMs = uploadAverage;
        results.downloadMs = downloadAverage;
        results.gpuEndToEndMs = gpuEndToEndAverage;

        results.speedups =
        {
            {"CPU to GPU compute", cpuAverage / gpuAverage},
            {"CPU to GPU end-to-end", cpuAverage / gpuEndToEndAverage}
        };

        benchmark::printResults(results);
        return resultsMatch ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Sharpen benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
