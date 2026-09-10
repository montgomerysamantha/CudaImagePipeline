#include "benchmarks/benchmark_utils.hpp"
#include "core/cuda_check.hpp"
#include "core/device_image.hpp"
#include "filters/grayscale.hpp"

#include <exception>
#include <iostream>
#include <string>

constexpr int RUNS = 20;

void grayscaleCpu(
    int width,
    int height,
    const unsigned char* input,
    unsigned char* output)
{
    const int pixelCount = width * height;

    for (int pixel = 0; pixel < pixelCount; ++pixel)
    {
        const int index = pixel * 3;
        const int red = input[index];
        const int green = input[index + 1];
        const int blue = input[index + 2];

        const unsigned char gray =
            static_cast<unsigned char>(
                (299 * red + 587 * green + 114 * blue) / 1000
            );

        output[index] = gray;
        output[index + 1] = gray;
        output[index + 2] = gray;
    }
}

void warmUpCpu(benchmark::BenchmarkContext& context)
{
    grayscaleCpu(
        context.input.width,
        context.input.height,
        context.input.pixels.data(),
        context.cpuOutput.pixels.data()
    );
}

void warmUpGpu(benchmark::BenchmarkContext& context)
{
    const DeviceImage& readOnlyInput = context.deviceInput;

    launchGrayscale(
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

        warmUpCpu(context);
        warmUpGpu(context);

        const DeviceImage& readOnlyInput = context.deviceInput;

        const float cpuAverage =
            benchmark::averageCpu(RUNS, [&]()
        {
            grayscaleCpu(
                context.input.width,
                context.input.height,
                context.input.pixels.data(),
                context.cpuOutput.pixels.data()
            );
        });

        const float gpuAverage =
            benchmark::averageCuda(
                RUNS,
                context.stream,
                [&]()
        {
            launchGrayscale(
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
                launchGrayscale
            );

        const float uploadAverage =
            benchmark::averageUpload(RUNS, context);

        const float downloadAverage =
            benchmark::averageDownload(RUNS, context);

        const std::string cpuOutputPath =
            benchmark::makeOutputPath(inputPath, "_grayscale_cpu");
        const std::string gpuOutputPath =
            benchmark::makeOutputPath(inputPath, "_grayscale_gpu");

        savePngImage(cpuOutputPath, context.cpuOutput);
        savePngImage(gpuOutputPath, context.gpuOutput);

        std::cout << "Saved CPU grayscale result to " << cpuOutputPath << '\n'
                  << "Saved GPU grayscale result to " << gpuOutputPath << '\n';

        benchmark::BenchmarkResults results;
        results.title = "Grayscale Benchmark";
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
            {"GPU grayscale kernel", gpuAverage}
        };

        results.productionKernelLabel = "GPU grayscale kernel";
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
        std::cerr << "Grayscale benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
