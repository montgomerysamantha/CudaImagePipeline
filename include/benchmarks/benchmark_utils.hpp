#pragma once

#include "core/cuda_check.hpp"
#include "core/device_image.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace benchmark
{

enum class InputRequirement
{
    Rgb,
    Grayscale
};

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

    for (int run = 0; run < runs; run++)
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

    for (int run = 0; run < runs; run++)
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

    std::string inputPath;
    bool sourceWasGrayscale = false;
    bool convertedToGrayscale = false;

    dim3 threads{16, 16};
    dim3 blocks{1, 1};

    ~BenchmarkContext()
    {
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
    }

    BenchmarkContext() = default;
    BenchmarkContext(const BenchmarkContext&) = delete;
    BenchmarkContext& operator=(const BenchmarkContext&) = delete;
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
    std::string inputPath;
    std::string inputPreparation;

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

inline std::string getInputPath(
    int argc,
    char** argv,
    const std::string& defaultPath)
{
    if (argc > 2)
    {
        throw std::invalid_argument(
            "Expected zero or one argument: [image-path]"
        );
    }

    if (argc == 2)
    {
        if (argv[1] == nullptr || argv[1][0] == '\0')
        {
            throw std::invalid_argument("Image path cannot be empty");
        }

        return argv[1];
    }

    return defaultPath;
}

inline void validateImage(
    const HostImage& image,
    int minimumWidth = 1,
    int minimumHeight = 1)
{
    if (image.width < minimumWidth || image.height < minimumHeight)
    {
        throw std::invalid_argument(
            "Image must be at least " +
            std::to_string(minimumWidth) + " x " +
            std::to_string(minimumHeight) + " pixels"
        );
    }

    if (image.channels != 3)
    {
        throw std::invalid_argument(
            "Benchmark images must use three-channel RGB storage"
        );
    }

    if (image.pixels.size() != image.view().bytes())
    {
        throw std::invalid_argument(
            "Image data size does not match its dimensions"
        );
    }
}

inline bool isGrayscale(const HostImage& image)
{
    validateImage(image);

    for (std::size_t index = 0;
         index < image.pixels.size();
         index += 3)
    {
        const unsigned char red = image.pixels[index];
        const unsigned char green = image.pixels[index + 1];
        const unsigned char blue = image.pixels[index + 2];

        if (red != green || red != blue)
        {
            return false;
        }
    }

    return true;
}

inline void convertToGrayscale(HostImage& image)
{
    validateImage(image);

    for (std::size_t index = 0;
         index < image.pixels.size();
         index += 3)
    {
        const int red = image.pixels[index];
        const int green = image.pixels[index + 1];
        const int blue = image.pixels[index + 2];

        const unsigned char gray =
            static_cast<unsigned char>(
                (299 * red + 587 * green + 114 * blue) / 1000
            );

        image.pixels[index] = gray;
        image.pixels[index + 1] = gray;
        image.pixels[index + 2] = gray;
    }
}

inline std::string describeInputPreparation(
    const BenchmarkContext& context)
{
    if (context.convertedToGrayscale)
    {
        return "Converted RGB to grayscale before timing";
    }

    if (context.sourceWasGrayscale)
    {
        return "Source was already grayscale";
    }

    return "Source RGB used without preprocessing";
}

inline void setupBenchmark(
    BenchmarkContext& context,
    const std::string& imagePath,
    InputRequirement requirement = InputRequirement::Rgb,
    int minimumWidth = 1,
    int minimumHeight = 1)
{
    if (context.stream != nullptr)
    {
        throw std::logic_error(
            "BenchmarkContext has already been set up"
        );
    }

    HostImage preparedInput = loadRgbImage(imagePath);
    validateImage(preparedInput, minimumWidth, minimumHeight);

    context.inputPath = imagePath;
    context.sourceWasGrayscale = isGrayscale(preparedInput);
    context.convertedToGrayscale =
        requirement == InputRequirement::Grayscale &&
        !context.sourceWasGrayscale;

    if (context.convertedToGrayscale)
    {
        convertToGrayscale(preparedInput);
    }

    context.input = std::move(preparedInput);

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

inline std::string makeOutputPath(
    const std::string& inputPath,
    const std::string& suffix)
{
    const std::filesystem::path path(inputPath);
    const std::string stem = path.stem().string();

    if (stem.empty())
    {
        throw std::invalid_argument(
            "Could not determine an output name from the image path"
        );
    }

    std::filesystem::create_directories("output");
    return (std::filesystem::path("output") /
            (stem + suffix + ".png")).string();
}

template <typename Launcher>
float averageGpuEndToEnd(
    int runs,
    BenchmarkContext& context,
    Launcher launcher)
{
    const HostImage& readOnlyHostInput = context.input;
    const DeviceImage& readOnlyDeviceInput = context.deviceInput;

    return averageCpu(runs, [&]()
    {
        context.deviceInput.uploadAsync(
            readOnlyHostInput.view(),
            context.stream
        );

        launcher(
            readOnlyDeviceInput.view(),
            context.deviceOutput.view(),
            context.stream
        );

        context.deviceOutput.downloadAsync(
            context.gpuOutput.view(),
            context.stream
        );

        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    });
}

inline float averageUpload(
    int runs,
    BenchmarkContext& context)
{
    const HostImage& readOnlyInput = context.input;

    return averageCpu(runs, [&]()
    {
        context.deviceInput.uploadAsync(
            readOnlyInput.view(),
            context.stream
        );
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    });
}

inline float averageDownload(
    int runs,
    BenchmarkContext& context)
{
    const DeviceImage& readOnlyOutput = context.deviceOutput;

    return averageCpu(runs, [&]()
    {
        readOnlyOutput.downloadAsync(
            context.gpuOutput.view(),
            context.stream
        );
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    });
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
        results.uploadMs +
        results.downloadMs;

    const float componentTotal =
        transferMs +
        results.productionKernelMs;

    const float transferPercent =
        componentTotal > 0.0f
            ? 100.0f * transferMs / componentTotal
            : 0.0f;

    const float kernelPercent =
        componentTotal > 0.0f
            ? 100.0f * results.productionKernelMs /
                componentTotal
            : 0.0f;

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

    if (!results.inputPath.empty())
    {
        std::cout << "Input file:   "
                  << results.inputPath << '\n';
    }

    if (!results.inputPreparation.empty())
    {
        std::cout << "Preparation:  "
                  << results.inputPreparation << '\n';
    }

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

    std::cout << "\nMeasured component breakdown\n";
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
}

} // namespace benchmark
