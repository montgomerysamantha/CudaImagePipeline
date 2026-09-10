#include "benchmarks/benchmark_utils.hpp"
#include "filters/heart_bokeh.hpp"
#include "reference/heart_bokeh_cpu.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace
{
constexpr int demoWidth = 480;
constexpr int demoHeight = 270;
constexpr int demoBackground = 4;
constexpr int demoLightCount = 30;
constexpr int horizontalMargin = 20;
constexpr int verticalMargin = 15;
// Coprime strides distribute repeatable light positions across the image.
constexpr int horizontalStride = 137;
constexpr int verticalStride = 73;
constexpr int rgbChannels = 3;
constexpr int maximumChannel = 255;
constexpr int defaultThreshold = 200;
constexpr float defaultIntensity = 0.35f;
constexpr int defaultRuns = 5;
constexpr int cyanRed = 170;
constexpr int magentaGreen = 190;
constexpr int yellowBlue = 120;

// Deterministic point lights demonstrate the aperture without photographic clutter.
HostImage makeDemo()
{
    HostImage image;
    image.width = demoWidth;
    image.height = demoHeight;
    image.pixels.assign(image.view().bytes(), demoBackground);

    for (int light = 0; light < demoLightCount; light++)
    {
        const int column = horizontalMargin + (light * horizontalStride) %
                                                  (demoWidth - 2 * horizontalMargin);
        const int row =
            verticalMargin + (light * verticalStride) % (demoHeight - 2 * verticalMargin);
        const std::size_t index = (row * image.width + column) * rgbChannels;
        image.pixels[index] = light % rgbChannels == 0 ? cyanRed : maximumChannel;
        image.pixels[index + 1] =
            light % rgbChannels == 1 ? magentaGreen : maximumChannel;
        image.pixels[index + 2] = light % rgbChannels == 2 ? yellowBlue : maximumChannel;
    }
    return image;
}

int parseInteger(const char *text)
{
    std::size_t used = 0;

    const int value = std::stoi(text, &used);
    if (used != std::string(text).size())
        throw std::invalid_argument("Expected an integer argument");
    return value;
}
struct BenchmarkOptions
{
    std::string source;

    int threshold;
    float intensity;

    int runs;
};

BenchmarkOptions parseOptions(int argc, char **argv)
{
    if (argc > 5)
        throw std::invalid_argument("Usage: benchmark_heart_bokeh [image-path|--demo] "
                                    "[threshold 0..255] [intensity 0..1] [runs]");

    const std::string source = argc > 1 ? argv[1] : "--demo";

    const int threshold = argc > 2 ? parseInteger(argv[2]) : defaultThreshold;
    std::size_t used = 0;

    const float intensity = argc > 3 ? std::stof(argv[3], &used) : defaultIntensity;

    const int runs = argc > 4 ? parseInteger(argv[4]) : defaultRuns;
    if (threshold < 0 || threshold > maximumChannel || runs <= 0 ||
        !std::isfinite(intensity) || intensity < 0 || intensity > 1 ||
        (argc > 3 && used != std::string(argv[3]).size()))
        throw std::invalid_argument(
            "Invalid threshold, intensity, or runCpuFilter count");

    return {source, threshold, intensity, runs};
}

// Allocate once; only transfers and kernel work belong in timed regions.
void prepareGpuContext(benchmark::BenchmarkContext &context, const HostImage &input)
{
    context.input = input;
    context.gpuOutput = input;
    context.deviceInput.allocate(input.width, input.height, input.channels);
    context.deviceOutput.allocate(input.width, input.height, input.channels);

    CUDA_CHECK(cudaStreamCreate(&context.stream));
    context.deviceInput.uploadAsync(input.view(), context.stream);
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const BenchmarkOptions options = parseOptions(argc, argv);
        const auto &source = options.source;
        const int threshold = options.threshold;
        const float intensity = options.intensity;
        const int runs = options.runs;

        const HostImage input = source == "--demo" ? makeDemo() : loadRgbImage(source);
        benchmark::validateImage(input);
        HostImage output = input;
        const auto runCpuFilter = [&]() {
            reference::heartBokeh(input.view(), output.view(),
                                  static_cast<unsigned char>(threshold), intensity);
        };
        runCpuFilter(); // Warm up; allocations and image I/O are excluded from timing.
        // CPU reference timing after an untimed warm-up.
        const float cpuAverageMs = benchmark::averageCpu(runs, runCpuFilter);
        benchmark::BenchmarkContext context;
        prepareGpuContext(context, input);

        const auto launcher = [&](ConstImageView sourceImage, ImageView destinationImage,
                                  cudaStream_t stream) {
            launchHeartBokeh(sourceImage, destinationImage,
                             static_cast<unsigned char>(threshold), intensity, stream);
        };
        const DeviceImage &deviceInput = context.deviceInput;
        // Keep the global baseline separate from the production kernel.
        const auto globalRun = [&]() {
            launchHeartBokehGlobal(deviceInput.view(), context.deviceOutput.view(),
                                   static_cast<unsigned char>(threshold), intensity,
                                   context.stream);
        };
        globalRun();
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        const float globalAverageMs =
            benchmark::averageCuda(runs, context.stream, globalRun);
        context.deviceOutput.downloadAsync(context.gpuOutput.view(), context.stream);
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        const bool globalMatch = output.pixels == context.gpuOutput.pixels;
        const auto gpuRun = [&]() {
            launcher(deviceInput.view(), context.deviceOutput.view(), context.stream);
        };
        gpuRun();
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        const float sharedAverageMs =
            benchmark::averageCuda(runs, context.stream, gpuRun);
        const float endToEndMs = benchmark::averageGpuEndToEnd(runs, context, launcher);
        const bool match = globalMatch && output.pixels == context.gpuOutput.pixels;
        const float uploadMs = benchmark::averageUpload(runs, context);
        const float downloadMs = benchmark::averageDownload(runs, context);
        // Save and report outside all timed regions.
        const std::string stem = source == "--demo" ? "heart_bokeh_demo" : source;
        const auto inputPath = benchmark::makeOutputPath(stem, "_bokeh_input");
        const auto outputPath = benchmark::makeOutputPath(stem, "_bokeh_cpu");
        savePngImage(inputPath, input);
        savePngImage(outputPath, output);
        const auto gpuPath = benchmark::makeOutputPath(stem, "_bokeh_gpu");
        savePngImage(gpuPath, context.gpuOutput);
        std::cout << "Heart bokeh CPU/GPU benchmark\n"
                  << input.width << " x " << input.height
                  << " RGB; threshold=" << threshold << "; intensity=" << intensity
                  << "; runs=" << runs << "\nCPU compute average: " << cpuAverageMs
                  << " ms\n"
                  << "GPU global average: " << globalAverageMs << " ms\n"
                  << "Global/shared speedup: " << globalAverageMs / sharedAverageMs
                  << "x\n"
                  << "Upload: " << uploadMs << " ms; Download: " << downloadMs << " ms\n"
                  << "GPU compute average: " << sharedAverageMs << " ms\n"
                  << "GPU end-to-end: " << endToEndMs << " ms\n"
                  << "Compute speedup: " << cpuAverageMs / sharedAverageMs << "x\n"
                  << "CPU/GPU byte agreement: " << (match ? "PASS" : "FAIL") << '\n'
                  << "GPU output: " << gpuPath << '\n'
                  << "Input: " << inputPath << "\nOutput: " << outputPath << '\n';
        return match ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Heart bokeh benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
