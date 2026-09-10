#include "benchmarks/benchmark_utils.hpp"
#include "reference/heart_bokeh_cpu.hpp"
#include "filters/heart_bokeh.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace
{
// Deterministic point lights demonstrate the aperture without photographic clutter.
HostImage makeDemo()
{
    HostImage image;
    image.width = 480;
    image.height = 270;
    image.pixels.assign(image.view().bytes(), 4);
    for (int light = 0; light < 30; ++light)
    {
        const int column = 20 + (light * 137) % 440;
        const int row = 15 + (light * 73) % 240;
        const std::size_t index = (row * image.width + column) * 3;
        image.pixels[index] = light % 3 == 0 ? 170 : 255;
        image.pixels[index + 1] = light % 3 == 1 ? 190 : 255;
        image.pixels[index + 2] = light % 3 == 2 ? 120 : 255;
    }
    return image;
}

int parseInteger(const char* text)
{
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != std::string(text).size())
        throw std::invalid_argument("Expected an integer argument");
    return value;
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc > 5)
            throw std::invalid_argument("Usage: benchmark_heart_bokeh [image-path|--demo] [threshold 0..255] [intensity 0..1] [runs]");
        const std::string source = argc > 1 ? argv[1] : "--demo";
        const int threshold = argc > 2 ? parseInteger(argv[2]) : 200;
        std::size_t used = 0;
        const float intensity = argc > 3 ? std::stof(argv[3], &used) : 0.35f;
        const int runs = argc > 4 ? parseInteger(argv[4]) : 5;
        if (threshold < 0 || threshold > 255 || runs <= 0 ||
            !std::isfinite(intensity) || intensity < 0 || intensity > 1 ||
            (argc > 3 && used != std::string(argv[3]).size()))
            throw std::invalid_argument("Invalid threshold, intensity, or run count");

        const HostImage input = source == "--demo" ? makeDemo() : loadRgbImage(source);
        benchmark::validateImage(input);
        HostImage output = input;
        const auto run = [&]()
        {
            reference::heartBokeh(input.view(), output.view(),
                static_cast<unsigned char>(threshold), intensity);
        };
        run(); // Warm up; allocations and image I/O are excluded from timing.
        const float average = benchmark::averageCpu(runs, run);
        benchmark::BenchmarkContext context;
        context.input = input;
        context.gpuOutput = input;
        context.deviceInput.allocate(input.width,input.height,3);
        context.deviceOutput.allocate(input.width,input.height,3);
        CUDA_CHECK(cudaStreamCreate(&context.stream));
        context.deviceInput.uploadAsync(input.view(),context.stream);
        const auto launcher = [&](ConstImageView a, ImageView b, cudaStream_t stream)
        { launchHeartBokeh(a,b,static_cast<unsigned char>(threshold),intensity,stream); };
        const DeviceImage& deviceInput = context.deviceInput;
        const auto gpuRun = [&]()
        { launcher(deviceInput.view(),context.deviceOutput.view(),context.stream); };
        gpuRun();
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        const float gpuMs = benchmark::averageCuda(runs,context.stream,gpuRun);
        const float endToEndMs = benchmark::averageGpuEndToEnd(runs,context,launcher);
        const bool match = output.pixels == context.gpuOutput.pixels;
        const std::string stem = source == "--demo" ? "heart_bokeh_demo" : source;
        const auto inputPath = benchmark::makeOutputPath(stem, "_bokeh_input");
        const auto outputPath = benchmark::makeOutputPath(stem, "_bokeh_cpu");
        savePngImage(inputPath, input);
        savePngImage(outputPath, output);
        const auto gpuPath = benchmark::makeOutputPath(stem, "_bokeh_gpu");
        savePngImage(gpuPath, context.gpuOutput);
        std::cout << "Heart bokeh CPU/GPU benchmark\n"
                  << input.width << " x " << input.height << " RGB; threshold="
                  << threshold << "; intensity=" << intensity << "; runs=" << runs
                  << "\nCPU compute average: " << average << " ms\n"
                  << "GPU compute average: " << gpuMs << " ms\n"
                  << "GPU end-to-end: " << endToEndMs << " ms\n"
                  << "Compute speedup: " << average / gpuMs << "x\n"
                  << "CPU/GPU byte agreement: " << (match ? "PASS" : "FAIL") << '\n'
                  << "GPU output: " << gpuPath << '\n'
                  << "Input: " << inputPath << "\nOutput: " << outputPath << '\n';
        return match ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Heart bokeh benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
