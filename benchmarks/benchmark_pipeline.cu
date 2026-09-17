#include "core/cuda_check.hpp"
#include "pipeline/image_pipeline.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr int warmupRuns = 3;
constexpr int stageCount = 6;

PipelineOptions disabledOptions()
{
    PipelineOptions options;
    options.grayscale = false;
    options.gaussianBlur = false;
    return options;
}

HostImage makeInput(int width, int height)
{
    HostImage input;
    input.allocate(width, height, 3);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const std::size_t offset = (static_cast<std::size_t>(y) * width + x) * 3;
            input.pixels[offset] = static_cast<unsigned char>((x * 13 + y * 7) % 256);
            input.pixels[offset + 1] = static_cast<unsigned char>((x * 3 + y * 17) % 256);
            input.pixels[offset + 2] = static_cast<unsigned char>((x * 11 + y * 5) % 256);
        }
    }
    return input;
}

struct Sample
{
    double wallMs = 0;
    PipelineTimings device;
};

void addTimings(PipelineTimings& total, const PipelineTimings& stage)
{
    total.uploadMs += stage.uploadMs;
    total.processingMs += stage.processingMs;
    total.downloadMs += stage.downloadMs;
}

// The separate path uses the same production implementation and options, but
// each pipeline enables just one stage. Its output returns to host memory
// before becoming the next stage's input. Device buffers persist across runs.
HostImage processSeparate(
    std::vector<std::unique_ptr<ImagePipeline>>& stages,
    const HostImage& input,
    PipelineTimings& total)
{
    HostImage output;
    const HostImage* current = &input;
    for (auto& stage : stages)
    {
        PipelineTimings timings;
        output = stage->process(*current, &timings);
        current = &output;
        addTimings(total, timings);
    }
    return output;
}

template <typename Function>
Sample measure(Function function, HostImage& output)
{
    Sample sample;
    const auto start = std::chrono::steady_clock::now();
    output = function(sample.device);
    const auto stop = std::chrono::steady_clock::now();
    sample.wallMs = std::chrono::duration<double, std::milli>(stop - start).count();
    return sample;
}

void printResult(int width, int height, const char* mode, const Sample& sum, int runs)
{
    std::cout << width << 'x' << height << ',' << mode << ','
        << sum.device.uploadMs / runs << ','
        << sum.device.processingMs / runs << ','
        << sum.device.downloadMs / runs << ','
        << sum.wallMs / runs << '\n';
}

void benchmarkSize(int width, int height, int runs)
{
    const HostImage input = makeInput(width, height);
    PipelineOptions options;
    options.heartBokeh = true;
    options.edgeDetection = true;
    options.sharpen = true;
    options.resize = true;
    options.outputWidth = width / 2;
    options.outputHeight = height / 2;
    ImagePipeline resident(options);

    std::vector<std::unique_ptr<ImagePipeline>> separate;
    for (int stage = 0; stage < stageCount; stage++)
    {
        PipelineOptions single = disabledOptions();
        single.grayscale = stage == 0;
        single.gaussianBlur = stage == 1;
        single.heartBokeh = stage == 2;
        single.edgeDetection = stage == 3;
        single.sharpen = stage == 4;
        single.resize = stage == 5;
        single.outputWidth = options.outputWidth;
        single.outputHeight = options.outputHeight;
        separate.push_back(std::make_unique<ImagePipeline>(single));
    }

    Sample residentTotal;
    Sample separateTotal;
    for (int run = -warmupRuns; run < runs; run++)
    {
        HostImage residentOutput;
        HostImage separateOutput;
        Sample residentSample;
        Sample separateSample;
        auto runResident = [&]()
        {
            residentSample = measure(
                [&](PipelineTimings& timings) { return resident.process(input, &timings); },
                residentOutput
            );
        };
        auto runSeparate = [&]()
        {
            separateSample = measure(
                [&](PipelineTimings& timings) { return processSeparate(separate, input, timings); },
                separateOutput
            );
        };

        // Alternate order to reduce consistent first/second-run bias.
        if (run % 2 == 0)
        {
            runResident();
            runSeparate();
        }
        else
        {
            runSeparate();
            runResident();
        }

        // Compare outside the timed intervals, including every warm-up run.
        if (residentOutput.width != separateOutput.width ||
            residentOutput.height != separateOutput.height ||
            residentOutput.channels != separateOutput.channels ||
            residentOutput.pixels != separateOutput.pixels)
        {
            throw std::runtime_error("Resident and separate-stage outputs differ");
        }
        if (run >= 0)
        {
            residentTotal.wallMs += residentSample.wallMs;
            separateTotal.wallMs += separateSample.wallMs;
            addTimings(residentTotal.device, residentSample.device);
            addTimings(separateTotal.device, separateSample.device);
        }
    }

    printResult(width, height, "resident", residentTotal, runs);
    printResult(width, height, "separate", separateTotal, runs);
    std::cout << "# " << width << 'x' << height << " outputs match; end-to-end speedup: "
        << separateTotal.wallMs / residentTotal.wallMs << "x\n";
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc > 2)
        {
            throw std::invalid_argument("Usage: benchmark_pipeline [positive-run-count]");
        }
        int runs = 20;
        if (argc == 2)
        {
            std::size_t consumed = 0;
            runs = std::stoi(argv[1], &consumed);
            if (consumed != std::string(argv[1]).size() || runs <= 0)
            {
                throw std::invalid_argument("Run count must be a positive integer");
            }
        }

        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        std::cout << "# GPU: " << properties.name << '\n'
            << "# Chain: grayscale -> blur -> bokeh -> Sobel -> sharpen -> half-size resize\n"
            << "# Bokeh threshold=200, intensity=0.05; sharpen strength=1\n"
            << "# Runs: " << runs << "; warm-ups per mode/size: " << warmupRuns << '\n'
            << "# Resident: 2 transfers; separate: 12 transfers per chain\n"
            << "# CUDA events: upload/processing/download; steady_clock: synchronous API wall time\n"
            << "# Device buffers warmed/reused; host allocations, events and synchronization included in wall time\n"
            << "# Input generation and output comparison excluded; pageable host memory\n"
            << "size,mode,upload_ms,processing_ms,download_ms,wall_ms\n"
            << std::fixed << std::setprecision(3);

        for (const auto size : {std::pair<int, int>{320, 240}, {640, 480}, {1280, 720}, {1920, 1080}})
        {
            benchmarkSize(size.first, size.second, runs);
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Pipeline benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
