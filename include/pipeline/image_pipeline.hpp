#pragma once

#include "core/device_image.hpp"
#include "core/image.hpp"

#include <cuda_runtime.h>

struct PipelineOptions
{
    bool grayscale = true;
    bool gaussianBlur = true;
    bool edgeDetection = false;
    bool sharpen = false;
    bool resize = false;

    float sharpenStrength = 1.0f;
    int outputWidth = 0;
    int outputHeight = 0;

    // Runs after Gaussian blur and before edge detection.
    bool heartBokeh = false;
    unsigned char bokehThreshold = 200;
    float bokehIntensity = 0.05f;
};

struct PipelineTimings
{
    float uploadMs = 0.0f;
    float processingMs = 0.0f;
    float downloadMs = 0.0f;
};

class ImagePipeline
{
public:
    explicit ImagePipeline(PipelineOptions options = {});
    ~ImagePipeline();

    ImagePipeline(const ImagePipeline&) = delete;
    ImagePipeline& operator=(const ImagePipeline&) = delete;

    HostImage process(const HostImage& input, PipelineTimings* timings = nullptr);

private:
    void ensureBuffers(int width, int height, int channels);

    PipelineOptions options_;
    DeviceImage bufferA_;
    DeviceImage bufferB_;
    cudaStream_t stream_ = nullptr;
    int bufferWidth_ = 0;
    int bufferHeight_ = 0;
    int bufferChannels_ = 0;
};
