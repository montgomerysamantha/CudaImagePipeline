#include "pipeline/image_pipeline.hpp"

#include "core/cuda_check.hpp"
#include "filters/edge_detection.hpp"
#include "filters/gaussian_blur.hpp"
#include "filters/grayscale.hpp"
#include "filters/heart_bokeh.hpp"
#include "filters/resize.hpp"
#include "filters/sharpen.hpp"

#include <stdexcept>
#include <cmath>
#include <utility>

namespace
{
class Event
{
public:
    Event()
    {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~Event()
    {
        if (event_ != nullptr)
        {
            cudaEventDestroy(event_);
        }
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    operator cudaEvent_t() const
    {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

ConstImageView readView(const DeviceImage& image)
{
    return image.view();
}
}

ImagePipeline::ImagePipeline(PipelineOptions options)
    : options_(options)
{
    if (options_.resize && (options_.outputWidth <= 0 || options_.outputHeight <= 0))
    {
        throw std::invalid_argument("Resize output dimensions must be positive");
    }
    if (options_.heartBokeh &&
        (!std::isfinite(options_.bokehIntensity) ||
         options_.bokehIntensity < 0 || options_.bokehIntensity > 1))
    {
        throw std::invalid_argument("Bokeh intensity must be finite and in [0,1]");
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

ImagePipeline::~ImagePipeline()
{
    if (stream_ != nullptr)
    {
        cudaStreamDestroy(stream_);
    }
}

void ImagePipeline::ensureBuffers(int width, int height, int channels)
{
    if (width == bufferWidth_ && height == bufferHeight_ && channels == bufferChannels_)
    {
        return;
    }

    bufferA_.allocate(width, height, channels);
    bufferB_.allocate(width, height, channels);
    bufferWidth_ = width;
    bufferHeight_ = height;
    bufferChannels_ = channels;
}

HostImage ImagePipeline::process(const HostImage& input, PipelineTimings* timings)
{
    if (input.width <= 0 || input.height <= 0 || input.channels != 3 ||
        input.pixels.size() != input.view().bytes())
    {
        throw std::invalid_argument("The pipeline expects a valid, tightly packed RGB image");
    }

    if (options_.resize)
    {
        // Keep this independent of the input-size cache: A/B retain their
        // original dimensions, while this buffer owns the final output shape.
        const auto resizedView = resizedBuffer_.view();
        if (resizedView.width != options_.outputWidth ||
            resizedView.height != options_.outputHeight ||
            resizedView.channels != input.channels)
        {
            resizedBuffer_.allocate(options_.outputWidth, options_.outputHeight, input.channels);
        }
    }

    ensureBuffers(input.width, input.height, input.channels);

    HostImage output;
    output.allocate(
        options_.resize ? options_.outputWidth : input.width,
        options_.resize ? options_.outputHeight : input.height,
        input.channels
    );

    Event start;
    Event uploaded;
    Event processed;
    Event downloaded;

    CUDA_CHECK(cudaEventRecord(start, stream_));
    bufferA_.uploadAsync(input.view(), stream_);
    CUDA_CHECK(cudaEventRecord(uploaded, stream_));

    DeviceImage* current = &bufferA_;
    DeviceImage* next = &bufferB_;
    auto advance = [&]() { std::swap(current, next); };

    if (options_.grayscale)
    {
        launchGrayscale(readView(*current), next->view(), stream_);
        advance();
    }

    if (options_.gaussianBlur)
    {
        launchGaussianBlur(readView(*current), next->view(), stream_);
        advance();
    }

    if (options_.heartBokeh)
    {
        launchHeartBokeh(readView(*current), next->view(),
            options_.bokehThreshold, options_.bokehIntensity, stream_);
        advance();
    }

    if (options_.edgeDetection)
    {
        launchEdgeDetection(readView(*current), next->view(), stream_);
        advance();
    }

    if (options_.sharpen)
    {
        launchSharpen(readView(*current), next->view(), options_.sharpenStrength, stream_);
        advance();
    }

    if (options_.resize)
    {
        launchResize(readView(*current), resizedBuffer_.view(), stream_);
        current = &resizedBuffer_;
    }

    CUDA_CHECK(cudaEventRecord(processed, stream_));
    current->downloadAsync(output.view(), stream_);
    CUDA_CHECK(cudaEventRecord(downloaded, stream_));
    CUDA_CHECK(cudaEventSynchronize(downloaded));

    if (timings != nullptr)
    {
        CUDA_CHECK(cudaEventElapsedTime(&timings->uploadMs, start, uploaded));
        CUDA_CHECK(cudaEventElapsedTime(&timings->processingMs, uploaded, processed));
        CUDA_CHECK(cudaEventElapsedTime(&timings->downloadMs, processed, downloaded));
    }

    return output;
}
