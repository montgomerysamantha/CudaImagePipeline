#pragma once

#include "core/cuda_check.hpp"
#include "core/device_image.hpp"
#include "core/image.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace test
{

inline void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

inline HostImage makeRgbImage(
    int width,
    int height,
    std::vector<unsigned char> pixels)
{
    HostImage image;
    image.width = width;
    image.height = height;
    image.channels = 3;
    image.pixels = std::move(pixels);

    require(
        image.pixels.size() == image.view().bytes(),
        "Test pixel count does not match the image dimensions"
    );

    return image;
}

class CudaStream
{
public:
    CudaStream()
    {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaStream()
    {
        if (stream_ != nullptr)
        {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const
    {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

template <typename Launcher>
HostImage runFilter(const HostImage& input, Launcher launcher)
{
    CudaStream stream;
    DeviceImage deviceInput(input.width, input.height, input.channels);
    DeviceImage deviceOutput(input.width, input.height, input.channels);

    HostImage output;
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.pixels.resize(input.pixels.size());

    deviceInput.uploadAsync(input.view(), stream.get());

    const DeviceImage& readOnlyInput = deviceInput;
    launcher(readOnlyInput.view(), deviceOutput.view(), stream.get());

    deviceOutput.downloadAsync(output.view(), stream.get());
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    return output;
}

inline void requirePixelsEqual(
    const HostImage& actual,
    const HostImage& expected,
    const std::string& testName)
{
    require(
        actual.width == expected.width &&
        actual.height == expected.height &&
        actual.channels == expected.channels,
        testName + ": output shape does not match"
    );

    require(
        actual.pixels == expected.pixels,
        testName + ": output pixels do not match"
    );
}

} // namespace test
