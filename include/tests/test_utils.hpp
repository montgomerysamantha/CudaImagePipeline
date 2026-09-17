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

struct RgbPixel
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

template <typename PixelGenerator>
HostImage makeGeneratedRgbImage(
    int width,
    int height,
    PixelGenerator generatePixel)
{
    require(
        width > 0 && height > 0,
        "Generated test image dimensions must be positive"
    );

    std::vector<unsigned char> pixels(
        static_cast<std::size_t>(width) * height * 3
    );

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const RgbPixel pixel = generatePixel(x, y);
            const int index = (y * width + x) * 3;

            pixels[index] = pixel.red;
            pixels[index + 1] = pixel.green;
            pixels[index + 2] = pixel.blue;
        }
    }

    return makeRgbImage(width, height, std::move(pixels));
}

inline HostImage makeSolidRgbImage(
    int width,
    int height,
    RgbPixel color)
{
    return makeGeneratedRgbImage(
        width,
        height,
        [color](int, int)
        {
            return color;
        }
    );
}

inline HostImage makeCoordinatePatternImage(
    int width,
    int height)
{
    return makeGeneratedRgbImage(
        width,
        height,
        [](int x, int y)
        {
            return RgbPixel{
                static_cast<unsigned char>((x * 13 + y * 7) % 256),
                static_cast<unsigned char>((x * 3 + y * 17) % 256),
                static_cast<unsigned char>((x * 11 + y * 5) % 256)
            };
        }
    );
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
    output.allocate(input.width, input.height, input.channels);

    deviceInput.uploadAsync(input.view(), stream.get());

    const DeviceImage& readOnlyInput = deviceInput;
    launcher(readOnlyInput.view(), deviceOutput.view(), stream.get());

    deviceOutput.downloadAsync(output.view(), stream.get());
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    return output;
}

template <typename Launcher>
HostImage runResize(
    const HostImage& input,
    int outputWidth,
    int outputHeight,
    Launcher launcher)
{
    CudaStream stream;
    DeviceImage deviceInput(input.width, input.height, input.channels);
    DeviceImage deviceOutput(outputWidth, outputHeight, input.channels);

    HostImage output;
    output.allocate(outputWidth, outputHeight, input.channels);

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

inline void setRgbPixel(
    HostImage& image,
    int x,
    int y,
    unsigned char red,
    unsigned char green,
    unsigned char blue)
{
    require(
        image.channels == 3,
        "setRgbPixel expects a three-channel image"
    );

    require(
        x >= 0 && x < image.width &&
        y >= 0 && y < image.height,
        "setRgbPixel coordinates are outside the image"
    );

    const int index =
        (y * image.width + x) * 3;

    image.pixels[index] = red;
    image.pixels[index + 1] = green;
    image.pixels[index + 2] = blue;
}

inline void setGrayscalePixel(
    HostImage& image,
    int x,
    int y,
    unsigned char value)
{
    setRgbPixel(
        image,
        x,
        y,
        value,
        value,
        value
    );
}

} // namespace test
