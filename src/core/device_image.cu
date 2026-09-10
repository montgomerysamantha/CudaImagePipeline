#include "core/device_image.hpp"

#include "core/cuda_check.hpp"

#include <stdexcept>
#include <utility>

namespace
{
void requireSameShape(ConstImageView source, ConstImageView destination)
{
    if (source.width != destination.width ||
        source.height != destination.height ||
        source.channels != destination.channels)
    {
        throw std::invalid_argument("Host and device images must have the same shape");
    }
}
}

DeviceImage::DeviceImage(int width, int height, int channels)
{
    allocate(width, height, channels);
}

DeviceImage::~DeviceImage()
{
    reset();
}

DeviceImage::DeviceImage(DeviceImage&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      width_(std::exchange(other.width_, 0)),
      height_(std::exchange(other.height_, 0)),
      channels_(std::exchange(other.channels_, 0))
{
}

DeviceImage& DeviceImage::operator=(DeviceImage&& other) noexcept
{
    if (this != &other)
    {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        width_ = std::exchange(other.width_, 0);
        height_ = std::exchange(other.height_, 0);
        channels_ = std::exchange(other.channels_, 0);
    }
    return *this;
}

void DeviceImage::allocate(int width, int height, int channels)
{
    if (width <= 0 || height <= 0 || channels <= 0)
    {
        throw std::invalid_argument("Device image dimensions must be positive");
    }

    reset();
    width_ = width;
    height_ = height;
    channels_ = channels;

    try
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()));
    }
    catch (...)
    {
        width_ = height_ = channels_ = 0;
        throw;
    }
}

void DeviceImage::reset() noexcept
{
    if (data_ != nullptr)
    {
        cudaFree(data_);
    }
    data_ = nullptr;
    width_ = height_ = channels_ = 0;
}

ImageView DeviceImage::view()
{
    return {data_, width_, height_, channels_};
}

ConstImageView DeviceImage::view() const
{
    return {data_, width_, height_, channels_};
}

std::size_t DeviceImage::bytes() const
{
    return static_cast<std::size_t>(width_) * height_ * channels_;
}

void DeviceImage::uploadAsync(ConstImageView source, cudaStream_t stream)
{
    const ConstImageView destination{data_, width_, height_, channels_};
    requireSameShape(source, destination);
    CUDA_CHECK(cudaMemcpyAsync(data_, source.data, bytes(), cudaMemcpyHostToDevice, stream));
}

void DeviceImage::downloadAsync(ImageView destination, cudaStream_t stream) const
{
    const ConstImageView hostDestination{
        destination.data, destination.width, destination.height, destination.channels};
    requireSameShape(view(), hostDestination);
    CUDA_CHECK(cudaMemcpyAsync(destination.data, data_, bytes(), cudaMemcpyDeviceToHost, stream));
}
