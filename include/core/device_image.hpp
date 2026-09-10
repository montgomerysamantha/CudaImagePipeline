#pragma once

#include "core/image.hpp"

#include <cuda_runtime.h>

#include <cstddef>

class DeviceImage
{
public:
    DeviceImage() = default;
    DeviceImage(int width, int height, int channels);
    ~DeviceImage();

    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;

    DeviceImage(DeviceImage&& other) noexcept;
    DeviceImage& operator=(DeviceImage&& other) noexcept;

    void allocate(int width, int height, int channels);
    void reset() noexcept;

    ImageView view();
    ConstImageView view() const;
    std::size_t bytes() const;

    void uploadAsync(ConstImageView source, cudaStream_t stream);
    void downloadAsync(ImageView destination, cudaStream_t stream) const;

private:
    unsigned char* data_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
};

