#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <stdexcept>

struct ImageView
{
    unsigned char* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;

    std::size_t bytes() const
    {
        return static_cast<std::size_t>(width) * height * channels;
    }
};

struct ConstImageView
{
    const unsigned char* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;

    std::size_t bytes() const
    {
        return static_cast<std::size_t>(width) * height * channels;
    }
};

struct HostImage
{
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<unsigned char> pixels;

    ImageView view()
    {
        return {pixels.data(), width, height, channels};
    }

    ConstImageView view() const
    {
        return {pixels.data(), width, height, channels};
    }

    void allocate(int newWidth, int newHeight, int newChannels)
    {
        if (newWidth <= 0 || newHeight <= 0 || newChannels <= 0)
        {
            throw std::invalid_argument("Host image dimensions must be positive");
        }

        // Calculate the required size without overflowing.
        std::size_t count = static_cast<std::size_t>(newWidth);

        if (static_cast<std::size_t>(newHeight) > pixels.max_size() / count)
        {
            throw std::length_error("Image allocation too big");
        }
        count *= newHeight;

        if (static_cast<std::size_t>(newChannels) > pixels.max_size() / count)
        {
            throw std::length_error("Image allocation too big");
        }
        count *= newChannels;

        pixels.resize(count);

        width = newWidth;
        height = newHeight;
        channels = newChannels;
    }
};

HostImage loadRgbImage(const std::string& path);
void savePngImage(const std::string& path, const HostImage& image);

