#pragma once

#include <cstddef>
#include <string>
#include <vector>

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
};

HostImage loadRgbImage(const std::string& path);
void savePngImage(const std::string& path, const HostImage& image);

