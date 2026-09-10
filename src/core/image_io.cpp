#include "core/image.hpp"

#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

HostImage loadRgbImage(const std::string& path)
{
    int width = 0;
    int height = 0;
    int originalChannels = 0;
    constexpr int requestedChannels = 3;

    unsigned char* loaded = stbi_load(
        path.c_str(), &width, &height, &originalChannels, requestedChannels);

    if (loaded == nullptr)
    {
        throw std::runtime_error("Failed to load image '" + path + "': " + stbi_failure_reason());
    }

    HostImage image;
    image.width = width;
    image.height = height;
    image.channels = requestedChannels;
    const std::size_t byteCount = image.view().bytes();
    image.pixels.assign(loaded, loaded + byteCount);
    stbi_image_free(loaded);
    return image;
}

void savePngImage(const std::string& path, const HostImage& image)
{
    if (image.pixels.size() != image.view().bytes())
    {
        throw std::invalid_argument("Image pixel count does not match its dimensions");
    }

    const int stride = image.width * image.channels;
    if (stbi_write_png(
            path.c_str(), image.width, image.height, image.channels,
            image.pixels.data(), stride) == 0)
    {
        throw std::runtime_error("Failed to save image '" + path + "'");
    }
}

