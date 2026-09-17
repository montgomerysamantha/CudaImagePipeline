#include "reference/sharpen_cpu.hpp"

#include <algorithm>  // std::min, std::max
#include <cmath>      // floating-point min/max
#include <cstddef>
#include <stdexcept>

namespace
{

void validateRgbPair(ConstImageView input, ImageView output)
{
    if (input.data == nullptr || output.data == nullptr ||
        input.width <= 0 || input.height <= 0 ||
        input.width != output.width || input.height != output.height ||
        input.channels != 3 || output.channels != 3)
    {
        throw std::invalid_argument(
            "CPU sharpen expects equally sized RGB input and output images"
        );
    }

    if (input.data == output.data)
    {
        throw std::invalid_argument("CPU sharpen expects different input and output pointers");
    }
}

unsigned char calculateNeighborhood(
    int x, int y, int width, int height,
    const unsigned char* input, int channel, float strength)
{
    int left  = std::max(x - 1, 0);
    int right = std::min(x + 1, width - 1);
    int up    = std::max(y - 1, 0);
    int down  = std::min(y + 1, height - 1);

    float center = input[(static_cast<std::size_t>(y) * width + x) * 3 + channel];
    float east  = input[(static_cast<std::size_t>(y) * width + right) * 3 + channel];
    float west   = input[(static_cast<std::size_t>(y) * width + left) * 3 + channel];
    float north = input[(static_cast<std::size_t>(up) * width + x) * 3 + channel];
    float south = input[(static_cast<std::size_t>(down) * width + x) * 3 + channel];
    float value = center + strength *
    (4.0f * center - north - south - east - west);

    return static_cast<unsigned char>(
    fminf(255.0f, fmaxf(0.0f, value)));
}

void sharpen(ConstImageView input, ImageView output, float strength)
{
    for (int y = 0; y < input.height; y++)
    {
        for (int x = 0; x < input.width; x++)
        {
            const std::size_t index = (static_cast<std::size_t>(y) * input.width + x) * 3;

            for (int channel = 0; channel < 3; channel++)
            {
                output.data[index + channel] = calculateNeighborhood(
                    x, y, input.width, input.height, input.data, channel, strength);
            }
        }
    }
}

} // namespace

namespace reference
{
    void launchSharpenCPU(ConstImageView input, ImageView output, float strength)
    {
        validateRgbPair(input, output);

        sharpen(input, output, strength);
    }

} // namespace reference