#include "reference/gaussian_blur_cpu.hpp"

#include <stdexcept>

namespace
{

constexpr int weights[3][3] =
{
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

void validateRgbPair(ConstImageView input, ImageView output)
{
    if (input.data == nullptr || output.data == nullptr ||
        input.width <= 0 || input.height <= 0 ||
        input.width != output.width || input.height != output.height ||
        input.channels != 3 || output.channels != 3)
    {
        throw std::invalid_argument(
            "CPU Gaussian blur expects equally sized RGB input and output images"
        );
    }
}

void blurPixel(
    int x,
    int y,
    ConstImageView input,
    ImageView output)
{
    int sums[3] = {0, 0, 0};
    int weightSum = 0;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int neighborX = x + dx;
            const int neighborY = y + dy;

            if (neighborX < 0 || neighborX >= input.width ||
                neighborY < 0 || neighborY >= input.height)
            {
                continue;
            }

            const int weight = weights[dy + 1][dx + 1];
            const int neighborIndex =
                (neighborY * input.width + neighborX) * 3;

            for (int channel = 0; channel < 3; ++channel)
            {
                sums[channel] += input.data[neighborIndex + channel] * weight;
            }

            weightSum += weight;
        }
    }

    const int outputIndex = (y * input.width + x) * 3;

    for (int channel = 0; channel < 3; ++channel)
    {
        output.data[outputIndex + channel] =
            static_cast<unsigned char>(sums[channel] / weightSum);
    }
}

} // namespace

namespace reference
{

void gaussianBlur(ConstImageView input, ImageView output)
{
    validateRgbPair(input, output);

    for (int y = 0; y < input.height; ++y)
    {
        for (int x = 0; x < input.width; ++x)
        {
            blurPixel(x, y, input, output);
        }
    }
}

} // namespace reference
