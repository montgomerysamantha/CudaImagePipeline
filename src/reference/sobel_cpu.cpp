#include "reference/sobel_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{

constexpr int sobelX[3][3] =
{
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

constexpr int sobelY[3][3] =
{
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

struct SobelGradients
{
    int gx;
    int gy;
};

void validateRgbPair(ConstImageView input, ImageView output)
{
    if (input.data == nullptr || output.data == nullptr ||
        input.width < 3 || input.height < 3 ||
        input.width != output.width || input.height != output.height ||
        input.channels != 3 || output.channels != 3)
    {
        throw std::invalid_argument(
            "CPU Sobel expects equally sized RGB images of at least 3 x 3 pixels"
        );
    }
}

bool isBorder(int x, int y, int width, int height)
{
    return x == 0 || y == 0 || x == width - 1 || y == height - 1;
}

SobelGradients calculateGradients(
    int x,
    int y,
    ConstImageView input)
{
    SobelGradients gradients{0, 0};

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int neighborIndex =
                ((y + dy) * input.width + (x + dx)) * 3;
            const int gray = input.data[neighborIndex];

            gradients.gx += gray * sobelX[dy + 1][dx + 1];
            gradients.gy += gray * sobelY[dy + 1][dx + 1];
        }
    }

    return gradients;
}

unsigned char calculateEdge(SobelGradients gradients)
{
    const float magnitude = std::sqrt(
        static_cast<float>(
            gradients.gx * gradients.gx +
            gradients.gy * gradients.gy
        )
    );

    return static_cast<unsigned char>(
        std::min(255.0f, magnitude)
    );
}

void writeGrayscalePixel(
    unsigned char value,
    ImageView output,
    int index)
{
    output.data[index] = value;
    output.data[index + 1] = value;
    output.data[index + 2] = value;
}

} // namespace

namespace reference
{

void sobel(ConstImageView input, ImageView output)
{
    validateRgbPair(input, output);

    for (int y = 0; y < input.height; ++y)
    {
        for (int x = 0; x < input.width; ++x)
        {
            const int outputIndex = (y * input.width + x) * 3;

            if (isBorder(x, y, input.width, input.height))
            {
                writeGrayscalePixel(0, output, outputIndex);
                continue;
            }

            const unsigned char edge =
                calculateEdge(calculateGradients(x, y, input));

            writeGrayscalePixel(edge, output, outputIndex);
        }
    }
}

} // namespace reference
