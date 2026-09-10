#include "reference/grayscale_cpu.hpp"

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
            "CPU grayscale expects equally sized RGB input and output images"
        );
    }
}

} // namespace

namespace reference
{

void grayscale(ConstImageView input, ImageView output)
{
    validateRgbPair(input, output);

    const int pixelCount = input.width * input.height;

    for (int pixel = 0; pixel < pixelCount; ++pixel)
    {
        const int index = pixel * 3;
        const int red = input.data[index];
        const int green = input.data[index + 1];
        const int blue = input.data[index + 2];

        const unsigned char gray =
            static_cast<unsigned char>(
                (299 * red + 587 * green + 114 * blue) / 1000
            );

        output.data[index] = gray;
        output.data[index + 1] = gray;
        output.data[index + 2] = gray;
    }
}

} // namespace reference
