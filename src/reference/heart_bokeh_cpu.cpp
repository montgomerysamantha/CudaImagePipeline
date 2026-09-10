#include "reference/heart_bokeh_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <stdexcept>

namespace
{

// A '#' marks an open aperture sample and a '.' marks a blocked sample.
constexpr int rgbChannels = 3;
constexpr int maximumChannel = 255;
constexpr int redWeight = 77;
constexpr int greenWeight = 150;
constexpr int blueWeight = 29;
constexpr int luminanceDivisor = 256;
constexpr int luminanceRounding = luminanceDivisor / 2;

std::size_t pixelIndex(int row, int column, int width)
{
    return (static_cast<std::size_t>(row) * width + column) * rgbChannels;
}

constexpr int apertureWidth = 21;
constexpr int apertureHeight = 20;

// Each row has apertureWidth visible characters plus its terminating null
// character. Access a position with heartAperture[row][column].
constexpr char heartAperture[apertureHeight][apertureWidth + 1] = {
    "....####.....####....", "..########.########..", ".#########.#########.",
    "##########.##########", "#####################", "#####################",
    "#####################", "#####################", ".###################.",
    ".###################.", "..#################..", "..#################..",
    "...###############...", "....#############....", ".....###########.....",
    "......#########......", ".......#######.......", "........#####........",
    ".........###.........", "..........#.........."};

struct ApertureCoordinate
{
    int row;
    int column;
};

ApertureCoordinate calcHeartCenter()
{
    int openCellCount = 0;
    int columnSum = 0;
    int rowSum = 0;

    for (int row = 0; row < apertureHeight; row++)
    {
        for (int column = 0; column < apertureWidth; column++)
        {
            if (heartAperture[row][column] == '#')
            {
                columnSum += column;
                rowSum += row;
                openCellCount++;
            }
        }
    }

    const int anchorColumn =
        static_cast<int>(std::round(static_cast<double>(columnSum) / openCellCount));

    const int anchorRow =
        static_cast<int>(std::round(static_cast<double>(rowSum) / openCellCount));

    return {anchorRow, anchorColumn};
}

const ApertureCoordinate apertureAnchor = calcHeartCenter();

// Convert an offset relative to the anchor into a mask row and column.
ApertureCoordinate calcRowCol(int offsetColumn, int offsetRow)
{
    const int row = offsetRow + apertureAnchor.row;
    const int column = offsetColumn + apertureAnchor.column;

    return {row, column};
}

// Return whether that position lies inside the heart aperture.
bool apertureContains(int offsetColumn, int offsetRow)
{
    constexpr int height = static_cast<int>(std::size(heartAperture));
    constexpr int width = static_cast<int>(std::size(heartAperture[0])) - 1;
    const ApertureCoordinate position = calcRowCol(offsetColumn, offsetRow);

    if (position.row < 0 || position.row >= height || position.column < 0 ||
        position.column >= width)
    {
        return false;
    }

    return heartAperture[position.row][position.column] == '#';
}

// Validate that input and output are non-null, equally sized RGB images.
// Validate intensity is within supported range.
void validateArguments(ConstImageView input, ImageView output, float intensity)
{
    if (input.data == nullptr || output.data == nullptr || input.width <= 0 ||
        input.height <= 0 || input.width != output.width ||
        input.height != output.height || input.channels != rgbChannels ||
        output.channels != rgbChannels)
    {
        throw std::invalid_argument("Heart-shaped bokeh filter expects equally sized RGB "
                                    "input and output images");
    }

    if (!std::isfinite(intensity) || intensity < 0.0f || intensity > 1.0f)
    {
        throw std::invalid_argument(
            "Heart-shaped bokeh filter expects intensity between 0.0 and 1.0");
    }

    // std::less provides a total order even for pointers to separate allocations.
    const auto less = std::less<const unsigned char *>{};
    if (less(input.data, output.data + output.bytes()) &&
        less(output.data, input.data + input.bytes()))
    {
        throw std::invalid_argument("Heart bokeh requires non-overlapping buffers");
    }
}

// Calculate a pixel's luminance for comparison with brightnessThreshold.
unsigned char calcLuminance(unsigned char red, unsigned char green, unsigned char blue)
{
    const int pixelLuminance =
        (redWeight * red + greenWeight * green + blueWeight * blue + luminanceRounding) /
        luminanceDivisor;

    return static_cast<unsigned char>(pixelLuminance);
}

// Gather bright neighbors and add their light to the original pixel.
void processPixel(int outputColumn, int outputRow, ConstImageView input, ImageView output,
                  unsigned char brightnessThreshold, float intensity)
{
    int redSum = 0;
    int greenSum = 0;
    int blueSum = 0;

    for (int row = 0; row < apertureHeight; row++)
    {
        for (int column = 0; column < apertureWidth; column++)
        {
            if (heartAperture[row][column] == '.')
            {
                continue;
            }

            const int offsetRow = row - apertureAnchor.row;
            const int offsetColumn = column - apertureAnchor.column;

            const int sourceColumn = outputColumn - offsetColumn;
            const int sourceRow = outputRow - offsetRow;

            if (sourceRow < 0 || sourceRow >= input.height || sourceColumn < 0 ||
                sourceColumn >= input.width)
            {
                continue;
            }

            const std::size_t sourceIndex =
                pixelIndex(sourceRow, sourceColumn, input.width);

            const unsigned char red = input.data[sourceIndex];
            const unsigned char green = input.data[sourceIndex + 1];
            const unsigned char blue = input.data[sourceIndex + 2];

            const unsigned char luminance = calcLuminance(red, green, blue);

            if (luminance < brightnessThreshold)
            {
                continue;
            }

            redSum += red;
            greenSum += green;
            blueSum += blue;
        }
    }

    const std::size_t outputIndex = pixelIndex(outputRow, outputColumn, output.width);

    const auto compositeChannel = [intensity](unsigned char original,
                                              int accumulatedLight) {
        const int scaledLight = static_cast<int>(
            std::round(static_cast<float>(accumulatedLight) * intensity));
        const int combined = static_cast<int>(original) + scaledLight;
        return static_cast<unsigned char>(std::min(combined, maximumChannel));
    };

    output.data[outputIndex] = compositeChannel(input.data[outputIndex], redSum);
    output.data[outputIndex + 1] =
        compositeChannel(input.data[outputIndex + 1], greenSum);
    output.data[outputIndex + 2] = compositeChannel(input.data[outputIndex + 2], blueSum);
}

} // namespace

namespace reference
{

void heartBokeh(ConstImageView input, ImageView output, unsigned char brightnessThreshold,
                float intensity)
{
    validateArguments(input, output, intensity);

    for (int row = 0; row < output.height; row++)
    {
        for (int column = 0; column < output.width; column++)
        {
            processPixel(column, row, input, output, brightnessThreshold, intensity);
        }
    }
}

} // namespace reference
