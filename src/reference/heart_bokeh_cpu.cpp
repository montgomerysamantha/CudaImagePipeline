#include "reference/heart_bokeh_cpu.hpp"

#include <cmath>
#include <iterator>

namespace
{

// A '#' marks an open aperture sample and a '.' marks a blocked sample.
constexpr int apertureWidth = 29;
constexpr int apertureHeight = 14;

// Each row has apertureWidth visible characters plus its terminating null
// character. Access a position with heartAperture[row][column].
constexpr char heartAperture[apertureHeight][apertureWidth + 1] =
{
    ".....######.......######.....",
    "...##########...##########...",
    ".#############.#############.",
    "#############################",
    "#############################",
    "#############################",
    ".###########################.",
    "...#######################...",
    ".....###################.....",
    ".......###############.......",
    ".........###########.........",
    "...........#######...........",
    ".............###.............",
    "..............#.............."
};

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

    for (int row = 0; row < apertureHeight; ++row)
    {
        for (int column = 0; column < apertureWidth; ++column)
        {
            if (heartAperture[row][column] == '#')
            {
                columnSum += column;
                rowSum += row;
                ++openCellCount;
            }
        }
    }

    const int anchorColumn =
        static_cast<int>(std::round(
            static_cast<double>(columnSum) / openCellCount));

    const int anchorRow =
        static_cast<int>(std::round(
            static_cast<double>(rowSum) / openCellCount));

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

    if (position.row < 0 || position.row >= height ||
        position.column < 0 || position.column >= width)
    {
        return false;
    }

    return heartAperture[position.row][position.column] == '#';
}

// TODO: Validate that input and output are non-null, equally sized RGB images.
// TODO: Validate that intensity is in the range you decide to support.
void validateArguments(...)
{

}

// TODO: Calculate a pixel's luminance for comparison with brightnessThreshold.
// unsigned char luminance(...);

// TODO: Gather bright neighboring pixels selected by the heart aperture.
// TODO: Combine the gathered highlight with the original pixel.
// void processPixel(...);

} // namespace

namespace reference
{

// TODO: Define heartBokeh after completing the helpers above.
//
// Suggested outline:
//   1. Validate the arguments.
//   2. Visit every output pixel.
//   3. Process that pixel using the heart aperture.

} // namespace reference
