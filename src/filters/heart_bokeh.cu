#include "filters/heart_bokeh.hpp"
#include "core/cuda_check.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>

namespace
{

// Same 21x20 mask as the CPU reference; rounded centroid is (column 10, row 8).
// Near-square proportions preserve a natural silhouette in square image pixels.
__constant__ char aperture[20][22] = {
    "....####.....####....",
    "..########.########..",
    ".#########.#########.",
    "##########.##########",
    "#####################",
    "#####################",
    "#####################",
    "#####################",
    ".###################.",
    ".###################.",
    "..#################..",
    "..#################..",
    "...###############...",
    "....#############....",
    ".....###########.....",
    "......#########......",
    ".......#######.......",
    "........#####........",
    ".........###.........",
    "..........#.........."
};

__global__ void heartBokehKernel(ConstImageView input, ImageView output,
    unsigned char threshold, float intensity)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= input.width || row >= input.height) return;
    int sums[3] = {0, 0, 0};
    for (int maskRow = 0; maskRow < 20; ++maskRow)
    {
        for (int maskColumn = 0; maskColumn < 21; ++maskColumn)
        {
            if (aperture[maskRow][maskColumn] != '#') continue;
            // Subtract the offset to preserve the asymmetric heart orientation.
            const int sourceRow = row - (maskRow - 8);
            const int sourceColumn = column - (maskColumn - 10);
            if (sourceRow < 0 || sourceRow >= input.height ||
                sourceColumn < 0 || sourceColumn >= input.width) continue;
            const std::size_t index =
                (static_cast<std::size_t>(sourceRow) * input.width + sourceColumn) * 3;
            const int red = input.data[index];
            const int green = input.data[index + 1];
            const int blue = input.data[index + 2];
            if ((77 * red + 150 * green + 29 * blue + 128) / 256 < threshold)
                continue;
            sums[0] += red;
            sums[1] += green;
            sums[2] += blue;
        }
    }
    const std::size_t index = (static_cast<std::size_t>(row) * input.width + column) * 3;
    for (int channel = 0; channel < 3; ++channel)
    {
        // Round the product separately before adding, as in the CPU reference.
        const int scaled = static_cast<int>(roundf(sums[channel] * intensity));
        output.data[index + channel] =
            static_cast<unsigned char>(min(255, input.data[index + channel] + scaled));
    }
}

} // namespace

void launchHeartBokeh(ConstImageView input, ImageView output,
    unsigned char brightnessThreshold, float intensity, cudaStream_t stream)
{
    if (!input.data || !output.data || input.width <= 0 || input.height <= 0 ||
        input.width != output.width || input.height != output.height ||
        input.channels != 3 || output.channels != 3)
        throw std::invalid_argument("Heart bokeh expects equally sized positive RGB images");
    if (!std::isfinite(intensity) || intensity < 0 || intensity > 1)
        throw std::invalid_argument("Heart bokeh intensity must be finite and in [0,1]");
    const auto less = std::less<const unsigned char*>{};
    if (less(input.data, output.data + output.bytes()) &&
        less(output.data, input.data + input.bytes()))
        throw std::invalid_argument("Heart bokeh requires non-overlapping buffers");
    const dim3 threads(16, 16);
    const dim3 blocks((input.width - 1) / 16 + 1, (input.height - 1) / 16 + 1);
    heartBokehKernel<<<blocks, threads, 0, stream>>>(
        input, output, brightnessThreshold, intensity);
    CUDA_CHECK(cudaGetLastError());
}
