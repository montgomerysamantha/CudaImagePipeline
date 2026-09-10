#include "core/cuda_check.hpp"
#include "filters/heart_bokeh.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>

namespace
{
constexpr int rgbChannels = 3;
constexpr int maximumChannel = 255;
constexpr int bitsPerChannel = 8;
constexpr int greenShift = bitsPerChannel;
constexpr int blueShift = bitsPerChannel * 2;
constexpr int redWeight = 77;
constexpr int greenWeight = 150;
constexpr int blueWeight = 29;
constexpr int luminanceDivisor = 256;
constexpr int luminanceRounding = luminanceDivisor / 2;
constexpr int apertureWidth = 21;
constexpr int apertureHeight = 20;
constexpr int anchorColumn = 10;
constexpr int anchorRow = 8;
constexpr int blockWidth = 16;
constexpr int blockHeight = 16;
constexpr int threadsPerBlock = blockWidth * blockHeight;
// Gathering reverses the mask offsets, hence the asymmetric top halo.
constexpr int haloTop = apertureHeight - 1 - anchorRow;
constexpr int haloLeft = apertureWidth - 1 - anchorColumn;
constexpr int tileWidth = blockWidth + apertureWidth - 1;
constexpr int tileHeight = blockHeight + apertureHeight - 1;

__device__ int calculateLuminance(int red, int green, int blue)
{
    return (redWeight * red + greenWeight * green + blueWeight * blue +
            luminanceRounding) /
           luminanceDivisor;
}

__device__ std::size_t pixelIndex(int row, int column, int width)
{
    return (static_cast<std::size_t>(row) * width + column) * rgbChannels;
}

__device__ bool isInsideImage(int row, int column, ConstImageView input)
{
    return row >= 0 && row < input.height && column >= 0 && column < input.width;
}

// Match CPU rounding before saturation; do not combine multiplication and addition.
__device__ void writeComposite(ConstImageView input, ImageView output, int row,
                               int column, const int *channelSums, float intensity)
{
    const std::size_t index = pixelIndex(row, column, input.width);

    for (int channel = 0; channel < rgbChannels; channel++)
    {
        const int scaledLight =
            static_cast<int>(roundf(channelSums[channel] * intensity));
        output.data[index + channel] = static_cast<unsigned char>(
            min(maximumChannel, input.data[index + channel] + scaledLight));
    }
}

// Same 21x20 mask as the CPU reference; rounded centroid is (column 10, row 8).
// Near-square proportions preserve a natural silhouette in square image pixels.
__constant__ char aperture[apertureHeight][apertureWidth + 1] = {
    "....####.....####....", "..########.########..", ".#########.#########.",
    "##########.##########", "#####################", "#####################",
    "#####################", "#####################", ".###################.",
    ".###################.", "..#################..", "..#################..",
    "...###############...", "....#############....", ".....###########.....",
    "......#########......", ".......#######.......", "........#####........",
    ".........###.........", "..........#.........."};

__global__ void heartBokehKernel(ConstImageView input, ImageView output,
                                 unsigned char threshold, float intensity)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= input.width || row >= input.height)
        return;

    int channelSums[rgbChannels] = {0, 0, 0};

    for (int maskRow = 0; maskRow < apertureHeight; maskRow++)
    {
        for (int maskColumn = 0; maskColumn < apertureWidth; maskColumn++)
        {
            if (aperture[maskRow][maskColumn] != '#')
                continue;
            // Subtract the offset to preserve the asymmetric heart orientation.
            const int sourceRow = row - (maskRow - anchorRow);
            const int sourceColumn = column - (maskColumn - anchorColumn);
            if (!isInsideImage(sourceRow, sourceColumn, input))
            {
                continue;
            }
            const std::size_t index = pixelIndex(sourceRow, sourceColumn, input.width);
            const int red = input.data[index];
            const int green = input.data[index + 1];
            const int blue = input.data[index + 2];
            if (calculateLuminance(red, green, blue) < threshold)
                continue;
            channelSums[0] += red;
            channelSums[1] += green;
            channelSums[2] += blue;
        }
    }
    writeComposite(input, output, row, column, channelSums, intensity);
}

// 256 threads cooperatively load a 36x35 halo tile (5,040 bytes).
// Pack RGB into one 32-bit word and threshold once per source sample.
__global__ void heartBokehSharedKernel(ConstImageView input, ImageView output,
                                       unsigned char threshold, float intensity)
{
    __shared__ unsigned int tile[tileHeight][tileWidth];

    const int linearThreadIndex = threadIdx.y * blockWidth + threadIdx.x;

    for (int tilePixelIndex = linearThreadIndex; tilePixelIndex < tileHeight * tileWidth;
         tilePixelIndex += threadsPerBlock)
    {
        const int tileRow = tilePixelIndex / tileWidth;
        const int tileColumn = tilePixelIndex % tileWidth;
        const int sourceRow = blockIdx.y * blockHeight + tileRow - haloTop;
        const int sourceColumn = blockIdx.x * blockWidth + tileColumn - haloLeft;
        unsigned int packedHighlight = 0;
        if (isInsideImage(sourceRow, sourceColumn, input))
        {
            const std::size_t index = pixelIndex(sourceRow, sourceColumn, input.width);
            const unsigned int red = input.data[index];
            const unsigned int green = input.data[index + 1];
            const unsigned int blue = input.data[index + 2];
            if (calculateLuminance(red, green, blue) >= threshold)
                packedHighlight = red | (green << greenShift) | (blue << blueShift);
        }
        tile[tileRow][tileColumn] = packedHighlight;
    }
    // Partial blocks must also participate before any thread returns.
    __syncthreads();

    const int column = blockIdx.x * blockWidth + threadIdx.x;

    const int row = blockIdx.y * blockHeight + threadIdx.y;
    if (column >= input.width || row >= input.height)
        return;

    int channelSums[rgbChannels] = {0, 0, 0};

    for (int maskRow = 0; maskRow < apertureHeight; maskRow++)
    {
        for (int maskColumn = 0; maskColumn < apertureWidth; maskColumn++)
        {
            if (aperture[maskRow][maskColumn] != '#')
                continue;
            const unsigned int sample =
                tile[threadIdx.y + apertureHeight - 1 - maskRow]
                    [threadIdx.x + apertureWidth - 1 - maskColumn];
            channelSums[0] += sample & maximumChannel;
            channelSums[1] += (sample >> greenShift) & maximumChannel;
            channelSums[2] += (sample >> blueShift) & maximumChannel;
        }
    }

    writeComposite(input, output, row, column, channelSums, intensity);
}

} // namespace

static void launchBokeh(ConstImageView input, ImageView output,
                        unsigned char brightnessThreshold, float intensity,
                        cudaStream_t stream, bool useSharedMemory)
{
    if (!input.data || !output.data || input.width <= 0 || input.height <= 0 ||
        input.width != output.width || input.height != output.height ||
        input.channels != rgbChannels || output.channels != rgbChannels)
        throw std::invalid_argument(
            "Heart bokeh expects equally sized positive RGB images");
    if (!std::isfinite(intensity) || intensity < 0 || intensity > 1)
        throw std::invalid_argument("Heart bokeh intensity must be finite and in [0,1]");

    const auto less = std::less<const unsigned char *>{};
    if (less(input.data, output.data + output.bytes()) &&
        less(output.data, input.data + input.bytes()))
        throw std::invalid_argument("Heart bokeh requires non-overlapping buffers");

    const dim3 threads(blockWidth, blockHeight);

    const dim3 blocks((input.width - 1) / blockWidth + 1,
                      (input.height - 1) / blockHeight + 1);
    if (useSharedMemory)
        heartBokehSharedKernel<<<blocks, threads, 0, stream>>>(
            input, output, brightnessThreshold, intensity);
    else
        heartBokehKernel<<<blocks, threads, 0, stream>>>(input, output,
                                                         brightnessThreshold, intensity);

    CUDA_CHECK(cudaGetLastError());
}

void launchHeartBokehGlobal(ConstImageView input, ImageView output,
                            unsigned char brightnessThreshold, float intensity,
                            cudaStream_t stream)
{
    launchBokeh(input, output, brightnessThreshold, intensity, stream, false);
}

void launchHeartBokehShared(ConstImageView input, ImageView output,
                            unsigned char brightnessThreshold, float intensity,
                            cudaStream_t stream)
{
    launchBokeh(input, output, brightnessThreshold, intensity, stream, true);
}

void launchHeartBokeh(ConstImageView input, ImageView output,
                      unsigned char brightnessThreshold, float intensity,
                      cudaStream_t stream)
{
    launchHeartBokehShared(input, output, brightnessThreshold, intensity, stream);
}
