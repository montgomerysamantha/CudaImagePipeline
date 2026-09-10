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

// 256 threads cooperatively load a 36x35 halo tile (5,040 bytes).
// Pack RGB into one 32-bit word and threshold once per source sample.
__global__ void heartBokehSharedKernel(ConstImageView input, ImageView output,
    unsigned char threshold, float intensity)
{
    __shared__ unsigned int tile[35][36];
    const int thread = threadIdx.y * 16 + threadIdx.x;
    for (int cell = thread; cell < 35 * 36; cell += 256)
    {
        const int y = cell / 36;
        const int x = cell % 36;
        const int sourceRow = blockIdx.y * 16 + y - 11;
        const int sourceColumn = blockIdx.x * 16 + x - 10;
        unsigned int packed = 0;
        if (sourceRow >= 0 && sourceRow < input.height &&
            sourceColumn >= 0 && sourceColumn < input.width)
        {
            const std::size_t index =
                (static_cast<std::size_t>(sourceRow) * input.width + sourceColumn) * 3;
            const unsigned int r = input.data[index];
            const unsigned int g = input.data[index+1];
            const unsigned int b = input.data[index+2];
            if ((77*r + 150*g + 29*b + 128)/256 >= threshold)
                packed = r | (g << 8) | (b << 16);
        }
        tile[y][x] = packed;
    }
    // Partial blocks must also participate before any thread returns.
    __syncthreads();
    const int column = blockIdx.x * 16 + threadIdx.x;
    const int row = blockIdx.y * 16 + threadIdx.y;
    if (column >= input.width || row >= input.height) return;
    int sums[3] = {0,0,0};
    for (int y = 0; y < 20; ++y)
        for (int x = 0; x < 21; ++x)
        {
            if (aperture[y][x] != '#') continue;
            const unsigned int sample = tile[threadIdx.y + 19 - y][threadIdx.x + 20 - x];
            sums[0] += sample & 255;
            sums[1] += (sample >> 8) & 255;
            sums[2] += (sample >> 16) & 255;
        }
    const std::size_t index = (static_cast<std::size_t>(row) * input.width + column) * 3;
    for (int channel = 0; channel < 3; ++channel)
    {
        const int scaled = static_cast<int>(roundf(sums[channel] * intensity));
        output.data[index+channel] =
            static_cast<unsigned char>(min(255, input.data[index+channel] + scaled));
    }
}

} // namespace

static void launchBokeh(ConstImageView input, ImageView output,
    unsigned char brightnessThreshold, float intensity, cudaStream_t stream, bool shared)
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
    if (shared)
        heartBokehSharedKernel<<<blocks, threads, 0, stream>>>(
            input, output, brightnessThreshold, intensity);
    else
        heartBokehKernel<<<blocks, threads, 0, stream>>>(
            input, output, brightnessThreshold, intensity);
    CUDA_CHECK(cudaGetLastError());
}

void launchHeartBokehGlobal(ConstImageView a, ImageView b, unsigned char t, float i, cudaStream_t s)
{ launchBokeh(a,b,t,i,s,false); }

void launchHeartBokehShared(ConstImageView a, ImageView b, unsigned char t, float i, cudaStream_t s)
{ launchBokeh(a,b,t,i,s,true); }

void launchHeartBokeh(ConstImageView a, ImageView b, unsigned char t, float i, cudaStream_t s)
{ launchHeartBokehShared(a,b,t,i,s); }
