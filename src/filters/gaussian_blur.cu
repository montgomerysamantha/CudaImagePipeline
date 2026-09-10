#include "filters/gaussian_blur.hpp"

#include "core/cuda_check.hpp"

#include <stdexcept>

namespace
{
constexpr int blockWidth = 16;
constexpr int blockHeight = 16;
constexpr int radius = 1;
constexpr int tileWidth = blockWidth + radius * 2;
constexpr int tileHeight = blockHeight + radius * 2;

__global__ void gaussianBlurKernel(
    int width,
    int height,
    const unsigned char* input,
    unsigned char* output)
{
    __shared__ unsigned char tile[tileHeight][tileWidth][3];

    // Every block cooperatively fills its complete tile, including halos. This also
    // works for partial blocks at the right and bottom edges of the image.
    for (int tileY = threadIdx.y; tileY < tileHeight; tileY += blockDim.y)
    {
        for (int tileX = threadIdx.x; tileX < tileWidth; tileX += blockDim.x)
        {
            const int globalX = blockIdx.x * blockDim.x + tileX - radius;
            const int globalY = blockIdx.y * blockDim.y + tileY - radius;

            for (int channel = 0; channel < 3; ++channel)
            {
                unsigned char value = 0;
                if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height)
                {
                    value = input[(globalY * width + globalX) * 3 + channel];
                }
                tile[tileY][tileX][channel] = value;
            }
        }
    }
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    constexpr int weights[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    int sums[3] = {0, 0, 0};
    int weightSum = 0;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int neighborX = x + dx;
            const int neighborY = y + dy;
            if (neighborX < 0 || neighborX >= width || neighborY < 0 || neighborY >= height)
            {
                continue;
            }

            const int weight = weights[dy + radius][dx + radius];
            for (int channel = 0; channel < 3; ++channel)
            {
                sums[channel] += tile[threadIdx.y + radius + dy]
                                     [threadIdx.x + radius + dx][channel] * weight;
            }
            weightSum += weight;
        }
    }

    const int outputIndex = (y * width + x) * 3;
    for (int channel = 0; channel < 3; ++channel)
    {
        output[outputIndex + channel] =
            static_cast<unsigned char>(sums[channel] / weightSum);
    }
}

void validateRgbPair(ConstImageView input, ImageView output)
{
    if (input.data == nullptr || output.data == nullptr ||
        input.width != output.width || input.height != output.height ||
        input.channels != 3 || output.channels != 3)
    {
        throw std::invalid_argument("Gaussian blur expects equally sized RGB input and output images");
    }
}
}

void launchGaussianBlur(ConstImageView input, ImageView output, cudaStream_t stream)
{
    validateRgbPair(input, output);
    constexpr dim3 threads(blockWidth, blockHeight);
    const dim3 blocks(
        (input.width + threads.x - 1) / threads.x,
        (input.height + threads.y - 1) / threads.y);

    gaussianBlurKernel<<<blocks, threads, 0, stream>>>(
        input.width, input.height, input.data, output.data);
    CUDA_CHECK(cudaGetLastError());
}
