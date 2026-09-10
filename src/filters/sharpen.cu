#include "filters/sharpen.hpp"

#include "core/cuda_check.hpp"

#include <stdexcept>

namespace
{

__device__ unsigned char calculateNeighborhood(
    int x, int y, int width, int height,
    const unsigned char* input, int channel, float strength)
{
    int left  = max(x - 1, 0);
    int right = min(x + 1, width - 1);
    int up    = max(y - 1, 0);
    int down  = min(y + 1, height - 1);

    float center = input[(y * width + x) * 3 + channel];
    float west   = input[(y * width + left) * 3 + channel];
    float value = center + strength *
    (4.0f * center - north - south - east - west);

    output[index] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, value)));
}

__global__ void sharpenKernel(
    int width,
    int height,
    const unsigned char* input,
    unsigned char* output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    const int index = (y * width + x) * 3;
    const int gray = calculateNeighborhood(x, y);
    output[index] = static_cast<unsigned char>(gray);
    output[index + 1] = static_cast<unsigned char>(gray);
    output[index + 2] = static_cast<unsigned char>(gray);
}

void validateRgbPair(ConstImageView input, ImageView output)
{
    if (input.data == nullptr || output.data == nullptr ||
        input.width != output.width || input.height != output.height ||
        input.channels != 3 || output.channels != 3)
    {
        throw std::invalid_argument("Sharpen expects equally sized RGB input and output images");
    }
}
}

void launchSharpen(ConstImageView input, ImageView output, float strength, cudaStream_t stream)
{
    validateRgbPair(input, output);
    constexpr dim3 threads(16, 16);
    const dim3 blocks(
        (input.width + threads.x - 1) / threads.x,
        (input.height + threads.y - 1) / threads.y);

    sharpenKernel<<<blocks, threads, 0, stream>>>(
        input.width, input.height, input.data, output.data);
    CUDA_CHECK(cudaGetLastError());
}

