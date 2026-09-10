#include "filters/grayscale.hpp"

#include "core/cuda_check.hpp"

#include <stdexcept>

namespace
{
__device__ int calculateGray(int red, int green, int blue)
{
    return (299 * red + 587 * green + 114 * blue) / 1000;
}

__global__ void grayscaleKernel(
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
    const int gray = calculateGray(input[index], input[index + 1], input[index + 2]);
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
        throw std::invalid_argument("Grayscale expects equally sized RGB input and output images");
    }
}
}

void launchGrayscale(ConstImageView input, ImageView output, cudaStream_t stream)
{
    validateRgbPair(input, output);
    constexpr dim3 threads(16, 16);
    const dim3 blocks(
        (input.width + threads.x - 1) / threads.x,
        (input.height + threads.y - 1) / threads.y);

    grayscaleKernel<<<blocks, threads, 0, stream>>>(
        input.width, input.height, input.data, output.data);
    CUDA_CHECK(cudaGetLastError());
}

