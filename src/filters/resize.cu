#include "filters/resize.hpp"

#include "core/cuda_check.hpp"

#include <cstddef>
#include <stdexcept>

namespace
{
__global__ void resizeKernel(
    int inputWidth,
    int inputHeight,
    const unsigned char* input,
    int outputWidth,
    int outputHeight,
    unsigned char* output)
{
    const int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outputX >= outputWidth || outputY >= outputHeight)
    {
        return;
    }

    const int sourceX = static_cast<int>(static_cast<long long>(outputX) * inputWidth / outputWidth);
    const int sourceY = static_cast<int>(static_cast<long long>(outputY) * inputHeight / outputHeight);

    const std::size_t inputIndex = (static_cast<std::size_t>(sourceY) * inputWidth + sourceX) * 3;
    const std::size_t outputIndex = (static_cast<std::size_t>(outputY) * outputWidth + outputX) * 3;

    // Copy R, G, and B.
    unsigned char r = input[inputIndex];
    unsigned char g = input[inputIndex + 1];
    unsigned char b = input[inputIndex + 2];

    output[outputIndex] = r;
    output[outputIndex + 1] = g;
    output[outputIndex + 2] = b;
}

void verifyInputOutputPointers(ConstImageView input, ImageView output)
{
    if (input.data == nullptr || output.data == nullptr)
    {
        throw std::invalid_argument("Resize requires non-null image data");
    }

    if (input.width <= 0 || input.height <= 0 || output.width <= 0 || output.height <= 0)
    {
        throw std::invalid_argument("Resize requires positive dimensions");
    }

    if (input.channels != 3 || output.channels != 3)
    {
        throw std::invalid_argument("Resize requires RGB images");
    }

    if (input.data == output.data)
    {
        throw std::invalid_argument("Resize requires separate input and output storage");
    }
}

}

void launchResize(ConstImageView input, ImageView output, cudaStream_t stream)
{
    verifyInputOutputPointers(input, output);

    constexpr dim3 threads(16, 16);
    const dim3 blocks(
        (output.width + threads.x - 1) / threads.x,
        (output.height + threads.y - 1) / threads.y);

    resizeKernel<<<blocks, threads, 0, stream>>>
    (   input.width,
        input.height,
        input.data,
        output.width,
        output.height,
        output.data);

    CUDA_CHECK(cudaGetLastError());
}

