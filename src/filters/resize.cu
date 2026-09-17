#include "filters/resize.hpp"

#include "core/cuda_check.hpp"

#include <stdexcept>
/*
inputWidth  = 5    outputWidth  = 3
inputHeight = 1    outputHeight = 1
Letter:    A  B  C  D  E
Input X:   0  1  2  3  4
Input Y:   0  0  0  0  0

sourceX = floor(outputX × inputWidth  / outputWidth)
sourceY = floor(outputY × inputHeight / outputHeight)

Which source indices do output positions 0, 1, and 2 select?
outputX = 0

sourceX = floor(0 × 5 / 3) = 0
sourceY = floor(0 × 1 / 1) = 0
A

outputX = 1

sourceX = floor(1 × 5 / 3) = 1
sourceY = floor(0 × 1 / 1) = 0

B

outputX = 2

sourceX = floor(2 × 5 / 3) = 3
sourceY = floor(0 × 1 / 1) = 0

D

Resize [A B C] from width 3 to 5:
sourceX = floor(outputX × 3 / 5)

outputX:  0  1  2  3  4
sourceX:  ?  ?  ?  ?  ?

outputX = 0
sourceX = floor(0 × 3 / 5) = 0

outputX = 1
sourceX = floor(1 x 3 / 5) = 0

outputX = 2
sourceX = floor(2 x 3 / 5) = 1

outputX = 3
sourceX = floor(3 x 3 / 5) = 1

outputX = 4
sourceX = floor(4 x 3 / 5) = 2

outputX:  0  1  2  3  4
sourceX:  0  0  1  1  2
*/

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

    output[outputIndex] = static_cast<unsigned char>(r);
    output[outputIndex + 1] = static_cast<unsigned char>(g);
    output[outputIndex + 2] = static_cast<unsigned char>(b);
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

void launchResize(ConstImageView input, ImageView output, cudaStream_t stream)
{
    if (input.data == nullptr || output.data == nullptr)
    {
        throw std::invalid_argument("Resize requires non-null image data");
    }

    if (input.width < 0 || input.height < 0 || output.width < 0 || output.height < 0)
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

