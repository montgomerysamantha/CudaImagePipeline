#include "filters/edge_detection.hpp"

#include "core/cuda_check.hpp"

#include <stdexcept>

namespace
{
constexpr int blockWidth = 16;
constexpr int blockHeight = 16;
constexpr int radius = 1;
constexpr int tileWidth = blockWidth + radius * 2;
constexpr int tileHeight = blockHeight + radius * 2;

__constant__ int sobelXDevice[3][3] =
{
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int sobelYDevice[3][3] =
{
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

__device__ void writeRgbDevice(
    unsigned char value,
    unsigned char* output,
    int outputIndex)
{
    output[outputIndex] = value;
    output[outputIndex + 1] = value;
    output[outputIndex + 2] = value;
}

struct SobelGradients
{
    int gx;
    int gy;
};

__device__  SobelGradients getGxGyValues(
    int x,
    int y,
    int width,
    const unsigned char* input)
{
    int gx = 0;
    int gy = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            const int neighborX = x + dx;
            const int neighborY = y + dy;

            const int neighborIndex =
                (neighborY * width + neighborX) * 3;

            const int gray = input[neighborIndex];

            gx += gray * sobelXDevice[dy + 1][dx + 1];
            gy += gray * sobelYDevice[dy + 1][dx + 1];
        }
    }

    return {gx, gy};
}

__device__  unsigned char calcEdge(int gx, int gy)
{
    const float magnitude =
        sqrtf(static_cast<float>(
            gx * gx + gy * gy
        ));

    const float clampedMagnitude =
        fminf(255.0f, magnitude);

    return static_cast<unsigned char>(
        clampedMagnitude
    );
}

__device__  bool isBorder(int x, int y, int width, int height)
{
    const bool isBorderCoord =
                x == 0 ||
                y == 0 ||
                x == width - 1 ||
                y == height - 1;

    return isBorderCoord;
}

__global__ void sobelGlobalMemKernel(
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

    const int outputIndex =
        (y * width + x) * 3;

    if (isBorder(x, y, width, height))
    {
        writeRgbDevice(0, output, outputIndex);
        return;
    }

    const SobelGradients gradients =
        getGxGyValues(
            x,
            y,
            width,
            input
        );

    const unsigned char edge =
        calcEdge(gradients.gx, gradients.gy);

    writeRgbDevice(edge, output, outputIndex);
}

__device__ SobelGradients getSharedGradients(
    int sharedX,
    int sharedY,
    const unsigned char tile[tileHeight][tileWidth])
{
    int gx = 0;
    int gy = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            const int gray = tile[sharedY + dy][sharedX + dx];

            gx += gray * sobelXDevice[dy + 1][dx + 1];
            gy += gray * sobelYDevice[dy + 1][dx + 1];
        }
    }

    return {gx, gy};
}

__global__ void sobelSharedMemKernel(
    int width,
    int height,
    const unsigned char* input,
    unsigned char* output)
{
    // output image pixel that this worker owns
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // where owned pixel sits inside the shared tile.
    const int sharedX = threadIdx.x + radius;
    const int sharedY = threadIdx.y + radius;

    // shared grayscale tile.
    __shared__ unsigned char tile[tileHeight][tileWidth];

    // load global pixels into the shared tile and its halo.
    for (int tileY = threadIdx.y; tileY < tileHeight; tileY += blockDim.y)
    {
        for (int tileX = threadIdx.x; tileX < tileWidth; tileX += blockDim.x)
        {
            // find global location of pixel that needs to be saved to tile
            const int globalX = blockIdx.x * blockDim.x + tileX - radius;
            const int globalY = blockIdx.y * blockDim.y + tileY - radius;

            unsigned char value = 0;
            if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height)
            {
                value = input[(globalY * width + globalX) * 3];
            }
            tile[tileY][tileX] = value;
        }
    }

    // wait for the entire block to finish loading pixels
    __syncthreads();

    // return if this thread is outside the image.
    if (x >= width || y >= height)
    {
        return;
    }

    // for image-border pixels, change to black and return
    const int outputIndex = (y * width + x) * 3;
    if (isBorder(x, y, width, height))
    {
        writeRgbDevice(0, output, outputIndex);
        return;
    }

    // calculate Gx and Gy using the shared tile
    const SobelGradients gradients = getSharedGradients(sharedX, sharedY, tile);

    // calculate the edge magnitude and write RGB output
    const unsigned char edge =
        calcEdge(gradients.gx, gradients.gy);

    writeRgbDevice(edge, output, outputIndex);
}

void validateRgbPair(
    ConstImageView input,
    ImageView output)
{
    const bool invalidPointers =
        input.data == nullptr ||
        output.data == nullptr;

    const bool differentDimensions =
        input.width != output.width ||
        input.height != output.height;

    const bool invalidChannels =
        input.channels != 3 ||
        output.channels != 3;

    if (invalidPointers ||
        differentDimensions ||
        invalidChannels)
    {
        throw std::invalid_argument(
            "Edge detection expects equally sized "
            "RGB input and output images"
        );
    }
}

}

void launchEdgeDetection(ConstImageView input, ImageView output, cudaStream_t stream)
{
    validateRgbPair(input, output);
    constexpr dim3 threads(blockWidth, blockHeight);
    const dim3 blocks(
        (input.width + threads.x - 1) / threads.x,
        (input.height + threads.y - 1) / threads.y);

    sobelGlobalMemKernel<<<blocks, threads, 0, stream>>>(
        input.width,
        input.height,
        input.data,
        output.data
    );

    CUDA_CHECK(cudaGetLastError());
}

void launchEdgeDetectionSharedMemory(ConstImageView input, ImageView output, cudaStream_t stream)
{
    validateRgbPair(input, output);
    constexpr dim3 threads(blockWidth, blockHeight);
    const dim3 blocks(
        (input.width + threads.x - 1) / threads.x,
        (input.height + threads.y - 1) / threads.y);

    sobelSharedMemKernel<<<blocks, threads, 0, stream>>>(
        input.width,
        input.height,
        input.data,
        output.data
    );

    CUDA_CHECK(cudaGetLastError());
}
